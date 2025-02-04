"""
Crypto Momentum Trading Strategy Implementation
Uses aggressive price/volume moves to identify and trade momentum in crypto assets
with strong risk management and market neutrality
"""

import uuid, os, requests
from vault.base_strategy import BaseStrategy, Signal, SignalResponse, Order
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict, Any
from systems.utils import create_logger

class Strategy(BaseStrategy):
    def __init__(self, config_dict=None):
        super().__init__()
        self.logger = create_logger(self.strategy_name)
        self.strategy_name = 'crypto_trader'
        self.granularity = "1m"
        
        # Core Parameters
        self.atr_window = 14  # Window for ATR calculation
        self.volume_window = 30  # Window for volume average
        self.aggressive_move_atr_multiple = 2.0  # Price move threshold
        self.volume_spike_multiple = 3.0  # Volume spike threshold
        self.position_size_limit = 0.10  # Max 10% per position
        
        # Risk Management
        self.initial_stop_atr = 1.5  # Initial stop loss distance in ATR
        self.trailing_stop_activation = 2.0  # Move to break-even after 2x ATR
        self.max_total_exposure = 2.0  # Maximum total exposure
        self.min_liquidity_threshold = 100000  # Minimum 24h volume in USD
        
        # Market Data Requirements
        self.min_history_bars = 100  # Minimum bars needed for signals
        self.update_interval = 60  # Update universe every 60 seconds
        self.last_update = None
        self.universe = []  # Current trading universe
        self.asset_metrics = {}  # Store ATR, volume metrics per asset
        
        # Data inputs and tickers
        self.data_inputs, self.tickers = self.datafeeder_inputs()

    def get_name(self):
        """Return the strategy name"""
        return self.strategy_name

    def datafeeder_inputs(self):
        """Define required data inputs"""
        data_inputs = {
            '1m': {
                'columns': ['open', 'high', 'low', 'close', 'volume'],
                'lookback': self.min_history_bars
            }
        }
        return data_inputs, []  # Tickers will be updated dynamically

    def fetch_kraken_assets(self, sort_order: str) -> pd.DataFrame:
        """Fetch top/bottom assets from Kraken"""
        try:
            url = "https://iapi.kraken.com/api/internal/markets/all/assets"
            params = {
                "sort_by": "change_pct_24h",
                "page": 0,
                "sort_order": sort_order,
                "quote_symbol": "usd",
                "tradable": "true",
                "page_size": 100
            }
            headers = {
                "authority": "iapi.kraken.com",
                "accept": "*/*",
                "origin": "https://www.kraken.com",
                "referer": "https://www.kraken.com/",
                "user-agent": "Mozilla/5.0"
            }
            response = requests.get(url, params=params, headers=headers)
            if response.status_code != 200:
                return pd.DataFrame()
            
            data = response.json()
            assets = data.get("result", {}).get("data", [])
            df = pd.DataFrame(assets)
            return df[["symbol", "price", "volume_24h", "market_cap", "change_pct_24h"]]
            
        except Exception as e:
            self.logger.error(f"Error fetching Kraken assets: {str(e)}")
            return pd.DataFrame()

    def update_universe(self):
        """Update trading universe with top/bottom 100 assets"""
        try:
            symbols_files_list = os.listdir
            
            
            top_100 = self.fetch_kraken_assets("descending")
            bottom_100 = self.fetch_kraken_assets("ascending")
            
            # Combine and filter assets
            all_assets = pd.concat([top_100, bottom_100])
            filtered_assets = all_assets[all_assets['volume_24h'] > self.min_liquidity_threshold]
            
            self.universe = filtered_assets.to_dict('records')
            self.tickers = filtered_assets['symbol'].tolist()
            
            self.logger.info(f"Updated universe with {len(self.universe)} assets")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating universe: {str(e)}")
            return False

    def calculate_atr(self, ohlcv: pd.DataFrame) -> float:
        """Calculate Average True Range"""
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=self.atr_window).mean().iloc[-1]

    def detect_aggressive_move(self, current_data: pd.Series, asset_metrics: Dict) -> Optional[str]:
        """Detect aggressive price and volume moves"""
        if 'atr' not in asset_metrics or 'avg_volume' not in asset_metrics:
            return None
            
        price_change = abs(current_data['close'] - current_data['open'])
        volume = current_data['volume']
        
        # Check for aggressive move
        if price_change > self.aggressive_move_atr_multiple * asset_metrics['atr']:
            if volume > self.volume_spike_multiple * asset_metrics['avg_volume']:
                return "LONG" if current_data['close'] > current_data['open'] else "SHORT"
        
        return None

    def calculate_position_size(self, price: float, atr: float, total_buying_power: float) -> int:
        """Calculate position size with risk management"""
        max_position_value = min(
            total_buying_power * self.position_size_limit,
            total_buying_power * (atr / price)  # Scale by volatility
        )
        return int(max_position_value / price)

    def generate_signals(
        self,
        next_rows,
        market_data_df,
        system_timestamp,
        total_buying_power: float = 0.0,
        buying_power_used: float = 0.0,
        open_signals: Optional[List[Signal]] = None
    ) -> Tuple[Optional[str], List[Signal], List[str]]:
        """Generate trading signals based on momentum"""
        signal_response = SignalResponse(return_type=None, signals=[], tickers=self.tickers)
        
        try:
            # Update universe if needed
            if not self.last_update or (system_timestamp - self.last_update).seconds > self.update_interval:
                if self.update_universe():
                    self.last_update = system_timestamp

            # Process open signals - update trailing stops
            if open_signals:
                for signal in open_signals:
                    for order in signal.orders:
                        if order.status == "open" and order.entryOrderBool:
                            symbol = order.symbol
                            if symbol not in next_rows:
                                continue
                                
                            current_price = next_rows[symbol]['close']
                            atr = self.asset_metrics.get(symbol, {}).get('atr', 0)
                            
                            # Update trailing stop
                            if order.orderDirection == "BUY":
                                profit_distance = current_price - order.filled_price
                                if profit_distance > self.trailing_stop_activation * atr:
                                    new_stop = current_price - self.initial_stop_atr * atr
                                    # Update stop if it would move higher
                                    for stop_order in signal.orders:
                                        if stop_order.order_type == "STOPLOSS" and stop_order.price < new_stop:
                                            stop_order.price = new_stop
                                            stop_order.fresh_update = True
                            else:  # SHORT position
                                profit_distance = order.filled_price - current_price
                                if profit_distance > self.trailing_stop_activation * atr:
                                    new_stop = current_price + self.initial_stop_atr * atr
                                    # Update stop if it would move lower
                                    for stop_order in signal.orders:
                                        if stop_order.order_type == "STOPLOSS" and stop_order.price > new_stop:
                                            stop_order.price = new_stop
                                            stop_order.fresh_update = True

            # Calculate current exposure
            long_exposure = sum(s.buying_power_used for s in (open_signals or [])
                              if any(o.orderDirection == "BUY" and o.status == "open" for o in s.orders))
            short_exposure = sum(s.buying_power_used for s in (open_signals or [])
                               if any(o.orderDirection == "SELL" and o.status == "open" for o in s.orders))

            # Update metrics and check for new signals
            for asset in self.universe:
                symbol = asset['symbol']
                if symbol not in next_rows or symbol not in market_data_df:
                    continue

                # Update asset metrics
                ohlcv_data = market_data_df[symbol]
                self.asset_metrics[symbol] = {
                    'atr': self.calculate_atr(ohlcv_data),
                    'avg_volume': ohlcv_data['volume'].rolling(window=self.volume_window).mean().iloc[-1]
                }

                # Check for aggressive moves
                move_direction = self.detect_aggressive_move(next_rows[symbol], self.asset_metrics[symbol])
                if not move_direction:
                    continue

                # Enforce market neutrality
                if move_direction == "LONG" and long_exposure > short_exposure:
                    continue
                if move_direction == "SHORT" and short_exposure > long_exposure:
                    continue

                # Calculate position size
                current_price = next_rows[symbol]['close']
                position_size = self.calculate_position_size(
                    current_price,
                    self.asset_metrics[symbol]['atr'],
                    total_buying_power
                )

                if position_size > 0:
                    # Generate orders
                    entry_order = Order(
                        symbol=symbol,
                        orderQuantity=position_size,
                        orderDirection="BUY" if move_direction == "LONG" else "SELL",
                        order_type="MARKET",
                        symbol_ltp={system_timestamp: current_price},
                        timeInForce="DAY",
                        status="pending",
                        entryOrderBool=True
                    )

                    # Calculate stop loss level
                    stop_price = current_price * (1 - self.initial_stop_atr) if move_direction == "LONG" else \
                               current_price * (1 + self.initial_stop_atr)

                    stop_order = Order(
                        symbol=symbol,
                        orderQuantity=position_size,
                        orderDirection="SELL" if move_direction == "LONG" else "BUY",
                        order_type="STOPLOSS",
                        price=stop_price,
                        symbol_ltp={system_timestamp: current_price},
                        timeInForce="GTC",
                        status="pending"
                    )

                    # Create signal
                    strategy_order_id = f"{symbol}_{uuid.uuid4().hex}"
                    signal = Signal(
                        strategy_name=self.strategy_name,
                        timestamp=system_timestamp,
                        orders=[entry_order, stop_order],
                        signal_strength=1.0,
                        granularity=self.granularity,
                        signal_type="BUY_SELL",
                        market_neutral=True,
                        total_buying_power=total_buying_power,
                        buying_power_used=position_size * current_price,
                        strategy_inputs={
                            'strategy_order_id': strategy_order_id,
                            'atr': self.asset_metrics[symbol]['atr'],
                            'move_direction': move_direction
                        }
                    )

                    signal_response.signals.append(signal)
                    if not signal_response.return_type:
                        signal_response.return_type = "signals"

            return signal_response.return_type, signal_response.signals, signal_response.tickers

        except Exception as e:
            error_msg = f"Error generating signals: {str(e)}"
            self.logger.error(error_msg)
            return error_msg, [], []

    def set_params(self, config: Dict[str, Any]):
        """Update strategy parameters"""
        if not config:
            return
            
        for param in [
            'atr_window', 'volume_window', 'aggressive_move_atr_multiple',
            'volume_spike_multiple', 'position_size_limit', 'initial_stop_atr',
            'trailing_stop_activation', 'max_total_exposure', 'min_liquidity_threshold'
        ]:
            if param in config:
                setattr(self, param, config[param])