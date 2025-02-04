"""
Simple SMA crossover strategy using 5 and 15 period moving averages on 1-minute data
"""

from vault.base_strategy import BaseStrategy, Signal, Order
import numpy as np
import sys, time
from systems.utils import create_logger, project_path

class Strategy(BaseStrategy):
    def __init__(self, config_dict):
        super().__init__()
        self.logger = create_logger(logger_name='sma_crossover', log_level='INFO')
        self.strategy_name = 'crypto_sma_crossover'
        self.granularity = "1m"
        self.orderType = "MARKET"
        self.stoploss_pct = 0.01  # 1% stoploss for quick moves
        self.exit_order_type = "stoploss_pct"
        self.timeInForce = "DAY"
        self.orderQuantity = 1
        self.data_inputs, self.tickers = self.datafeeder_inputs()
        
    def get_name(self):
        return self.strategy_name
        
    def datafeeder_inputs(self):
        # Get major USD pairs from Kraken 1m data folder
        tickers = [
            'BTC', 'ETH', 'SOL', 'DOT', 'ADA', 'LINK',
            'MATIC', 'ATOM', 'AVAX', 'FIL', 'LTC', 'XRP'
        ]
        data_inputs = {'1m': {'columns': ['open', 'high', 'close', 'low', 'volume'], 'lookback': 20}}
        return data_inputs, tickers
        
    def generate_signals(self, next_rows, market_data_df, system_timestamp, open_signals=None):
        """
        Generate signals based on SMA crossover strategy (5 and 15 period).
        Takes into account any open signals.
        """
        signals = []
        return_type = None
        open_signals = open_signals or []

        for symbol in set(market_data_df["open"].columns):
            if(self.granularity not in market_data_df.index.levels[0] or len(market_data_df.loc[self.granularity]) <= self.data_inputs[self.granularity]['lookback']):
                continue
                
            asset_data_df = market_data_df.loc[self.granularity].xs(symbol, axis=1, level='symbol').reset_index()
            asset_data_df['SMA5'] = asset_data_df['close'].rolling(window=5).mean()
            asset_data_df['SMA15'] = asset_data_df['close'].rolling(window=15).mean()

            current_price = asset_data_df.iloc[-1]['close']
            
            # Generate buy signal when fast SMA crosses above slow SMA
            if (round(asset_data_df.iloc[-1]['SMA5'], 2) > round(asset_data_df.iloc[-1]['SMA15'], 2)) and (round(asset_data_df.iloc[-2]['SMA5'], 2) <= round(asset_data_df.iloc[-2]['SMA15'], 2)):
                signal_strength = 1
                orderDirection = "BUY"
            # Generate sell signal when fast SMA crosses below slow SMA
            elif (round(asset_data_df.iloc[-1]['SMA5'], 2) < round(asset_data_df.iloc[-1]['SMA15'], 2)) and (round(asset_data_df.iloc[-2]['SMA5'], 2) >= round(asset_data_df.iloc[-2]['SMA15'], 2)):
                signal_strength = 1
                orderDirection = "SELL"
            else:
                signal_strength = 0
                
            if signal_strength != 0:
                # Check if we have an open signal for this symbol
                existing_signal = None
                for signal in open_signals:
                    if signal.status not in ['closed', 'rejected']:
                        for order in signal.orders:
                            if order.symbol == symbol and order.status == 'closed' and order.entryOrderBool:
                                existing_signal = signal
                                break
                        if existing_signal:
                            break
                
                # If we have an existing position and get a reverse signal, add exit order
                if existing_signal:
                    existing_entry = None
                    for order in existing_signal.orders:
                        if order.symbol == symbol and order.entryOrderBool and order.status == 'closed':
                            existing_entry = order
                            break
                    
                    if existing_entry and existing_entry.orderDirection != orderDirection:
                        # Cancel any existing stoploss orders
                        for order in existing_signal.orders:
                            if order.order_type == "STOPLOSS" and order.status == "open":
                                order.status = "cancel"
                                order.message = "Cancelled due to exit signal"
                                order.fresh_update = True
                        
                        # Add market exit order
                        exit_direction = "SELL" if existing_entry.orderDirection == "BUY" else "BUY"
                        exit_order = Order(
                            symbol=symbol,
                            orderQuantity=self.orderQuantity,
                            orderDirection=orderDirection,
                            order_type=self.orderType,
                            symbol_ltp={system_timestamp: current_price},
                            timeInForce=self.timeInForce,
                            entryOrderBool=False,
                            status="pending"
                        )
                        existing_signal.orders.append(exit_order)
                        existing_signal.signal_update = True
                        signals.append(existing_signal)
                        return_type = 'signals'
                
                # If no existing position and we get a signal, create new entry
                elif not existing_signal:
                    # Create Order object
                    order = Order(
                        symbol=symbol,
                        orderQuantity=self.orderQuantity,
                        orderDirection=orderDirection,
                        order_type=self.orderType,
                        symbol_ltp={system_timestamp: current_price},
                        timeInForce=self.timeInForce,
                        entryOrderBool=True,
                        status="pending"
                    )

                    # Create stoploss order
                    stoploss_price = current_price * (1 - self.stoploss_pct) if orderDirection == "BUY" else current_price * (1 + self.stoploss_pct)
                    stoploss_order = Order(
                        symbol=symbol,
                        orderQuantity=self.orderQuantity,
                        orderDirection="SELL" if orderDirection == "BUY" else "BUY",
                        order_type="STOPLOSS",
                        price=stoploss_price,
                        symbol_ltp={system_timestamp: current_price},
                        timeInForce=self.timeInForce,
                        entryOrderBool=False,
                        status="pending"
                    )
                    
                    # Create Signal object
                    signal = Signal(
                        strategy_name=self.strategy_name,
                        timestamp=system_timestamp,
                        orders=[order, stoploss_order],
                        signal_strength=signal_strength,
                        granularity=self.granularity,
                        signal_type="BUY_SELL",
                        market_neutral=False
                    )
                    signals.append(signal)
                    self.logger.info({f'SIGNAL GENERATED: {symbol}'})
                    return_type = 'signals'
                    
        return return_type, signals, self.tickers