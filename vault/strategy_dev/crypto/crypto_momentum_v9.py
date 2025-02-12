"""
SMA 10 and SMA 50 crossover momentum strategy with dynamic ATR-based stops
"""

from vault.base_strategy import BaseStrategy, Signal, Order
import logging
import numpy as np
import pandas as pd
from systems.utils import create_logger
import time
from datetime import datetime, timedelta

class Strategy(BaseStrategy):
    def __init__(self, config_dict=None):
        super().__init__()
        self.logger = create_logger(logger_name='trader', log_level='INFO')
        self.strategy_name = 'crypto_momentum_last'
        self.granularity = "15m"
        self.higher_timeframe = "12h"  # Can be configured to different timeframes
        self.orderType = "MARKET"
        self.trail_step_pct = 0.003  # Trail step for profit locking
        self.atr_period = 20  # ATR lookback period
        self.atr_multiplier = 4.0  # ATR multiplier for stops
        self.pullback_atr_ratio = 0.3  # Required ATR pullback for entry
        self.max_risk_pct = 0.04  # Maximum risk per trade as % of account
        self.min_win_ratio = 1.5  # Minimum win/loss ratio required
        self.timeInForce = "DAY"
        self.minimum_order_value = 10  # Minimum $10 position
        self.position_value = 10000  # Fixed $10,000 position size
        self.regime_confirmation_periods = 5
        self.previous_regimes = {}  # Track regime history
        self.min_hold_time = 15  # Minimum holding time in minutes
        self.entry_times = {}  # Track entry times
        self.funds_available = 0
        self.regime_strength_threshold = 1.5  # Minimum regime strength required
        self.data_inputs, self.tickers = self.datafeeder_inputs()
        
    def generate_signals(self, next_rows, market_data_df, system_timestamp, strategy_margins, open_signals=None):
        """
        Generate trading signals based on defined rules and conditions
        """
        rules = {
            'regime': {
                'inputs': ['regime_value', 'regime_stability', 'market_regime'],
                'output': lambda x: 1 if x['regime_value'] > self.regime_strength_threshold and x['regime_stability'] and x['market_regime'] > self.regime_strength_threshold else -1 if x['regime_value'] < -self.regime_strength_threshold and x['regime_stability'] and x['market_regime'] < -self.regime_strength_threshold else 0,
                'name': 'Regime Alignment'
            },
            'sma_crossover': {
                'inputs': ['sma10_current', 'sma10_prev', 'sma50_current', 'sma50_prev'],
                'output': lambda x: 1 if x['sma10_current'] > x['sma50_current'] and x['sma10_prev'] <= x['sma50_prev'] else -1 if x['sma10_current'] < x['sma50_current'] and x['sma10_prev'] >= x['sma50_prev'] else 0,
                'name': 'SMA Crossover'
            },
            'volume': {
                'inputs': ['current_volume', 'volume_sma'],
                'output': lambda x: True if x['current_volume'] > x['volume_sma'] * 1.5 else False,
                'name': 'Volume Threshold'
            },
            'rsi': {
                'inputs': ['rsi_value'],
                'output': lambda x: 1 if x['rsi_value'] < 70 else -1 if x['rsi_value'] > 30 else 0,
                'name': 'RSI Filter'
            },
            'macd': {
                'inputs': ['macd_histogram'],
                'output': lambda x: 1 if x['macd_histogram'] > 0 else -1 if x['macd_histogram'] < 0 else 0,
                'name': 'MACD Direction'
            },
            'pullback': {
                'inputs': ['pullback_size', 'bounce_size'],
                'output': lambda x: 1 if x['pullback_size'] > self.pullback_atr_ratio else -1 if x['bounce_size'] > self.pullback_atr_ratio else 0,
                'name': 'Price Pullback'
            }
        }
        
        signals = []
        return_type = 'signals'
        open_signals = open_signals or []
        
        # First calculate market index state
        market_index_state, market_regime = self.calculate_market_index(market_data_df, system_timestamp)
        if market_index_state['calculation']['status'] == 'failed':
            return return_type, signals, self.tickers
            
        # Get unique symbols from market data
        try:
            symbols = market_data_df.xs('open', axis=1, level=0).columns.tolist()
        except:
            return return_type, signals, self.tickers
            
        for symbol in symbols:
            if not self.check_data_sufficiency(market_data_df, symbol):
                continue
                
            # Calculate technical metrics
            metrics = self.calculate_metrics(market_data_df, symbol, market_regime)
            if not metrics:
                continue
                
            # Calculate rule outputs
            rule_outputs = []
            rule_names = []
            for rule_name, rule in rules.items():
                try:
                    rule_inputs = {input_name: metrics[input_name] for input_name in rule['inputs']}
                    rule_outputs.append(rule['output'](rule_inputs))
                    rule_names.append(rule['name'])
                except KeyError as e:
                    self.logger.warning(f"Missing metric for rule {rule_name}: {str(e)}")
                    continue
            
            if rule_outputs[rule_names.index('Regime Alignment')] != 0:
                self.logger.info(f"Rule outputs for {symbol}: {dict(zip(rule_names, rule_outputs))}")
            
            # Generate signal if all rules align
            signal_direction = None
            if all(output == 1 or output == True for output in rule_outputs):
                signal_direction = "BUY"
            elif all(output == -1 or output == True for output in rule_outputs):
                signal_direction = "SELL"
                
            if signal_direction:
                # Calculate position size and validate funds
                position_size, stoploss_price = self.calculate_position_details(
                    metrics['current_price'], 
                    metrics['atr'],
                    strategy_margins
                )
                
                if position_size > 0:
                    signal = self.create_signal(
                        symbol, 
                        signal_direction,
                        position_size,
                        stoploss_price,
                        metrics['current_price'],
                        system_timestamp,
                        metrics['regime_value']
                    )
                    signals.append(signal)
                    
            # Update trailing stops for existing positions
            existing_signal = self.get_existing_signal(symbol, open_signals)
            if existing_signal:
                updated_signal = self.trail_stoploss(
                    existing_signal, 
                    metrics['current_price'],
                    system_timestamp
                )
                if updated_signal:
                    signals.append(updated_signal)
                    
        return return_type, signals, self.tickers

    def calculate_metrics(self, market_data_df, symbol, market_regime):
        """Calculate all technical metrics needed for signal generation"""
        try:
            # Get price data
            asset_data_df = market_data_df.loc[self.granularity].xs(symbol, axis=1, level='symbol').copy()
            
            # Basic price data
            close = pd.to_numeric(asset_data_df['close'], errors='coerce')
            high = pd.to_numeric(asset_data_df['high'], errors='coerce')
            low = pd.to_numeric(asset_data_df['low'], errors='coerce')
            volume = pd.to_numeric(asset_data_df['volume'], errors='coerce')
            
            if close.isna().any() or volume.isna().any():
                return None
                
            current_price = float(close.iloc[-1])
            
            # Volume metrics
            volume_sma = volume.rolling(window=20).mean()
            current_volume = float(volume.iloc[-1])
            
            # Moving averages
            sma10 = close.rolling(window=10).mean()
            sma50 = close.rolling(window=50).mean()
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            
            # Calculate ATR
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=self.atr_period).mean()
            
            # Pullback/Bounce calculations
            recent_high = high.rolling(window=5).max().iloc[-1]
            recent_low = low.rolling(window=5).min().iloc[-1]
            pullback_size = (recent_high - close.iloc[-1]) / atr.iloc[-1]
            bounce_size = (close.iloc[-1] - recent_low) / atr.iloc[-1]
            
            # Calculate regime
            regime_value = self.get_market_regime(asset_data_df)
            
            # Update regime history
            if symbol not in self.previous_regimes:
                self.previous_regimes[symbol] = []
            self.previous_regimes[symbol].append(regime_value)
            
            # Maintain fixed history length
            max_history = self.regime_confirmation_periods * 2
            if len(self.previous_regimes[symbol]) > max_history:
                self.previous_regimes[symbol] = self.previous_regimes[symbol][-max_history:]
            
            # Calculate regime stability
            recent_regimes = self.previous_regimes[symbol][-self.regime_confirmation_periods:]
            regime_signs = [1 if r > 0 else -1 if r < 0 else 0 for r in recent_regimes]
            sign_matches = sum(1 for x in regime_signs if x == regime_signs[-1]) if regime_signs else 0
            regime_stable = (len(regime_signs) >= self.regime_confirmation_periods and
                           sign_matches >= self.regime_confirmation_periods - 1 and
                           abs(regime_value) > self.regime_strength_threshold and
                           regime_signs[-1] != 0)
            
            return {
                'current_price': current_price,
                'current_volume': current_volume,
                'volume_sma': float(volume_sma.iloc[-1]),
                'sma10_current': float(sma10.iloc[-1]),
                'sma10_prev': float(sma10.iloc[-2]),
                'sma50_current': float(sma50.iloc[-1]),
                'sma50_prev': float(sma50.iloc[-2]),
                'rsi_value': float(rsi.iloc[-1]),
                'macd_histogram': float(histogram.iloc[-1]),
                'atr': float(atr.iloc[-1]),
                'pullback_size': float(pullback_size),
                'bounce_size': float(bounce_size),
                'regime_value': regime_value,
                'regime_stability': regime_stable,
                'market_regime': market_regime
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics for {symbol}: {str(e)}")
            return None
            
    def check_data_sufficiency(self, market_data_df, symbol):
        """Check if we have enough data for signal generation"""
        if self.granularity not in market_data_df.index.levels[0]:
            return False
            
        min_bars = max(50, self.data_inputs[self.granularity]['lookback'])
        if len(market_data_df.loc[self.granularity]) <= min_bars:
            return False
            
        return True

    def create_signal(self, symbol, direction, position_size, stoploss_price, current_price,
                     system_timestamp, regime_value):
        """Create a new signal with entry and stoploss orders"""
        # Track entry time
        self.entry_times[symbol] = pd.to_datetime(system_timestamp)
        
        # Create entry order
        entry_order = Order(
            symbol=symbol,
            orderQuantity=position_size,
            orderDirection=direction,
            order_type=self.orderType,
            symbol_ltp={system_timestamp: current_price},
            timeInForce=self.timeInForce,
            entryOrderBool=True,
            status="pending"
        )
        
        # Create stoploss order
        stoploss_direction = "SELL" if direction == "BUY" else "BUY"
        stoploss_order = Order(
            symbol=symbol,
            orderQuantity=position_size,
            orderDirection=stoploss_direction,
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
            orders=[entry_order, stoploss_order],
            signal_strength=1,
            granularity=self.granularity,
            signal_type="BUY_SELL",
            market_neutral=False,
            strategy_inputs={'regime_filter': regime_value}
        )
        
        return signal
        
    def calculate_position_details(self, current_price, atr, strategy_margins):
        """Calculate position size and stoploss based on risk parameters"""
        # Calculate stoploss distance using ATR
        stop_distance = atr * self.atr_multiplier
        stoploss_price = current_price - stop_distance if current_price > 0 else current_price + stop_distance
        
        # Calculate position value based on fixed size
        position_value = self.position_value
        
        # Validate minimum order value
        if position_value < self.minimum_order_value:
            return 0, stoploss_price
            
        # Check available funds
        if strategy_margins and 'USD' in strategy_margins:
            self.funds_available = float(strategy_margins['USD'].get('buying_power_available', 0))
            if position_value > self.funds_available:
                return 0, stoploss_price
                
        # Calculate final position size
        position_size = position_value / current_price
        return position_size, stoploss_price
        
    def get_market_regime(self, asset_data_df):
        """Calculate market regime score between -5 (strongly bearish) and +5 (strongly bullish)"""
        close = pd.to_numeric(asset_data_df['close'], errors='coerce')
        volume = pd.to_numeric(asset_data_df['volume'], errors='coerce')
        
        if close.isna().any() or volume.isna().any():
            return 0
            
        # Trend Factors (0 to ±3)
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        
        trend_score = 0
        if len(ema20) >= 5 and len(ema50) >= 5:
            ema20_slope = (ema20.iloc[-1] - ema20.iloc[-5]) / ema20.iloc[-5]
            ema50_slope = (ema50.iloc[-1] - ema50.iloc[-5]) / ema50.iloc[-5]
            
            if not np.isnan(ema20_slope) and not np.isnan(ema50_slope):
                if ema20.iloc[-1] > ema50.iloc[-1] and ema20_slope > 0:
                    trend_score += 1.5
                elif ema20.iloc[-1] < ema50.iloc[-1] and ema20_slope < 0:
                    trend_score -= 1.5
                    
        # Long-term trend
        if ema50.iloc[-1] > ema200.iloc[-1]:
            trend_score += 1.5
        elif ema50.iloc[-1] < ema200.iloc[-1]:
            trend_score -= 1.5
            
        # Momentum (0 to ±1.5)
        momentum_score = 0
        if len(close) >= 50:
            roc20 = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] * 100
            roc50 = (close.iloc[-1] - close.iloc[-50]) / close.iloc[-50] * 100
            
            if roc20 > 2 and roc50 > 0:
                momentum_score = 1.5
            elif roc20 < -2 and roc50 < 0:
                momentum_score = -1.5
            elif roc20 > 1:
                momentum_score = 0.75
            elif roc20 < -1:
                momentum_score = -0.75
                
        # Volume trend (0 to ±1)
        volume_sma = volume.rolling(window=20).mean()
        vol_ratio = volume.iloc[-1] / volume_sma.iloc[-1]
        vol_score = 0
        
        if vol_ratio > 1.5:
            vol_score = 1 if close.iloc[-1] > close.iloc[-2] else -1
            
        final_score = trend_score + momentum_score + vol_score
        return np.clip(final_score, -5, 5)
        
    def trail_stoploss(self, signal, current_price, system_timestamp):
        """Update trailing stoploss if conditions are met"""
        entry_order = None
        stoploss_order = None
        
        for order in signal.orders:
            if order.status == 'closed' and order.entryOrderBool:
                entry_order = order
            elif order.order_type == 'STOPLOSS' and order.status == 'open':
                stoploss_order = order
                
        if not entry_order or not stoploss_order:
            return None
            
        # Only trail after minimum hold time
        entry_time = self.entry_times.get(entry_order.symbol)
        if not entry_time:
            return None
            
        current_time = pd.to_datetime(system_timestamp)
        elapsed_minutes = (current_time - entry_time).total_seconds() / 60
        if elapsed_minutes < self.min_hold_time:
            return None
            
        if entry_order.orderDirection == "BUY":
            distance_to_stoploss = (current_price - stoploss_order.price) / current_price
            steps = int(distance_to_stoploss / self.trail_step_pct)
            if steps > 0:
                new_stoploss = current_price * (1 - self.trail_step_pct * (steps - 1))
                if new_stoploss > stoploss_order.price:
                    stoploss_order.price = new_stoploss
                    stoploss_order.symbol_ltp[system_timestamp] = current_price
                    signal.signal_update = True
                    return signal
        else:  # SELL position
            distance_to_stoploss = (stoploss_order.price - current_price) / current_price
            steps = int(distance_to_stoploss / self.trail_step_pct)
            if steps > 0:
                new_stoploss = current_price * (1 + self.trail_step_pct * (steps - 1))
                if new_stoploss < stoploss_order.price:
                    stoploss_order.price = new_stoploss
                    stoploss_order.symbol_ltp[system_timestamp] = current_price
                    signal.signal_update = True
                    return signal
                    
        return None
        
    def get_existing_signal(self, symbol, open_signals):
        """Check if we have an open signal for the given symbol"""
        if not open_signals:
            return None
            
        for signal in open_signals:
            if signal.status not in ['closed', 'rejected']:
                for order in signal.orders:
                    if order.symbol == symbol and order.status == 'closed' and order.entryOrderBool:
                        return signal
        return None
        
    def calculate_market_index(self, market_data_df, system_timestamp):
        """Calculate market index using major coins"""
        market_index_state = {
            'calculation': {'status': 'pending'},
            'regime': None
        }
        
        try:
            # Get major coins data
            major_coins = ['ETHUSD', 'XRPUSD']
            market_index = pd.DataFrame()
            
            for coin in major_coins:
                if coin not in market_data_df.xs('close', axis=1, level=0).columns:
                    continue
                    
                coin_data = market_data_df.loc[self.granularity].xs(coin, axis=1, level='symbol').copy()
                close = pd.to_numeric(coin_data['close'], errors='coerce')
                volume = pd.to_numeric(coin_data['volume'], errors='coerce')
                
                if close.isna().any() or volume.isna().any():
                    continue
                    
                market_index[f"{coin}_close"] = close
                market_index[f"{coin}_volume"] = volume
                
            if len(market_index.columns) < 4:  # Need both price and volume for both coins
                market_index_state['calculation']['status'] = 'failed'
                return market_index_state, 0
                
            # Calculate composite index
            market_index['index'] = market_index[[c for c in market_index.columns if 'close' in c]].mean(axis=1)
            market_regime = self.get_market_regime(pd.DataFrame({
                'close': market_index['index'],
                'volume': market_index[[c for c in market_index.columns if 'volume' in c]].mean(axis=1)
            }))
            
            market_index_state['calculation']['status'] = 'success'
            market_index_state['regime'] = {
                'value': market_regime,
                'timestamp': str(system_timestamp)
            }
            
            return market_index_state, market_regime
            
        except Exception as e:
            market_index_state['calculation']['status'] = 'failed'
            market_index_state['calculation']['error'] = str(e)
            return market_index_state, 0
            
    def datafeeder_inputs(self):
        """Define data requirements for the strategy"""
        tickers = ['ETHUSD', 'SOLUSD', 'ADAUSD', 'XRPUSD']
        
        data_inputs = {
            self.granularity: {
                'columns': ['open', 'high', 'close', 'low', 'volume'],
                'lookback': 100
            },
            self.higher_timeframe: {
                'columns': ['open', 'high', 'close', 'low', 'volume'],
                'lookback': 10
            }
        }
        
        return data_inputs, tickers
        
    def get_name(self):
        """Return strategy name"""
        return self.strategy_name