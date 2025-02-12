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
        self.trail_step_pct = 0.003  # Decreased trail step for better profit locking
        self.atr_period = 20  # ATR lookback period
        self.atr_multiplier = 4.0  # Increased ATR multiplier for wider stops
        self.pullback_atr_ratio = 0.3  # Require 30% of ATR pullback for entry
        self.max_risk_pct = 0.04  # Maximum risk per trade as % of account
        self.min_win_ratio = 1.5  # Minimum win/loss ratio required for position sizing
        self.timeInForce = "DAY"
        self.position_value = 10000  # Fixed $10,000 position size
        self.regime_confirmation_periods = 5  # Increased from 3 to 5
        self.previous_regimes = {}  # Dictionary to track regime history for each symbol
        self.min_hold_time = 15  # Minimum holding time in minutes
        self.entry_times = {}  # Track entry times for minimum hold period
        self.data_inputs, self.tickers = self.datafeeder_inputs()
        self.funds_available = 0
        self.regime_strength_threshold = 1.5  # Minimum regime strength required

    def log_strategy_decisions(self, symbol_state, symbol, strategy_state=None):
        """Log all strategy decisions and debugging information from the symbol state"""
        # Log market data structure if available
        if strategy_state and 'market_data' in strategy_state:
            structure = strategy_state['market_data']['structure']
            # self.logger.info(f"Market data index structure: {structure['index_names']}")
            # self.logger.info(f"Market data column structure: {structure['column_names']}")
            # self.logger.info(f"Market data shape: {structure['shape']}")
            
            if 'error' in strategy_state:
                self.logger.error(f"Error getting symbols from market data: {strategy_state['error']}")
                return

        # Log symbol processing
        self.logger.info(f"\nProcessing symbol: {symbol}")

        # Log data sufficiency
        if 'data_check' in symbol_state:
            data_check = symbol_state['data_check']
            if not data_check.get('sufficient', True):
                self.logger.info(f"{data_check.get('message', 'Insufficient data')}")
                return

        # Log market index status with defensive checks
        if ('market_index' in symbol_state and
            isinstance(symbol_state.get('market_index'), dict) and
            isinstance(symbol_state['market_index'].get('calculation'), dict)):
            
            calc_status = symbol_state['market_index']['calculation'].get('status', 'unknown')
            # self.logger.info(f"Market Index Status: {calc_status}")
            
            if calc_status == 'failed':
                self.logger.info(f"Market index calculation failed for {symbol}")
                return
            
            # if (isinstance(symbol_state['market_index'].get('regime'), dict) and
            #     'value' in symbol_state['market_index']['regime']):
                # self.logger.info(f"Market Regime: {symbol_state['market_index']['regime']['value']:.2f}")
        
        # Log asset regime and status with defensive checks
        if isinstance(symbol_state.get('asset'), dict):
            regime_data = symbol_state['asset'].get('regime', {})
            if isinstance(regime_data, dict) and all(k in regime_data for k in ['status', 'value', 'market_regime']):
                self.logger.info(f"REGIME {regime_data['status'].upper()}: Asset={regime_data['value']:.2f}, Market={regime_data['market_regime']:.2f}")
            
        # Log regime stability if available
        if ('asset' in symbol_state and
            symbol_state.get('asset') is not None and
            symbol_state['asset'].get('regime') is not None and
            isinstance(symbol_state['asset']['regime'], dict) and
            'stability' in symbol_state['asset']['regime']):
            
            regime_stability = symbol_state['asset']['regime']['stability']
            if regime_stability:
                self.logger.info(f"REGIME STABILITY: {symbol} | "
                               f"Stable: {regime_stability['is_stable']} | "
                               f"Sign Matches: {regime_stability['sign_matches']}/{regime_stability['confirmation_periods']} | "
                               f"Current Sign: {regime_stability['current_sign']}")

        # Log SMA values and signal generation
        if isinstance(symbol_state.get('asset'), dict):
            tech_data = symbol_state['asset'].get('technical', {})
            if isinstance(tech_data, dict) and 'sma_values' in tech_data:
                sma = tech_data['sma_values']
                self.logger.info(f'SMA VALUES: {symbol} | '
                               f'Current SMA10: {sma["current"]["sma10"]} | '
                               f'Current SMA50: {sma["current"]["sma50"]} | '
                               f'Prev SMA10: {sma["previous"]["sma10"]} | '
                               f'Prev SMA50: {sma["previous"]["sma50"]}')

        # Log signal generation with enhanced checks
        if isinstance(symbol_state.get('signal'), dict):
            signal_data = symbol_state['signal']
            regime_data = symbol_state['asset'].get('regime', {})
            
            if signal_data.get('generated'):
                # Log crossover type
                if signal_data.get('type') in ['bullish_crossover', 'bearish_crossover']:
                    self.logger.info(f'{signal_data["type"].upper()} DETECTED: {symbol}')
                
                # Log signal details
                if all(k in regime_data for k in ['value', 'market_regime', 'combined']):
                    self.logger.info(f'SIGNAL GENERATED: {symbol} | '
                                   f'Direction: {signal_data["direction"]} | '
                                   f'Asset Regime: {regime_data["value"]:.2f} | '
                                   f'Market Regime: {regime_data["market_regime"]:.2f} | '
                                   f'Combined Regime: {regime_data["combined"]:.2f}')
                
                # Log validation result
                if 'validation' in signal_data:
                    validation = signal_data['validation']
                    if signal_data['strength'] > 0:
                        self.logger.info(f'VALID {signal_data["direction"]} SIGNAL: {symbol} | '
                                       f'Regime: {validation["regime"]:.2f} | '
                                       f'Stable: {validation["stable"]}')
                    else:
                        self.logger.info(f'REJECTED {signal_data["direction"]} SIGNAL: {symbol} | '
                                       f'Regime: {validation["regime"]:.2f} | '
                                       f'Stable: {validation["stable"]} | '
                                       f'Required: {validation["threshold"]}')
            elif signal_data.get('type') == 'no_crossover':
                sma = tech_data['sma_values']['current']
                self.logger.info(f'NO CROSSOVER: {symbol} | SMA10: {sma["sma10"]} | SMA50: {sma["sma50"]}')

    def calculate_market_index(self, market_data_df, system_timestamp):
        """Calculate market index using major coins"""
        market_index_state = {
            'calculation': {
                'status': 'pending',
                'index_components': {},
                'composite_index': None
            },
            'components': {},
            'regime': None,
            'error': None
        }
        
        # Initialize market index calculation
        market_index = pd.DataFrame()
        position_values = {}
        
        # First pass to get initial prices for index components
        for major_coin in ['ETHUSD', 'XRPUSD']:
            try:
                coin_data = market_data_df.loc[self.granularity].xs(major_coin, axis=1, level='symbol').copy()
                close_prices = pd.to_numeric(coin_data['close'], errors='coerce')
                initial_price = float(close_prices.iloc[0])
                position_size = 1000 / initial_price  # $1000 worth of each coin
                
                market_index_state['components'][major_coin] = {
                    'initial_price': initial_price,
                    'position_size': position_size,
                    'last_price': float(close_prices.iloc[-1]),
                    'return': float(close_prices.iloc[-1]/initial_price - 1)
                }
                position_values[major_coin] = position_size
                
            except Exception as e:
                market_index_state['components'][major_coin] = {'error': str(e)}
                continue
                
        if len(position_values) != 2:
            market_index_state['calculation']['status'] = 'failed'
            market_index_state['calculation']['reason'] = 'Missing major coins data'
            return market_index_state, None
            
        # Calculate full market index
        for major_coin in ['ETHUSD', 'XRPUSD']:
            try:
                coin_data = market_data_df.loc[self.granularity].xs(major_coin, axis=1, level='symbol').copy()
                close_prices = pd.to_numeric(coin_data['close'], errors='coerce')
                coin_returns = close_prices.pct_change()
                
                position_value = (1 + coin_returns) * position_values[major_coin]
                volume_value = pd.to_numeric(coin_data['volume'], errors='coerce')
                
                market_index[f"{major_coin}_pos"] = position_value
                market_index[f"{major_coin}_vol"] = volume_value
                
                market_index_state['calculation']['index_components'][major_coin] = {
                    'returns': {
                        'latest': float(coin_returns.iloc[-1]) if not pd.isna(coin_returns.iloc[-1]) else None,
                        'mean': float(coin_returns.mean()),
                        'std': float(coin_returns.std())
                    },
                    'position': {
                        'latest': float(position_value.iloc[-1]),
                        'mean': float(position_value.mean())
                    },
                    'volume': {
                        'latest': float(volume_value.iloc[-1]),
                        'mean': float(volume_value.mean())
                    }
                }
                
            except Exception as e:
                market_index_state['calculation']['index_components'][major_coin] = {'error': str(e)}
                continue
                
        # Calculate composite index
        market_index['close'] = market_index[['ETHUSD_pos', 'XRPUSD_pos']].sum(axis=1)
        market_index['volume'] = market_index[['ETHUSD_vol', 'XRPUSD_vol']].sum(axis=1)
        
        market_index_state['calculation']['status'] = 'success'
        market_index_state['calculation']['composite_index'] = {
            'close': {
                'latest': float(market_index['close'].iloc[-1]),
                'mean': float(market_index['close'].mean()),
                'std': float(market_index['close'].std())
            },
            'volume': {
                'latest': float(market_index['volume'].iloc[-1]),
                'mean': float(market_index['volume'].mean()),
                'std': float(market_index['volume'].std())
            }
        }
        
        # Calculate market regime
        market_regime = self.get_market_regime(market_index.ffill())
        market_index_state['regime'] = {
            'value': market_regime,
            'timestamp': str(system_timestamp)
        }
        
        return market_index_state, market_regime

    def calculate_atr(self, asset_data_df, period=None):
        """Calculate Average True Range"""
        if period is None:
            period = self.atr_period
            
        high = pd.to_numeric(asset_data_df['high'], errors='coerce')
        low = pd.to_numeric(asset_data_df['low'], errors='coerce')
        close = pd.to_numeric(asset_data_df['close'], errors='coerce')
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
        
    def get_higher_timeframe_trend(self, market_data_df, symbol, direction):
        """Check if higher timeframe trend aligns with trade direction"""
        if self.higher_timeframe not in market_data_df.index.levels[0]:
            return False
            
        # Get higher timeframe data
        htf_data = market_data_df.loc[self.higher_timeframe].xs(symbol, axis=1, level='symbol').copy()
        if len(htf_data) < 5:
            return False
            
        close = pd.to_numeric(htf_data['close'], errors='coerce')
        ema3 = close.ewm(span=3, adjust=False).mean()
        ema5 = close.ewm(span=5, adjust=False).mean()
        
        ema3_slope = (ema3.iloc[-1] - ema3.iloc[-2]) / ema3.iloc[-2]
        ema5_slope = (ema5.iloc[-1] - ema5.iloc[-2]) / ema5.iloc[-2]
        
        if direction == "BUY":
            return ema3_slope > 0 and ema5_slope > 0
        else:
            return ema3_slope < 0 and ema5_slope < 0
            
    def check_min_hold_time(self, symbol, system_timestamp):
        """Check if minimum holding time has elapsed"""
        if symbol not in self.entry_times:
            return True
            
        entry_time = self.entry_times[symbol]
        current_time = pd.to_datetime(system_timestamp)
        elapsed_minutes = (current_time - entry_time).total_seconds() / 60
        
        return elapsed_minutes >= self.min_hold_time
            
    def validate_entry(self, asset_data_df, direction, market_data_df, symbol):
        """Validate entry with additional filters including pullback and higher timeframe confirmation"""
        validation_result = {'passed': False, 'reason': None, 'metrics': {}}
        
        close = pd.to_numeric(asset_data_df['close'], errors='coerce')
        high = pd.to_numeric(asset_data_df['high'], errors='coerce')
        low = pd.to_numeric(asset_data_df['low'], errors='coerce')
        volume = pd.to_numeric(asset_data_df['volume'], errors='coerce')
        
        atr = self.calculate_atr(asset_data_df)
        if pd.isna(atr.iloc[-1]):
            validation_result['reason'] = f'Invalid ATR value'
            return validation_result
            
        required_pullback = atr.iloc[-1] * self.pullback_atr_ratio
        volume_sma = volume.rolling(window=20).mean()
        current_volume = volume.iloc[-1]
        
        # Volume validation
        if current_volume <= volume_sma.iloc[-1] * 1.5:
            validation_result['reason'] = f'Volume {current_volume:.2f} below threshold {(volume_sma.iloc[-1] * 1.5):.2f}'
            validation_result['metrics']['volume'] = {'current': current_volume, 'threshold': volume_sma.iloc[-1] * 1.5}
            return validation_result
        
        # Technical indicators
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        # Higher timeframe trend check
        trend_metrics = {}
        if self.higher_timeframe in market_data_df.index.levels[0]:
            htf_data = market_data_df.loc[self.higher_timeframe].xs(symbol, axis=1, level='symbol').copy()
            if len(htf_data) >= 5:
                hourly_close = pd.to_numeric(htf_data['close'], errors='coerce')
                ema3 = hourly_close.ewm(span=3, adjust=False).mean()
                ema5 = hourly_close.ewm(span=5, adjust=False).mean()
                ema3_slope = (ema3.iloc[-1] - ema3.iloc[-2]) / ema3.iloc[-2] * 100
                ema5_slope = (ema5.iloc[-1] - ema5.iloc[-2]) / ema5.iloc[-2] * 100
                trend_metrics = {'3h_slope': ema3_slope, '5h_slope': ema5_slope}
                validation_result['metrics']['trend'] = trend_metrics

        if direction == "BUY":
            recent_high = high.rolling(window=5).max().iloc[-1]
            pullback_size = (recent_high - close.iloc[-1]) / atr.iloc[-1]
            validation_result['metrics']['pullback'] = pullback_size

            if len(trend_metrics) > 0 and (trend_metrics['3h_slope'] <= 0 or trend_metrics['5h_slope'] <= 0):
                validation_result['reason'] = f'Higher timeframe trend not aligned'
                return validation_result
            if rsi.iloc[-1] >= 70:
                validation_result['reason'] = f'RSI overbought: {rsi.iloc[-1]:.2f}'
                validation_result['metrics']['rsi'] = float(rsi.iloc[-1])
                return validation_result
            if histogram.iloc[-1] <= 0:
                validation_result['reason'] = f'MACD histogram negative: {histogram.iloc[-1]:.2f}'
                validation_result['metrics']['macd'] = float(histogram.iloc[-1])
                return validation_result
            if pullback_size < self.pullback_atr_ratio:
                validation_result['reason'] = f'Insufficient pullback: {pullback_size:.2f} ATR vs required {self.pullback_atr_ratio:.2f} ATR'
                return validation_result
        else:
            recent_low = low.rolling(window=5).min().iloc[-1]
            bounce_size = (close.iloc[-1] - recent_low) / atr.iloc[-1]
            validation_result['metrics']['bounce'] = bounce_size

            if len(trend_metrics) > 0 and (trend_metrics['3h_slope'] >= 0 or trend_metrics['5h_slope'] >= 0):
                validation_result['reason'] = f'Higher timeframe trend not aligned'
                return validation_result
            if rsi.iloc[-1] <= 30:
                validation_result['reason'] = f'RSI oversold: {rsi.iloc[-1]:.2f}'
                validation_result['metrics']['rsi'] = float(rsi.iloc[-1])
                return validation_result
            if histogram.iloc[-1] >= 0:
                validation_result['reason'] = f'MACD histogram positive: {histogram.iloc[-1]:.2f}'
                validation_result['metrics']['macd'] = float(histogram.iloc[-1])
                return validation_result
            if bounce_size < self.pullback_atr_ratio:
                validation_result['reason'] = f'Insufficient bounce: {bounce_size:.2f} ATR vs required {self.pullback_atr_ratio:.2f} ATR'
                return validation_result

        validation_result['passed'] = True
        return validation_result

    def get_name(self):
        return self.strategy_name
        
    def datafeeder_inputs(self):
        tickers = ['ETHUSD', 'SOLUSD', 'ADAUSD', 'XRPUSD',
            # 'DOTUSD', 'DOGEUSD', 'LTCUSD', 'UNIUSD', 'LINKUSD'
        ]
        # Reduced lookback to ensure data availability while maintaining strategy requirements
        data_inputs = {
            self.granularity: {
                'columns': ['open', 'high', 'close', 'low', 'volume'],
                'lookback': 100  # Reduced from 250 to ensure data availability
            },
            self.higher_timeframe: {
                'columns': ['open', 'high', 'close', 'low', 'volume'],
                'lookback': 10
            }
        }
        self.logger.info(f"Initialized strategy with tickers: {tickers}")
        self.logger.info(f"Data requirements - {self.granularity}: {data_inputs[self.granularity]['lookback']} bars, {self.higher_timeframe}: {data_inputs[self.higher_timeframe]['lookback']} bars")
        return data_inputs, tickers
    
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
        if len(ema20) >= 5 and ema20.iloc[-5] != 0 and len(ema50) >= 5 and ema50.iloc[-5] != 0:
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
        
        if vol_ratio > 1.5:  # Increased threshold
            vol_score = 1 if close.iloc[-1] > close.iloc[-2] else -1
            
        final_score = trend_score + momentum_score + vol_score
        return np.clip(final_score, -5, 5)

    def get_existing_signal(self, symbol, open_signals):
        """Check if we have an open signal for the given symbol"""
        for signal in open_signals:
            if signal.status not in ['closed', 'rejected']:
                for order in signal.orders:
                    if order.symbol == symbol and order.status == 'closed' and order.entryOrderBool:
                        return signal, order
        return None, None

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
        if not self.check_min_hold_time(entry_order.symbol, system_timestamp):
            return None
            
        if entry_order.orderDirection == "BUY":
            distance_to_stoploss = (current_price - stoploss_order.price) / current_price
            steps = int(distance_to_stoploss / self.trail_step_pct)
            if steps > 0:
                atr = self.calculate_atr(pd.DataFrame({'high': [current_price], 'low': [stoploss_order.price], 'close': [current_price]}))
                stop_distance = atr.iloc[-1] * self.atr_multiplier if not pd.isna(atr.iloc[-1]) else current_price * 0.01
                new_stoploss = current_price - stop_distance
                if new_stoploss > stoploss_order.price:
                    stoploss_order.price = new_stoploss
                    stoploss_order.symbol_ltp[system_timestamp] = current_price
                    return signal
        else:  # SELL position
            distance_to_stoploss = (stoploss_order.price - current_price) / current_price
            steps = int(distance_to_stoploss / self.trail_step_pct)
            if steps > 0:
                atr = self.calculate_atr(pd.DataFrame({'high': [current_price], 'low': [stoploss_order.price], 'close': [current_price]}))
                stop_distance = atr.iloc[-1] * self.atr_multiplier if not pd.isna(atr.iloc[-1]) else current_price * 0.01
                new_stoploss = current_price + stop_distance
                if new_stoploss < stoploss_order.price:
                    stoploss_order.price = new_stoploss
                    stoploss_order.symbol_ltp[system_timestamp] = current_price
                    return signal
                    
        return None
        
    def generate_signals(self, next_rows, market_data_df, system_timestamp, strategy_margins, open_signals=None):
        """Generate signals based on SMA crossover with enhanced regime filters"""
        strategy_state = {
            'system': {
                'timestamp': system_timestamp,
                'market_data_shape': market_data_df.shape,
                'available_granularities': market_data_df.index.levels[0].tolist()
            },
            'signals': [],
            'symbols': {}
        }
        
        signals = []
        return_type = 'signals'
        open_signals = open_signals or []

        # Store market data info in strategy state
        strategy_state['market_data'] = {
            'structure': {
                'index_names': market_data_df.index.names,
                'column_names': market_data_df.columns.names,
                'shape': market_data_df.shape
            }
        }
        
        try:
            # Get unique symbols from multi-index columns
            symbols = []
            if isinstance(market_data_df.columns, pd.MultiIndex):
                symbols = market_data_df.xs('open', axis=1, level=0).columns.tolist()
            else:
                symbols = set(col[1] for col in market_data_df.columns if col[0] == 'open')
                
            strategy_state['symbols_found'] = symbols
        except Exception as e:
            strategy_state['error'] = str(e)
            return return_type, signals, self.tickers

        for symbol in symbols:
            
            if self.granularity not in market_data_df.index.levels[0]:
                self.logger.info(f"Granularity {self.granularity} not found in market data")
                continue
            if len(market_data_df.loc[self.granularity]) <= self.data_inputs[self.granularity]['lookback']:
                # self.logger.info(f"Insufficient data for {symbol}: {len(market_data_df.loc[self.granularity])} rows vs required {self.data_inputs[self.granularity]['lookback']}")
                continue
                
            # Initialize symbol state in strategy state
            strategy_state['symbols'][symbol] = {
                'market_index': {
                    'calculation': {},
                    'components': {},
                    'regime': None
                },
                'asset': {
                    'data': {},
                    'technical': {},
                    'regime': None
                },
                'validation': {
                    'checks': {},
                    'passed': False
                },
                'signal': {
                    'generated': False,
                    'direction': None,
                    'strength': 0,
                    'reason': None
                }
            }
            
            # Calculate market index and regime
            symbol_state = strategy_state['symbols'][symbol]
            market_index_state, market_regime = self.calculate_market_index(market_data_df, system_timestamp)
            symbol_state['market_index'] = market_index_state
            
            if market_index_state['calculation']['status'] == 'failed':
                self.logger.info(f"Market index calculation failed for {symbol}")
                continue
            
            # Individual asset analysis
            asset_data_df = market_data_df.loc[self.granularity].xs(symbol, axis=1, level='symbol').reset_index()
            
            # Calculate moving averages
            sma10 = asset_data_df['close'].rolling(window=10).mean()
            sma50 = asset_data_df['close'].rolling(window=50).mean()
            asset_data_df['SMA10'] = sma10
            asset_data_df['SMA50'] = sma50

            # Store technical indicators
            symbol_state['asset']['technical'] = {
                'moving_averages': {
                    'SMA10': {
                        'current': float(sma10.iloc[-1]),
                        'previous': float(sma10.iloc[-2]),
                        'slope': float((sma10.iloc[-1] - sma10.iloc[-2]) / sma10.iloc[-2]) if not pd.isna(sma10.iloc[-2]) else None
                    },
                    'SMA50': {
                        'current': float(sma50.iloc[-1]),
                        'previous': float(sma50.iloc[-2]),
                        'slope': float((sma50.iloc[-1] - sma50.iloc[-2]) / sma50.iloc[-2]) if not pd.isna(sma50.iloc[-2]) else None
                    },
                    'crossover': {
                        'current': sma10.iloc[-1] > sma50.iloc[-1],
                        'previous': sma10.iloc[-2] > sma50.iloc[-2],
                        'direction': 'bullish' if (sma10.iloc[-1] > sma50.iloc[-1] and sma10.iloc[-2] <= sma50.iloc[-2]) else
                                   'bearish' if (sma10.iloc[-1] < sma50.iloc[-1] and sma10.iloc[-2] >= sma50.iloc[-2]) else 'none'
                    }
                }
            }

            current_price = float(asset_data_df.iloc[-1]['close'])
            symbol_state['asset']['data']['price'] = current_price
            
            # Calculate and store asset regime
            asset_regime = self.get_market_regime(asset_data_df)
            
            # Calculate combined regime
            regime = 0
            regime_status = 'weak'
            if abs(asset_regime) > self.regime_strength_threshold and abs(market_regime) > self.regime_strength_threshold:
                if np.sign(asset_regime) == np.sign(market_regime):
                    regime = asset_regime
                    regime_status = 'aligned'
                else:
                    regime_status = 'mismatched'
                    
            symbol_state['asset']['regime'] = {
                'value': asset_regime,
                'market_regime': market_regime,
                'combined': regime,
                'status': regime_status,
                'threshold': self.regime_strength_threshold,
                'message': f'REGIME {regime_status.upper()}: Asset={asset_regime:.2f}, Market={market_regime:.2f}'
            }
            
            # Initialize regime history for symbol if not exists
            if symbol not in self.previous_regimes:
                self.previous_regimes[symbol] = []
            
            # Update regime history
            self.previous_regimes[symbol].append(regime)
            max_history = self.regime_confirmation_periods * 2
            if len(self.previous_regimes[symbol]) > max_history:
                self.previous_regimes[symbol] = self.previous_regimes[symbol][-max_history:]
            
            # Calculate regime stability metrics
            recent_regimes = self.previous_regimes[symbol][-self.regime_confirmation_periods:] if len(self.previous_regimes[symbol]) >= self.regime_confirmation_periods else []
            regime_signs = [1 if r > 0 else -1 if r < 0 else 0 for r in recent_regimes]
            sign_matches = sum(1 for x in regime_signs if x == regime_signs[-1]) if regime_signs else 0
            regime_stable = (len(regime_signs) >= self.regime_confirmation_periods and
                           sign_matches >= self.regime_confirmation_periods - 1 and
                           abs(regime) > self.regime_strength_threshold and
                           regime_signs[-1] != 0)
            
            # Store regime stability analysis
            symbol_state['asset']['regime']['stability'] = {
                'history': self.previous_regimes[symbol],
                'recent_signs': regime_signs,
                'sign_matches': sign_matches,
                'confirmation_periods': self.regime_confirmation_periods,
                'is_stable': regime_stable,
                'current_sign': regime_signs[-1] if regime_signs else 0
            }
            
            # Log the full state with strategy context
            self.log_strategy_decisions(symbol_state, symbol, strategy_state)

            # Store SMA values and evaluate crossovers
            current_sma10 = round(asset_data_df.iloc[-1]['SMA10'], 2)
            current_sma50 = round(asset_data_df.iloc[-1]['SMA50'], 2)
            prev_sma10 = round(asset_data_df.iloc[-2]['SMA10'], 2)
            prev_sma50 = round(asset_data_df.iloc[-2]['SMA50'], 2)
            
            symbol_state['asset']['technical']['sma_values'] = {
                'current': {'sma10': current_sma10, 'sma50': current_sma50},
                'previous': {'sma10': prev_sma10, 'sma50': prev_sma50}
            }
            
            # Evaluate buy signal
            if current_sma10 > current_sma50 and prev_sma10 <= prev_sma50:
                symbol_state['signal'].update({
                    'generated': True,
                    'type': 'bullish_crossover',
                    'direction': 'BUY',
                    'strength': 1 if (regime > self.regime_strength_threshold and regime_stable) or regime > 2 else 0,
                    'validation': {
                        'regime': regime,
                        'stable': regime_stable,
                        'threshold': self.regime_strength_threshold
                    }
                })
            
            # Evaluate sell signal
            elif current_sma10 < current_sma50 and prev_sma10 >= prev_sma50:
                symbol_state['signal'].update({
                    'generated': True,
                    'type': 'bearish_crossover',
                    'direction': 'SELL',
                    'strength': 1 if (regime < -self.regime_strength_threshold and regime_stable) or regime < -2 else 0,
                    'validation': {
                        'regime': regime,
                        'stable': regime_stable,
                        'threshold': -self.regime_strength_threshold
                    }
                })
            else:
                symbol_state['signal'].update({
                    'generated': False,
                    'type': 'no_crossover',
                    'direction': None,
                    'strength': 0
                })
            
            existing_signal, existing_entry = self.get_existing_signal(symbol, open_signals)
            
            # Check minimum holding period for exits
            if existing_signal and not self.check_min_hold_time(symbol, system_timestamp):
                continue

            if symbol_state['signal']['strength'] > 0:
                # Validate entry conditions
                validation_result = self.validate_entry(asset_data_df, symbol_state['signal']['direction'], market_data_df, symbol)
                if not validation_result['passed']:
                    self.logger.info(f'ENTRY VALIDATION FAILED: {symbol} | Direction: {symbol_state["signal"]["direction"]} | {validation_result["reason"]}')
                    continue
                    
                # Calculate ATR-based stop loss
                atr = self.calculate_atr(asset_data_df)
                if pd.isna(atr.iloc[-1]):
                    continue
                    
                stop_distance = atr.iloc[-1] * self.atr_multiplier
                stoploss_price = (current_price - stop_distance) if symbol_state['signal']['direction'] == "BUY" else (current_price + stop_distance)
                
                # Calculate position size
                account_value = float(strategy_margins['USD'].get('equity', 0)) if strategy_margins and 'USD' in strategy_margins else self.position_value
                position_size = self.calculate_position_size(current_price, stoploss_price, account_value)
                position_value = position_size * current_price
                
                # Check funds for new positions
                if not existing_signal and not self.validate_funds_for_order(position_value, strategy_margins):
                    continue
                
                # Handle existing positions
                if existing_signal and existing_entry and existing_entry.orderDirection != orderDirection:
                    # Cancel existing stoploss orders
                    for order in existing_signal.orders:
                        if order.order_type == "STOPLOSS" and order.status == "open":
                            order.status = "cancel"
                            order.message = "Cancelled due to exit signal"
                    
                    # Use same position size as entry
                    position_size = existing_entry.orderQuantity
                    
                    # Add market exit order
                    exit_order = Order(
                        symbol=symbol,
                        orderQuantity=position_size,
                        orderDirection=symbol_state['signal']['direction'],
                        order_type=self.orderType,
                        symbol_ltp={system_timestamp: current_price},
                        timeInForce=self.timeInForce,
                        entryOrderBool=False,
                        status="pending"
                    )
                    
                    existing_signal.orders.append(exit_order)
                    existing_signal.signal_update = True
                    existing_signal.strategy_inputs['regime_filter'] = regime
                    signals.append(existing_signal)
                    return_type = 'signals'
                
                # Create new position if none exists
                elif not existing_signal:
                    # Track entry time
                    self.entry_times[symbol] = pd.to_datetime(system_timestamp)
                    
                    # Calculate position size
                    position_size = round(self.position_value / current_price, 8)
                    
                    # Create entry order
                    order = Order(
                        symbol=symbol,
                        orderQuantity=position_size,
                        orderDirection=symbol_state['signal']['direction'],
                        order_type=self.orderType,
                        symbol_ltp={system_timestamp: current_price},
                        timeInForce=self.timeInForce,
                        entryOrderBool=True,
                        status="pending"
                    )

                    # Create stoploss order
                    stoploss_order = Order(
                        symbol=symbol,
                        orderQuantity=position_size,
                        orderDirection="SELL" if symbol_state['signal']['direction'] == "BUY" else "BUY",
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
                        signal_strength=symbol_state['signal']['strength'],
                        granularity=self.granularity,
                        signal_type="BUY_SELL",
                        market_neutral=False,
                        strategy_inputs={'regime_filter': regime}
                    )
                    signals.append(signal)
                    return_type = 'signals'
            
            # Check for trailing stoploss updates
            if existing_signal:
                updated_signal = self.trail_stoploss(existing_signal, current_price, system_timestamp)
                if updated_signal:
                    updated_signal.signal_update = True
                    signals.append(existing_signal)
                    
        return return_type, signals, self.tickers
    
    def validate_funds_for_order(self, position_value, strategy_margins):
        """Check if we have enough funds for the order"""
        # Update available funds from latest margin info
        if strategy_margins and 'USD' in strategy_margins:
            self.funds_available = float(strategy_margins['USD'].get('buying_power_available', 0))
        
        if position_value > self.funds_available:
            # self.logger.warning(f"Insufficient funds - Required: ${position_value:.2f}, Available: ${self.funds_available:.2f}")
            return False
        return True
    
    def calculate_position_size(self, entry_price, stop_price, account_value):
        """Calculate position size based on risk and minimum win ratio"""
        if not entry_price or not stop_price or entry_price == stop_price:
            return 0
            
        # Calculate stop distance and required target distance
        stop_distance = abs(entry_price - stop_price)
        target_distance = stop_distance * self.min_win_ratio
        
        # For long positions, target should be achievable (not too far above recent highs)
        if entry_price > stop_price:  # Long position
            target_price = entry_price + target_distance
        else:  # Short position
            target_price = entry_price - target_distance
            
        # Calculate position size based on risk
        risk_amount = account_value * self.max_risk_pct
        position_size = risk_amount / stop_distance
        
        # Apply minimum position size
        min_position_size = 10 / entry_price  # Minimum $10 position
        position_size = max(position_size, min_position_size)
        
        # # Log position details for debugging
        # self.logger.info(f"Position Sizing Details:")
        # self.logger.info(f"  Entry: ${entry_price:.2f}")
        # self.logger.info(f"  Stop: ${stop_price:.2f} (${stop_distance:.2f} away)")
        # self.logger.info(f"  Target: ${target_price:.2f} (${target_distance:.2f} away)")
        # self.logger.info(f"  Risk/Reward: 1:{self.min_win_ratio:.1f}")
        # self.logger.info(f"  Risk Amount: ${risk_amount:.2f}")
        # self.logger.info(f"  Position Size: {position_size:.8f}")
        
        return position_size
    