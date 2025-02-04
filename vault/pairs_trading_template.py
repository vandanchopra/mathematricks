"""
Advanced Pairs Trading Strategy Implementation
Uses statistical arbitrage with dynamic pair selection, 
adaptive thresholds, and market regime detection
"""

from vault.base_strategy import BaseStrategy, Signal, SignalResponse, Order
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from scipy import stats
import statsmodels.api as sm
import sys, time
from typing import Tuple, Optional, List, Dict, Any
import uuid

class Strategy(BaseStrategy):
    def __init__(self, config_dict):
        super().__init__()
        self.strategy_name = 'pairs_trading'
        self.granularity = "1d"
        # Core Parameters - Adapted for limited historical data
        self.lookback_window = 20  # Minimum window for statistical validity
        self.cointegration_threshold = 0.3  # Very relaxed threshold for startup period
        self.min_half_life = 2  # Very aggressive mean reversion period
        self.max_half_life = 30  # Shorter maximum for faster trades
        
        # Entry/Exit Parameters - Optimized for current market conditions
        self.entry_z = 1.5  # More aggressive entry for more trades
        self.exit_z = 0.75  # Allow more room for mean reversion
        self.profit_target_z = 1.25  # Faster profit taking
        self.stop_loss_pct = 0.03  # Wider stop loss for more room    
        self.max_position_pct = 0.1  # Higher position sizing for more opportunities    
        self.debug_mode = True  # Enable detailed validation logging
        
        # Risk Management - Enhanced
        self.orderType = "MARKET"
        self.timeInForce = "DAY"
        self.orderQuantity = 30  # Higher base quantity for 2011 volatility
        
        # Market Regime Parameters - More adaptive
        self.volatility_window = 20  # Shorter window for faster reaction
        self.regime_lookback = 30  # Shorter lookback for more responsive regime detection
        self.correlation_threshold = 0.55  # More lenient correlation requirement
        self.vol_threshold = 0.25  # Higher volatility tolerance for more opportunities
        
        # Historically reliable pairs for 2011 environment
        self.potential_pairs = [
            ('XOM', 'CVX'),     # Energy - strong historical relationship
            ('JPM', 'GS'),      # Financials - similar business models
            ('MS', 'GS'),       # Investment banking focus
            # ('BAC', 'JPM'),     # Large bank correlation
            # ('COP', 'CVX'),     # Oil & gas integration
            # ('HD', 'LOW'),      # Home improvement retail
            # ('USB', 'WFC'),     # Regional banking focus
            # ('MRK', 'JNJ'),     # Healthcare/pharma
        ]
        
        # Load required data
        self.data_inputs, self.tickers = self.datafeeder_inputs()
        self.validated_pairs = []

    def get_name(self):
        """Return the strategy name"""
        return self.strategy_name
        
    def datafeeder_inputs(self):
        tickers = []
        for pair in self.potential_pairs:
            tickers.extend(pair)
        tickers = list(set(tickers))
        
        data_inputs = {
            '1d': {
                'columns': ['open', 'high', 'close', 'low', 'volume'],
                'lookback': max(252, self.lookback_window)
            }
        }
        return data_inputs, tickers

    def check_correlation(self, price1, price2, min_correlation=0.6):
        """Check if prices maintain minimum correlation"""
        try:
            # Convert to pandas Series if needed
            if not isinstance(price1, pd.Series):
                price1 = pd.Series(price1)
            if not isinstance(price2, pd.Series):
                price2 = pd.Series(price2)
            
            # Calculate log returns
            returns1 = np.log(price1).diff().dropna()
            returns2 = np.log(price2).diff().dropna()
            
            # Check if we have enough data points
            if len(returns1) < 2 or len(returns2) < 2:
                return 0.0
                
            correlation = returns1.corr(returns2)
            return abs(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            self.logger.error(f"Error in correlation calculation: {e}")
            return 0.0

    def calculate_half_life(self, spread):
        """Calculate mean reversion speed using OLS"""
        lag_spread = spread.shift(1)
        delta_spread = spread - lag_spread
        lag_spread = lag_spread[1:]
        delta_spread = delta_spread[1:]
        beta = np.polyfit(lag_spread, delta_spread, 1)[0]
        half_life = -np.log(2) / beta if beta < 0 else np.inf
        return half_life

    def validate_pair(self, price1, price2):
        """Test pair for cointegration and mean reversion properties with adaptive criteria"""
        try:
            # Use log prices for better statistical properties
            log_price1 = np.log(price1)
            log_price2 = np.log(price2)
            
            # Calculate returns and correlation
            returns1 = log_price1.diff().dropna()
            returns2 = log_price2.diff().dropna()
            correlation = returns1.corr(returns2)
            # self.logger.debug(f"Return correlation: {correlation:.4f}")
            
            # Minimal data requirement for startup period
            window_size = min(len(price1), self.lookback_window)
            if window_size < 20:  # Only require 20 days of data
                self.logger.debug(f"Insufficient data points: {window_size} < 20")
                return False, None
                
            # Use available data more effectively
            lookback = min(window_size, 30)  # Cap lookback at 30 days
                
            # Use both log prices and levels for cointegration testing
            _, pvalue_levels, _ = coint(price1[-window_size:], price2[-window_size:])
            _, pvalue_logs, _ = coint(log_price1[-window_size:], log_price2[-window_size:])
            
            # Take best result from levels or logs
            best_pvalue = min(pvalue_levels, pvalue_logs)
            
            # Calculate market regime indicators
            spread = price1 - price2
            recent_vol = np.std(spread[-20:])
            full_vol = np.std(spread)
            vol_ratio = recent_vol / full_vol

            # Dynamic threshold adjustment based on market conditions
            base_coint_threshold = self.cointegration_threshold
            # Relax threshold in high volatility periods
            if vol_ratio > 1.2:
                base_coint_threshold *= 1.3
            # Relax threshold if correlation is very strong
            if correlation > 0.8:
                base_coint_threshold *= 1.2
            
            cointegration_threshold = min(0.3, base_coint_threshold * (1 + 0.5 * vol_ratio))
            correlation_threshold = 0.5 if vol_ratio < 1.2 else 0.6  # More lenient correlation requirements
            
            # self.logger.info(f"""Market condition adjustments:
            #     - Base cointegration threshold: {base_coint_threshold:.4f}
            #     - Adjusted cointegration threshold: {cointegration_threshold:.4f}
            #     - Correlation threshold: {correlation_threshold:.2f}
            #     - Volatility ratio: {vol_ratio:.2f}
            # """)

            # Evaluate pair based on combined metrics with adaptive thresholds
            strong_correlation = correlation > correlation_threshold
            good_cointegration = best_pvalue < cointegration_threshold

            # Simplified scoring system with lower thresholds
            score = 0
            
            # More lenient correlation scoring
            if correlation > 0.55:  # Lowered from 0.7
                score += 2
            elif correlation > 0.4:  # Lowered from correlation_threshold
                score += 1
            
            # More lenient cointegration scoring
            if best_pvalue < 0.25:  # Lowered from cointegration_threshold
                score += 2
            elif best_pvalue < 0.4:  # Increased from 0.3
                score += 1
            
            # Broader volatility acceptance (0-1)
            if 0.5 <= vol_ratio <= 2.0:  # Much wider acceptable range
                score += 1
            elif 0.3 <= vol_ratio <= 2.5:  # Partial credit for less ideal but still usable volatility
                score += 0.5
            
            # self.logger.info(f"""Pair validation scoring:
            #     - Correlation score: {2 if correlation > 0.55 else 1 if correlation > 0.4 else 0}
            #     - Cointegration score: {2 if best_pvalue < 0.25 else 1 if best_pvalue < 0.4 else 0}
            #     - Volatility score: {1 if 0.5 <= vol_ratio <= 2.0 else 0.5 if 0.3 <= vol_ratio <= 2.5 else 0}
            #     - Total score: {score}/5
            # """)
            
            # Extremely lenient validation for startup period
            min_score = 1.0 if window_size < 30 else 1.5  # Even lower requirement during startup
            
            if score < min_score:  # Dynamic scoring threshold
                # self.logger.debug(f"""Pair validation details:
                #     - Cointegration p-value: {best_pvalue:.4f} (threshold: {cointegration_threshold:.4f})
                #     - Correlation: {correlation:.2f} (threshold: {correlation_threshold:.2f})
                #     - Volatility ratio: {vol_ratio:.2f}
                #     - Recent volatility: {recent_vol:.4f}
                #     - Historical volatility: {full_vol:.4f}
                #     """)
                return False, None
            
            # Calculate hedge ratio using robust regression
            recent_window = min(window_size, 63)  # Last quarter for hedge ratio
            x = sm.add_constant(price2[-recent_window:])
            model = sm.RLM(price1[-recent_window:], x).fit()
            hedge_ratio = model.params.iloc[1]  # Use iloc instead of [] for positional indexing
            
            # Calculate spread
            spread = price1 - hedge_ratio * price2
            
            # More flexible mean reversion test
            half_life = self.calculate_half_life(spread)
            half_life_score = 0
            if self.min_half_life <= half_life <= self.max_half_life:
                half_life_score = 1
            elif half_life <= self.max_half_life * 1.5:  # Allow slightly longer half-lives
                half_life_score = 0.5
                
            # Relaxed stationarity test
            adf_result = adfuller(spread, maxlag=5, regression='ct')
            adf_threshold = 0.25  # More lenient threshold
            
            # Only enforce strict stationarity test for pairs with weaker correlation
            if correlation < 0.65 and adf_result[1] > adf_threshold:
                # self.logger.debug(f"Failed stationarity test for weakly correlated pair - ADF p-value: {adf_result[1]:.4f}")
                return False, None
                
            # Add half-life score to total score
            score += half_life_score
            
            # self.logger.info(f"""Additional validation metrics:
            #     - Half-life: {half_life:.1f} days (score: {half_life_score})
            #     - ADF p-value: {adf_result[1]:.4f}
            #     - Final total score: {score}/6
            # """)
            
            # Final validation check - require minimum total score
            if score < 2.5:  # Allowing for partial scores
                return False, None
            
            # Success metrics
            # self.logger.info(f"Pair validation metrics - Correlation: {correlation:.2f}, "
                        # f"Cointegration p-value: {best_pvalue:.4f}, "
                        # f"Half-life: {half_life:.1f}, Hedge ratio: {hedge_ratio:.2f}, "
                        # f"ADF p-value: {adf_result[1]:.4f}")
            return True, hedge_ratio
            
        except Exception as e:
            self.logger.error(f"Error in pair validation: {str(e)}")
            return False, None
            
    def calculate_position_sizes(self, price1, price2, hedge_ratio, vol_factor, market_data_df, system_timestamp, symbol1, symbol2):
        """Calculate position sizes based on volatility and price ratio"""
        self.market_data_df = market_data_df
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        notional1 = price1 * self.orderQuantity
        notional2 = price2 * round(self.orderQuantity * hedge_ratio)
        
        # Enhanced volatility adjustment
        vol_scaling = 1.0 / vol_factor if vol_factor > 1.0 else 1.0
        
        try:
            # Get historical prices for correlation calculation
            price1_hist = self.market_data_df.loc[self.granularity].xs(self.symbol1, axis=1, level='symbol')['close'].astype(float)
            price2_hist = self.market_data_df.loc[self.granularity].xs(self.symbol2, axis=1, level='symbol')['close'].astype(float)
            correlation_scaling = min(abs(self.check_correlation(price1_hist, price2_hist, 0.0)), 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating correlation scaling: {e}")
            correlation_scaling = 0.5  # Use conservative default
            
        size_scalar = vol_scaling * correlation_scaling
        
        # Get account size from available buying power
        account_size = self.get_account_size(market_data_df, system_timestamp)
        
        # Ensure balanced notional exposure with dynamic account size
        total_notional = notional1 + notional2
        max_notional = self.max_position_pct * account_size
        if total_notional > max_notional:
            size_scalar *= max_notional / total_notional
            
        return (
            round(self.orderQuantity * size_scalar),
            round(self.orderQuantity * hedge_ratio * size_scalar)
        )

    def detect_market_regime(self, spread, vol_window=20):
        """Detect current market regime using volatility and trend"""
        recent_vol = spread.rolling(vol_window).std()
        long_vol = spread.rolling(self.regime_lookback).std()
        
        # Volatility regime
        vol_ratio = recent_vol.iloc[-1] / long_vol.iloc[-1]
        high_vol = vol_ratio > 1.2
        
        # Trend regime
        spread_sma = spread.rolling(self.regime_lookback).mean()
        trend = (spread.iloc[-1] - spread_sma.iloc[-1]) / long_vol.iloc[-1]
        strong_trend = abs(trend) > 1.0
        
        return high_vol, strong_trend

    def get_account_size(self, market_data_df, system_timestamp):
        """Get current account size from available buying power metrics"""
        try:
            # Default to conservative estimate if metrics unavailable
            return 1000000
        except:
            self.logger.warning("Could not get account metrics, using default size")
            return 100000
            
    def adjust_thresholds(self, base_threshold, high_vol, strong_trend):
        """Adjust thresholds based on market regime"""
        if high_vol:
            base_threshold *= 1.25  # Higher threshold in high volatility
        if strong_trend:
            base_threshold *= 1.1  # Higher threshold in trending markets
        return base_threshold

    def calculate_spread(self, market_data_df, pair, system_timestamp):
        """Calculate spread with dynamic hedge ratio updates"""
        stock1, stock2 = pair
        try:
            try:
                price1 = market_data_df.loc[self.granularity].xs(stock1, axis=1, level='symbol')['close'].astype(float)
                price2 = market_data_df.loc[self.granularity].xs(stock2, axis=1, level='symbol')['close'].astype(float)
            except (KeyError, ValueError) as e:
                self.logger.debug(f"Data access error for {pair}: {e}")
                return None, None
            
            # Handle missing or invalid data
            valid_data = ~(np.isnan(price1) | np.isnan(price2) | np.isinf(price1) | np.isinf(price2))
            price1 = price1[valid_data]
            price2 = price2[valid_data]
            
            # Minimum data requirement aligned with validation
            min_required = 20  # Match the minimum requirement in validate_pair
            if len(price1) < min_required:
                # self.logger.debug(f"Insufficient data points for {pair}: {len(price1)} < {min_required}")
                return None, None
                
            # self.logger.info(f"Processing {pair} with {len(price1)} data points")
            
            # Log data availability
            # self.logger.info(f"Processing {pair} with {len(price1)} valid data points")
                
            # Allow more pairs through to validation by skipping initial correlation filter
            is_valid, hedge_ratio = self.validate_pair(price1, price2)
            if not is_valid:
                # self.logger.debug(f"Pair {pair} failed validation tests")
                return None, None
            
            # self.logger.info(f"Pair {pair} passed all validation tests with hedge ratio {hedge_ratio:.2f}")
                
            spread = price1 - hedge_ratio * price2
            return spread, hedge_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating spread for pair {pair}: {e}")
            return None, None

    def compute_zscore(self, spread, window=None):
        """Calculate z-score with dynamic lookback"""
        if spread is None or len(spread) < 2:
            return None
            
        if window is None:
            window = min(len(spread), self.lookback_window)
            
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        zscore = (spread - rolling_mean) / rolling_std
        return zscore.iloc[-1]

    def generate_signals(self, next_rows, market_data_df, system_timestamp, total_buying_power=0.0, buying_power_used=0.0, open_signals=None):
        """Generate trading signals with dynamic pair selection and risk management"""
        if open_signals is None:
            open_signals = []
            
        signals = []
        return_type = None

        if self.granularity not in market_data_df.index.levels[0]:
            return None, signals, self.tickers

        
        # Track active pairs to avoid duplicate trades
        active_pairs = set()
        for signal in open_signals:
            # Get the strategy_order_id from signal inputs
            open_pair = set([order.symbol for order in signal.orders])
            active_pairs.update(open_pair)
        
        # Validate and update pairs
        self.validated_pairs = []
        
        for pair in self.potential_pairs:
            spread, hedge_ratio = self.calculate_spread(market_data_df, pair, system_timestamp)
            if spread is not None and hedge_ratio is not None:
                self.validated_pairs.append((pair, hedge_ratio))

        # Generate signals for validated pairs
        for (pair, hedge_ratio) in self.validated_pairs:
            stock1, stock2 = pair
            
            # Skip pairs with active positions            
            # self.logger.info({'pair': pair, 'active_pairs': active_pairs})
            # check if the two symbols in the pair are in any of the active pairs
            if pair in active_pairs:
                self.logger.info(f"Skipping pair {pair} due to active positions")
                self.logger.info({'active_pairs': active_pairs})
                continue
                
            spread, _ = self.calculate_spread(market_data_df, pair, system_timestamp)
            if spread is None:
                continue
            
            # Market regime detection
            high_vol, strong_trend = self.detect_market_regime(spread)
            
            # Calculate adjusted thresholds
            entry_threshold = self.adjust_thresholds(self.entry_z, high_vol, strong_trend)
            current_z = self.compute_zscore(spread)
            
            if current_z is None:
                self.logger.debug(f"Could not compute z-score for {pair}")
                continue

            # Calculate volatility factor for position sizing
            vol_factor = spread.rolling(self.volatility_window).std().iloc[-1] / spread.std()
            vol_factor = np.clip(vol_factor, 0.8, 1.5)

            # Log pair evaluation details
            # self.logger.info(f"Evaluating pair {pair} - Z-score: {current_z:.2f}, "
            #                f"Entry threshold: {entry_threshold:.2f}, "
            #                f"Vol factor: {vol_factor:.2f}")

            if abs(current_z) > entry_threshold:
                # self.logger.info(f"Generating signal for pair {pair} with z-score {current_z:.2f}")
                is_long = current_z < -entry_threshold
                
                try:
                    price1 = float(market_data_df.loc[self.granularity].xs(stock1, axis=1, level='symbol')['close'].iloc[-1])
                    price2 = float(market_data_df.loc[self.granularity].xs(stock2, axis=1, level='symbol')['close'].iloc[-1])
                    
                    # Dynamic position sizing with market data context
                    size1, size2 = self.calculate_position_sizes(
                        price1, price2, hedge_ratio, vol_factor,
                        market_data_df, system_timestamp,
                        stock1, stock2  # Pass symbol names for correlation calculation
                    )
                    
                    # Enhanced dynamic stop loss based on volatility and z-score
                    base_stop = self.stop_loss_pct * (1 + (vol_factor - 1) * 0.5)
                    z_factor = min(abs(current_z) / self.entry_z, 2.0)
                    stop_loss = base_stop * (2.0 - z_factor)  # Tighter stops for extreme z-scores
                    strategy_order_id = f"{stock1}_{stock2}_{uuid.uuid4().hex}"
                    
                    # Create a single signal for the pair
                    signal = Signal(
                        strategy_name=self.strategy_name,
                        timestamp=system_timestamp,
                        signal_strength=int(min(round(abs(current_z)), 10)),
                        granularity=self.granularity,
                        signal_type='BUY_SELL',
                        market_neutral=True,
                        total_buying_power=total_buying_power,
                        buying_power_used=buying_power_used
                    )
                    
                    # Add orders for both legs of the pair
                    orders = []
                    for stock, size, price, is_first_leg in [(stock1, size1, price1, True), (stock2, size2, price2, False)]:
                        # Main order
                        order = Order(
                            symbol=stock,
                            orderQuantity=abs(size),
                            orderDirection="BUY" if (is_long == is_first_leg) else "SELL",
                            order_type=self.orderType,
                            entryOrderBool=True,
                            symbol_ltp={system_timestamp: price},
                            status="pending",
                            timeInForce=self.timeInForce
                        )
                        orders.append(order)
                        
                        # Stoploss order (5% away from entry)
                        is_buy = order.orderDirection == "BUY"
                        stoploss_price = price * (0.95 if is_buy else 1.05)  # 5% below for buys, 5% above for sells
                        stoploss_order = Order(
                            symbol=stock,
                            orderQuantity=abs(size),
                            orderDirection="SELL" if is_buy else "BUY",  # Opposite of main order
                            order_type="STOPLOSS",
                            price=stoploss_price,
                            entryOrderBool=False,
                            symbol_ltp={system_timestamp: price},
                            status="pending",
                            timeInForce=self.timeInForce
                        )
                        orders.append(stoploss_order)
                        
                    signal.orders = orders
                    signal.strategy_inputs = {'strategy_order_id': strategy_order_id}
                    signals.append(signal)
                    self.logger.info(f"""
                                        SIGNAL GENERATED:
                                        - Strategy: {signal.strategy_name}
                                        - Strength: {signal.signal_strength}
                                        - Orders:
                                        - {stock1}: Direction={orders[0].orderDirection}, Qty={orders[0].orderQuantity}, Price=${price1:.2f}, Value=${(orders[0].orderQuantity * price1):.2f}
                                        - {stock2}: Direction={orders[1].orderDirection}, Qty={orders[1].orderQuantity}, Price=${price2:.2f}, Value=${(orders[1].orderQuantity * price2):.2f}
                                        - Z-Score: {current_z:.2f}
                                        - Status: {signal.status}
                                        """)
                    # input("Press Enter to continue...")
                    return_type = 'signals'

                except (ValueError, TypeError) as e:
                    self.logger.error(f"Error processing prices for pair {pair}: {e}")
                    continue
        return return_type, signals, self.tickers