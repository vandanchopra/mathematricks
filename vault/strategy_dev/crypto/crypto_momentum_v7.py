"""
SMA 10 and SMA 50 crossover momentum strategy with 0.25% stoploss on 1-minute data
"""

from vault.base_strategy import BaseStrategy, Signal, Order
import numpy as np
import pandas as pd
from systems.utils import create_logger

class Strategy(BaseStrategy):
    def __init__(self, config_dict):
        super().__init__()
        self.logger = create_logger(logger_name='trader', log_level='INFO')
        self.strategy_name = 'crypto_momentum_v7'
        self.granularity = "1m"
        self.orderType = "MARKET"
        self.trail_step_pct = 0.005  # Increased trail step for less frequent updates
        self.atr_period = 20  # ATR lookback period
        self.atr_multiplier = 3.0  # Increased ATR multiplier for wider stops
        self.pullback_atr_ratio = 0.3  # Require 30% of ATR pullback for entry
        self.max_risk_pct = 0.02  # Maximum risk per trade as % of account
        self.min_win_ratio = 1.5  # Minimum win/loss ratio required for position sizing
        self.timeInForce = "DAY"
        self.position_value = 20000  # Fixed $10,000 position size
        self.regime_confirmation_periods = 3  # Number of periods regime must remain stable
        self.previous_regimes = {}  # Dictionary to track regime history for each symbol
        self.data_inputs, self.tickers = self.datafeeder_inputs()
        self.funds_available = 0
        
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
        if '1h' not in market_data_df.index.levels[0]:
            return False
            
        # Get hourly data
        hourly_data = market_data_df.loc['1h'].xs(symbol, axis=1, level='symbol').copy()
        if len(hourly_data) < 5:  # Need at least 5 hours of data
            return False
            
        # Calculate 3-hour and 5-hour EMAs
        close = pd.to_numeric(hourly_data['close'], errors='coerce')
        ema3 = close.ewm(span=3, adjust=False).mean()
        ema5 = close.ewm(span=5, adjust=False).mean()
        
        # Calculate slopes
        ema3_slope = (ema3.iloc[-1] - ema3.iloc[-2]) / ema3.iloc[-2]
        ema5_slope = (ema5.iloc[-1] - ema5.iloc[-2]) / ema5.iloc[-2]
        
        if direction == "BUY":
            return ema3_slope > 0 and ema5_slope > 0  # Both trends must be up
        else:
            return ema3_slope < 0 and ema5_slope < 0  # Both trends must be down
            
    def validate_entry(self, asset_data_df, direction, market_data_df, symbol):
        """Validate entry with additional filters including pullback and higher timeframe confirmation"""
        close = pd.to_numeric(asset_data_df['close'], errors='coerce')
        high = pd.to_numeric(asset_data_df['high'], errors='coerce')
        low = pd.to_numeric(asset_data_df['low'], errors='coerce')
        volume = pd.to_numeric(asset_data_df['volume'], errors='coerce')
        
        # Calculate ATR for pullback validation
        atr = self.calculate_atr(asset_data_df)
        if pd.isna(atr.iloc[-1]):
            return False
            
        required_pullback = atr.iloc[-1] * self.pullback_atr_ratio
        
        # Volume confirmation - 20-period volume SMA
        volume_sma = volume.rolling(window=20).mean()
        current_volume = volume.iloc[-1]
        volume_filter = current_volume > volume_sma.iloc[-1] * 1.2
        
        # RSI for momentum confirmation
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD for trend confirmation
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        # Higher timeframe trend check
        trend_metrics = {}
        if '1h' in market_data_df.index.levels[0]:
            hourly_data = market_data_df.loc['1h'].xs(symbol, axis=1, level='symbol').copy()
            if len(hourly_data) >= 5:
                hourly_close = pd.to_numeric(hourly_data['close'], errors='coerce')
                ema3 = hourly_close.ewm(span=3, adjust=False).mean()
                ema5 = hourly_close.ewm(span=5, adjust=False).mean()
                ema3_slope = (ema3.iloc[-1] - ema3.iloc[-2]) / ema3.iloc[-2] * 100  # Convert to percentage
                ema5_slope = (ema5.iloc[-1] - ema5.iloc[-2]) / ema5.iloc[-2] * 100  # Convert to percentage
                trend_metrics = {'3h_slope': ema3_slope, '5h_slope': ema5_slope}

        # Current timeframe metrics
        if direction == "BUY":
            recent_high = high.rolling(window=5).max().iloc[-1]
            pullback_size = (recent_high - close.iloc[-1]) / atr.iloc[-1]  # Size in ATR units
            current_metrics = {
                'vol_ratio': current_volume / volume_sma.iloc[-1],
                'rsi': rsi.iloc[-1],
                'macd_hist': histogram.iloc[-1],
                'pullback': pullback_size
            }

            # Check all conditions and construct failure message if any fail
            if len(trend_metrics) > 0 and (trend_metrics['3h_slope'] <= 0 or trend_metrics['5h_slope'] <= 0):
                # self.logger.info(f"{symbol}: Failed validation - Direction=BUY, 3h_slope={trend_metrics['3h_slope']:.2f}%, 5h_slope={trend_metrics['5h_slope']:.2f}%, Required: both > 0%")
                return False
            if current_volume <= volume_sma.iloc[-1] * 1.2:
                # self.logger.info(f"{symbol}: Failed validation - Direction=BUY, vol_ratio={current_metrics['vol_ratio']:.2f}, Required: > 1.2")
                return False
            if rsi.iloc[-1] >= 70:
                # self.logger.info(f"{symbol}: Failed validation - Direction=BUY, RSI={current_metrics['rsi']:.1f}, Required: < 70")
                return False
            if histogram.iloc[-1] <= 0:
                # self.logger.info(f"{symbol}: Failed validation - Direction=BUY, MACD_hist={current_metrics['macd_hist']:.6f}, Required: > 0")
                return False
            if pullback_size < self.pullback_atr_ratio:
                # self.logger.info(f"{symbol}: Failed validation - Direction=BUY, Pullback={current_metrics['pullback']:.2f}ATR, Required: >= {self.pullback_atr_ratio:.2f}ATR")
                return False
            return True
        else:
            recent_low = low.rolling(window=5).min().iloc[-1]
            bounce_size = (close.iloc[-1] - recent_low) / atr.iloc[-1]  # Size in ATR units
            current_metrics = {
                'vol_ratio': current_volume / volume_sma.iloc[-1],
                'rsi': rsi.iloc[-1],
                'macd_hist': histogram.iloc[-1],
                'bounce': bounce_size
            }

            # Check all conditions and construct failure message if any fail
            if len(trend_metrics) > 0 and (trend_metrics['3h_slope'] >= 0 or trend_metrics['5h_slope'] >= 0):
                # self.logger.info(f"{symbol}: Failed validation - Direction=SELL, 3h_slope={trend_metrics['3h_slope']:.2f}%, 5h_slope={trend_metrics['5h_slope']:.2f}%, Required: both < 0%")
                return False
            if current_volume <= volume_sma.iloc[-1] * 1.2:
                # self.logger.info(f"{symbol}: Failed validation - Direction=SELL, vol_ratio={current_metrics['vol_ratio']:.2f}, Required: > 1.2")
                return False
            if rsi.iloc[-1] <= 30:
                # self.logger.info(f"{symbol}: Failed validation - Direction=SELL, RSI={current_metrics['rsi']:.1f}, Required: > 30")
                return False
            if histogram.iloc[-1] >= 0:
                # self.logger.info(f"{symbol}: Failed validation - Direction=SELL, MACD_hist={current_metrics['macd_hist']:.6f}, Required: < 0")
                return False
            if bounce_size < self.pullback_atr_ratio:
                # self.logger.info(f"{symbol}: Failed validation - Direction=SELL, Bounce={current_metrics['bounce']:.2f}ATR, Required: >= {self.pullback_atr_ratio:.2f}ATR")
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
        
    def validate_funds_for_order(self, position_value, strategy_margins):
        """Check if we have enough funds for the order"""
        # Update available funds from latest margin info
        if strategy_margins and 'USD' in strategy_margins:
            self.funds_available = float(strategy_margins['USD'].get('buying_power_available', 0))
        
        if position_value > self.funds_available:
            # self.logger.warning(f"Insufficient funds - Required: ${position_value:.2f}, Available: ${self.funds_available:.2f}")
            return False
        return True
        
    def get_name(self):
        return self.strategy_name
        
    def datafeeder_inputs(self):
        tickers = [
            'BTCUSD', 'ETHUSD', 'SOLUSD', 'ADAUSD', 'XRPUSD', 'DOTUSD', 'DOGEUSD', 'LTCUSD', 'UNIUSD', 'LINKUSD'
        ]
        # Increased lookback to accommodate regime calculation
        data_inputs = {'1m': {'columns': ['open', 'high', 'close', 'low', 'volume'], 'lookback': 250},
                       '1h': {'columns': ['open', 'high', 'close', 'low', 'volume'], 'lookback': 10}}
        return data_inputs, tickers

    def get_market_regime(self, asset_data_df):
        """
        Calculate market regime score between -5 (strongly bearish) and +5 (strongly bullish)
        Uses multiple factors:
        1. Trend direction and strength (EMAs)
        2. Momentum (ROC)
        3. Volume trend
        4. Volatility state
        """
        close = pd.to_numeric(asset_data_df['close'], errors='coerce')
        volume = pd.to_numeric(asset_data_df['volume'], errors='coerce')
        
        if close.isna().any() or volume.isna().any():
            return 0  # Return neutral if data is invalid
        
        # 1. Trend Factors (0 to ±3) - Enhanced stability
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        
        trend_score = 0
        # Calculate trend slopes with safety checks
        if len(ema20) >= 5 and ema20.iloc[-5] != 0 and len(ema50) >= 5 and ema50.iloc[-5] != 0:
            ema20_slope = (ema20.iloc[-1] - ema20.iloc[-5]) / ema20.iloc[-5]
            ema50_slope = (ema50.iloc[-1] - ema50.iloc[-5]) / ema50.iloc[-5]
            
            # Short-term trend with slope confirmation
            if not np.isnan(ema20_slope) and not np.isnan(ema50_slope):
                if ema20.iloc[-1] > ema50.iloc[-1] and ema20_slope > 0:
                    trend_score += 1.5
                elif ema20.iloc[-1] < ema50.iloc[-1] and ema20_slope < 0:
                    trend_score -= 1.5
            
        # Long-term trend with strength factor
        if ema50.iloc[-1] > ema200.iloc[-1]:
            strength = min((ema50.iloc[-1] / ema200.iloc[-1] - 1) * 100, 1.5)
            trend_score += strength
        elif ema50.iloc[-1] < ema200.iloc[-1]:
            strength = min((1 - ema50.iloc[-1] / ema200.iloc[-1]) * 100, 1.5)
            trend_score -= strength
            
        # 2. Momentum Factor (0 to ±1.5) - Use EMA for smoothing with safety checks
        momentum_score = 0
        if len(close) >= 50:  # Ensure we have enough data
            if close.iloc[-20] != 0 and close.iloc[-50] != 0:  # Prevent division by zero
                roc_fast = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] * 100
                roc_slow = (close.iloc[-1] - close.iloc[-50]) / close.iloc[-50] * 100
                
                if not np.isnan(roc_fast) and not np.isnan(roc_slow):
                    roc_ema = pd.Series(roc_fast).ewm(span=10).mean().iloc[-1]  # Smooth ROC
                    
                    if not np.isnan(roc_ema):
                        if roc_ema > 1.5 and roc_slow > 0: momentum_score = 1.5
                        elif roc_ema < -1.5 and roc_slow < 0: momentum_score = -1.5
                        elif roc_ema > 1: momentum_score = 0.75
                        elif roc_ema < -1: momentum_score = -0.75
        
        # 3. Volume Factor (0 to ±0.5) - Reduced impact
        vol_ema = volume.ewm(span=20).mean()
        vol_score = 0
        if volume.iloc[-1] > vol_ema.iloc[-1] * 1.3:  # Reduced threshold
            vol_score = 0.5 if close.iloc[-1] > close.iloc[-2] else -0.5
            
        # 4. Volatility Factor (0 to ±1) - Use longer lookback and EMA
        price_returns = close.pct_change()
        current_vol = price_returns.ewm(span=30).std() * np.sqrt(252)  # Use EMA instead of simple rolling
        hist_vol = current_vol.rolling(120).mean()  # Longer historical lookback
        
        vol_score = 0
        if (len(hist_vol) > 0 and
            not np.isnan(hist_vol.iloc[-1]) and
            not np.isnan(current_vol.iloc[-1]) and
            hist_vol.iloc[-1] > 0):  # Prevent division by zero
            
            vol_ratio = current_vol.iloc[-1] / hist_vol.iloc[-1]
            if not np.isnan(vol_ratio):  # Check calculated ratio
                if vol_ratio < 0.9:  # Less strict thresholds
                    vol_score = 1
                elif vol_ratio > 1.1:
                    vol_score = -1
            
        # Combine all factors with higher weight on trend
        final_score = (trend_score * 1.5) + momentum_score + (vol_score * 0.5)
        
        # Ensure score is between -5 and +5
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
        """
        Update trailing stoploss price if price has moved by at least one trail_step_pct chunk.
        Returns the signal if stoploss was updated, None otherwise.
        """
        entry_order = None
        stoploss_order = None
        
        # Find entry and stoploss orders
        for order in signal.orders:
            if order.status == 'closed' and order.entryOrderBool:
                entry_order = order
            elif order.order_type == 'STOPLOSS' and order.status == 'open':
                stoploss_order = order
                
        if not entry_order or not stoploss_order:
            return None
            
        # Update stoploss based on entry direction
        if entry_order.orderDirection == "BUY":
            # For long positions, calculate distance to current stoploss
            distance_to_stoploss = (current_price - stoploss_order.price) / current_price
            # Only update if we can move up by at least one trail step
            steps = int(distance_to_stoploss / self.trail_step_pct)
            if steps > 0:
                # Calculate new stop based on ATR
                atr = self.calculate_atr(pd.DataFrame({'high': [current_price], 'low': [stoploss_order.price], 'close': [current_price]}))
                stop_distance = atr.iloc[-1] * self.atr_multiplier if not pd.isna(atr.iloc[-1]) else current_price * 0.01
                new_stoploss = current_price - stop_distance
                if new_stoploss > stoploss_order.price:
                    stoploss_order.price = new_stoploss
                    stoploss_order.symbol_ltp[system_timestamp] = current_price
                    return signal
        else:  # SELL position
            # For short positions, calculate distance to current stoploss
            distance_to_stoploss = (stoploss_order.price - current_price) / current_price
            # Only update if we can move down by at least one trail step
            steps = int(distance_to_stoploss / self.trail_step_pct)
            if steps > 0:
                # Calculate new stop based on ATR
                atr = self.calculate_atr(pd.DataFrame({'high': [current_price], 'low': [stoploss_order.price], 'close': [current_price]}))
                stop_distance = atr.iloc[-1] * self.atr_multiplier if not pd.isna(atr.iloc[-1]) else current_price * 0.01
                new_stoploss = current_price + stop_distance
                if new_stoploss < stoploss_order.price:
                    stoploss_order.price = new_stoploss
                    stoploss_order.symbol_ltp[system_timestamp] = current_price
                    return signal
                
        return None

    def generate_signals(self, next_rows, market_data_df, system_timestamp, strategy_margins, open_signals=None):
        """
        Generate signals based on SMA crossover strategy (10 and 50 period) with market regime filter.
        Takes into account any open signals.
        """
        signals = []
        return_type = 'signals'
        open_signals = open_signals or []

        for symbol in set(market_data_df["open"].columns):
            if(self.granularity not in market_data_df.index.levels[0] or len(market_data_df.loc[self.granularity]) <= self.data_inputs[self.granularity]['lookback']):
                continue
                
            # Calculate market index first
            market_index = pd.DataFrame()
            position_values = {}
            
            # First pass to get initial prices
            for major_coin in ['BTCUSD', 'ETHUSD', 'XRPUSD']:
                coin_data = market_data_df.loc[self.granularity].xs(major_coin, axis=1, level='symbol').copy()
                close_prices = pd.to_numeric(coin_data['close'], errors='coerce')
                initial_price = float(close_prices.iloc[0])
                position_size = 1000 / initial_price  # $1000 worth of each coin
                position_values[major_coin] = position_size
            
            # Calculate index with both price and volume
            for major_coin in ['BTCUSD', 'ETHUSD', 'XRPUSD']:
                coin_data = market_data_df.loc[self.granularity].xs(major_coin, axis=1, level='symbol').copy()
                # Convert to numeric and calculate returns
                close_prices = pd.to_numeric(coin_data['close'], errors='coerce')
                coin_returns = close_prices.pct_change()
                market_index[f"{major_coin}_pos"] = (1 + coin_returns) * position_values[major_coin]
                # Track volume-weighted contributions
                market_index[f"{major_coin}_vol"] = pd.to_numeric(coin_data['volume'], errors='coerce')
            
            # Equal-weighted market index value
            market_index['close'] = market_index[['BTCUSD_pos', 'ETHUSD_pos', 'XRPUSD_pos']].sum(axis=1)
            # Sum of volumes
            market_index['volume'] = market_index[['BTCUSD_vol', 'ETHUSD_vol', 'XRPUSD_vol']].sum(axis=1)
            market_regime = self.get_market_regime(market_index.ffill())
            
            # Individual asset analysis
            asset_data_df = market_data_df.loc[self.granularity].xs(symbol, axis=1, level='symbol').reset_index()
            asset_data_df['SMA10'] = asset_data_df['close'].rolling(window=10).mean()
            asset_data_df['SMA50'] = asset_data_df['close'].rolling(window=50).mean()

            current_price = float(asset_data_df.iloc[-1]['close'])
            
            # Calculate individual asset regime
            asset_regime = self.get_market_regime(asset_data_df)
            
            # Only consider regime valid if it aligns with market regime
            regime = asset_regime if np.sign(asset_regime) == np.sign(market_regime) else 0
            
            # Initialize regime history for symbol if not exists
            if symbol not in self.previous_regimes:
                self.previous_regimes[symbol] = []
            
            # Update regime history
            self.previous_regimes[symbol].append(regime)
            # Keep only last 2N periods to prevent memory bloat
            max_history = self.regime_confirmation_periods * 2
            if len(self.previous_regimes[symbol]) > max_history:
                self.previous_regimes[symbol] = self.previous_regimes[symbol][-max_history:]
            
            # Check regime stability and trend using only the last N periods
            regime_stable = False
            if len(self.previous_regimes[symbol]) >= self.regime_confirmation_periods:
                # Get the last N periods
                recent_regimes = self.previous_regimes[symbol][-self.regime_confirmation_periods:]
                # Check if regimes are consistently trending in one direction
                regime_signs = [1 if r > 0 else -1 if r < 0 else 0 for r in recent_regimes]
                sign_matches = sum(1 for x in regime_signs if x == regime_signs[-1])
                # Regime is stable if majority of periods have same sign and current regime is significant
                regime_stable = (sign_matches >= self.regime_confirmation_periods - 1 and
                               abs(regime) > 0.5 and regime_signs[-1] != 0)
                
            # Generate buy signal when fast SMA crosses above slow SMA and regime is trending positive
            if (round(asset_data_df.iloc[-1]['SMA10'], 2) > round(asset_data_df.iloc[-1]['SMA50'], 2)) and (round(asset_data_df.iloc[-2]['SMA10'], 2) <= round(asset_data_df.iloc[-2]['SMA50'], 2)):
                # Generate signal if regime is positive and stable, or if regime is very strongly positive
                if (regime > 0 and regime_stable) or regime > 2:
                    signal_strength = 1
                    orderDirection = "BUY"
                else:
                    signal_strength = 0
                    orderDirection = "BUY"
            # Generate sell signal when fast SMA crosses below slow SMA and regime is trending negative
            elif (round(asset_data_df.iloc[-1]['SMA10'], 2) < round(asset_data_df.iloc[-1]['SMA50'], 2)) and (round(asset_data_df.iloc[-2]['SMA10'], 2) >= round(asset_data_df.iloc[-2]['SMA50'], 2)):
                # Generate signal if regime is negative and stable, or if regime is very strongly negative
                if (regime < 0 and regime_stable) or regime < -2:
                    signal_strength = 1
                    orderDirection = "SELL"
                else:
                    signal_strength = 0
                    orderDirection = "SELL"
            else:
                signal_strength = 0
            
            existing_signal, existing_entry = self.get_existing_signal(symbol, open_signals)
            
            if signal_strength != 0:
                # Validate entry conditions including higher timeframe trend
                if not self.validate_entry(asset_data_df, orderDirection, market_data_df, symbol):
                    # self.logger.info(f"{symbol}: Failed validation checks")
                    continue
                    
                # Calculate ATR-based stop loss for position sizing
                atr = self.calculate_atr(asset_data_df)
                if pd.isna(atr.iloc[-1]):
                    continue
                    
                stop_distance = atr.iloc[-1] * self.atr_multiplier
                stoploss_price = (current_price - stop_distance) if orderDirection == "BUY" else (current_price + stop_distance)
                
                # Calculate position size based on risk
                account_value = float(strategy_margins['USD'].get('equity', 0)) if strategy_margins and 'USD' in strategy_margins else self.position_value
                position_size = self.calculate_position_size(current_price, stoploss_price, account_value)
                position_value = position_size * current_price
                
                # Check if we have enough funds (only for new positions)
                if not existing_signal and not self.validate_funds_for_order(position_value, strategy_margins):
                    self.logger.info("Signal dropped due to insufficient funds")
                    continue
                
                # Check for existing signal using helper function
                # If we have an existing position and get a reverse signal, add exit order
                if existing_signal and existing_entry and existing_entry.orderDirection != orderDirection:
                    # Cancel any existing stoploss orders
                    for order in existing_signal.orders:
                        if order.order_type == "STOPLOSS" and order.status == "open":
                            order.status = "cancel"
                            order.message = "Cancelled due to exit signal"
                    
                    # Use the same position size as entry order for exit
                    position_size = existing_entry.orderQuantity
                    
                    # Add market exit order
                    exit_direction = "SELL" if existing_entry.orderDirection == "BUY" else "BUY"
                    exit_order = Order(
                        symbol=symbol,
                        orderQuantity=position_size,
                        orderDirection=orderDirection,
                        order_type=self.orderType,
                        symbol_ltp={system_timestamp: current_price},
                        timeInForce=self.timeInForce,
                        entryOrderBool=False,
                        status="pending"
                    )
                    
                    # Log exit details
                    # self.logger.info(f"Exit Position - {symbol}:")
                    # self.logger.info(f"  Exit Price: ${current_price:.2f}")
                    # self.logger.info(f"  Size: {position_size:.8f}")
                    # self.logger.info(f"  Value: ${position_size * current_price:.2f}")
                    existing_signal.orders.append(exit_order)
                    existing_signal.signal_update = True
                    existing_signal.strategy_inputs['regime_filter'] = regime
                    signals.append(existing_signal)
                    return_type = 'signals'
                
                # If no existing position and we get a signal, create new entry
                elif not existing_signal:
                    # Calculate position size based on fixed position value
                    position_size = round(self.position_value / current_price, 8)  # Round to 8 decimals for crypto
                    
                    # Log position details
                    # self.logger.info(f"Position Details - {symbol}:")
                    # self.logger.info(f"  Price: ${current_price:.2f}")
                    # self.logger.info(f"  Size: {position_size:.8f}")
                    # self.logger.info(f"  Value: ${position_size * current_price:.2f}")
                    
                    # Create Order object
                    order = Order(
                        symbol=symbol,
                        orderQuantity=position_size,
                        orderDirection=orderDirection,
                        order_type=self.orderType,
                        symbol_ltp={system_timestamp: current_price},
                        timeInForce=self.timeInForce,
                        entryOrderBool=True,
                        status="pending"
                    )

                    # Create stoploss order with ATR-based stop loss calculated earlier
                    stoploss_order = Order(
                        symbol=symbol,
                        orderQuantity=position_size,
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
                        market_neutral=False,
                        strategy_inputs={'regime_filter': regime}  # Store regime value
                    )
                    signals.append(signal)
                    self.logger.info(f'SIGNAL GENERATED: {symbol} | Direction: {orderDirection} | Asset Regime: {asset_regime:.2f} | Market Regime: {market_regime:.2f} | Combined Regime: {regime:.2f}')
                    return_type = 'signals'
            
            # Check for trailing stoploss updates on existing positions
            if existing_signal:
                updated_signal = self.trail_stoploss(existing_signal, current_price, system_timestamp)
                if updated_signal:
                    updated_signal.signal_update = True
                    signals.append(existing_signal)
                    
        return return_type, signals, self.tickers