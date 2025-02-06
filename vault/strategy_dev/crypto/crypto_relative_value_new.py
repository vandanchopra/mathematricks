from vault.base_strategy import BaseStrategy, Signal, Order
import numpy as np
import pandas as pd
from systems.utils import create_logger
pd.set_option('future.no_silent_downcasting', True)

class Strategy(BaseStrategy):
    def __init__(self, config_dict):
        super().__init__()
        self.logger = create_logger(logger_name='trader', log_level='INFO')
        self.strategy_name = 'crypto_relative_value_new'
        
        # Core Parameters
        self.granularity = "1m"
        self.orderType = "MARKET"
        self.timeInForce = "DAY"
        
        # Z-Score Parameters
        self.lookback_window = 90       # 90-minute lookback for z-score
        self.entry_z_score = 2.3        # Entry threshold
        self.exit_z_score = 0.5         # Exit threshold
        
        # Index Construction
        self.index_assets = {
            'BTCUSD': 0.50,    # 50% weight
            'ETHUSD': 0.30,    # 30% weight
            'XRPUSD': 0.20     # 20% weight
        }
        
        # Risk Parameters
        self.risk_per_trade = 0.0025    # Risk 0.25% per trade
        self.fixed_risk_pct = 0.0025    # 0.25% stop loss
        self.min_rr_ratio = 3.0         # Minimum risk-reward ratio
        self.take_profit_pct = 0.0075   # 0.75% take profit (3:1 RR)
        self.trail_pct = 0.0015         # 0.15% trailing stop buffer
        self.primary_tp_ratio = 0.7     # 70% at fixed target, 30% trailing
        
        # Volume Analysis - Only detect toxic trades
        self.vol_ma_window = 15         # 15-period volume MA for confirmation
        self.vol_spike_threshold = 10.0  # Toxic volume detection (only extremely unusual volume)
        self.vol_slope_window = 5        # Window for volume trend analysis
        self.cumulative_delta_window = 5 # 5-minute window for buy/sell pressure
        
        # HMA Parameters
        self.hma_period = 50            # Hull Moving Average period
        self.hma_min_slope = 15         # Minimum slope in degrees
        
        # Circuit Breakers
        self.daily_loss_limit = 0.015   # Stop trading after 1.5% drawdown
        self.daily_profit_target = 0.03  # Lock profits at 3% gain
        self.max_concurrent_trades = 30   # Maximum number of open positions
        self.cooldown_minutes = 3        # Minutes between trades
        self.max_trade_duration = 180      # Maximum trade duration in minutes
        
        # Position Tracking
        self.active_positions = {}       # Track active positions
        self.daily_pnl = 0              # Track daily PnL
        self.last_trade_time = {}       # Track last trade time per symbol
        self.funds_available = 0
        
        # Data Requirements
        self.data_inputs, self.tickers = self.datafeeder_inputs()
        
    def datafeeder_inputs(self):
        """Define required data inputs for the strategy."""
        tickers = [
            'BTCUSD', 'ETHUSD', 'XRPUSD',  # Index components
            'SOLUSD', 'AVAXUSD', 'MATICUSD', 'DOGEUSD', 
            'LTCUSD', 'DOTUSD', 'LINKUSD'
        ]
        
        # Calculate maximum lookback needed
        max_lookback = max(
            self.lookback_window * 2,    # Z-score calculation
            self.hma_period * 2,         # HMA calculation
            self.vol_ma_window * 2       # Volume analysis
        )
        
        data_inputs = {
            self.granularity: {
                'columns': ['close', 'high', 'low', 'volume'],
                'lookback': max_lookback
            }
        }
        return data_inputs, tickers
    
    def _calculate_weighted_index(self, market_df):
        """Calculate the weighted crypto index."""
        index_returns_list = []
        weights = []
        
        for symbol, weight in self.index_assets.items():
            if symbol not in market_df.columns.levels[1]:
                continue
                
            df_symbol = market_df.xs(symbol, level=1, axis=1)
            closes = df_symbol['close'].ffill().dropna()
            
            # Calculate returns
            returns = closes.pct_change().fillna(0)
            index_returns_list.append(returns * weight)
            weights.append(weight)
            
        if not index_returns_list:
            return None
            
        # Normalize weights if some components are missing
        weights_sum = sum(weights)
        if weights_sum < 1:
            index_returns_list = [ret * (1/weights_sum) for ret in index_returns_list]
            
        weighted_returns = sum(index_returns_list)
        return (1 + weighted_returns).cumprod()
    
    def _calculate_relative_zscore(self, asset_prices, index_prices):
        """Calculate z-score of relative performance."""
        asset_returns = asset_prices.pct_change().fillna(0)
        asset_cumperf = (1 + asset_returns).cumprod()
        
        # Get relative performance
        rel_perf = asset_cumperf - index_prices
        
        # Calculate rolling z-score
        roll_mean = rel_perf.rolling(self.lookback_window).mean()
        roll_std = rel_perf.rolling(self.lookback_window).std()
        
        if pd.isna(roll_mean.iloc[-1]) or pd.isna(roll_std.iloc[-1]) or roll_std.iloc[-1] == 0:
            return None
            
        return (rel_perf.iloc[-1] - roll_mean.iloc[-1]) / roll_std.iloc[-1]
    
    def _calculate_volatility_metrics(self, df_symbol):
        """
        Calculate volatility metrics including ATR and median comparisons.
        Used for position sizing and spread toxicity checks.
        """
        closes = df_symbol['close'].astype(float)
        highs = df_symbol['high'].astype(float)
        lows = df_symbol['low'].astype(float)
        
        # Calculate True Range
        prev_close = closes.shift(1)
        tr1 = highs - lows
        tr2 = (highs - prev_close).abs()
        tr3 = (lows - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Current ATR (14 periods)
        current_atr = float(true_range.rolling(window=14).mean().iloc[-1])
        
        # 24h median ATR (if we have enough data)
        atr_24h = float(true_range.rolling(window=1440).median().iloc[-1])
        
        # Position size adjustment based on volatility
        vol_adjustment = 0.6 if current_atr > 2.0 * atr_24h else 1.0
            
        return {
            'current_atr': current_atr,
            'atr_24h_median': atr_24h,
            'position_size_adjustment': vol_adjustment
        }

    def _calculate_hma(self, series, period):
        """Calculate Hull Moving Average."""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # Calculate weighted moving averages
        wma1 = series.rolling(window=half_period).mean()
        wma2 = series.rolling(window=period).mean()
        
        # Calculate HMA: WMA(2*WMA(n/2) - WMA(n))
        wma_diff = 2 * wma1 - wma2
        hma = wma_diff.rolling(window=sqrt_period).mean()
        
        return hma
    
    def _analyze_volume(self, df_symbol, direction=None):
        """Analyze volume conditions and buying/selling pressure."""
        volume = df_symbol['volume']
        current_volume = volume.iloc[-1]
        
        # 1. Toxic volume check only
        vol_ma = volume.rolling(window=self.vol_ma_window).mean()
        vol_ratio = current_volume / vol_ma.iloc[-1]
        is_spike = vol_ratio > self.vol_spike_threshold
        
        # 2. Only detect heavy opposing pressure
        highs = df_symbol['high'][-self.cumulative_delta_window:].astype(float)
        lows = df_symbol['low'][-self.cumulative_delta_window:].astype(float)
        closes = df_symbol['close'][-self.cumulative_delta_window:].astype(float)
        volumes = volume[-self.cumulative_delta_window:].astype(float)
        
        # Calculate net pressure
        pos_in_candle = (closes - lows) / (highs - lows)
        buy_pressure = (pos_in_candle * volumes).sum()
        sell_pressure = ((1 - pos_in_candle) * volumes).sum()
        cum_delta = buy_pressure - sell_pressure
        
        # Only check for strong counter pressure
        # Check for extreme counter pressure (2x average volume)
        pressure_toxic = False
        if direction:
            avg_volume = volume.mean()
            pressure_toxic = (
                (direction == "BUY" and cum_delta < -2 * avg_volume) or  # Extreme selling
                (direction == "SELL" and cum_delta > 2 * avg_volume)     # Extreme buying
            )
        
        return {
            'is_spike': is_spike,
            'pressure_toxic': pressure_toxic,
            'entry_volume': current_volume  # Store for reference
        }
    
    def _validate_entry(self, df_symbol, direction, z_score):
        """
        Validate trade entry conditions.
        Returns: (is_valid, reason)
        """
        current_price = df_symbol['close'].iloc[-1]
        
        # Only check for toxic conditions
        vol_analysis = self._analyze_volume(df_symbol, direction)
        
        # 1. Check for extreme volume spike
        if vol_analysis['is_spike']:
            return False, "Toxic volume spike detected"
        
        # 2. Check for extreme counter pressure
        if vol_analysis['pressure_toxic']:
            return False, "Strong adverse pressure"
            
        # 3. Check HMA trend alignment (only for clear trends)
        hma = self._calculate_hma(df_symbol['close'], self.hma_period)
        if not pd.isna(hma.iloc[-1]):
            # Calculate HMA slope in degrees
            hma_slope = np.arctan((hma.iloc[-1] - hma.iloc[-2])/current_price) * 180/np.pi
            
            # Only block trades against very strong trends (reduced sensitivity)
            strong_counter_trend = (
                (direction == "BUY" and hma_slope < -45) or
                (direction == "SELL" and hma_slope > 45)
            )
            if strong_counter_trend:
                return False, f"Strong counter trend: {hma_slope:.1f}°"
            
        # 3. Check risk-reward ratio
        stop_distance = current_price * self.fixed_risk_pct
        target_distance = current_price * self.take_profit_pct
        
        rr_ratio = target_distance / stop_distance
        if rr_ratio < self.min_rr_ratio:
            return False, f"Insufficient RR ratio: {rr_ratio:.1f}"
            
        return True, "Entry validated"
    
    def _check_circuit_breakers(self, symbol, system_timestamp):
        """Check various circuit breakers before allowing new trades."""
        
        # 1. Check daily loss limit
        if self.daily_pnl <= -self.daily_loss_limit * self.funds_available:
            return False, "Daily loss limit reached"
            
        # 2. Check daily profit target
        if self.daily_pnl >= self.daily_profit_target * self.funds_available:
            return False, "Daily profit target reached"
            
        # 3. Check concurrent trades limit
        if len(self.active_positions) >= self.max_concurrent_trades:
            return False, "Maximum concurrent trades reached"
            
        # 4. Check symbol cooldown
        if symbol in self.last_trade_time:
            time_since_last = system_timestamp - self.last_trade_time[symbol]
            if time_since_last < pd.Timedelta(minutes=self.cooldown_minutes):
                return False, f"Symbol in cooldown for {self.cooldown_minutes - time_since_last.total_seconds()/60:.1f} more minutes"
                
        # 5. Check trade duration for existing positions
        if symbol in self.active_positions:
            trade_duration = system_timestamp - self.active_positions[symbol]['entry_time']
            if trade_duration >= pd.Timedelta(minutes=self.max_trade_duration):
                return False, "Maximum trade duration reached"
                
        return True, "Circuit breakers clear"
    
    def _compute_position_size(self, current_price, df_symbol, strategy_margins):
        """Calculate position size based on risk parameters, volatility, and max value."""
        risk_amount = float(self.funds_available) * self.risk_per_trade
        stop_distance = float(current_price) * self.fixed_risk_pct
        
        # Calculate base position size based on risk
        position_size = risk_amount / stop_distance
        
        # Apply volatility adjustment
        vol_metrics = self._calculate_volatility_metrics(df_symbol)
        position_size = float(position_size) * vol_metrics['position_size_adjustment']
        
        # Calculate maximum position size based on 5% of available funds
        max_trade_value = float(self.funds_available) * 0.05  # 5% limit
        max_size_by_value = int(max_trade_value / current_price)
        
        # Ensure we don't exceed available margin or 5% limit
        max_margin_size = int(strategy_margins['USD']['buying_power_available'] / current_price)
        position_size = min(position_size, max_size_by_value, max_margin_size)
        
        # Round to whole units and enforce minimum
        position_size = max(1, round(position_size))
        
        # Split for primary/trailing take profit
        primary_size = int(position_size * self.primary_tp_ratio)  # 70% fixed
        trailing_size = position_size - primary_size  # 30% trailing
        
        return position_size, primary_size, trailing_size

    def _create_orders(self, symbol, direction, current_price, position_sizes, system_timestamp, entry_vol_data):
        """Create entry, stop loss, and take profit orders with split targets."""
        orders = []
        total_size, primary_size, trailing_size = position_sizes
        
        # 1. Entry Order
        entry_order = Order(
            symbol=symbol,
            orderQuantity=total_size,
            orderDirection=direction,
            order_type=self.orderType,
            symbol_ltp={system_timestamp: current_price},
            timeInForce=self.timeInForce,
            entryOrderBool=True,
            status="pending"
        )
        orders.append(entry_order)
        
        # Calculate stop and target levels
        if direction == "BUY":
            sl_price = current_price * (1 - self.fixed_risk_pct)
            tp_price = current_price * (1 + self.take_profit_pct)
            trail_activation = current_price * (1 + self.take_profit_pct * 0.5)  # Activate at 50% of target
            sl_direction = "SELL"
        else:
            sl_price = current_price * (1 + self.fixed_risk_pct)
            tp_price = current_price * (1 - self.take_profit_pct)
            trail_activation = current_price * (1 - self.take_profit_pct * 0.5)
            sl_direction = "BUY"
            
        # 2. Stop Loss Order
        stop_loss_order = Order(
            symbol=symbol,
            orderQuantity=total_size,
            orderDirection=sl_direction,
            order_type="STOPLOSS",
            price=sl_price,
            symbol_ltp={system_timestamp: current_price},
            timeInForce=self.timeInForce,
            entryOrderBool=False,
            status="pending"
        )
        orders.append(stop_loss_order)
        
        # 3. Primary Take Profit Order (70%)
        if primary_size > 0:
            take_profit_order = Order(
                symbol=symbol,
                orderQuantity=primary_size,
                orderDirection=sl_direction,
                order_type="LIMIT",
                price=tp_price,
                symbol_ltp={system_timestamp: current_price},
                timeInForce=self.timeInForce,
                entryOrderBool=False,
                status="pending"
            )
            orders.append(take_profit_order)
        
        # 4. Initial Stop for Trailing Portion (30%)
        if trailing_size > 0:
            trailing_stop = Order(
                symbol=symbol,
                orderQuantity=trailing_size,
                orderDirection=sl_direction,
                order_type="STOPLOSS",
                price=sl_price,  # Start at regular stop level
                symbol_ltp={system_timestamp: current_price},
                timeInForce=self.timeInForce,
                entryOrderBool=False,
                status="pending"
            )
            orders.append(trailing_stop)
            
            # Store trailing stop info in position data
            # Create position data if it doesn't exist yet
            if symbol not in self.active_positions:
                self.active_positions[symbol] = {}
                
            self.active_positions[symbol].update({
                'trailing_size': trailing_size,
                'trailing_stop': sl_price,
                'highest_price': current_price if direction == "BUY" else float('inf'),
                'lowest_price': current_price if direction == "SELL" else float('-inf'),
                'direction': direction  # Ensure direction is set
            })
            
        return orders
        
    def _check_exit_conditions(self, df_symbol, position_data, system_timestamp):
        """Check time-based exit and trailing stop conditions."""
        exit_reason = None
        
        # 1. Check time-based exit (3 minute max)
        trade_duration = system_timestamp - position_data['entry_time']
        if trade_duration >= pd.Timedelta(minutes=self.max_trade_duration):
            exit_reason = f"Time-based exit ({self.max_trade_duration} mins)"
            
        # 2. Update and check trailing stop (if applicable)
        if 'trailing_stop' in position_data:
            current_price = float(df_symbol['close'].iloc[-1])
            direction = position_data['direction']
            
            if direction == "BUY":
                # Update highest price seen
                if current_price > position_data['highest_price']:
                    position_data['highest_price'] = current_price
                    # New stop is highest price minus trail buffer
                    new_stop = current_price * (1 - self.trail_pct)
                    # Only move stop up, never down
                    if new_stop > position_data['trailing_stop']:
                        position_data['trailing_stop'] = new_stop
                # Check if trailing stop hit
                if current_price <= position_data['trailing_stop']:
                    exit_reason = "Trailing stop hit"
            else:  # SELL
                # Update lowest price seen
                if current_price < position_data['lowest_price']:
                    position_data['lowest_price'] = current_price
                    # New stop is lowest price plus trail buffer
                    new_stop = current_price * (1 + self.trail_pct)
                    # Only move stop down, never up
                    if new_stop < position_data['trailing_stop']:
                        position_data['trailing_stop'] = new_stop
                # Check if trailing stop hit
                if current_price >= position_data['trailing_stop']:
                    exit_reason = "Trailing stop hit"
            
        return exit_reason is not None, exit_reason
    
    def _find_signal_for_symbol(self, symbol, open_signals):
        """Find the existing signal for a symbol."""
        for signal in open_signals:
            if (signal.strategy_name == self.strategy_name and
                any(o.symbol == symbol and o.entryOrderBool for o in signal.orders)):
                return signal
        return None
    
    def generate_signals(self, next_rows, market_data_df, system_timestamp, strategy_margins, open_signals=None):
        """Main signal generation logic."""
        signals = []
        return_type = None
        open_signals = open_signals or []
        self.funds_available = strategy_margins['USD']['buying_power_available']
        
        # Basic data validation
        if self.granularity not in market_data_df.index.levels[0]:
            return return_type, signals, self.tickers
            
        market_df = market_data_df.loc[self.granularity]
        if len(market_df) < self.lookback_window * 2:
            return return_type, signals, self.tickers
            
        # Calculate weighted index
        market_index = self._calculate_weighted_index(market_df)
        if market_index is None:
            return return_type, signals, self.tickers
            
        # Process each symbol
        for symbol in self.tickers:
            # Skip index components
            if symbol in self.index_assets:
                continue
                
            # Get symbol data
            if symbol not in market_df.columns.levels[1]:
                continue
                
            df_symbol = market_df.xs(symbol, level=1, axis=1)
            closes = df_symbol['close'].ffill().dropna()
            current_price = closes.iloc[-1]
            
            # Calculate z-score
            z_score = self._calculate_relative_zscore(closes, market_index)
            if z_score is None:
                continue
                
            # 1. Check existing positions first
            if symbol in self.active_positions:
                position_data = self.active_positions[symbol]
                should_exit = False
                exit_reason = None
                
                # Check exits based on z-score mean reversion
                if abs(z_score) <= self.exit_z_score:
                    should_exit = True
                    exit_reason = "Mean reversion target reached"
                
                # Check volume and time-based exits
                if not should_exit:
                    should_exit, exit_reason = self._check_exit_conditions(
                        df_symbol, position_data, system_timestamp
                    )
                
                if should_exit:
                    # Find the original signal
                    signal = self._find_signal_for_symbol(symbol, open_signals)
                    if signal:
                        # Create exit order
                        exit_direction = "SELL" if position_data['direction'] == "BUY" else "BUY"
                        exit_order = Order(
                            symbol=symbol,
                            orderQuantity=position_data['size'],
                            orderDirection=exit_direction,
                            order_type="MARKET",
                            symbol_ltp={system_timestamp: current_price},
                            timeInForce=self.timeInForce,
                            entryOrderBool=False,
                            status="pending"
                        )
                        
                        # Cancel existing orders
                        for order in signal.orders:
                            if not order.entryOrderBool and order.status not in ["closed", "cancelled"]:
                                order.status = "cancel"
                        
                        signal.orders.append(exit_order)
                        signals.append(signal)
                        
                        self.logger.info(f"[{symbol}] Exiting position: {exit_reason}")
                    
                    return_type = "signals"
                    continue
            
            # 2. Entry Logic
            # Determine potential trade direction
            new_direction = None
            if abs(z_score) >= self.entry_z_score:
                new_direction = "SELL" if z_score > 0 else "BUY"
                
            if new_direction:
                # Check circuit breakers
                can_trade, reason = self._check_circuit_breakers(symbol, system_timestamp)
                if not can_trade:
                    self.logger.info(f"Circuit breaker triggered for {symbol}: {reason}")
                    continue
                    
                # Get volume data for entry validation
                vol_analysis = self._analyze_volume(df_symbol, new_direction)
                
                # Validate entry conditions
                is_valid, reason = self._validate_entry(df_symbol, new_direction, z_score)
                if not is_valid:
                    self.logger.info(f"{symbol} {new_direction} signal rejected - {reason}")
                    continue
                    
                # Calculate position sizes (total, primary, trailing)
                position_sizes = self._compute_position_size(current_price, df_symbol, strategy_margins)
                if position_sizes[0] == 0:
                    self.logger.info(f"{symbol} - Invalid position size")
                    continue
                    
                # Create orders (entry, stop loss, split take profits)
                orders = self._create_orders(
                    symbol, new_direction, current_price,
                    position_sizes, system_timestamp,
                    vol_analysis
                )
                
                # Create signal
                new_signal = Signal(
                    strategy_name=self.strategy_name,
                    timestamp=system_timestamp,
                    orders=orders,
                    signal_strength=abs(z_score) / self.entry_z_score,
                    granularity=self.granularity,
                    signal_type="BUY_SELL",
                    market_neutral=True,
                    status="pending"
                )
                
                # Update tracking
                self.active_positions[symbol] = {
                    'direction': new_direction,
                    'entry_time': system_timestamp,
                    'entry_price': current_price,
                    'size': position_sizes[0],
                    'entry_volume': vol_analysis['entry_volume']
                }
                self.last_trade_time[symbol] = system_timestamp
                
                # Log the signal
                self.logger.info(f"""
                [{symbol}] {new_direction} SIGNAL
                - Z-Score: {z_score:.2f}
                - Entry: {current_price:.8f}
                - Stop: {orders[1].price:.8f} ({self.fixed_risk_pct:.2%})
                - Target 1 ({position_sizes[1]}): {orders[2].price:.8f} ({self.take_profit_pct:.2%})
                - Target 2 ({position_sizes[2]}): Trailing {self.trail_pct:.2%}
                - Total Size: {position_sizes[0]}
                - Value: ${position_sizes[0] * current_price:.2f}
                """)
                
                signals.append(new_signal)
                return_type = "signals"
        
        return return_type, signals, self.tickers
