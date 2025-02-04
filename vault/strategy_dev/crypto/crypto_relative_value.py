from vault.base_strategy import BaseStrategy, Signal, Order
import numpy as np
import pandas as pd
from systems.utils import create_logger, MarketDataExtractor, sleeper
pd.set_option('future.no_silent_downcasting', True)

class Strategy(BaseStrategy):
    def __init__(self, config_dict):
        super().__init__()
        self.logger = create_logger(logger_name='trader', log_level='INFO')
        self.strategy_name = 'crypto_relative_value'
        
        # -----------------------------------------
        # Existing Core Parameters
        # -----------------------------------------
        self.granularity = "1m"
        self.orderType = "MARKET"
        self.timeInForce = "DAY"
        self.stoploss_pct = 0.05      # % stop loss
        self.risk_per_trade = 0.03     # Risk 5% of available funds per trade
        self.z_score_threshold = 3.15   # Z-score threshold for trading signals
        self.lookback_window = 60      # Window for calculating mean and std dev
        
        # Weighted index assets & weights
        self.index_assets = ['BTCUSD', 'ETHUSD', 'XRPUSD']
        self.index_weights = {'BTCUSD': 1000, 'ETHUSD': 1000, 'XRPUSD': 1000}
        
        # Capital (simulated)
        self.funds_available = 0

        # -----------------------------------------
        # New Features (Configurable)
        # -----------------------------------------
        # 1) Trailing Stop
        self.trail_stop_activation = True
        self.trail_stop_buffer = self.stoploss_pct/3  # If price moves 5% from entry in our favor, shift SL

        # 2) Take-Profit
        self.take_profit_activation = True
        self.take_profit_pct = 0.07   # 5% profit

        # 3) Cooldown (bars) after closing a position
        self.cooldown_bars = 5
        self.last_close_bar = {}  # track bar index (or time) when a symbol closed

        # 4) ATR-based sizing
        self.use_atr_sizing = True
        self.atr_window = 14
        self.atr_multiplier = 1.0

        # Data input dictionary (must use existing structure)
        self.data_inputs, self.tickers = self.datafeeder_inputs()
        self.market_data_extractor = MarketDataExtractor()

    def get_name(self):
        return self.strategy_name

    def datafeeder_inputs(self):
        """Adjust columns/lookback to cover both z-score and ATR if needed."""
        tickers = [
            'BTCUSD', 'ETHUSD', 'XRPUSD', 
            'SOLUSD', 
            'DOGEUSD', 'ADAUSD', 'TRXUSD', 'SHIBUSD', 'LTCUSD', 'DOTUSD', 'UNIUSD',
            'LINKUSD', 'AVAXUSD', 'WBTCUSD', 'XLMUSD', 'SUIUSD'
        ]
        # Make the lookback big enough for both z-score and ATR
        needed_lookback = max(self.lookback_window, self.atr_window + 1)

        data_inputs = {
            self.granularity: {
                'columns': ['close', 'high', 'low'], 
                'lookback': needed_lookback
            }
        }
        return data_inputs, tickers
    
    # ----------------------------------------------------------------------
    # Helper Methods (called inside generate_signals, no signature changes)
    # ----------------------------------------------------------------------
    def _update_trailing_stops_internally(self, market_df, open_signals, current_bar):
        """
        Handle trailing stop logic inside `generate_signals` 
        without changing function signature or return type.
        """
        # We'll look at each open signal. If there's a position filled,
        # and price has moved in our favor by self.trail_stop_buffer, 
        # we adjust the stop price accordingly.
        for signal in open_signals:
            if signal.strategy_name != self.strategy_name:
                continue
            for order in signal.orders:
                # Find the entry order
                if order.entryOrderBool and order.status in ['filled', 'pending', 'open']:
                    sym         = order.symbol
                    direction   = order.orderDirection
                    # If your system sets the "executionPrice" upon fill, use that
                    entry_price = getattr(order, 'executionPrice', None)
                    if entry_price is None:
                        # fallback if no fill price is tracked:
                        entry_price = getattr(order, 'price', None)
                    # If still None, we can't do trailing
                    if entry_price is None or sym not in market_df.columns.levels[1]:
                        continue

                    # Current LTP
                    sym_closes = market_df.xs(sym, level=1, axis=1)['close']
                    current_price = sym_closes.iloc[-1]

                    # Movement in favor
                    if direction == "BUY":
                        move_in_favor = (current_price - entry_price) / entry_price
                        if move_in_favor >= self.trail_stop_buffer:
                            # Update the existing stoploss
                            self._adjust_stoploss(signal, sym, entry_price, current_price, direction)
                    else:
                        # SELL => if price goes down, it is in our favor
                        move_in_favor = (entry_price - current_price) / entry_price
                        if move_in_favor >= self.trail_stop_buffer:
                            self._adjust_stoploss(signal, sym, entry_price, current_price, direction)

    def _adjust_stoploss(self, signal, symbol, entry_price, current_price, direction):
        """
        Example logic: move the stop halfway from entry to current price.
        """
        for odr in signal.orders:
            if (odr.symbol == symbol 
                and odr.order_type == "STOPLOSS"
                and odr.status not in ["closed", "cancelled"]):
                
                if direction == "BUY":
                    # new stop = entry + 50% of (current - entry)
                    new_sl = entry_price + 0.5 * (current_price - entry_price)
                    # Ensure we never place stop above current price
                    new_sl = min(new_sl, current_price * 0.99)
                else:
                    # SELL => trailing stop moves downward
                    new_sl = entry_price - 0.5 * (entry_price - current_price)
                    # Ensure we never place stop below current price
                    new_sl = max(new_sl, current_price * 1.01)

                odr.price = new_sl
                self.logger.info(f"TRAIL-STOP Updated for {symbol} => {new_sl:.4f}")
                break

    def _calculate_atr(self, closes, highs, lows, window=14):
        """
        Basic ATR Calculation:
         1) True Range = max( high-low, abs(high-prev_close), abs(low-prev_close) )
         2) ATR = rolling mean of True Range
        """
        prev_close = closes.shift(1)
        tr1 = highs - lows
        tr2 = (highs - prev_close).abs()
        tr3 = (lows - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr

    def _compute_position_size(self, current_price, latest_atr, direction, strategy_margins):
        """
        Decide how large a position to open.
        - If ATR-based sizing is on, we use ATR to define the 'stop distance' 
        or at least to scale the final position.
        - Otherwise, fallback to a simple risk calculation = (risk amount / (price * stoploss_pct)).
        - Ensures total position value doesn't exceed 20% of available funds.
        """
        self.funds_available = strategy_margins['USD']['buying_power_available']
        risk_amount = self.funds_available * self.risk_per_trade
        max_position_value = self.funds_available * 0.2  # 20% of available funds
        position_size = 0
        # If ATR is missing, do the fallback
        if not self.use_atr_sizing or latest_atr is None or np.isnan(latest_atr):
            position_size = risk_amount / (current_price * self.stoploss_pct)
            self.logger.info({'position_size':position_size, 'current_price':current_price, 'stoploss_pct':self.stoploss_pct, 'risk_amount':risk_amount, 'risk_math':abs(current_price - (current_price * self.stoploss_pct))})
            position_size = max(1, round(position_size))
            # Ensure position value doesn't exceed max allowed
            position_size = min(position_size, max_position_value / current_price)
            position_size = max(1, round(position_size))

        # ATR-based logic:
        # e.g. set a "stop distance" to the smaller of (stoploss_pct * price) or (atr * multiplier).
        atr_stop_distance = latest_atr * self.atr_multiplier
        sl_stop_distance  = self.stoploss_pct * current_price
        if direction == "BUY":
            actual_stop_dist  = min(atr_stop_distance, sl_stop_distance)
        else:
            actual_stop_dist  = max(atr_stop_distance, sl_stop_distance)

        if actual_stop_dist <= 0:
            position_size = risk_amount / (current_price * self.stoploss_pct)
        else:
            position_size = risk_amount / actual_stop_dist

        position_size = max(1, round(position_size))
        # Ensure position value doesn't exceed max allowed
        position_size = min(position_size, max_position_value / current_price)
        
        # self.logger.info({
        #     'position_size': position_size, 
        #     'current_price': current_price, 
        #     'stoploss_pct': self.stoploss_pct, 
        #     'risk_amount': risk_amount,
        #     'atr_stop_distance': atr_stop_distance, 
        #     'sl_stop_distance': sl_stop_distance, 
        #     'actual_stop_dist': actual_stop_dist,
        #     'total_position_value': current_price * position_size,
        #     'max_position_value': max_position_value
        # })

        position_size = max(1, round(position_size))
        
        if position_size * current_price > strategy_margins['USD']['buying_power_available']:
            position_size = 0
            
        return position_size

    def generate_signals(self, next_rows, market_data_df, system_timestamp, strategy_margins, open_signals=None):
            """
            1) Builds a weighted 'market index' based on BTC, ETH, and XRP.
            2) Calculates z-score of each altcoin's relative performance vs that index.
            3) Generates or closes positions based on:
            - z-score threshold
            - trailing stops
            - take-profit
            - cooldown logic
            - ATR-based position sizing (optional)
            4) Must return (return_type, signals, self.tickers).
            """
            # ------------------------------------------------------------------
            # REQUIRED: Must end with (return return_type, signals, self.tickers)
            # ------------------------------------------------------------------
            signals = []
            return_type = None
            open_signals = open_signals or []

            # ========== Identify the numeric "bar index" for cooldown logic ==========
            # Some systems pass an integer bar count or time. Adjust to your system.
            # If your "next_rows" or "system_timestamp" is your bar index, you can store that below.
            # We'll pretend 'system_timestamp' increments by 1 each bar for demonstration.
            current_bar = system_timestamp if isinstance(system_timestamp, int) else 0
            # self.logger.info({f"{self.strategy_name} - strategy_margins": strategy_margins})
            
            # ========== Ensure we have data ==========  
            if self.granularity not in market_data_df.index.levels[0]:
                return return_type, signals, self.tickers
            market_df = market_data_df.loc[self.granularity]
            if len(market_df) < max(self.lookback_window, self.atr_window + 1):
                return return_type, signals, self.tickers

            # ------------------------------------------------------------------
            # 1) Possibly update trailing stops or handle partial exits 
            #    (We do it inside generate_signals for "single function" constraint)
            # ------------------------------------------------------------------
            if self.trail_stop_activation:
                self._update_trailing_stops_internally(market_df, open_signals, current_bar)

            # ------------------------------------------------------------------
            # 2) Construct Weighted "Market Index"
            # ------------------------------------------------------------------
            index_returns_list = []
            for symbol in self.index_assets:
                if symbol not in market_df.columns.levels[1]:
                    continue
                df_symbol = market_df.xs(symbol, level=1, axis=1)
                closes_symbol = df_symbol['close'].ffill().dropna()
                symbol_returns = closes_symbol.pct_change().fillna(0) * self.index_weights[symbol]
                index_returns_list.append(symbol_returns)

            if not index_returns_list:
                # No index can be formed
                return return_type, signals, self.tickers
            
            weighted_index_returns = pd.concat(index_returns_list, axis=1).sum(axis=1)
            # Turn returns into a "price" series for the index
            market_index_series = (1 + weighted_index_returns).cumprod()

            # ------------------------------------------------------------------
            # 3) Generate/Close Positions Per Symbol
            # ------------------------------------------------------------------
            for symbol in self.tickers:
                # Skip if symbol is an index component or not in data
                if symbol in self.index_assets or symbol not in market_df.columns.levels[1]:
                    continue

                # Enough data for this symbol?
                df_symbol = market_df.xs(symbol, level=1, axis=1)
                if len(df_symbol) < max(self.lookback_window, self.atr_window + 1):
                    continue

                # Check cooldown: if we closed a position recently, skip if not enough bars have passed
                if symbol in self.last_close_bar:
                    if (current_bar - self.last_close_bar[symbol]) < self.cooldown_bars:
                        # Skip opening a new trade in cooldown
                        continue

                # Extract close/high/low
                # Explicitly convert them to floats after ffill
                close_series = df_symbol['close'].ffill().dropna().astype(float)
                high_series  = df_symbol['high'].ffill().dropna().astype(float)
                low_series   = df_symbol['low'].ffill().dropna().astype(float)

                # 3a) ATR-based sizing calculation (optional)
                if self.use_atr_sizing:
                    atr_series = self._calculate_atr(close_series, high_series, low_series, self.atr_window)
                    latest_atr = atr_series.iloc[-1] if len(atr_series) > 0 else None
                else:
                    latest_atr = None

                # 3b) Compute relative performance vs the market index & z-score
                asset_returns = close_series.pct_change().fillna(0)
                asset_cumperf = (1 + asset_returns).cumprod()
                
                # Align index data to these timestamps
                common_index = asset_cumperf.index.intersection(market_index_series.index)
                rel_perf = asset_cumperf.loc[common_index] - market_index_series.loc[common_index]

                rolling_mean = rel_perf.rolling(self.lookback_window).mean()
                rolling_std  = rel_perf.rolling(self.lookback_window).std()

                # If we can't compute a valid z-score, skip
                if pd.isna(rolling_mean.iloc[-1]) or pd.isna(rolling_std.iloc[-1]) or rolling_std.iloc[-1] == 0:
                    continue

                z_score = (rel_perf.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
                current_price = close_series.iloc[-1]

                # 3c) See if we already have an open position for this symbol
                has_open_position = False
                new_direction     = None
                if abs(z_score) >= self.z_score_threshold:
                    new_direction = "SELL" if z_score > 0 else "BUY"
                for open_signal in open_signals:
                    if not open_signal.orders:
                        continue
                    entry_orders = [
                        o for o in open_signal.orders
                        if o.symbol == symbol and o.entryOrderBool
                    ]
                    
                    if entry_orders:
                        has_open_position = True
                        entry_order = entry_orders[0]
                        if entry_order.status not in ["closed", "cancelled"]:
                            # => We have an active position
                            # If new signal is opposite, close the old position
                            if new_direction and new_direction != entry_order.orderDirection:
                                # Cancel old stoploss/tp orders
                                for odr in open_signal.orders:
                                    if (odr.symbol == symbol and 
                                        odr.order_type in ["STOPLOSS", "LIMIT"] and
                                        odr.status not in ["closed", "cancelled"]):
                                        odr.status = "cancel"
                                        self.logger.info(f"[{symbol}] Cancelled {odr.order_type} due to opposite signal")

                                # Create exit order
                                exit_order = Order(
                                    symbol=symbol,
                                    orderQuantity=entry_order.orderQuantity,
                                    orderDirection="SELL" if entry_order.orderDirection == "BUY" else "BUY",
                                    order_type=self.orderType,
                                    symbol_ltp={system_timestamp: current_price},
                                    timeInForce=self.timeInForce,
                                    entryOrderBool=False,
                                    status="pending"
                                )
                                self.logger.info(f"""
    [{symbol}] 
    - {new_direction} EXIT TRIGGERED
    - Z-Score={z_score:.2f}
    - Current Price={current_price:.8f}
    - Position Size={position_size}
    - Stoploss={sl_price:.4f}
    - Take Profit={tp_price}
    - Position Value={position_size * current_price:.2f}
    """)
                                
                                open_signal.orders.append(exit_order)
                                signals.append(open_signal)
                                return_type = "signals"
                        break  # no need to check other open_signals
                        

                # 3d) If no open position => we may open a new one
                if (new_direction is not None) and (not has_open_position):
                    # Position sizing
                    position_size = self._compute_position_size(
                        current_price, latest_atr, new_direction, strategy_margins
                    )
                    if position_size == 0:
                        continue
                    # Market entry
                    entry_order = Order(
                        symbol=symbol,
                        orderQuantity=position_size,
                        orderDirection=new_direction,
                        order_type=self.orderType,
                        symbol_ltp={system_timestamp: current_price},
                        timeInForce=self.timeInForce,
                        entryOrderBool=True,
                        status="pending"
                    )

                    # Stop-loss
                    if new_direction == "BUY":
                        sl_price = current_price * (1 - self.stoploss_pct)
                        sl_side = "SELL"
                    else:
                        sl_price = current_price * (1 + self.stoploss_pct)
                        sl_side = "BUY"
                    stoploss_order = Order(
                        symbol=symbol,
                        orderQuantity=position_size,
                        orderDirection=sl_side,
                        order_type="STOPLOSS",
                        price=sl_price,
                        symbol_ltp={system_timestamp: current_price},
                        timeInForce=self.timeInForce,
                        entryOrderBool=False,
                        status="pending"
                    )

                    # Take-profit (if activated)
                    tp_orders = []
                    if self.take_profit_activation:
                        if new_direction == "BUY":
                            tp_price = current_price * (1 + self.take_profit_pct)
                            tp_side  = "SELL"
                        else:
                            tp_price = current_price * (1 - self.take_profit_pct)
                            tp_side  = "BUY"

                        tp_order = Order(
                            symbol=symbol,
                            orderQuantity=position_size,
                            orderDirection=tp_side,
                            order_type="LIMIT",
                            price=tp_price,
                            symbol_ltp={system_timestamp: current_price},
                            timeInForce=self.timeInForce,
                            entryOrderBool=False,
                            status="pending"
                        )
                        tp_orders.append(tp_order)

                    # Combine orders into a signal
                    all_orders = [entry_order, stoploss_order] + tp_orders
                    new_signal = Signal(
                        strategy_name=self.strategy_name,
                        timestamp=system_timestamp,
                        orders=all_orders,
                        signal_strength=abs(z_score) / self.z_score_threshold,
                        granularity=self.granularity,
                        signal_type="BUY_SELL",
                        market_neutral=True,
                        status="pending"
                    )

                    signals.append(new_signal)
                    return_type = "signals"

                    self.logger.info(f"""
    [{symbol}] 
    - {new_direction} SIGNAL TRIGGERED
    - Z-Score={z_score:.2f}
    - Current Price={current_price:.8f}
    - Position Size={position_size}
    - Stoploss={sl_price:.4f}
    - Take Profit={tp_price}
    - Position Value={position_size * current_price:.2f}
    """)
                    # sleeper(5, 'Sleeping for 5 seconds to update the list of symbols')

            # ------------------------------------------------------------------
            # END: Must return (return_type, signals, tickers)
            # ------------------------------------------------------------------
            return return_type, signals, self.tickers
