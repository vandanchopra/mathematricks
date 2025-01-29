"""
Create a dual SMA crossover strategy without stop losses
"""

from vault.base_strategy import BaseStrategy, Signal, Order
import numpy as np
import sys, time

class Strategy(BaseStrategy):
    def __init__(self, config_dict):
        super().__init__()
        self.strategy_name = 'strategy_1'
        self.granularity = "1d"
        self.orderType = "MARKET"
        self.stoploss_pct = 0.05  # 5% stoploss
        self.exit_order_type = "stoploss_pct" #sl_pct , sl_abs
        self.timeInForce = "DAY"    #DAY, Expiry, IoC (immediate or cancel) , TTL (Order validity in minutes) 
        self.orderQuantity = 10
        self.data_inputs, self.tickers = self.datafeeder_inputs()
        
    def get_name(self):
        return self.strategy_name
        
    def datafeeder_inputs(self):
        tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'HBNC', 'NFLX', 'GS', 'AMD', 'XOM', 'JNJ', 'JPM', 'V', 'PG', 'UNH', 'DIS', 'HD', 'CRM', 'NKE']
        data_inputs = {'1d': {'columns': ['open', 'high', 'close', 'low', 'volume'] , 'lookback':52}}
        return data_inputs, tickers
        
    def generate_signals(self, next_rows, market_data_df, system_timestamp, open_signals=None):
        """
        Generate signals based on dual SMA crossover strategy (20 and 50 period). Takes into account any open signals.
        """
        signals = []
        return_type = None
        open_signals = open_signals or []

        for symbol in set(market_data_df["open"].columns):
            # If 
            if(self.granularity not in market_data_df.index.levels[0] or len(market_data_df.loc[self.granularity]) <= self.data_inputs[self.granularity]['lookback']):
                continue
            asset_data_df = market_data_df.loc[self.granularity].xs(symbol, axis=1, level='symbol').reset_index()
            asset_data_df['SMA20'] = asset_data_df['close'].rolling(window=20).mean()
            asset_data_df['SMA50'] = asset_data_df['close'].rolling(window=50).mean()

            asset_data_df['signal_strength'] = 0
            asset_data_df.loc[50:, 'signal_strength'] = np.where(asset_data_df.loc[50:, 'SMA20'] > asset_data_df.loc[50:, 'SMA50'], 1, 0)
            asset_data_df['position'] = asset_data_df['signal_strength'].diff()
            current_price = asset_data_df.iloc[-1]['close']
            
            # Generate buy signal when fast SMA crosses above slow SMA
            if (round(asset_data_df.iloc[-1]['SMA20'], 2) > round(asset_data_df.iloc[-1]['SMA50'], 2)) and (round(asset_data_df.iloc[-2]['SMA20'], 2) <= round(asset_data_df.iloc[-2]['SMA50'], 2)):
                signal_strength = 1
                orderDirection = "BUY"
            # Generate sell signal when fast SMA crosses below slow SMA
            elif (round(asset_data_df.iloc[-1]['SMA20'], 2) < round(asset_data_df.iloc[-1]['SMA50'], 2)) and (round(asset_data_df.iloc[-2]['SMA20'], 2) >= round(asset_data_df.iloc[-2]['SMA50'], 2)):
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
                    return_type = 'signals'
        return return_type, signals, self.tickers