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
        self.exit_order_type = "stoploss_pct" #sl_pct , sl_abs
        self.timeInForce = "DAY"    #DAY, Expiry, IoC (immediate or cancel) , TTL (Order validity in minutes) 
        self.orderQuantity = 10
        self.data_inputs, self.tickers = self.datafeeder_inputs()
        
    def get_name(self):
        return self.strategy_name
        
    def datafeeder_inputs(self):
        tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'HBNC', 'NFLX', 'GS', 'AMD', 'XOM', 'JNJ', 'JPM', 'V', 'PG', 'UNH', 'DIS', 'HD', 'CRM', 'NKE']
        data_inputs = {'1d': {'columns': ['open', 'high', 'close', 'volume'] , 'lookback':52}}
        return data_inputs, tickers
        
    def generate_signals(self, next_rows, market_data_df, system_timestamp, open_signals=None):
        """
        Generate signals based on dual SMA crossover strategy (20 and 50 period). Takes into account any open signals.
        """
        signals = []
        return_type = None

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
                # Create Order object
                order = Order(
                    symbol=symbol,
                    orderQuantity=self.orderQuantity,
                    orderDirection=orderDirection,
                    order_type=self.orderType,
                    symbol_ltp={system_timestamp: current_price},
                    timeInForce=self.timeInForce,
                    entryOrderBool=True
                )
                
                # Create Signal object
                signal = Signal(
                    strategy_name=self.strategy_name,
                    timestamp=system_timestamp,
                    orders=[order],
                    signal_strength=signal_strength,
                    granularity=self.granularity,
                    signal_type="BUY_SELL",
                    market_neutral=False
                )
                signals.append(signal)
                return_type = 'signals'
        return return_type, signals, self.tickers