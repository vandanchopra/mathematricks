"""
Create a simple Single Moving Crossover Strategy as a demo
"""

from vault.base_strategy import BaseStrategy
import numpy as np
import sys

class Strategy (BaseStrategy):
    def __init__(self):
        self.strategy_name = 'strategy_1'
        self.granularity = "1m"
        self.stop_loss_abs = 0.3    #absolute value
        self.stop_loss_pct = 0.2    #percentage
        self.target_abs = 0.3       #absolute value
        self.target_pct = 0.2       #percentage
        
        self.orderType = "MARKET" #MARKET, STOPLOSS-MARKET
        self.exit_order_type = "stoploss_pct" #sl_pct , sl_abs
        self.timeInForce = "DAY"    #DAY, Expiry, IoC (immediate or cancel) , TTL (Order validity in minutes) 
        self.orderQuantity = 10
        
    def get_name(self):
        return self.strategy_name
        
    def datafeeder_inputs(self):
        tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'MTSI', 'GOOGL', 'HBNC', 'NFLX', 'GS', 'AMD', 'XOM', 'JNJ', 'JPM', 'V', 'PG', 'UNH', 'DIS', 'HD', 'CRM', 'NKE']
        #tickers = ['MSFT', 'NVDA']
        data_inputs = {'1m': {'columns': ['open', 'high', 'close', 'volume'] , 'lookback':100}, '1d': {'columns': ['open', 'high', 'close', 'volume'] , 'lookback':100}}
        return data_inputs, tickers
        
    def generate_signals(self, market_data_df):
        """
        Generate signals based on the strategy. THIS IS DUMMY CODE FOR CREATING SIGNALS.
        """
        signals = []
        ideal_portfolio = False

        for symbol in set(market_data_df["open"].columns):
            if(self.granularity not in market_data_df.index.levels[0] or len(market_data_df.loc[self.granularity]) <= 30):
                continue
            asset_data_df = market_data_df.loc[self.granularity].xs(symbol, axis=1, level='symbol').reset_index()
            asset_data_df['SMA15'] = asset_data_df['close'].rolling(window=15).mean()
            asset_data_df['SMA30'] = asset_data_df['close'].rolling(window=30).mean()

            asset_data_df['signal_strength'] = 0
            asset_data_df['signal_strength'][30:] = np.where(asset_data_df['SMA15'][30:] > asset_data_df['SMA30'][30:], 1, 0) 
            asset_data_df['position'] = asset_data_df['signal_strength'].diff() 
            #in the above df the position column will have a +1 value in case of a positive sma crossover and -1 in the opposite case
            #otherwise it will have 0 value
            #so we could long if position value is 1 and short when its -1

            #but for testing purposes I want to generate signals for each instruments 
            
            #long if above sma15 else short
            if (asset_data_df.iloc[-1]['SMA15'] > asset_data_df.iloc[-1]['SMA30']) and (asset_data_df.iloc[-2]['SMA15'] <= asset_data_df.iloc[-2]['SMA30']):
                signal_strength = 1
            elif (asset_data_df.iloc[-1]['SMA15'] < asset_data_df.iloc[-1]['SMA30']) and (asset_data_df.iloc[-2]['SMA15'] >= asset_data_df.iloc[-2]['SMA30']):
                signal_strength = -1
            else:
                signal_strength = 0
            #signal_strength = int(asset_data_df.iloc[-1]['position'])
            if signal_strength != 0:
                signal = {"symbol": symbol, 
                        "signal_strength":signal_strength, 
                        "strategy_name": self.strategy_name, 
                        "timestamp": asset_data_df.iloc[-1]['datetime'], 
                        "entry_order_type": self.orderType, 
                        "exit_order_type":self.exit_order_type, 
                        "sl_pct": self.stop_loss_pct, 
                        "sl_abs": self.stop_loss_abs, 
                        "symbol_ltp" : asset_data_df.iloc[-1]['close'], 
                        "timeInForce" : self.timeInForce , 
                        "orderQuantity" : self.orderQuantity
                        }
                signal["timestamp"] = signal["timestamp"].strftime('%Y-%m-%d %H:%M:%S')
                signals.append(signal)
            
        return signals, ideal_portfolio