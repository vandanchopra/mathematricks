"""
Create a simple Single Moving Crossover Strategy as a demo
"""

from vault.base_strategy import BaseStrategy
import numpy as np

class Strategy (BaseStrategy):
    def __init__(self):
        self.strategy_name = 'strategy_1'
        self.granularity = "id"
        
    def get_name(self):
        return self.strategy_name
        
    def datafeeder_inputs(self):
        tickers = ['AAPL', 'MSFT', 'NVDA']
        return {'granularity': self.granularity, 'lookback':100, 'columns': ['open', 'high', 'low', 'close', 'volume']}, tickers
        
    def generate_signals (self, market_data_df):
        """
        Generate signals based on the strategy. THIS IS DUMMY CODE FOR CREATING SIGNALS.
        """
        signals = []
        for symbol in set(market_data_df["Open"].columns):
            asset_data_df = market_data_df.loc[self.granularity].xs(symbol, axis=1, level='Ticker').reset_index()
            asset_data_df['SMA15'] = asset_data_df['close'].rolling(window=15).mean()
            asset_data_df['SMA30'] = asset_data_df['close'].rolling(window=30).mean()
            
            asset_data_df['signal_strength'] = 0
            asset_data_df['signal_strength'][30:] = np.where(asset_data_df['SMA15'][30:] > asset_data_df['SMA30'][30:], 1, 0) 
            asset_data_df['position'] = asset_data_df['signal_strength'].diff() 

            signal = {'symbol': symbol, 'signal_strength':0.8, self.strategy_name: self.strategy_name, 'timestamp': asset_data_df.iloc[-1]['Datetime'], 'entry_order_type': 'market','exit_order_type':'stoploss_pct'}
            
            signals.append(signal)
            
        """
        eg. signal = {'symbol': 'AAPL', 'signal_strength':0.8, strategy_name: 'strat_2', 'timestamp': datetime.now, entry_order_type: 'market', 
                  'exit_order_type':'stoploss_pct'}
        """
    
        return signals , False