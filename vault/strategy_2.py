"""
get demo strategy.py to work with the vault system
"""

from vault.base_strategy import BaseStrategy

class Strategy (BaseStrategy):
    def __init__(self):
        self.strategy_name = 'strategy_2'
        self.granularity = "1d"

    def get_name(self):
        return self.strategy_name

    def datafeeder_inputs(self):
        tickers = ['AAPL', 'MSFT', 'NVDA']
        return {'granularity': self.granularity, 'lookback':100, 'columns': ['Open', 'High', 'Low', 'Close', 'Volume']}, tickers
        
    def generate_signals (self, market_data_df):
        """
        Generate signals based on the strategy. THIS IS DUMMY CODE FOR CREATING IDEAL PORTFOLIO.
        """
        symbol_scores = {}
        for symbol in set(market_data_df["Open"].columns):
            # calculate a score for each symbol, and go long on the top 10 and short on the bottom 10, based on 'demo_strategy.py'
            df = market_data_df.loc[self.granularity].xs(symbol, axis=1, level='Ticker')
            symbol_scores[symbol] = df.iloc[-1]['Close'] / df.iloc[-200]['Close']
        
        sorted_symbols = sorted (symbol_scores, key=symbol_scores.get)
        top_symbols = sorted_symbols [-10:]
        bottom_symbols = sorted_symbols [:10]
        
        ideal_portfolio = {'symbols':{}}
        for symbol in top_symbols:
            ideal_portfolio['symbols'][symbol] = 1
        for symbol in bottom_symbols:
            ideal_portfolio['symbols'][symbol] = -1
        ideal_portfolio['strategy_name'] = self.strategy_name
        ideal_portfolio['timestamp'] = market_data_df.loc[self.granularity].xs(symbol, axis=1, level='Ticker').reset_index().iloc[-1]['Datetime'] # Find the latest timestamp.
        """
        ideal_portfolio = {'symbols':{'AAPL':1, 'MSFT':1, 'NVDA':1, 'TSLA':1, 'AMZN':1, 'GOOGL':1, 'FB':1, 'NFLX':1, 'INTC':1, 'AMD':1, 'XOM':-1, 'JNJ':-1, 
                                      'JPM':-1, 'V':-1, 'PG':-1, 'UNH':-1, 'DIS':-1, 'HD':-1, 'CRM':-1, 'NKE':-1},strategy_name: 'strat_2', 'timestamp': 
                                      datetime.now}
        """
        return False , ideal_portfolio