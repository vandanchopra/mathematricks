"""
get demo strategy.py to work with the vault system
"""

from vault.base_strategy import BaseStrategy
import numpy as np

class Strategy (BaseStrategy):
    def __init__(self):
        self.strategy_name = 'strategy_2'
        self.granularity = "1d"
        self.stop_loss_abs = 0.3
        self.stop_loss_pct = 0.2    #percentage
        self.target_abs = 0.3
        self.target_pct = 0.2       #percentage
        
        self.orderType = "MARKET" #MARKET, LIMIT, STOPLOSS  
        self.exit_order_type = "stoploss_pct" #sl_pct , sl_abs
        self.timeInForce = "DAY"    #DAY, Expiry, IoC (immediate or cancel) , TTL (Order validity in minutes) 
        self.orderQuantity = 100
        

    def get_name(self):
        return self.strategy_name

    
    def datafeeder_inputs(self):
        tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'FB', 'NFLX', 'INTC', 'AMD', 'XOM', 'JNJ', 'JPM', 'V', 'PG', 'UNH', 'DIS', 'HD', 'CRM', 'NKE']
        return { self.granularity : {'columns': ['Open', 'High', 'Low', 'Close', 'Volume'] , 'lookback':100}}, tickers

    
    def run_strategy(self,market_data_df):
        symbol_scores = {}
        ltp = {}
        scores = []
        for symbol in set(market_data_df["Open"].columns):
            # calculate a score for each symbol, and go long on the top 10 and short on the bottom 10, based on 'demo_strategy.py'
            df = market_data_df.loc[self.granularity].xs(symbol, axis=1, level='Ticker')
            symbol_scores[symbol] = df.iloc[-1]['Close'] / df.iloc[-200]['Close']
            scores.append(df.iloc[-1]['Close'] / df.iloc[-200]['Close'])
            ltp[symbol] = df.iloc[-1]['Close']
        
        sorted_symbols = sorted (symbol_scores, key=symbol_scores.get)
        scores = sorted(scores)
        top_symbols = sorted_symbols [-10:]
        top_scores = np.array(scores[-10:])
        # finding mean of each element so the sum of top scores is +1
        top_scores = (top_scores/sum(top_scores)).round(decimals=2) 
        
        bottom_symbols = sorted_symbols [:10]
        bottom_scores = np.array(scores[:10])
        # finding mean of each element so the sum of bottom scores is -1
        bottom_scores = (-1*bottom_scores/sum(bottom_scores)).round(decimals=2)

        return top_symbols, top_scores, bottom_symbols, bottom_scores, ltp

    
    def generate_signals (self, market_data_df):
        """
        Generate signals based on the strategy. THIS IS DUMMY CODE FOR CREATING IDEAL PORTFOLIO.
        """
        #run strategy and get result data
        top_symbols, top_scores, bottom_symbols, bottom_scores, ltp = self.run_strategy(market_data_df)

        #above this line was the strategy portion and below is generation of the ideal portfolio signal
        ideal_portfolio = {
            'strategy_name':self.strategy_name,
            'timestamp':market_data_df.loc[self.granularity].xs(top_symbols[0], axis=1, level='Ticker').reset_index().iloc[-1]['Datetime'],
            'entry_order_type':self.orderType,
            'exit_order_type':self.exit_order_type,
            'sl_pct':self.stop_loss_pct,
            'sl_abs':self.stop_loss_abs,
            'timeInForce':self.timeInForce,
            'orderQuantity':self.orderQuantity
        }
        ideal_portfolio['symbols'] = {}
        for i in range(len(top_symbols)):
            ideal_portfolio['symbols'][top_symbols[i]] = [top_scores[i], ltp[top_symbols[i]]]
        for i in range(len(bottom_symbols)):
            ideal_portfolio['symbols'][bottom_symbols[i]] = [bottom_scores[i], ltp[bottom_symbols[i]]]
        ideal_portfolio["timestamp"] = ideal_portfolio["timestamp"].strftime('%Y-%m-%d %H:%M:%S')

        return False , ideal_portfolio