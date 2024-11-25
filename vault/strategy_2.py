"""
get demo strategy.py to work with the vault system
"""

from vault.base_strategy import BaseStrategy
import numpy as np

class Strategy (BaseStrategy):
    def __init__(self, config_dict):
        super().__init__()
        self.strategy_name = 'strategy_2'
        self.granularity = "1d"
        self.stop_loss_pct = 0.2    #percentage in decimals
        self.target_pct = 0.2       #percentage in decimals
        self.orderType = "MARKET" #MARKET, LIMIT, STOPLOSS  
        self.exit_order_type = "stoploss_pct" #sl_pct , sl_abs
        self.timeInForce = "DAY"    #DAY, Expiry, IoC (immediate or cancel) , TTL (Order validity in minutes) 
        self.orderQuantity = 100
        self.data_inputs, self.tickers = self.datafeeder_inputs()
        self.long_short_symbol_count = 2

    def get_name(self):
        return self.strategy_name

    
    def datafeeder_inputs(self):
        self.tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'MTSI', 'GOOGL', 'HBNC', 'NFLX', 'GS', 'AMD', 'XOM', 'JNJ', 'JPM', 'V', 'PG', 'UNH', 'DIS', 'HD', 'CRM', 'NKE']
        # tickers = ['AAPL', 'MSFT']
        return { "1d" : {'columns': ['open', 'high', 'low', 'close', 'volume'] , 'lookback':100}}, self.tickers

    
    def run_strategy(self,market_data_df):
        symbol_scores = {}
        ltp = {}
        scores = []
        for symbol in set(market_data_df["open"].columns):
            if(self.granularity not in market_data_df.index.levels[0] or len(market_data_df.loc[self.granularity]) >= self.data_inputs[self.granularity]['lookback']):
                # calculate a score for each symbol, and go long on the top 10 and short on the bottom 10, based on 'demo_strategy.py'
                df = market_data_df.loc[self.granularity].xs(symbol, axis=1, level='symbol')
                df.dropna(inplace=True)
                if len(df) >= self.data_inputs[self.granularity]['lookback']:
                    symbol_scores[symbol] = df.iloc[-1]['close'] / df.iloc[-100]['close']
                    scores.append(df.iloc[-1]['close'] / df.iloc[-100]['close'])
                    ltp[symbol] = df.iloc[-1]['close']
        
        sorted_symbols = sorted(symbol_scores, key=symbol_scores.get)
        scores = sorted(scores)
        top_symbols = sorted_symbols [-self.long_short_symbol_count:]
        # self.logger.debug({'sorted_symbols':sorted_symbols})
        # self.logger.debug({'scores':scores})
        top_scores = np.array(scores[-self.long_short_symbol_count:])
        # self.logger.debug({'top_scores':top_scores})
        # finding mean of each element so the sum of top scores is +1
        top_scores = (top_scores/sum(top_scores)).round(decimals=2) 
        # self.logger.debug({'top_scores':top_scores})
        
        bottom_symbols = sorted_symbols [:self.long_short_symbol_count]
        bottom_scores = np.array(scores[:self.long_short_symbol_count])
        # finding mean of each element so the sum of bottom scores is -1
        bottom_scores = (-1*bottom_scores/sum(bottom_scores)).round(decimals=2)

        return top_symbols, top_scores, bottom_symbols, bottom_scores, ltp

    
    def generate_signals(self, next_rows, market_data_df, system_timestamp):
        """
        Generate signals based on the strategy. THIS IS DUMMY CODE FOR CREATING IDEAL PORTFOLIO.
        """
        
        #run strategy and get result data
        top_symbols, top_scores, bottom_symbols, bottom_scores, ltp = self.run_strategy(market_data_df)
        # self.logger.debug({'top_symbols':top_symbols})
        # self.logger.debug({'top_scores':top_scores})
        # self.logger.debug({'bottom_symbols':bottom_symbols})
        # self.logger.debug({'bottom_scores':bottom_scores})
        # self.logger.debug({'ltp':ltp})
        
        ideal_portfolio_entry = {}
        
        if len(top_symbols) != 0 or len(bottom_symbols) != 0:  
            #above this line was the strategy portion and below is generation of the ideal portfolio signal
            ideal_portfolio_entry = {
                'strategy_name':self.strategy_name,
                'timestamp':system_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'entry_order_type':self.orderType,
                'exit_order_type':self.exit_order_type,
                'stoploss_pct':self.stop_loss_pct,
                'timeInForce':self.timeInForce,
                'orderQuantity':self.orderQuantity,
                'granularity':self.granularity,
                'signal_type':'BUY_SELL',
                'market_neutral':False,
            }
            ideal_portfolio_entry['ideal_portfolio'] = {}
            
            normalized_top_scores = np.array(top_scores) / np.sum(top_scores)
            normalized_bottom_scores = np.array(bottom_scores) / np.sum(bottom_scores)
            
            # self.logger.debug({'normalized_top_scores':normalized_top_scores, 'normalized_bottom_scores':normalized_bottom_scores})
            
            for i in range(len(top_symbols)):
                ideal_portfolio_entry['ideal_portfolio'][top_symbols[i]] = {'orderDirection':'BUY', 'signal_strength':abs(normalized_top_scores[i]), 'current_price':ltp[top_symbols[i]]}
            for i in range(len(bottom_symbols)):
                ideal_portfolio_entry['ideal_portfolio'][bottom_symbols[i]] = {'orderDirection':'SELL', 'signal_strength':abs(normalized_bottom_scores[i]), 'current_price':ltp[bottom_symbols[i]]}
            # ideal_portfolio["timestamp"] = ideal_portfolio["timestamp"].strftime('%Y-%m-%d %H:%M:%S')
            # self.logger.debug({'ideal_portfolio_entry':ideal_portfolio_entry})
            return_type = 'ideal_portfolios'
        else:
            return_type = None
        
        ideal_portfolio_entry = [ideal_portfolio_entry]
        
        return return_type, ideal_portfolio_entry, self.tickers

#['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'FB', 'NFLX', 'INTC', 'AMD', 'XOM', 'JNJ', 'JPM', 'V', 'PG', 'UNH', 'DIS', 'HD', 'CRM', 'NKE']
#PG, XOM, MSFT, JPM, AAPL, NFLX, GOOGL, TSLA, NVDA, NKE, CRM, UNH, DIS, HD, JNJ, V, AMD, 
#AMZN, INTC, FB