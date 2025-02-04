"""
get demo strategy.py to work with the vault system
"""
from matplotlib import ticker
from vault.base_strategy import BaseStrategy
import numpy as np
import pandas as pd
import json, os
from tqdm import tqdm
import pickle
from systems.utils import project_path, sleeper, MarketDataExtractor
from copy import deepcopy

class Strategy(BaseStrategy):
    def __init__(self, config_dict):
        super().__init__()
        self.strategy_name = os.path.basename(__file__).split('.')[0]
        self.granularity = "1d"
        self.stop_loss_pct = 0.05  #percentage in decimals
        self.target_pct = 0.2       #percentage in decimals
        self.entry_order_type = "MARKET" #MARKET, LIMIT, STOPLOSS  
        self.exit_order_type = "stoploss_pct" #sl_pct , sl_abs
        self.timeInForce = "GTC"    #DAY, GTC Expiry, IoC (immediate or cancel) , Tgit TL (Order validity in minutes) 
        self.max_risk_per_bet = 0.05
        self.config_dict = config_dict
        # self.logger.debug({'config_dict':config_dict})
        self.start_time = self.config_dict['backtest_inputs']['start_time']
        self.end_time = self.config_dict['backtest_inputs']['end_time']
        self.days_since_rebalance = 0
        self.rebalance_frequency = 7
        self.universe_size = 200
        self.long_short_symbol_count = 5
        self.min_market_cap_in_billions = 30
        self.tickers = []
        self.prices_data_dict = {}
        self.lookback = 365
        self.market_data_extractor = MarketDataExtractor()
        # self.data_inputs, self.tickers = self.datafeeder_inputs()

    def return_notes(self):
        note = '''
        - Ran a 4 year test. It gave me a 37% CAGR as opposed to 11% CAGR for the NASDAQ. and a sharpe ratio of 0.93 as opposed to 0.46 for the NASDAQ. (Test: a2a439897d6955d379bf1b09c251fd0aab13fe6abb3e00dbf60021e44c9bf7bf)
        - 25 year test - 2000-2024: 11.7% CARG, same as nasdaq. Drawdown: 40 pct. sharpe ratio of 0.30 as opposed to 0.46 for the NASDAQ. (Test: 443fd442012eac677f0498423512c2232658f9a285cd51f3325c211bc62e812e)
            - The quality of the curve is good, however the overall return isn't good. 
            - The overall win pct is 30-35%. I think it's making a lot of false trades, before it can make a good one. Too many false positives. Basically, it keeps trying to enter every week, without learning from the past !!!
        - 78cd37c684bc4f90b7e8595ed0164637387b13a87126d783026b3369c96ee682_1 - without vwap.
        - 78cd37c684bc4f90b7e8595ed0164637387b13a87126d783026b3369c96ee682 - with vwap added as signal confirmation. Without vwap was better. vwap just reduced profit and increased drawdowns.
        - 78cd37c684bc4f90b7e8595ed0164637387b13a87126d783026b3369c96ee682_1 : vwap reverse. all the returns are wiped out. CAGR of -1.6%. So this doesn't work as well.
        - VWAP: The problem is that once the volume comes in, the prices move is already done. 
        - Increased the leway on updated trailing loss to 5%: CAGR 17%. Drawdown: 20% (Test: 78cd37c684bc4f90b7e8595ed0164637387b13a87126d783026b3369c96ee682_3) - 4 year test - 2010-2014
        - Increased the leway on updated trailing loss to 5%: CAGR 13%. Drawdown: 20% (Test: 78cd37c684bc4f90b7e8595ed0164637387b13a87126d783026b3369c96ee682_4) - 14 year test - 2010-2024 (Last time we ran a 25 year test, it gave a CAGR of 11.7%)
        - No use increasing this to 25. it'll just drop to 11pct again.
        
        '''
        return note
    
    def get_name(self):
        return self.strategy_name
    
    def datafeeder_inputs(self):
        # tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'AMD', 'NFLX', 'JNJ']
        # tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'HBNC', 'NFLX', 'GS', 'AMD', 'XOM', 'JNJ', 'JPM', 'V', 'PG', 'UNH', 'DIS', 'HD', 'CRM', 'NKE']
        # tickers = tickers[:10]
        # tickers = ['AAPL', 'MSFT', 'NVDA', 'TSM']
        self.tickers, self.stock_symbols_df = self.get_universe()
        self.data_inputs = { "1d" : {'columns': ['open', 'high', 'low', 'close', 'volume'] , 'lookback':self.lookback}, 
                                #  "1m" : {'columns': ['open', 'high', 'low', 'close', 'volume'] , 'lookback':365}
                                 }
        
        symbols_to_remove = ['BRK/A', 'BRK/B', 'MNST', 'LIN', 'TPL']
        for symbol_to_remove in symbols_to_remove:
            if symbol_to_remove in self.tickers:
                self.tickers.remove(symbol_to_remove)
                
        return self.data_inputs, self.tickers

    def get_universe(self):
        def remove_old_files(stocksymbolslists_folder, days_old=30):    
            # Get file names from the folder
            file_names = [f for f in os.listdir(stocksymbolslists_folder) if f.endswith('.csv')]
            # Get the number portion of the file names
            file_numbers = [float(f.split('_')[-1].split('.')[0]) for f in file_names]
            # Now assume that these numbers are EPOCH timestamps in milliseconds and calculate the age of the timestamp in days
            from datetime import datetime
            from datetime import timezone
            from datetime import timedelta
            now = datetime.now(timezone.utc)
            file_numbers = [datetime.fromtimestamp(x/1000, timezone.utc) for x in file_numbers]
            file_numbers = [now - x for x in file_numbers]
            file_numbers = [x.days for x in file_numbers]
            # if any of those are more than 30 days old, then delete them from file_name
            file_names = [f for f, age in zip(file_names, file_numbers) if age < days_old]
            return file_names
        def get_list_of_all_symbols():
            # get file names from the folder and load all the csv files and concatenate them and return a pandas dataframe
            stocksymbolslists_folder = '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/db/data/stocksymbolslists'
            
            # Get file names from the folder
            file_names = [f for f in os.listdir(stocksymbolslists_folder) if f.endswith('.csv')]
            # Remove all files that are more than 30 days old
            file_names = remove_old_files(stocksymbolslists_folder, days_old=60)
            # Load all the CSV files and concatenate them into a single DataFrame
            dfs = [pd.read_csv(os.path.join(stocksymbolslists_folder, file)) for file in file_names]
            stock_symbols_df = pd.concat(dfs, ignore_index=True)
            # Now sort the combined_df by the market cap column in descending order
            stock_symbols_df = stock_symbols_df.sort_values(by='Market Cap', ascending=False)
            # Drop all rows where Market Cap is NaN
            stock_symbols_df = stock_symbols_df.dropna(subset=['Market Cap'])
            # Drop all rows where Market Cap is 0
            stock_symbols_df = stock_symbols_df[stock_symbols_df['Market Cap'] != 0]
            stock_symbols_df = stock_symbols_df.copy()
            return stock_symbols_df
        def filter_symbols_by_market_cap(stock_symbols_df, min_market_cap_in_billions=30):
            # Drop all rows where Market Cap is less than 3 billion
            stock_symbols_df = stock_symbols_df[stock_symbols_df['Market Cap'] > min_market_cap_in_billions * 1000000000]
            return stock_symbols_df
        
        stock_symbols_df = get_list_of_all_symbols()
        # self.logger.debug({'stock_symbols_df':stock_symbols_df})
        stock_symbols_df = filter_symbols_by_market_cap(stock_symbols_df, self.min_market_cap_in_billions)
        return list(set(list(stock_symbols_df['Symbol'].values))), stock_symbols_df
        
    def get_list_of_symbols_to_trade(self, start_time, market_data_df, universe_size=100):
        '''
        Get a list of all symbols.
        Sort them by Market Cap.
        Remove everything smaller than a certain {market cap}.
        Now calculate the growth rate of each stock over the last {n} days.
        Now sort them by growth rate, and only keep the top {m} stocks and the bottom {m} stocks.
        '''
        '''STEP 1: '''
        def calculate_growth_rate(stock_symbols_df, market_data_df):
            # Initialize lists to store the latest and 1-year-ago prices
            latest_prices = {}
            one_year_ago_prices = {}

            # Define the date range
            test_start_date = start_time
            lookback_start_date = test_start_date - pd.DateOffset(days=self.lookback)
            lookback_start_date = lookback_start_date
            
            market_data_df = market_data_df.loc['1d'].loc[lookback_start_date:].copy()
            if market_data_df.iloc[-1].name > lookback_start_date:
                first_last_df = market_data_df.iloc[[0, -1]]
                # Create a new df with only columns ['one_year_ago', 'today'] from new_df
                last_one_year_prices = first_last_df['close'].iloc[0]
                today_prices = first_last_df['close'].iloc[1]
                growth = (today_prices - last_one_year_prices) / last_one_year_prices * 100
                # make a new df with these values
                stock_symbols_df = pd.DataFrame({'last_one_year_prices': last_one_year_prices, 'today_prices': today_prices, 'growth_rate_pct': growth})
                stock_symbols_df['growth_rate_pct'] = stock_symbols_df['growth_rate_pct'].round(2)
                stock_symbols_df.dropna(inplace=True)
                stock_symbols_df = stock_symbols_df.sort_values(by='growth_rate_pct', ascending=False)
            else:
                stock_symbols_df = pd.DataFrame()
            
            return stock_symbols_df
        
        def get_top_bottom_symbols(stock_symbols_df, universe_size):
            # create a new DF with only stock_symbols_df[:100], stock_symbols_df[-100:] concatenated
            top_n = stock_symbols_df[:universe_size]
            bottom_n = stock_symbols_df[-universe_size:]
            top_bottom_n = pd.concat([top_n, bottom_n])
            # self.logger.debug({'top_bottom_n':top_bottom_n.index})
            top_bottom_n_symbols = list(top_bottom_n.index)
            return top_bottom_n_symbols
        def get_random_n_symbols(stock_symbols_df, n):
            # self.logger.debug({'stock_symbols_df':stock_symbols_df})
            stock_symbols_df = stock_symbols_df.sample(n)
            return list(stock_symbols_df['Symbol'].values)
        
        # self.logger.debug({'stock_symbols_df':stock_symbols_df})
        stock_symbols_df = deepcopy(self.stock_symbols_df)
        stock_symbols_df = calculate_growth_rate(stock_symbols_df, market_data_df)
        
        # if long_short_symbol_count is an odd number, add 1 to make it even
        if universe_size % 2 != 0:
            universe_size += 1
        list_of_symbols_to_trade = get_top_bottom_symbols(stock_symbols_df, universe_size)
        
        # for extraction_mode in ['top_bottom', 'random']:
        #     if extraction_mode == 'top_bottom':
        #         list_of_symbols_to_trade = get_top_bottom_symbols(stock_symbols_df, universe_size)
        #     elif extraction_mode == 'random':
        #         list_of_symbols_to_trade = None #get_random_n_symbols(stock_symbols_df, long_short_symbol_count)
        
        # self.logger.debug({'list_of_symbols_to_trade':list_of_symbols_to_trade})
        # input('Press Enter to continue...')
        return list_of_symbols_to_trade
    
    def run_strategy(self, next_rows, market_data_df, system_timestamp, long_short_symbol_count=5):
        
        '''STEP 2: Create the strategy dataframe with only specific days ago prices'''
        if self.granularity in next_rows.index and self.granularity in market_data_df.index.levels[0] and (len(market_data_df.loc[self.granularity]) > self.data_inputs[self.granularity]['lookback']):
            
            '''STEP 1: Reverse the order of the dataframe'''
            # Reverse the order of the dataframe
            market_data_df = market_data_df.iloc[::-1]
            # Only create data for tickers that are in market_data_df
            tickers_temp = list(set(self.symbols_to_trade) & set(market_data_df.columns.get_level_values(1)))
            # Only retain the columns that are in the tickers list
            market_data_df = market_data_df.loc[self.granularity, pd.IndexSlice[:, tickers_temp]]
            # In market_data_df only keep these rows on the index level
            days_ago_list_difference = 15
            days_ago_list = list(range(1, 365, days_ago_list_difference))
            strategy_df_orig = market_data_df.iloc[days_ago_list]['close']

            # Drop all assets with NaN values
            strategy_df_orig = strategy_df_orig.dropna(axis=1)
            strategy_df = strategy_df_orig.copy()
            
            # Create a list from days_ago_list that has each element followed by '_days_ago'
            price_days_ago_list_str = [str(x) + '_days_ago_price' for x in days_ago_list]
            # replace the index with the days_ago_list
            strategy_df.index = price_days_ago_list_str

            '''STEP 3: Calculate growth rate 1 on strategy dataframe: Growth Rate Period Based'''
            # Calculate the growth rate for each row
            growth_rate_1_df = (strategy_df - strategy_df.shift(-1))/strategy_df.shift(-1)
            growth_rate_1_df = growth_rate_1_df.dropna()
            growth_rate_1_days_ago_list_str = [str(x) + '_days_ago_growth_1' for x in days_ago_list[1:]]
            growth_rate_1_df.index = growth_rate_1_days_ago_list_str
            # Normalize growth_rate_1_df horizontally like this (x - min(x))/(max(x) - min(x))
            growth_rate_1_df_normalized_scores = growth_rate_1_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=1)

            # Join the growth rate dataframe with the strategy dataframe one on top of the other
            # strategy_df = pd.concat([strategy_df, growth_rate_df])

            '''STEP 4: Calculate growth rate 2 on strategy dataframe. Which is everything divided by the first row: Growth Rate Cumulative Based'''
            # Calculate the growth rate for each row
            growth_rate_2_df = (strategy_df.iloc[0]-strategy_df)/strategy_df
            growth_rate_2_days_ago_list_str = [str(x) + '_days_ago_growth_2' for x in days_ago_list[1:]]
            growth_rate_2_df = growth_rate_2_df.drop(growth_rate_2_df.index[0])
            growth_rate_2_df.index = growth_rate_2_days_ago_list_str

            # Normalize growth_rate_1_df horizontally like this (x - min(x))/(max(x) - min(x))
            growth_rate_2_df_normalized_scores = growth_rate_2_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=1)

            '''STEP 5: Get the Weights for each row'''
            def calculate_u_curve_weights(points_count, mode='normal'):
                    x = np.linspace(-5, 10, points_count)
                    
                    # Define a quadratic function that is high at the edges and lower in the middle
                    arr = ((-0.5 * x**2) + 4) * -1
                    # Ensure all values are positive
                    if np.min(arr) < 0:
                        arr = arr + abs(np.min(arr)-4)

                    # Normalize the array so the sum is 100
                    arr = (arr / np.sum(arr)) * 100
                    
                    return arr
            weights_1 = calculate_u_curve_weights(len(growth_rate_1_days_ago_list_str))
            weights_2 = calculate_u_curve_weights(len(growth_rate_2_days_ago_list_str))

            '''Step 6: Calculate the weight score for each row'''
            scores_1_df = growth_rate_1_df_normalized_scores.multiply(weights_1, axis=0)
            scores_2_df = growth_rate_2_df_normalized_scores.multiply(weights_2, axis=0)

            '''STEP 7: Combine the scores and get the final Score'''
            # Concatenate the scores_1_df and scores_2_df
            weights_for_scores_combination = [10, 10]
            scores_df = pd.concat([scores_1_df * weights_for_scores_combination[0], scores_2_df * weights_for_scores_combination[1]])
            final_scores = pd.DataFrame(scores_df.sum(axis=0))
            # give the columns names
            final_scores.columns = ['final_score']

            # self.logger.debug({"market_data_df[::-1].loc['1d']":market_data_df[::-1].loc['1d']})
            # Add the Latest price to the final_scores from the market_data_df
            final_scores['latest_price'] = market_data_df[::-1].iloc[-1]['close']
            # Sort final_scores by final_score
            final_scores = final_scores.sort_values(by='final_score', ascending=False)
            # Get the top 5 stocks
            top_stocks = final_scores.head(long_short_symbol_count)
            # Get the bottom 5 stocks
            bottom_stocks = final_scores.tail(long_short_symbol_count)
            
            # self.logger.debug({'top_stocks':top_stocks, 'bottom_stocks':bottom_stocks})
            # input('Press Enter to continue...')

            '''STEP 8: Get the Final Return Variables'''
            top_symbols = list(top_stocks.index)
            top_scores = list(top_stocks['final_score'])
            bottom_symbols = list(bottom_stocks.index)
            bottom_scores = list(bottom_stocks['final_score'])
            top_symbols_ltp = list(top_stocks['latest_price'])
            bottom_symbols_ltp = list(bottom_stocks['latest_price'])
            # self.logger.info({'top_symbols':top_symbols, 'bottom_symbols':bottom_symbols})
            
            return top_symbols, top_scores, bottom_symbols, bottom_scores, top_symbols_ltp, bottom_symbols_ltp
        else:
            return [], [], [], [], [], []
        
    def run_regime_filter(self, market_data_df, regime_lookback):
        # in market_data_df.loc['1d'], divide the last close price by the close price 7 days ago and get a matrix
        regime_calc_matrix = market_data_df.loc['1d'].iloc[-1]['close'] / market_data_df.loc['1d'].iloc[-regime_lookback]['close']
        bullish_score = (regime_calc_matrix.values > 1).sum() / len(regime_calc_matrix)
        return bullish_score
    
    def validate_signal_with_vwap(market_data_df, signal_data):
        """
        Validate signals based on VWAP.

        Args:
            market_data_df (pd.DataFrame): Market data with multi-level index (symbol and date),
                                        containing 'close' and 'volume' columns.
            signal_data (dict): Dictionary where keys are symbols and values are the signal price
                                to validate against VWAP.

        Returns:
            dict: Validation results with symbols as keys and validation status (True/False) as values.
        """
        results = {}

        # Ensure market_data_df has the necessary columns
        if 'close' not in market_data_df.columns or 'volume' not in market_data_df.columns:
            raise ValueError("market_data_df must contain 'close' and 'volume' columns.")

        for symbol, signal_price in signal_data.items():
            if symbol not in market_data_df.columns:
                results[symbol] = False
                continue

            # Calculate VWAP for the symbol
            data = market_data_df[[symbol]].copy()
            data['price_volume'] = data['close'] * data['volume']
            vwap = data['price_volume'].cumsum() / data['volume'].cumsum()

            # Validate the signal
            latest_vwap = vwap.iloc[-1]
            results[symbol] = signal_price > latest_vwap

        return results
    
    def get_symbol_vwap(self, symbol_data):
         # Calculate VWAP for the symbol
        data = symbol_data.copy()
        data['price_volume'] = data['close'] * data['volume']
        vwap = data['price_volume'].cumsum() / data['volume'].cumsum()
        # Validate the signal
        latest_vwap = vwap.iloc[-1]
        return latest_vwap      
      
    def generate_signals(self, next_rows, market_data_df, system_timestamp):
        """
        Generate signals based on the strategy. THIS IS DUMMY CODE FOR CREATING IDEAL PORTFOLIO.
        """
        ideal_portfolio_entry_list = []
        return_type = None
        
        #run strategy and get result data
        if self.granularity in next_rows.index and self.granularity in market_data_df.index.levels[0] and (len(market_data_df.loc[self.granularity]) > self.data_inputs[self.granularity]['lookback']):
            if system_timestamp > self.config_dict['backtest_inputs']['start_time']:
                if self.days_since_rebalance >= self.rebalance_frequency:
                    # tickers_new = self.get_list_of_symbols_to_trade(system_timestamp, universe_size=self.universe_size)
                    self.symbols_to_trade = self.get_list_of_symbols_to_trade(system_timestamp, market_data_df, self.universe_size)
                    # bullish_score = self.run_regime_filter(market_data_df, regime_lookback=7)
                    
                    # self.logger.debug({'self.symbols_to_trade':self.symbols_to_trade})
                    # sleeper(1)
                    self.days_since_rebalance = 0
                    top_symbols, top_scores, bottom_symbols, bottom_scores, top_symbols_ltp, bottom_symbols_ltp = self.run_strategy(next_rows, market_data_df, system_timestamp, self.long_short_symbol_count * 2)
                    # self.logger.debug({'top_symbols':top_symbols, 'top_scores':top_scores, 'bottom_symbols':bottom_symbols, 'bottom_scores':bottom_scores})
                    
                    if len(top_symbols) != 0 or len(bottom_symbols) != 0:  
                        #above this line was the strategy portion and below is generation of the ideal portfolio signal
                        ideal_portfolio_entry = {
                            'strategy_name':self.strategy_name,
                            'timestamp':system_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            'entry_order_type':self.entry_order_type,
                            'exit_order_type':self.exit_order_type,
                            'sl_pct':self.stop_loss_pct,
                            'timeInForce':self.timeInForce,
                            'granularity':self.granularity,
                            "stoploss_pct": self.stop_loss_pct,
                            'market_neutral':True,
                            'signal_type':'BUY_SELL',
                            'max_risk_per_bet':self.max_risk_per_bet,
                        }
                        ideal_portfolio_entry['ideal_portfolio'] = {}
                        
                        normalized_top_scores = np.array(top_scores) / np.sum(top_scores)
                        normalized_bottom_scores = np.array(bottom_scores) / np.sum(bottom_scores)
                        
                        # if bullish_score > 0.7:
                        for i in range(len(top_symbols)):
                            # latest_vwap = self.get_symbol_vwap(market_data_df.loc['1d'].xs(top_symbols[i], axis=1, level='symbol'))
                            signal_validation = True #top_symbols_ltp[i] < latest_vwap
                            # self.logger.debug({'Symbol':bottom_symbols[i], 'latest_vwap':latest_vwap, 'current_price':bottom_symbols_ltp[i], 'Signal Validation:':signal_validation})
                            if signal_validation:
                                ideal_portfolio_entry['ideal_portfolio'][top_symbols[i]] = {'orderDirection':'BUY', 'signal_strength':abs(normalized_top_scores[i]), 'current_price':top_symbols_ltp[i]}
                            if i > self.long_short_symbol_count:
                                break
                                # self.logger.debug("LONG Symbol: {}".format(top_symbols[i]))
                        # if bullish_score < 0.3:
                        for i in range(len(bottom_symbols)):
                            # latest_vwap = self.get_symbol_vwap(market_data_df.loc['1d'].xs(top_symbols[i], axis=1, level='symbol'))
                            signal_validation = True #bottom_symbols_ltp[i] > latest_vwap
                            # self.logger.debug({'Symbol':bottom_symbols[i], 'latest_vwap':latest_vwap, 'current_price':bottom_symbols_ltp[i], 'Signal Validation:':signal_validation})
                            if signal_validation:
                                ideal_portfolio_entry['ideal_portfolio'][bottom_symbols[i]] = {'orderDirection':'SELL', 'signal_strength':abs(normalized_bottom_scores[i]), 'current_price':bottom_symbols_ltp[i]}
                                # self.logger.debug("SHORT Symbol: {}".format(bottom_symbols[i]))
                            if i > self.long_short_symbol_count:
                                break
                            
                        # ideal_portfolio["timestamp"] = ideal_portfolio["timestamp"].strftime('%Y-%m-%d %H:%M:%S')
                        # self.logger.debug({'ideal_portfolio_entry':ideal_portfolio_entry})
                        return_type = 'ideal_portfolios'
                        # self.logger.info({"ideal_portfolio_entry['ideal_portfolio']":ideal_portfolio_entry['ideal_portfolio']})
                    else:
                        ideal_portfolio_entry = {}
                        return_type = None
                    
                    if len(ideal_portfolio_entry) > 0 :
                        ideal_portfolio_entry_list.append(ideal_portfolio_entry)
                        # self.logger.info({'ideal_portfolio_entry_list':ideal_portfolio_entry_list})
                    # self.logger.debug({'ideal_portfolio_entry_list':ideal_portfolio_entry_list})
                else:
                    if '1d' in market_data_df.index.levels[0]:
                        self.days_since_rebalance += 1
        self.logger.debug({'self.days_since_rebalance':self.days_since_rebalance, 'self.rebalance_frequency':self.rebalance_frequency})
            
        return return_type, ideal_portfolio_entry_list, self.tickers