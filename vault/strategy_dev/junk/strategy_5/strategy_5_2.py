"""
get demo strategy.py to work with the vault system
"""
import pickle, time, json, os
from vault.base_strategy import BaseStrategy
import numpy as np
import pandas as pd
from systems.utils import project_path, sleeper, MarketDataExtractor, create_open_positions_from_open_orders
from copy import deepcopy

pd.set_option('future.no_silent_downcasting', True)

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
        self.universe_size = 10
        self.long_short_symbol_count = 10
        self.min_market_cap_in_billions = 30
        self.tickers = []
        self.prices_data_dict = {}
        self.market_data_extractor = MarketDataExtractor()
        # self.data_inputs, self.tickers = self.datafeeder_inputs()
        
        ### New Strategy Parameters
        self.shortlen=1000
        self.longlen = 2 * self.shortlen
        self.basisLen=12
        self.offset=0.85
        self.lookback = 400
        self.offsetSigma=3

    def return_notes(self):
        notes = {'First Dev':'''
        Version 1:
        - For every timestamp
            - make a list of open signals for this strategy
            - if not: 
                - Taking the half-sine curve and creating trend scores
                - Creating a custom indicator for directional positions called flimsy_dingo
                - start by checking all open signals, and check if the flimsy_dingo position has flipped if it has, issue a market_exit for the same. if not, leave it to run it's course.
                - Then sorting them by Sine score score and creating signals to get to 20 long and 20 short signals based on the flimsy_dingo positions.
        '''}
        return notes
    
    def get_name(self):
        return self.strategy_name
    
    def datafeeder_inputs(self):
        # tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'AMD', 'NFLX', 'JNJ']
        # tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'HBNC', 'NFLX', 'GS', 'AMD', 'XOM', 'JNJ', 'JPM', 'V', 'PG', 'UNH', 'DIS', 'HD', 'CRM', 'NKE']
        # tickers = tickers[:10]
        # tickers = ['AAPL', 'MSFT', 'NVDA', 'TSM']
        self.tickers, self.stock_symbols_df = self.get_universe()
        # self.tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'HBNC', 'NFLX', 'GS', 'AMD', 'XOM', 'JNJ', 'JPM', 'V', 'PG', 'UNH', 'DIS', 'HD', 'CRM', 'NKE'] #self.tickers
        
        self.data_inputs = { "1d" : {'columns': ['open', 'high', 'low', 'close', 'volume'] , 'lookback':self.lookback}, 
                                #  "1m" : {'columns': ['open', 'high', 'low', 'close', 'volume'] , 'lookback':365}
                                 }
        
        symbols_to_remove = ['BRK/A', 'BRK/B', 'MNST', 'LIN', 'TPL', 'TBC']
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
    
    def create_trend_strength_scores(self, market_data_df_orig, symbols_to_trade, granularity='1d'):
        '''STEP 1: Create market_data_df_to_trade. Remove all unwanted stocks'''
        market_data_df = market_data_df_orig.copy()
        tickers_temp = list(set(symbols_to_trade) & set(market_data_df.columns.get_level_values(1)))
        market_data_df_to_trade = market_data_df.loc[granularity, pd.IndexSlice[:, tickers_temp]]

        '''STEP 2: Create the strategy dataframe with only specific days ago prices'''
        days_ago_list_difference = 15
        start_date_idx = 365
        days_ago_list = list(range(-start_date_idx, 0, days_ago_list_difference))
        days_ago_list.append(-1)
        strategy_df_orig = market_data_df_to_trade.iloc[days_ago_list]['close']
        # Forward fill strategy_df_orig
        strategy_df_orig.ffill(inplace=True)
        strategy_df = strategy_df_orig.copy()

        '''STEP 3: Calculate growth rate 1 on strategy dataframe: Growth Rate Period Based'''
        growth_rate_1_df = strategy_df.pct_change()
        growth_rate_1_df = growth_rate_1_df.dropna()
        growth_rate_1_df_normalized_scores = growth_rate_1_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=1)

        '''STEP 4: Calculate growth rate 2 on strategy dataframe. Which is everything divided by the first row: Growth Rate Cumulative Based'''
        growth_rate_2_df = strategy_df/strategy_df.iloc[0]
        growth_rate_2_df = growth_rate_2_df.drop(growth_rate_2_df.index[0])
        growth_rate_2_df_normalized_scores = growth_rate_2_df

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
        weights_1 = calculate_u_curve_weights(len(growth_rate_2_df_normalized_scores))
        weights_2 = calculate_u_curve_weights(len(growth_rate_2_df_normalized_scores))

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
        final_scores = final_scores.sort_values(by='final_score', ascending=False)
        return final_scores
    
    def sma(self, series, length):
        return series.rolling(length).mean()

    def computed_prices(self, series, length=9, offset=0.85, sigma=6):
        if length < 1:
            return series
        m = offset * (length - 1)
        s = length / sigma
        idx = np.arange(length)
        w = np.exp(-((idx - m) ** 2) / (2 * s * s))
        w = w / w.sum()
        return series.rolling(length).apply(lambda x: (x * w).sum(), raw=True)

    def get_entry_indicator(self, df_1d, shortlen, longlen, basisLen, offset, offsetSigma):
        # Calculate the moving averages
        short_sma = df_1d['close'].rolling(window=shortlen, min_periods=1, center=False).mean()
        long_sma = df_1d['close'].rolling(window=longlen, min_periods=1, center=False).mean()
        # self.logger.debug({'shortlen:':shortlen, 'longlen':longlen, 'basisLen':basisLen, 'offset':offset, 'offsetSigma':offsetSigma})
        # self.logger.debug({'short_sma':short_sma['MSFT']})
        # self.logger.debug({'long_sma':long_sma['MSFT']})
        closeSeries = df_1d['close'].apply(lambda x: self.computed_prices(x, length=basisLen, offset=offset, sigma=offsetSigma))
        openSeries = df_1d['open'].apply(lambda x: self.computed_prices(x, length=basisLen, offset=offset, sigma=offsetSigma))
        # Create Positions
        conditions = [
            closeSeries.gt(openSeries),
            closeSeries.lt(openSeries)
        ]
        choices = [1, -1]

        position = np.select(conditions, choices, default=0)
        position = pd.DataFrame(position, index=closeSeries.index, columns=closeSeries.columns)
        positions = position.iloc[-1]
        entry_indicator_strength = closeSeries.iloc[-1]/openSeries.iloc[-1]
        
        # If position == 0, then entry_indicator_strength = 0, if position == 1, then entry_indicator_strength = entry_indicator_strength, if position == -1, then entry_indicator_strength = 1/entry_indicator_strength
        entry_indicator_strength = entry_indicator_strength * positions
        entry_indicator_strength[positions == -1] = 1/entry_indicator_strength[positions == -1] * -1
        sma_trend_indicator = short_sma.iloc[-1]/long_sma.iloc[-1]
        
        return positions, entry_indicator_strength, sma_trend_indicator
        
    def run_strategy(self, next_rows, market_data_df, system_timestamp):
        if self.granularity in next_rows.index and self.granularity in market_data_df.index.levels[0] and (len(market_data_df.loc[self.granularity]) > self.data_inputs[self.granularity]['lookback']):
            # start = time.time()
            trend_strength_scores = self.create_trend_strength_scores(market_data_df, symbols_to_trade=self.tickers, granularity=self.granularity)
            # Validate the long_short_symbol_count to not be too high as compared to the number of assets
            # self.logger.debug(f"Time taken to create trend_strength_scores: {time.time() - start}")
            max_required_len = self.long_short_symbol_count * 4
            long_symbols = list(trend_strength_scores.head(int(len(trend_strength_scores)/2)).index)[:max_required_len]
            short_symbols = list(trend_strength_scores.tail(int(len(trend_strength_scores)/2)).index)[::-1][:max_required_len]

            long_signal_symbols = dict()
            short_signal_symbols = dict()
            non_signal_symbols = dict()
            df = deepcopy(market_data_df)
            
            # Only create data for tickers that are in market_data_df
            tickers_temp = long_symbols + short_symbols
            # Only retain the columns that are in the tickers list
            trimmed_input_df = df.loc['1d', pd.IndexSlice[:, tickers_temp]]
            
            positions, entry_indicator_strengths, sma_trend_indicator = self.get_entry_indicator(trimmed_input_df, self.shortlen, self.longlen, self.basisLen, self.offset, self.offsetSigma)
            self.minimum_granularity = self.market_data_extractor.get_market_data_df_minimum_granularity(market_data_df)
            
            for symbol, value in trend_strength_scores.iterrows():
                if symbol in long_symbols or symbol in short_symbols:
                    entry_indicator = positions[symbol]
                    entry_indicator_strength = entry_indicator_strengths[symbol]
                    symbol_ltp = df.loc[self.minimum_granularity , 'close'][symbol].iloc[-1]
                    if entry_indicator == 1 and symbol in long_symbols:
                        long_signal_symbols[symbol] = {'direction':entry_indicator, 'trend_indicator_strength':value['final_score'], 'entry_indicator_strength':entry_indicator_strength, 'symbol_ltp':symbol_ltp}
                    elif entry_indicator == -1 and symbol in short_symbols:
                        short_signal_symbols[symbol] = {'direction':entry_indicator, 'trend_indicator_strength':value['final_score'], 'entry_indicator_strength':entry_indicator_strength, 'symbol_ltp':symbol_ltp}
                    else:
                        non_signal_symbols[symbol] = {'direction':0, 'trend_indicator_strength':value['final_score'], 'entry_indicator_strength':0, 'symbol_ltp':None}
                else:
                    non_signal_symbols[symbol] = {'direction':0, 'trend_indicator_strength':value['final_score'], 'entry_indicator_strength':0, 'symbol_ltp':None}
            
            # self.logger.debug(f"Time taken to create Rest: {time.time() - start}")
            
            decision_matrix = {'trend_strength_scores':trend_strength_scores, 'entry_indicator_strengths':entry_indicator_strengths, 'sma_trend_indicator':sma_trend_indicator}
            
            return long_signal_symbols, short_signal_symbols, non_signal_symbols, decision_matrix
      
    def generate_signals(self, next_rows, market_data_df, system_timestamp, open_orders):
        """
        Generate signals based on the strategy. THIS IS DUMMY CODE FOR CREATING IDEAL PORTFOLIO.
        """
        signals = []
        return_type = None
        
        #run strategy and get result data
        if self.granularity in next_rows.index and self.granularity in market_data_df.index.levels[0] and (len(market_data_df.loc[self.granularity]) > self.data_inputs[self.granularity]['lookback']):
            if system_timestamp > self.config_dict['backtest_inputs']['start_time']:
                '''STEP 1: Trim the universe to only include high possibilities (THIS HAS NOT BEEN WRITTEN YET. INCLUDE IF NEEDED)'''
                # tickers_new = self.get_list_of_symbols_to_trade(system_timestamp, universe_size=self.universe_size)
                self.symbols_to_trade = self.tickers #self.get_list_of_symbols_to_trade(system_timestamp, market_data_df, self.universe_size)
                # bullish_score = self.run_regime_filter(market_data_df, regime_lookback=7)
                
                '''STEP 3: Convert Open Orders into an easily searchable dictionary'''
                current_open_positions = create_open_positions_from_open_orders(open_orders=open_orders, strategy_name=self.strategy_name)
                
                '''STEP 2: Run Strategy Rules and create long and short signals'''
                long_signal_symbols, short_signal_symbols, non_signal_symgols, decision_matrix = self.run_strategy(next_rows, market_data_df, system_timestamp)
                # self.logger.debug({'long_signal_symbols':long_signal_symbols.keys(), 'short_signal_symbols':short_signal_symbols.keys()})
                
                '''STEP 4: Now look at the current open symbols and check if there is any position that has flipped. If yes, then do a MARKET_EXIT'''
                updated_current_open_positions = deepcopy(current_open_positions)
                for symbol, open_position in current_open_positions.items():
                    if open_position['orderQuantity'] > 0 and symbol in short_signal_symbols.keys():
                        signal = {"symbol": symbol, 
                                "trend_strength":short_signal_symbols[symbol]['trend_indicator_strength'],
                                "entry_indicator_strength":short_signal_symbols[symbol]['entry_indicator_strength'],
                                "signal_strength":1, # Need to fix up signal strength
                                "strategy_name": self.strategy_name,
                                "timestamp": system_timestamp,
                                "entry_order_type": self.entry_order_type, 
                                "exit_order_type":self.exit_order_type, 
                                "stoploss_pct": self.stop_loss_pct,
                                # "sl_abs": (1-self.stop_loss_abs) * current_price, 
                                "symbol_ltp" : {system_timestamp:short_signal_symbols[symbol]['symbol_ltp']},
                                "timeInForce" : self.timeInForce, 
                                "orderQuantity" : open_position['orderQuantity'],
                                'orderDirection': 'BUY',
                                'granularity': self.granularity,
                                'signal_type':'MARKET_EXIT',
                                'market_neutral':False,
                                'decision_matrix':decision_matrix
                                }
                        signals.append(signal)
                        # sleeper(5, f'MARKET_EXIT ORDER SENT: {symbol}')
                        del updated_current_open_positions[symbol]
                    elif open_position['orderQuantity'] < 0 and symbol in long_signal_symbols.keys():
                        signal = {"symbol": symbol, 
                                "trend_strength":long_signal_symbols[symbol]['trend_indicator_strength'],
                                "entry_indicator_strength":long_signal_symbols[symbol]['entry_indicator_strength'],
                                "signal_strength":1, # Need to fix up signal strength
                                "strategy_name": self.strategy_name, 
                                "timestamp": system_timestamp,
                                "entry_order_type": self.entry_order_type, 
                                "exit_order_type":self.exit_order_type, 
                                "stoploss_pct": self.stop_loss_pct,
                                # "sl_abs": (1-self.stop_loss_abs) * current_price, 
                                "symbol_ltp" : {system_timestamp:long_signal_symbols[symbol]['symbol_ltp']},
                                "timeInForce" : self.timeInForce,
                                "orderQuantity" : open_position['orderQuantity'],
                                'orderDirection': 'SELL',
                                'granularity': self.granularity,
                                'signal_type':'MARKET_EXIT',
                                'market_neutral':False,
                                'decision_matrix':decision_matrix
                                }
                        signals.append(signal)
                        # sleeper(5, f'MARKET_EXIT ORDER SENT: {symbol}')
                        del updated_current_open_positions[symbol]
                
                current_open_positions = deepcopy(updated_current_open_positions)
                
                '''STEP 5: Now add the new orders to get to 5 long/5 short, add from the top, and make sure you aren't adding the same symbol again'''
                # list of all keys where the value > 0 in current_open_positions
                updated_long_positions = [key for key, value in current_open_positions.items() if value['orderQuantity'] > 0]
                updated_short_positions = [key for key, value in current_open_positions.items() if value['orderQuantity'] < 0]
                
                # self.logger.debug({'updated_long_positions':updated_long_positions, 'updated_short_positions':updated_short_positions})
                if len(updated_long_positions) < self.long_short_symbol_count:
                    for symbol in long_signal_symbols.keys():
                        if len(updated_long_positions) < self.long_short_symbol_count and symbol not in updated_long_positions:
                            signal = {"symbol": symbol, 
                                    "trend_strength":long_signal_symbols[symbol]['trend_indicator_strength'],
                                    "entry_indicator_strength":long_signal_symbols[symbol]['entry_indicator_strength'],
                                    "signal_strength":1, # Need to fix up signal strength
                                    "strategy_name": self.strategy_name,
                                    "timestamp": system_timestamp,
                                    "entry_order_type": self.entry_order_type, 
                                    "exit_order_type":self.exit_order_type, 
                                    "stoploss_pct": self.stop_loss_pct,
                                    # "sl_abs": (1-self.stop_loss_abs) * current_price, 
                                    "symbol_ltp" : {system_timestamp:long_signal_symbols[symbol]['symbol_ltp']},
                                    "timeInForce" : self.timeInForce, 
                                    "orderQuantity" : int(10000/long_signal_symbols[symbol]['symbol_ltp']),
                                    'orderDirection': 'BUY',
                                    'granularity': self.granularity,
                                    'signal_type':'BUY_SELL',
                                    'market_neutral':False,
                                    'decision_matrix':decision_matrix
                                    }
                            signals.append(signal)
                            updated_long_positions.append(symbol)
                            # sleeper(5, f'BUY_SELL ORDER SENT: {symbol}')
                        # else:
                            # self.logger.debug(f"Symbol: {symbol} not added to long positions")
                if len(updated_short_positions) < self.long_short_symbol_count:
                    for symbol in short_signal_symbols.keys():
                        if len(updated_short_positions) < self.long_short_symbol_count and symbol not in updated_short_positions:
                            signal = {"symbol": symbol, 
                                    "trend_strength":short_signal_symbols[symbol]['trend_indicator_strength'],
                                    "entry_indicator_strength":short_signal_symbols[symbol]['entry_indicator_strength'],
                                    "signal_strength":1, # Need to fix up signal strength
                                    "strategy_name": self.strategy_name, 
                                    "timestamp": system_timestamp,
                                    "entry_order_type": self.entry_order_type, 
                                    "exit_order_type":self.exit_order_type, 
                                    "stoploss_pct": self.stop_loss_pct,
                                    # "sl_abs": (1-self.stop_loss_abs) * current_price, 
                                    "symbol_ltp" : {system_timestamp:short_signal_symbols[symbol]['symbol_ltp']},
                                    "timeInForce" : self.timeInForce,
                                    "orderQuantity" : int(10000/short_signal_symbols[symbol]['symbol_ltp']),
                                    'orderDirection': 'SELL',
                                    'granularity': self.granularity,
                                    'signal_type':'BUY_SELL',
                                    'market_neutral':False,
                                    'decision_matrix':decision_matrix
                                    }
                            signals.append(signal)
                            updated_short_positions.append(symbol)
                            # sleeper(5, f'BUY_SELL ORDER SENT: {symbol}')
                        # else:
                        #     self.logger.debug(f"Symbol: {symbol} not added to short positions")

                # self.logger.debug({'system_timestamp':system_timestamp, 'signal':signal})
                # self.logger.debug(f"Previous SMA15: {round(asset_data_df.iloc[-2]['SMA15'], 2)}, Current SMA15: {round(asset_data_df.iloc[-1]['SMA15'], 2)}")
                # self.logger.debug(f"Previous SMA30: {round(asset_data_df.iloc[-2]['SMA30'], 2)}, Current SMA30: {round(asset_data_df.iloc[-1]['SMA30'], 2)}")
                # self.logger.debug("-" * 150)
                # self.sleeper(5, "Strategy 1 Manual Sleep")
                return_type = 'signals'
                    
        # self.logger.debug({'signals':signals})
        return return_type, signals, self.tickers