"""
get demo strategy.py to work with the vault system
"""
import pickle, time, json, os
from vault.base_strategy import BaseStrategy
import numpy as np
import pandas as pd
from systems.utils import project_path, sleeper, MarketDataExtractor, create_open_positions_from_open_orders
from copy import deepcopy
from datetime import datetime, timezone, timedelta


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
        self.long_short_symbol_count = 3
        self.min_market_cap_in_billions = 30
        self.tickers = []
        self.prices_data_dict = {}
        self.market_data_extractor = MarketDataExtractor()
        # self.data_inputs, self.tickers = self.datafeeder_inputs()
        ### New Strategy Parameters
        self.shortlen=30
        self.longlen = 2 * self.shortlen
        self.lookback = self.longlen
        self.basisLen=30
        self.offsetSigma=6
        self.offset=0.85

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
        - 5_1_1: Vectorized, so backtesting should be faster
        - 5_1_2: removing our own trend indicator and using a simple sma 180/365
        
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
        # self.tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'HBNC', 'NFLX', 'GS', 'AMD', 'XOM', 'JNJ', 'JPM','V', 'PG', 'UNH',  'DIS', 'HD', 'CRM', 'NKE'] #self.tickers
        # self.tickers = self.tickers[:40]
        # self.tickers = self.tickers[:100]
        
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
            stock_symbols_df = stock_symbols_df.sort_values(by='Market Cap', ascending=False)
            
            return stock_symbols_df
        
        stock_symbols_df = get_list_of_all_symbols()
        # self.logger.debug({'stock_symbols_df':stock_symbols_df})
        stock_symbols_df = filter_symbols_by_market_cap(stock_symbols_df, self.min_market_cap_in_billions)
        
        # sort stock_symbols_df by column Market Cap
        stock_symbols_df = stock_symbols_df.drop_duplicates(subset=['Symbol'])
        shortlisted_symbols = stock_symbols_df['Symbol'].tolist()

        return shortlisted_symbols, stock_symbols_df
        
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
    
    def sma(self, series, length):
        return series.rolling(length).mean()

    def computed_prices(self, df, length=9, offset=0.85, sigma=6):
        if length < 1:
            return df

        # Precompute Gaussian weights
        m = offset * (length - 1)
        s = length / sigma
        idx = np.arange(length)
        w = np.exp(-((idx - m) ** 2) / (2 * s * s))
        w /= w.sum()

        # Perform weighted rolling computation across all columns (vectorized)
        def rolling_weighted_sum(x):
            # Dynamically adjust weights to match the size of x
            current_len = len(x)
            w_adjusted = w[:current_len] / w[:current_len].sum()  # Normalize weights for the current window
            return np.dot(x, w_adjusted)

        # Apply rolling weighted sum across all symbols at once
        return df.rolling(window=length, min_periods=1).apply(rolling_weighted_sum, raw=True)
    
    def get_entry_indicator(self, df, shortlen, longlen, basisLen, offset, offsetSigma, open_symbols):
        df_1d = df.loc[self.granularity]
        short_sma = df_1d['close'].rolling(window=shortlen, min_periods=1, center=False).mean().iloc[-1]
        long_sma = df_1d['close'].rolling(window=longlen, min_periods=1, center=False).mean().iloc[-1]

        trend_strength = short_sma/long_sma
        sma_strength_long_symbols = trend_strength[(trend_strength > 1) & (trend_strength > 1.04)]
        sma_strength_short_symbols = trend_strength[(trend_strength < 1) & (trend_strength < 0.96)]

        # Sort long_signals in descending order
        sma_strength_long_symbols = sma_strength_long_symbols.sort_values(ascending=False)
        sma_strength_short_symbols = sma_strength_short_symbols.sort_values(ascending=True)

        # Trim the list of signals
        max_required_len = self.long_short_symbol_count * 2
        sma_strength_long_symbols = sma_strength_long_symbols[:max_required_len]
        sma_strength_short_symbols = sma_strength_short_symbols[:max_required_len]

        tickers_temp = list(set(list(sma_strength_long_symbols.index) + list(sma_strength_short_symbols.index) + open_symbols))
        if len(tickers_temp) == 0:
            tickers_temp = [df_1d['close'].columns[0]]
        self.logger.debug({'tickers_temp':tickers_temp})
        trimmed_input_df = df.loc['1d', pd.IndexSlice[:, tickers_temp]]
        df_1d_trimmed = trimmed_input_df.copy()

        short_sma_trimmed = short_sma[short_sma.index.isin(tickers_temp)]
        long_sma_trimmed = long_sma[long_sma.index.isin(tickers_temp)]

        def computed_prices(df, length=9, offset=0.85, sigma=6):
            if length < 1:
                return df

            # Precompute Gaussian weights
            m = offset * (length - 1)
            s = length / sigma
            idx = np.arange(length)
            w = np.exp(-((idx - m) ** 2) / (2 * s * s))
            w /= w.sum()

            # Perform weighted rolling computation across all columns (vectorized)
            def rolling_weighted_sum(x):
                # Dynamically adjust weights to match the size of x
                current_len = len(x)
                w_adjusted = w[:current_len] / w[:current_len].sum()  # Normalize weights for the current window
                return np.dot(x, w_adjusted)

            # Apply rolling weighted sum across all symbols at once
            return df.rolling(window=length, min_periods=1).apply(rolling_weighted_sum, raw=True)
        try:        
            closeSeries = computed_prices(df_1d_trimmed['close'], length=basisLen, offset=offset, sigma=offsetSigma).iloc[-1]
            openSeries = computed_prices(df_1d_trimmed['open'], length=basisLen, offset=offset, sigma=offsetSigma).iloc[-1]
            closeSeries = closeSeries.loc[tickers_temp]
            openSeries = openSeries.loc[tickers_temp]
        except Exception as e:
            self.logger.debug({'df_1d':df_1d})
            self.logger.debug({'df_1d_trimmed':df_1d_trimmed})
            raise Exception(e)

        # Create Positions
        # conditions = [
        #     ((closeSeries.gt(openSeries)) & (short_sma_trimmed.gt(long_sma_trimmed))),
        #     ((closeSeries.lt(openSeries)) & (short_sma_trimmed.lt(long_sma_trimmed)))
        # ]
        # choices = [1, -1]
        # positions = np.select(conditions, choices, default=0)
        decision_matrix_df = pd.DataFrame(index=tickers_temp)
        # Include symbol LTP        
        symbol_ltp = df_1d_trimmed.iloc[-1]['close']
        decision_matrix_df['symbol_ltp'] = symbol_ltp.loc[symbol_ltp.index.isin(decision_matrix_df.index)]

        decision_matrix_df['short_sma_trimmed'] = short_sma_trimmed
        decision_matrix_df['long_sma_trimmed'] = long_sma_trimmed
        decision_matrix_df['trend_strength'] = trend_strength.loc[trend_strength.index.isin(decision_matrix_df.index)]

        decision_matrix_df['trend_strength_direction'] = 0
        decision_matrix_df.loc[(decision_matrix_df['trend_strength'] > 1) & (decision_matrix_df['trend_strength'] > 1.04), 'trend_strength_direction'] = 1
        decision_matrix_df.loc[(decision_matrix_df['trend_strength'] < 1) & (decision_matrix_df['trend_strength'] < 0.96), 'trend_strength_direction'] = -1
        
        # self.logger.debug({'decision_matrix_df':decision_matrix_df})
        decision_matrix_df['trend_strength_score'] = float(0)
        decision_matrix_df.loc[(decision_matrix_df['trend_strength'] > 1) & (decision_matrix_df['trend_strength'] != 0), 'trend_strength_score'] = decision_matrix_df['trend_strength'].astype(float)
        decision_matrix_df.loc[(decision_matrix_df['trend_strength'] < 1) & (decision_matrix_df['trend_strength'] != 0), 'trend_strength_score'] = (1/decision_matrix_df['trend_strength']).astype(float)

        # Include closeSeries and openSeries
        decision_matrix_df['closeSeries'] = closeSeries.loc[closeSeries.index.isin(decision_matrix_df.index)]
        decision_matrix_df['openSeries'] = openSeries.loc[openSeries.index.isin(decision_matrix_df.index)]

        decision_matrix_df['entry_indicator_direction'] = decision_matrix_df['closeSeries'] / decision_matrix_df['openSeries']
        decision_matrix_df['entry_strength_indicator_direction'] = 0
        decision_matrix_df.loc[decision_matrix_df['closeSeries'] > decision_matrix_df['openSeries'], 'entry_strength_indicator_direction'] = 1
        decision_matrix_df.loc[decision_matrix_df['closeSeries'] < decision_matrix_df['openSeries'], 'entry_strength_indicator_direction'] = -1

        decision_matrix_df['entry_indicator_strength'] = decision_matrix_df['closeSeries'] / decision_matrix_df['openSeries']
        # if entry_indicator_strength < 1, then make it 1/entry_indicator_strength
        decision_matrix_df.loc[decision_matrix_df['entry_indicator_strength'] < 1, 'entry_indicator_strength'] = 1/decision_matrix_df['entry_indicator_strength']

        decision_matrix_df['position'] = 0
        decision_matrix_df.loc[(decision_matrix_df['closeSeries'] > decision_matrix_df['openSeries']) & (decision_matrix_df['trend_strength_direction'] == 1), 'position'] = 1
        decision_matrix_df.loc[(decision_matrix_df['closeSeries'] < decision_matrix_df['openSeries']) & (decision_matrix_df['trend_strength_direction'] == -1), 'position'] = -1

        decision_matrix_df['position_score'] = decision_matrix_df['position'] * decision_matrix_df['trend_strength_score']

        # sort by position_score
        decision_matrix_df = decision_matrix_df.sort_values(by='position_score', ascending=False)
        # # Remove all symbols where score is 
        # decision_matrix_df = decision_matrix_df[decision_matrix_df['position_score'] != 0]

        return decision_matrix_df
        
    def run_strategy(self, next_rows, market_data_df, system_timestamp, open_symbols):
        if self.granularity in next_rows.index and self.granularity in market_data_df.index.levels[0] and (len(market_data_df.loc[self.granularity]) > self.data_inputs[self.granularity]['lookback']):
            df = deepcopy(market_data_df)
            # Only create data for tickers that are in market_data_df
            decision_matrix_df = self.get_entry_indicator(df, self.shortlen, self.longlen, self.basisLen, self.offset, self.offsetSigma, open_symbols)
            self.minimum_granularity = self.market_data_extractor.get_market_data_df_minimum_granularity(market_data_df)
            
            return decision_matrix_df
      
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
                open_symbols = list(current_open_positions.keys())
                '''STEP 2: Run Strategy Rules and create long and short signals'''
                # start = time.time()
                decision_matrix_df = self.run_strategy(next_rows, market_data_df, system_timestamp, open_symbols)
                decision_matrix_long_df = decision_matrix_df[decision_matrix_df['position'] > 0]
                decision_matrix_short_df = decision_matrix_df[decision_matrix_df['position'] < 0][::-1]
                decision_matrix_none_df = decision_matrix_df[decision_matrix_df['position'] == 0]
                decision_matrix_open_orders_df = decision_matrix_df[decision_matrix_df.index.isin(open_symbols)]
                # self.logger.debug({'decision_matrix_df':decision_matrix_df})
                # self.logger.debug(f"Time taken to run_strategy: {time.time() - start}")
                # self.logger.debug({'long_signal_symbols':long_signal_symbols.keys(), 'short_signal_symbols':short_signal_symbols.keys()})
                
                '''STEP 4: Now look at the current open symbols and check if there is any position that has flipped. If yes, then do a MARKET_EXIT'''
                updated_current_open_positions = deepcopy(current_open_positions)
                for symbol, open_position in current_open_positions.items():
                    symbol_position = decision_matrix_df.loc[symbol]['position']
                    if open_position['orderQuantity'] > 0 and symbol_position != 1:
                        symbol_ltp = decision_matrix_short_df.loc[symbol]['symbol_ltp'] if symbol in decision_matrix_short_df.index else decision_matrix_none_df.loc[symbol]['symbol_ltp']
                        signal = {"symbol": symbol,
                                "signal_strength":1, # Need to fix up signal strength
                                "strategy_name": self.strategy_name,
                                "timestamp": system_timestamp,
                                "entry_order_type": self.entry_order_type,
                                "exit_order_type":self.exit_order_type,
                                "stoploss_pct": self.stop_loss_pct,
                                # "sl_abs": (1-self.stop_loss_abs) * current_price, 
                                "symbol_ltp" : {system_timestamp:symbol_ltp},
                                "timeInForce" : self.timeInForce, 
                                "orderQuantity" : open_position['orderQuantity'],
                                'orderDirection': 'BUY',
                                'granularity': self.granularity,
                                'signal_type':'MARKET_EXIT',
                                'market_neutral':False,
                                'decision_matrix':decision_matrix_df
                                }
                        signals.append(signal)
                        # self.logger.debug(f'decision_matrix_df:\n {decision_matrix_df}')
                        # sleeper(5, f'MARKET_EXIT ORDER SENT: {symbol}')
                        # raise AssertionError(f'MARKET_EXIT ORDER SENT: {symbol}')
                        del updated_current_open_positions[symbol]
                    elif open_position['orderQuantity'] < 0 and symbol_position != -1:
                        symbol_ltp = decision_matrix_long_df.loc[symbol]['symbol_ltp'] if symbol in decision_matrix_long_df.index else decision_matrix_none_df.loc[symbol]['symbol_ltp']
                        signal = {"symbol": symbol, 
                                "signal_strength":1, # Need to fix up signal strength
                                "strategy_name": self.strategy_name, 
                                "timestamp": system_timestamp,
                                "entry_order_type": self.entry_order_type, 
                                "exit_order_type":self.exit_order_type, 
                                "stoploss_pct": self.stop_loss_pct,
                                # "sl_abs": (1-self.stop_loss_abs) * current_price, 
                                "symbol_ltp" : {system_timestamp:symbol_ltp},
                                "timeInForce" : self.timeInForce,
                                "orderQuantity" : open_position['orderQuantity'],
                                'orderDirection': 'SELL',
                                'granularity': self.granularity,
                                'signal_type':'MARKET_EXIT',
                                'market_neutral':False,
                                'decision_matrix':decision_matrix_df
                                }
                        signals.append(signal)
                        # sleeper(5, f'MARKET_EXIT ORDER SENT: {symbol}')
                        # self.logger.debug(f'decision_matrix_df:\n {decision_matrix_df}')
                        # raise AssertionError(f'MARKET_EXIT ORDER SENT: {symbol}')
                        
                        del updated_current_open_positions[symbol]
                
                current_open_positions = deepcopy(updated_current_open_positions)
                
                '''STEP 5: Now add the new orders to get to 5 long/5 short, add from the top, and make sure you aren't adding the same symbol again'''
                # list of all keys where the value > 0 in current_open_positions
                updated_long_positions = [key for key, value in current_open_positions.items() if value['orderQuantity'] > 0]
                updated_short_positions = [key for key, value in current_open_positions.items() if value['orderQuantity'] < 0]
                
                # self.logger.debug({'updated_long_positions':updated_long_positions, 'updated_short_positions':updated_short_positions})
                if len(updated_long_positions) < self.long_short_symbol_count:
                    # self.logger.debug({'decision_matrix_long_df':decision_matrix_long_df})
                    if not decision_matrix_long_df.empty:
                        for symbol in decision_matrix_long_df.index:
                            if len(updated_long_positions) < self.long_short_symbol_count and symbol not in updated_long_positions:
                                signal = {"symbol": symbol, 
                                        "trend_strength":decision_matrix_long_df.loc[symbol]['trend_strength_score'],
                                        "entry_indicator_strength":decision_matrix_long_df.loc[symbol]['entry_indicator_strength'],
                                        "signal_strength":1, # Need to fix up signal strength
                                        "strategy_name": self.strategy_name,
                                        "timestamp": system_timestamp,
                                        "entry_order_type": self.entry_order_type, 
                                        "exit_order_type":self.exit_order_type, 
                                        "stoploss_pct": self.stop_loss_pct,
                                        # "sl_abs": (1-self.stop_loss_abs) * current_price, 
                                        "symbol_ltp" : {system_timestamp:decision_matrix_long_df.loc[symbol]['symbol_ltp']},
                                        "timeInForce" : self.timeInForce, 
                                        "orderQuantity" : int(10000/decision_matrix_long_df.loc[symbol]['symbol_ltp']),
                                        'orderDirection': 'BUY',
                                        'granularity': self.granularity,
                                        'signal_type':'BUY_SELL',
                                        'market_neutral':False,
                                        'decision_matrix':decision_matrix_df
                                        }
                                signals.append(signal)
                                updated_long_positions.append(symbol)
                            # sleeper(5, f'BUY_SELL ORDER SENT: {symbol}')
                        # else:
                            # self.logger.debug(f"Symbol: {symbol} not added to long positions")
                if len(updated_short_positions) < self.long_short_symbol_count:
                    if not decision_matrix_short_df.empty:
                        for symbol in decision_matrix_short_df.index:
                            if len(updated_short_positions) < self.long_short_symbol_count and symbol not in updated_short_positions:
                                signal = {"symbol": symbol, 
                                        "trend_strength":decision_matrix_short_df.loc[symbol]['trend_strength_score'],
                                        "entry_indicator_strength":decision_matrix_short_df.loc[symbol]['entry_indicator_strength'],
                                        "signal_strength":1, # Need to fix up signal strength
                                        "strategy_name": self.strategy_name, 
                                        "timestamp": system_timestamp,
                                        "entry_order_type": self.entry_order_type, 
                                        "exit_order_type":self.exit_order_type, 
                                        "stoploss_pct": self.stop_loss_pct,
                                        # "sl_abs": (1-self.stop_loss_abs) * current_price, 
                                        "symbol_ltp" : {system_timestamp:decision_matrix_short_df.loc[symbol]['symbol_ltp']},
                                        "timeInForce" : self.timeInForce,
                                        "orderQuantity" : int(10000/decision_matrix_short_df.loc[symbol]['symbol_ltp']),
                                        'orderDirection': 'SELL',
                                        'granularity': self.granularity,
                                        'signal_type':'BUY_SELL',
                                        'market_neutral':False,
                                        'decision_matrix':decision_matrix_df
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