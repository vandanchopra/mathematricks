"""
Create a simple Single Moving Crossover Strategy as a demo
"""

from vault.base_strategy import BaseStrategy
import numpy as np
import sys, time, os
import pandas as pd


class Strategy(BaseStrategy):
    def __init__(self, config_dict):
        super().__init__()
        self.strategy_name = 'strategy_1'
        self.granularity = "1d"
        self.stop_loss_pct = 0.05    #percentage
        self.target_pct = 0.08       #percentage
        self.orderType = "MARKET" #MARKET, STOPLOSS-MARKET
        self.exit_order_type = "stoploss_pct" #sl_pct , sl_abs
        self.timeInForce = "DAY"    #DAY, Expiry, IoC (immediate or cancel) , TTL (Order validity in minutes) 
        self.orderQuantity = 10
        self.min_market_cap_in_billions = 30
        self.lookback = 80
        self.data_inputs, self.tickers = self.datafeeder_inputs()
        
    def get_name(self):
        return self.strategy_name
    
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
    
    def datafeeder_inputs(self):
        # tickers = self.tickers, self.stock_symbols_df = self.get_universe()
        # # tickers = ['MSFT', 'NVDA']
        # data_inputs = {'1d': {'columns': ['open', 'high', 'close', 'volume'] , 'lookback':60}} #'1m': {'columns': ['open', 'high', 'close', 'volume'] , 'lookback':100}
        # return data_inputs, tickers
        self.tickers, self.stock_symbols_df = self.get_universe()
        self.data_inputs = { "1d" : {'columns': ['open', 'high', 'low', 'close', 'volume'] , 'lookback':self.lookback}, 
                                #  "1m" : {'columns': ['open', 'high', 'low', 'close', 'volume'] , 'lookback':365}
                                }
        
        symbols_to_remove = ['BRK/A', 'BRK/B', 'MNST', 'LIN', 'TPL']
        for symbol_to_remove in symbols_to_remove:
            if symbol_to_remove in self.tickers:
                self.tickers.remove(symbol_to_remove)
                
        return self.data_inputs, self.tickers
        
    def run_regime_filter(self, market_data_df, regime_lookback):
        # in market_data_df.loc['1d'], divide the last close price by the close price 7 days ago and get a matrix
        regime_calc_matrix = market_data_df.loc['1d'].iloc[-1]['close'] / market_data_df.loc['1d'].iloc[-regime_lookback]['close']
        bullish_score = (regime_calc_matrix.values > 1).sum() / len(regime_calc_matrix)
        return bullish_score
    
    def generate_signals(self, next_rows, market_data_df, system_timestamp):
        """
        Generate signals based on the strategy. THIS IS DUMMY CODE FOR CREATING SIGNALS.
        """
        signals = []
        return_type = None
        if(self.granularity in market_data_df.index.levels[0] and len(market_data_df.loc[self.granularity]) >= self.data_inputs[self.granularity]['lookback']):
            bullish_score = self.run_regime_filter(market_data_df, regime_lookback=7)
            self.logger.debug({'bullish_score':bullish_score})
            
            for symbol in set(market_data_df["open"].columns):
                asset_data_df = market_data_df.loc[self.granularity].xs(symbol, axis=1, level='symbol').reset_index()
                asset_data_df['SMA15'] = asset_data_df['close'].rolling(window=15).mean()
                asset_data_df['SMA60'] = asset_data_df['close'].rolling(window=60).mean()

                asset_data_df['signal_strength'] = 0
                asset_data_df.loc[60:, 'signal_strength'] = np.where(asset_data_df.loc[60:, 'SMA15'] > asset_data_df.loc[60:, 'SMA60'], 1, 0)
                asset_data_df['position'] = asset_data_df['signal_strength'].diff()
                current_price = asset_data_df.iloc[-1]['close']
                #in the above df the position column will have a +1 value in case of a positive sma crossover and -1 in the opposite case
                #otherwise it will have 0 value
                #so we could long if position value is 1 and short when its -1

                #but for testing purposes I want to generate signals for each instruments 
                
                #long if above sma15 else short
                if (round(asset_data_df.iloc[-1]['SMA15'], 2) > round(asset_data_df.iloc[-1]['SMA60'], 2)) and (round(asset_data_df.iloc[-2]['SMA15'], 2) <= round(asset_data_df.iloc[-2]['SMA60'], 2)) and (bullish_score >= 0.7):
                    signal_strength = 1
                    orderDirection = "BUY"
                elif (round(asset_data_df.iloc[-1]['SMA15'], 2) < round(asset_data_df.iloc[-1]['SMA60'], 2)) and (round(asset_data_df.iloc[-2]['SMA15'], 2) >= round(asset_data_df.iloc[-2]['SMA60'], 2)) and (bullish_score <= 0.3):
                    signal_strength = 1
                    orderDirection = "SELL"
                else:
                    signal_strength = 0
                #signal_strength = int(asset_data_df.iloc[-1]['position'])
                if signal_strength != 0:
                    signal = {"symbol": symbol, 
                            "signal_strength":signal_strength, 
                            "strategy_name": self.strategy_name, 
                            "timestamp": system_timestamp,
                            "entry_order_type": self.orderType, 
                            "exit_order_type":self.exit_order_type, 
                            "stoploss_pct": self.stop_loss_pct,
                            # "sl_abs": (1-self.stop_loss_abs) * current_price, 
                            "symbol_ltp" : {system_timestamp:current_price}, 
                            "timeInForce" : self.timeInForce , 
                            "orderQuantity" : self.orderQuantity,
                            'orderDirection': orderDirection,
                            'granularity': self.granularity,
                            'signal_type':'BUY_SELL',
                            'market_neutral':False,
                            }
                    signals.append(signal)
                    # self.logger.debug({'system_timestamp':system_timestamp, 'signal':signal})
                    # self.logger.debug(f"Previous SMA15: {round(asset_data_df.iloc[-2]['SMA15'], 2)}, Current SMA15: {round(asset_data_df.iloc[-1]['SMA15'], 2)}")
                    # self.logger.debug(f"Previous SMA30: {round(asset_data_df.iloc[-2]['SMA30'], 2)}, Current SMA30: {round(asset_data_df.iloc[-1]['SMA30'], 2)}")
                    # self.logger.debug("-" * 150)
                    # self.sleeper(5, "Strategy 1 Manual Sleep")
                    return_type = 'signals'
        # self.logger.debug({'signals':signals})
        return return_type, signals, self.tickers