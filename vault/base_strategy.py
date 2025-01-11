import numpy as np
from datetime import timedelta
import pandas as pd
from systems.utils import create_logger, sleeper

class BaseStrategy:
    def __init__(self, config_dict=None):
        self.strategy_name = 'BaseStrategy'
        self.logger = create_logger(log_level='DEBUG', logger_name='Vault-Strategy', print_to_console=True)
        self.sleeper = sleeper
        self.config_dict = config_dict
        
    def get_name(self):
        return self.strategy_name
    
    def get_signal(self, data):
        raise NotImplementedError

    def get_target(self, data):
        raise NotImplementedError

    def get_trades(self, data):
        raise NotImplementedError

    def get_metrics(self, data):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    def set_params(self, config):
        raise NotImplementedError
    
    def create_analysis_array_symbol_efficient(self, df, start_date_dt, days_ago_list):
        # Convert 'Date' column to datetime type
        df['Date'] = pd.to_datetime(df['Date'])

        # Filter the dataframe to include only the data needed
        df_pruned = df.iloc[-days_ago_list[-1]:] if len(df) >= days_ago_list[-1] else df

        # Ensure we have data to process
        if not df_pruned.empty:
            # Calculate the target dates for all 'days_ago' at once
            target_dates = start_date_dt - np.array(days_ago_list) * timedelta(days=1)

            # Find the closest date indices for all target dates
            closest_date_indices = (df_pruned['Date'] - target_dates[:, np.newaxis]).abs().idxmin()

            # Get the prices and dates for the closest date indices
            prices = df_pruned.loc[closest_date_indices, 'Adj Close'].tolist()
            dates = df_pruned.loc[closest_date_indices, 'Date'].tolist()

            # Generate the data index
            data_index = ['{}d_ago_price'.format(days_ago) for days_ago in days_ago_list]

            # Convert the list of prices into a numpy array
            analysis_array = np.array(prices)

            return analysis_array, dates, data_index
        else:
            return None, None, None
    
    def create_analysis_array_symbol(self, df, start_date_dt, days_ago_list):
        # Filter the dataframe to include only the data needed
        df_pruned = df.iloc[-days_ago_list[-1]:] if len(df) >= days_ago_list[-1] else df

        # Ensure we have data to process
        if not df_pruned.empty:
            # Define the days for which we want to get the prices
            # Initialize an empty list to hold the prices for the specified days ago
            prices = []
            dates = []
            data_index = []
            
            for days_ago in days_ago_list:
                # Calculate the target date for each 'days_ago'
                target_date = start_date_dt - timedelta(days=days_ago)
                
                # Find the row in df_last_380 that has the closest date to the target date
                closest_date_index = (df_pruned['Date'] - target_date).abs().idxmin()
                
                # Get the price for the closest date
                price_on_closest_date = df_pruned.loc[closest_date_index, 'Adj Close']
                prices.append(price_on_closest_date)
                date_on_closest_date = df_pruned.loc[closest_date_index, 'Date']
                dates.append(date_on_closest_date)
                data_index.append('{}d_ago_price'.format(days_ago))
                        
            # Convert the list of prices into a numpy array
            analysis_array = np.array(prices)

            return analysis_array, dates, data_index
        else:
            return None, None, None
        
    def get_analysis_array(self, symbols, start_date_dt, historical_data_interval, rebalance_frequency):
        symbols_array = []  # To hold symbols
        data_array = []  # To hold analysis arrays from each symbol
        dates_final = {}
        update_data_index_final = True
        days_ago_list = [180, 270, 365]
        data_index_final = []
        
        for count, symbol in enumerate(symbols):
            # Call your existing function for each symbol
            # try:
                analysis_array, dates, data_index = self.create_analysis_array_symbol(historical_data_interval[symbol], start_date_dt, days_ago_list)
                # Get the price difference at each period.
                analysis_np = np.array(analysis_array)
                if analysis_np.size > 2:
                    if update_data_index_final:
                        data_index_final = data_index
                        for days_ago in days_ago_list[:-1]:
                            data_index_final.append('{}d_pct_change'.format(days_ago))
                        for days_ago in days_ago_list[:-1]:
                            data_index_final.append('{}d_pct_growth'.format(days_ago))
                        update_data_index_final = False

                    # Append symbol to the index array
                    symbols_array.append(symbol)
                    
                    # Append analysis array to the data array
                    data_array.append(analysis_np)
                    
                    # Append dates to the dates dictionary
                    dates_final[symbol] = dates
                else:
                    pass
                    
            # except Exception as e:
            #     print(f'Error processing {symbol}: Exception: {Exception}, e: {e}')
        return symbols_array, np.array(data_array), data_index_final, dates_final
    
    def get_long_short_symbols(self, long_count, short_count, symbols_array, data_array, data_index):
        #### Get the Stocks to Short
        # Use lowest_10_indexes to filter other_array
        short_symbols = np.random.choice(symbols_array, short_count)

        #### Get the Stocks for LONG
        # Use lowest_10_indexes to filter other_array
        long_symbols = np.random.choice(symbols_array, long_count)

        return long_symbols, short_symbols, symbols_array, data_array, data_index

class Strategy(BaseStrategy):
    pass

if __name__ == '__main__':
    bs = Strategy()
    bs.logger.debug('STRATEGY object created.')
    