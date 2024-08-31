# Imports
from vault.base_strategy import BaseStrategy
import numpy as np
from datetime import timedelta
import pandas as pd

class Strategy(BaseStrategy):
    def __init__(self):
        self.strategy_name = 'smooth_operator_v1'
    
    def get_name(self):
        return self.strategy_name
    
    def create_analysis_array_symbol(self, df, start_date_dt, days_ago_list):
        # Filter the dataframe to include only the data needed
        df_pruned = df.iloc[-days_ago_list[-1]:] if len(df) >= days_ago_list[-1] else df
        # df_pruned = pd.DataFrame(df_pruned)
        # Creating a trading volume filter
        df_pruned['traded_value'] = df_pruned['Adj Close'] * df_pruned['Volume']
        df_pruned['30d_trading_vol_USD'] = df_pruned['traded_value'].rolling(30).sum()

        # Ensure we have data to process
        if not df_pruned.empty:
            # Define the days for which we want to get the prices
            # Initialize an empty list to hold the prices for the specified days ago
            prices = []
            dates = []
            data_index = []
            trading_vol = []
            
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
            trading_vol.append(int(df_pruned.iloc[-1]['30d_trading_vol_USD']))
            
            analysis_array = np.array(prices)
            trading_vol_array = np.array(trading_vol)

            return analysis_array, dates, data_index, trading_vol_array
        else:
            return None, None, None, None
        
    def get_analysis_array(self, symbols, start_date_dt, historical_data_interval, rebalance_frequency):
        symbols_array = []  # To hold symbols
        data_array = []  # To hold analysis arrays from each symbol
        dates_final = {}
        update_data_index_final = True
        days_ago_list = list(range(1, 365, rebalance_frequency))
        
        for count, symbol in enumerate(symbols):
            # Call your existing function for each symbol
            if symbol in historical_data_interval:
                # try:
                    analysis_array, dates, data_index, trading_vol_array = self.create_analysis_array_symbol(historical_data_interval[symbol], start_date_dt, days_ago_list)
                    # Get the price difference at each period.
                    analysis_np = np.array(analysis_array)
                    if analysis_np.size > 2:
                        percentage_differences = ((analysis_np[1:]-analysis_np[:-1]) * -1)/analysis_np[:-1]
                        # Get the price change from now to each period.
                        differences_from_first = (analysis_np[0]-analysis_np[1:])/analysis_np[1:]
                        # Concatenate the original array with the two new arrays
                        analysis_np = np.concatenate((analysis_np, percentage_differences))
                        analysis_np = np.concatenate((analysis_np, differences_from_first))
                        analysis_np = np.concatenate((analysis_np, trading_vol_array), axis=0)
                        
                        if update_data_index_final:
                            data_index_final = data_index
                            for days_ago in days_ago_list[:-1]:
                                data_index_final.append('{}d_pct_change'.format(days_ago))
                            for days_ago in days_ago_list[:-1]:
                                data_index_final.append('{}d_pct_growth'.format(days_ago))
                            data_index_final.append('30d_trading_vol_USD'.format(days_ago))
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
    
    import numpy as np
    def calculate_u_curve_weights(self, points_count, mode='normal'):
        x = np.linspace(-5, 5, points_count)
        
        # Define a quadratic function that is high at the edges and lower in the middle
        arr = ((-0.5 * x**2) + 4) * -1
        # Ensure all values are positive
        if np.min(arr) < 0:
            arr = arr + abs(np.min(arr)-4)

        # Normalize the array so the sum is 100
        arr = (arr / np.sum(arr)) * 100
        
        # if mode.lower() != 'fast':
        #     # Create an array of 10 numbers from 10 to 0
        #     mult = np.linspace(10, 7, 10)
        #     arr = mult * arr
        #     arr = (arr / np.sum(arr)) * 100
        
        # if mode.lower() != 'slow':
        #     # Create an array of 10 numbers from 10 to 0
        #     mult = np.linspace(7, 10, 10)
        #     arr = mult * arr
        #     arr = (arr / np.sum(arr)) * 100
        
        return arr
    
    def get_data_index_counts(self, data_index):
        prices_count = 0
        pct_change_count = 0
        pct_growth_count = 0

        for x in data_index:
            if 'price' in x:
                prices_count += 1
            elif 'pct_change' in x:
                pct_change_count += 1
            elif 'pct_growth' in x:
                pct_growth_count += 1
                
        return prices_count, pct_change_count, pct_growth_count
    
    def get_long_short_symbols(self, long_count, short_count, symbols_array, data_array, data_index):
        # On the outset, lets just remove all the assets that are not in the top 500 assets
        # Create a rank for the 30d_trading_vol_USD column
        sorted_indices = np.argsort(data_array[:, data_index.index('30d_trading_vol_USD')])
        print({'sorted_indices':sorted_indices})
        raise AssertionError('MS')        
        
        prices_count, pct_change_count, pct_growth_count = self.get_data_index_counts(data_index)
        change_weights = self.calculate_u_curve_weights(pct_change_count)
        growth_weights = self.calculate_u_curve_weights(pct_growth_count)
        
        weights = np.concatenate((change_weights, growth_weights))

        # Assuming 'array' is the complete NumPy array you provided
        # Extracting columns 14 onwards for normalization
        columns_to_normalize = data_array[:, prices_count:]

        # Normalizing these columns
        normalized_columns = (columns_to_normalize - columns_to_normalize.min(axis=0)) / (columns_to_normalize.max(axis=0) - columns_to_normalize.min(axis=0))

        # Multiplying normalized columns by weights and summing up for each row
        weighted_sums = np.dot(normalized_columns, weights)

        # Concatenating normalized columns and weighted sums back to the original array
        normalized_columns = np.concatenate((normalized_columns, weighted_sums.reshape(-1, 1)), axis=1)

        # Sort the array based on the last column (column index -1)
        sorted_indices = np.argsort(normalized_columns[:, -1])

        # Create a rank column by assigning rank values to the sorted indices
        rank_column = np.empty_like(sorted_indices, dtype=int)
        rank_column[sorted_indices] = np.arange(len(normalized_columns)) + 1

        # Add the rank column as the last column to the original array
        data_array = np.column_stack((normalized_columns, rank_column))

        # Sort the array based on the last column (column index -1)
        sorted_indices = np.argsort(data_array[:, -1])
        
        #### Get the Stocks to Short
        # Get the indexes of the lowest 10 values
        short_10_indexes = np.array(sorted_indices[:short_count], dtype=int)

        # Use lowest_10_indexes to filter other_array
        short_symbols = np.array(symbols_array)[short_10_indexes]

        #### Get the Stocks for LONG
        # Get the indexes of the highest 10 values
        long_10_indexes = np.array(sorted_indices[-long_count:], dtype=int)

        # Use lowest_10_indexes to filter other_array
        long_symbols = np.array(symbols_array)[long_10_indexes]

        return long_symbols, short_symbols, symbols_array, data_array, data_index