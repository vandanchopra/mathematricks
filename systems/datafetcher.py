import time
from systems.utils import create_logger
import logging
from brokers.brokers import Brokers
import sys
import pandas as pd


class DataFetcher:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        log_level = logging.DEBUG if self.config_dict['log_level'] == 'DEBUG' else logging.INFO
        self.logger = create_logger(log_level=log_level, logger_name='datafetcher', print_to_console=True)
        self.broker = Brokers()
    
    # def fetch_updated_price_data_single_asset_old(self, symbol):
    #     # update the data from the datasource
    #     data_sources = self.config_dict['data_update_inputs']['data_sources']
    #     for data_source in data_sources:
    #         if data_source == 'yahoo':
    #             asset_data_df = self.broker.yahoo.update_price_data_single_asset(symbol)
                
    #         elif data_source == 'ibkr':
    #             raise NotImplementedError('IBKR data source is not implemented yet.')
    #     return asset_data_df
    
    # def fetch_updated_price_data_old(self, market_data_df):
    #     list_of_symbols = self.config_dict['list_of_symbols']
        
    #     # Initialize an empty list to store DataFrames
    #     data_frames = []
            
    #     for symbol in tqdm(list_of_symbols, desc='Updating data: '):
    #         asset_data_df = self.fetch_updated_price_data_single_asset(symbol)
    #         # add the data to the dataframe
    #         if asset_data_df is not None:
    #             data_frames.append(asset_data_df)
    #     dataframes = 
    #     # Combine all DataFrames into a single DataFrame
    #     market_data_df = pd.concat(data_frames)

    #     # Set multi-index
    #     market_data_df.set_index(['symbol', 'date'], inplace=True)

    #     # Sort the index
    #     market_data_df.sort_index(inplace=True)   
            
    #     return market_data_df
    
    def fetch_updated_price_data(self, market_data_df):
        list_of_symbols = self.config_dict['list_of_symbols']
        self.logger.debug({'list_of_symbols': list_of_symbols})
        data_sources = self.config_dict['data_update_inputs']['data_sources']
        for data_source in data_sources:
            if data_source == 'yahoo':
                market_data_df = self.broker.yahoo.update_price_data(list_of_symbols)
                
            elif data_source == 'ibkr':
                raise NotImplementedError('IBKR data source is not implemented yet.')
        return market_data_df
        
    '''
    # DUMMY CODE FOR IBKR DATA DOWNLOAD
    ticker = "AMZN"
    contract = Stock(ticker, 'SMART', 'USD')

    trader.ib.reqContractDetails(contract)

    bars = trader.ib.reqHistoricalData(
        contract, endDateTime='', durationStr='1 D',
        barSizeSetting='1 min', whatToShow='TRADES', useRTH=True,
        formatDate=1, keepUpToDate=True)

    df = pd.DataFrame(bars)
    # print(df.head())
    # print ()
    for item in range(df.shape[0]):
        print ("The closing price was " + str(df['close'].iloc[item]) + " as of " + str(df['date'].iloc[item]))
    '''
        
if __name__ == '__main__':
    from config import config_dict
    datafetcher = DataFetcher(config_dict=config_dict)
    asset_data = datafetcher.fetch_updated_price_data_single_asset('AAPL')