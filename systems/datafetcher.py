import time
from systems.utils import create_logger
import logging
from brokers.brokers import Brokers
import sys
import pandas as pd

class DataFetcher:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.logger = create_logger(log_level='DEBUG', logger_name='datafetcher', print_to_console=True)
        self.broker = Brokers()
    
    def fetch_updated_price_data(self, market_data_df, start_date=None, end_date=None, lookback=None, update_data=True):
        list_of_symbols = self.config_dict['datafeeder_config']['list_of_symbols']
        interval_inputs = self.config_dict['datafeeder_config']['data_inputs']
        # self.logger.debug({'list_of_symbols': list_of_symbols})
        data_sources = self.config_dict['data_update_inputs']['data_sources']
        for data_source in data_sources:
            if data_source == 'yahoo':
                market_data_df = self.broker.sim.data.update_price_data(list_of_symbols,interval_inputs=interval_inputs,back_test_start_date=start_date,back_test_end_date=end_date, lookback=lookback, update_data=update_data)
                
            elif data_source == 'ibkr':
                market_data_df = self.broker.ib.data.update_price_data(list_of_symbols,interval_inputs=interval_inputs,back_test_start_date=start_date,back_test_end_date=end_date, lookback=lookback) 

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