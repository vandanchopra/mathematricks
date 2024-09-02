from tqdm import tqdm
import time
from utils import create_logger
import logging
from ..brokers import Brokers

class DataFetcher:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.logger = create_logger(log_level=logging.DEBUG, logger_name='datafetcher')
        self.broker = Brokers()
    
    def fetch_updated_data_single_asset(self, symbol):
        # update the data from the datasource
        data_sources = self.config_dict['data_update_inputs']['data_sources']
        for data_source in data_sources:
            if data_source == 'yahoo':
                self.broker.yahoo.update_data(symbol, start, end)
                
            elif data_source == 'ibkr':
                pass
        
        return asset_data
        
    def fetch_updated_data(self, market_data_df):
        list_of_symbols = self.config_dict['list_of_symbols']
        for symbol in tqdm(list_of_symbols, desc='Updating data'):
            asset_data = self.fetch_updated_data_single_asset(symbol)
            # update the data to the db
            # update the data to dataframe
        return data
    '''
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
        
    
    
    
    