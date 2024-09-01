from tqdm import tqdm
import time

class DataFetcher:
    def __init__(self, config_dict):
        self.config_dict = config_dict
    
    def fetch_updated_data_single_asset(symbol):
        # update the data from the datasource
        
        return asset_data
        
    def fetch_updated_data(self, list_of_symbols, data_df):
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
        
    
    
    
    