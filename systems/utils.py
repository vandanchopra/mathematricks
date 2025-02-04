# python: Create a logger for my system

import os, time, sys, logging, json, hashlib, datetime, pytz
import pandas as pd

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

def create_logger(log_level, logger_name='mathematricks', print_to_console=True):
    logger = logging.getLogger(logger_name)
    if not logger.hasHandlers():
        # Create formatter
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s- %(message)s - %(filename)s:%(lineno)d')
        
        # Ensure the logs directory exists
        os.makedirs('./logs', exist_ok=True)
        
        # Create file handler and set level to log_level
        logfile_path = f'./logs/{logger_name}.log'
        fh = logging.FileHandler(logfile_path, encoding='utf-8')
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Create console handler and set level to log_level
        if print_to_console:
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        # Set the logger level
        logger.setLevel(log_level)
    
    return logger

def sleeper(total_seconds, message="System Sleeping"):
    # Total time in seconds (e.g., 3 days)

    for remaining in range(total_seconds, 0, -1):
        days = remaining // (24 * 60 * 60)
        hours = (remaining % (24 * 60 * 60)) // (60 * 60)
        minutes = (remaining % (60 * 60)) // 60
        seconds = remaining % 60
        
        time_str = ""
        if days > 0:
            time_str += "{:2d} days,".format(days)
        if hours > 0:
            time_str += "{:2d} hours,".format(hours)
        if minutes > 0:
            time_str += "{:2d} minutes,".format(minutes)
        if seconds > 0:
            time_str += "{:2d} seconds".format(seconds)
        
        sys.stdout.write("\r")
        sys.stdout.write(f"{message}: " + time_str + " remaining.")
        sys.stdout.flush()
        time.sleep(1)  # Sleep for 1 second
    sys.stdout.write("\r" + " " * 100)  # Clear the line
    sys.stdout.write("\r")

def serialize_for_hash(obj):
    """Helper function to serialize objects for hashing"""
    if isinstance(obj, (pd.Timestamp, datetime.datetime)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {str(k): serialize_for_hash(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_hash(x) for x in obj]
    return str(obj)

def generate_hash_id(input_dict, system_timestamp):
    """Generate a hash ID for a dictionary, handling Timestamps and other complex types"""
    serialized_dict = serialize_for_hash(input_dict)
    json_str = json.dumps(serialized_dict) + serialize_for_hash(system_timestamp)
    json_bytes = json_str.encode('utf-8')
    order_id = hashlib.sha256(json_bytes).hexdigest()
    return order_id

def load_symbols_universe_df(broker_name):
    if broker_name in ['ibkr', 'yahoo']:
        '''URL TO DOWNLOAD THE STOCK SYMBOLS LIST: https://www.nasdaq.com/market-activity/stocks/screener'''
        ignored_symbols = ['VSTEW', 'RVPHW', 'SCLXW', 'CORZZ', 'TVGNW', 'ASTLW', 'SMXWW', 'ZOOZW', 'SHFSW', 'PCTTW', 'UKOMW', 'PRENW', 'AIRJW', 'CSLRW', 'HUBCW', 'ADSEW', 'PAMT', 'KITTW', 'ZEOWW', 'INVZW', 'RMCOW', 'BFRIW', 'NVNIW', 'SOUNW', 'NIOBW', 'COEPW', 'RVSNW', 'ZCARW', 'HUMAW', 'HTZWW', 'NCNCW', 'USGOW', 'DHCNL', 'RUMBW', 'SBCWW', 'ABPWW', 'PETWW', 'SQFTW', 'NVVEW', 'ABVEW', 'MSAIW', 'CEROW', 'ECXWW', 'BENFW', 'NNAVW', 'DTSTW', 'CORZW', 'ICUCW', 'FOXXW', 'FMSTW', 'BTCTW', 'ATNFW', 'GCMGW', 'LNZAW', 'LVROW', 'CRGOW', 'SWVLW', 'HYZNW', 'AENTW', 'OABIW', 'NIVFW', 'ADVWW', 'ESGLW', 'BLDEW', 'SAIHW', 'TCBPW', 'CRESW', 'LGHLW', 'BNZIW', 'EVLVW', 'LEXXW', 'ORGNW', 'SXTPW', 'SYTAW', 'TALKW']
        ignored_symbols += ['MAPSW', 'FATBW', 'CELUW', 'FBYDW', 'KTTAW', 'XBPEW', 'DFLIW', 'EDBLW', 'BRLSW', 'COCHW', 'MNTSW', 'CMPOW', 'PBMWW', 'CRMLW', 'GIPRW', 'HYMCW', 'NXGLW', 'KPLTW', 'FAASW', 'GOEVW', 'KLTOW', 'NXPLW', 'BCTXW', 'ARBEW', 'CINGW', 'VGASW', 'SFB', 'ARKOW', 'GSMGW', 'BFRGW', 'WLDSW', 'SRZNW', 'RGTIW', 'PAVMZ', 'OPTXW', 'PXSAW', 'ODVWZ', 'AISPW', 'AFRIW', 'LDTCW', 'CXAIW', 'NXLIW', 'STSSW', 'VEEAW', 'CURIW', 'UHGWW', 'KDLYW', 'BTMWW', 'SONDW', 'RVMDW', 'AILEW', 'LOTWW', 'LCFYW', 'SWAGW', 'APCXW', 'CDTTW', 'PITAW', 'SLDPW', 'HOVRW', 'TMTCR', 'NRSNW', 'OCSAW', 'FFIEW', 'WGSWW', 'CIFRW', 'LSBPW', 'TOIIW', 'HUBCZ', 'DHAIW', 'NMHIW', 'MOBXW', 'AVPTW']
        ignored_symbols += ['TNONW', 'MMVWW', 'ZAPPW', 'ACONW', 'JSPRW', 'SHOTW', 'MRNOW', 'IVDAW', 'DSYWW', 'NESRW', 'ONMDW', 'GRRRW', 'PIIIW', 'GGROW', 'CLNNW', 'HOFVW', 'PROCW', 'MTEKW', 'RCKTW', 'NIXXW', 'IMTXW', 'ALVOW', 'CMAXW', 'BIAFW', 'BCGWW', 'CDIOW', 'LUNRW', 'SDAWW', 'FUFUW', 'AUROW', 'AREBW', 'COOTW', 'XOSWW', 'LFLYW', 'GOVXW', 'ABLLW', 'SBFMW', 'NWTNW', 'BHST', 'SVMHW', 'LTRYW', 'REVBW', 'BEATW', 'MVSTW', 'CPTNW', 'CGBSW', 'RZLVW', 'TBLAW', 'MKDWW', 'BAERW', 'DAVEW', 'SLXNW', 'BZFDW', 'NRXPW', 'DBGIW', 'NCPLW', 'SVREW', 'CDROW', 'NEOVW', 'EUDAW', 'MDAIW', 'ANGHW', 'NKGNW', 'ILLRW', 'CAPTW', 'BDMDW', 'LSEAW', 'HOLOW', 'ADNWW', 'SABSW', 'KWESW', 'QSIAW', 'ENGNW', 'IZTC', 'GFAIW']
        ignored_symbols += ['MAPSW', 'FATBW', 'CELUW', 'FBYDW', 'KTTAW', 'XBPEW', 'DFLIW', 'EDBLW', 'BRLSW', 'COCHW', 'MNTSW', 'CMPOW', 'PBMWW', 'CRMLW', 'GIPRW', 'HYMCW', 'NXGLW', 'KPLTW', 'FAASW', 'GOEVW', 'KLTOW', 'NXPLW', 'BCTXW', 'ARBEW', 'CINGW', 'VGASW', 'SFB', 'ARKOW', 'GSMGW', 'BFRGW', 'WLDSW', 'SRZNW', 'RGTIW', 'PAVMZ', 'OPTXW', 'PXSAW', 'ODVWZ', 'AISPW', 'AFRIW', 'LDTCW', 'CXAIW', 'NXLIW', 'STSSW', 'VEEAW', 'CURIW', 'UHGWW', 'KDLYW', 'BTMWW', 'SONDW', 'RVMDW', 'AILEW', 'LOTWW', 'LCFYW', 'SWAGW', 'APCXW', 'CDTTW', 'PITAW', 'SLDPW', 'HOVRW', 'TMTCR', 'NRSNW', 'OCSAW', 'FFIEW', 'WGSWW', 'CIFRW', 'LSBPW', 'TOIIW', 'HUBCZ', 'DHAIW', 'NMHIW', 'MOBXW', 'AVPTW']
        
        def remove_old_files(stocksymbolslists_folder, days_old=30):    
            # Get file names from the folder
            file_names = [f for f in os.listdir(stocksymbolslists_folder) if f.endswith('.csv')]
            # Get the number portion of the file names
            file_numbers = [float(f.split('_')[-1].split('.')[0]) for f in file_names]
            # Now assume that these numbers are EPOCH timestamps in milliseconds and calculate the age of the timestamp in days
            from datetime import datetime, timezone, timedelta
            now = datetime.now(timezone.utc)
            file_numbers = [datetime.fromtimestamp(x/1000, timezone.utc) for x in file_numbers]
            file_numbers = [now - x for x in file_numbers]
            file_numbers = [x.days for x in file_numbers]
            # if any of those are more than 30 days old, then delete them from file_name
            file_names = [f for f, age in zip(file_names, file_numbers) if age < days_old]
            return file_names
        
        # get file names from the folder and load all the csv files and concatenate them and return a pandas dataframe
        stocksymbolslists_folder = '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/db/data/stocksymbolslists'
        
        # Get file names from the folder
        file_names = [f for f in os.listdir(stocksymbolslists_folder) if f.endswith('.csv')]
        # Remove all files that are more than 30 days old
        file_names = remove_old_files(stocksymbolslists_folder, days_old=30)
        # Load all the CSV files and concatenate them into a single DataFrame
        dfs = [pd.read_csv(os.path.join(stocksymbolslists_folder, file)) for file in file_names]
        symbols_universe_df = pd.concat(dfs, ignore_index=True)
        # Now sort the combined_df by the market cap column in descending order
        symbols_universe_df = symbols_universe_df.sort_values(by='Market Cap', ascending=False)
        # Drop all rows where Market Cap is NaN
        symbols_universe_df = symbols_universe_df.dropna(subset=['Market Cap'])
        # Drop all rows where Market Cap is 0
        symbols_universe_df = symbols_universe_df[symbols_universe_df['Market Cap'] != 0]
        # Drop all rows where Symbol is NaN
        symbols_universe_df = symbols_universe_df.dropna(subset=['Symbol'])
        
        # If there are spaces in the Symbol column, then remove them
        symbols_universe_df['Symbol'] = symbols_universe_df['Symbol'].str.replace(' ', '')
        
        # Remove all rows where Symbol is in the ignored_symbols list
        symbols_universe_df = symbols_universe_df[~symbols_universe_df['Symbol'].isin(ignored_symbols)]
    elif broker_name == 'kraken':
        symbols_universe_df = pd.read_csv(f'{project_path}db/data//stocksymbolslists/kraken_symbols_universe.csv')
        symbols_universe_df = symbols_universe_df.head(2)
    return symbols_universe_df

class MarketDataExtractor:
    def get_market_data_granularities(self, market_data_df):
        return list(market_data_df.index.get_level_values(0).unique())

    def get_market_data_df_columns(self, market_data_df, granularity):
        return list(market_data_df.loc[granularity].columns.get_level_values(0).unique())

    def get_market_data_df_symbols(self, market_data_df, granularity):
        return list(market_data_df.loc[granularity].columns.get_level_values(1).unique())

    def get_market_data_df_symbol_prices(self, market_data_df, granularity, symbol, column):
        if symbol not in market_data_df.loc[granularity].columns.get_level_values(1).unique():
            return None
        else:
            return market_data_df.loc[granularity].xs(symbol, level=1, axis=1)[column]

    def get_market_data_df_symbol_timestamps(self, market_data_df, granularity, symbol):
        return list(market_data_df.loc[granularity].xs(symbol, level=1, axis=1).index)
    
    def get_market_data_df_timestamps(self, market_data_df):
        return list(market_data_df.index.get_level_values(1).unique())
    
    def get_market_data_df_minimum_granularity(self, market_data_df):
        granularity_lookup_dict = {"1m":60,"2m":120,"5m":300,"1d":86400}
        available_granularities = self.get_market_data_granularities(market_data_df)
        min_granularity_val = min([granularity_lookup_dict[granularity] for granularity in available_granularities])
        min_granularity = list(granularity_lookup_dict.keys())[list(granularity_lookup_dict.values()).index(min_granularity_val)]
        
        return min_granularity

if __name__ == '__main__':
    logger = create_logger(log_level=logging.DEBUG, logger_name='mathematricks2', print_to_console=True)
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')