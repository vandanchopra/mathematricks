# python: Create a logger for my system

import os, time, sys, logging, json, hashlib, datetime, pytz
import pandas as pd

def create_logger(log_level, logger_name='mathematricks', print_to_console=True):
    logger = logging.getLogger(logger_name)
    if not logger.hasHandlers():
        # Create formatter
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d')
        
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

def generate_hash_id(input_dict, system_timestamp):
    json_str = json.dumps(input_dict, default=str) + str(system_timestamp)
    json_bytes = json_str.encode('utf-8')
    order_id = hashlib.sha256(json_bytes).hexdigest()
    return order_id

def load_symbols_universe_df():
    '''URL TO DOWNLOAD THE STOCK SYMBOLS LIST: https://www.nasdaq.com/market-activity/stocks/screener'''
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
    
    return symbols_universe_df

project_path = '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/'
               
class SystemTemplates:
    pass

if __name__ == '__main__':
    logger = create_logger(log_level=logging.DEBUG, logger_name='mathematricks2', print_to_console=True)
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')