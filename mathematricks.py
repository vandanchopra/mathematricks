import os, json, time, logging, sys, pickle, warnings
import pandas as pd
from config import config_dict
from systems.datafetcher import DataFetcher
from systems.datafeeder import DataFeeder
from systems.vault import Vault
from systems.rms import RMS
from systems.oms import OMS
from systems.utils import create_logger, sleeper
import datetime, pytz
warnings.filterwarnings("ignore")

'''
write the software for AAPL, MSFT only.
'''

class Mathematricks:
    def __init__(self, config_dict):
        self.logger = create_logger(log_level='DEBUG', logger_name='Mathematricks', print_to_console=True)
        self.config_dict = config_dict
        self.sleep_time = self.config_dict['sleep_time']
        self.market_data_df = pd.DataFrame()
        self.oms = OMS(self.config_dict)
        # Update the config_dict with the latest values from Vault
        self.config_dict = self.oms.config_dict
        self.vault = Vault(self.config_dict)
        # Update the config_dict with the latest values from Vault
        self.config_dict = self.vault.config_dict
        self.rms = RMS(self.config_dict)
        self.datafeeder = DataFeeder(self.config_dict)
        self.datafetcher = DataFetcher(self.config_dict)
    
    def run(self):
        run_mode = config_dict['run_mode']
        if run_mode in [1,2,3]:
            while True:
                try:
                    if run_mode in [1,2]: # live trading - real money
                        start_date = datetime.datetime.now(datetime.timezone.utc).astimezone(pytz.timezone('US/Eastern'))  # Convert to Eastern Time
                        end_date = None
                        run_mode = 'LIVE'
                    elif run_mode == 3: # backtesting
                        assert 'start_time' in self.config_dict['backtest_inputs'] and 'end_time' in self.config_dict['backtest_inputs'], 'start_time and end_time must be provided in backtest_inputs if run_mode is 3'
                        start_date = self.config_dict['backtest_inputs']['start_time']
                        end_date = self.config_dict['backtest_inputs']['end_time']
                        run_mode = 'BT'
                        update_data = self.config_dict['backtest_inputs']['update_data']
                        
                    next_rows = self.datafeeder.next(market_data_df=self.market_data_df, run_mode=run_mode, sleep_time=self.sleep_time, start_date=start_date, end_date=end_date, update_data=update_data)
                    if next_rows is not None:
                        self.system_timestamp = next_rows.index.get_level_values(1)[-1]
                        self.market_data_df = pd.concat([self.market_data_df, next_rows], axis=0)
                        self.market_data_df = self.market_data_df[~self.market_data_df.index.duplicated(keep='last')]
                        # for interval, next_datetime in next_rows.index:
                        #     # self.logger.debug(f"Interval: {interval}, Datetime: {next_datetime}, system_timestamp: {self.system_timestamp}")
                        #     self.logger.debug(next_rows)
                        #     # self.logger.debug({'self.market_data_df':self.market_data_df.shape})
                        #     time.sleep(1)
                        
                        # execute_signals(signals)
                        new_signals = self.vault.generate_signals(self.market_data_df, self.system_timestamp)
                        # ## PRINT THE SIGNALS GENERATED IF NEEDED
                        # if len(new_signals['signals']) > 0 or len(new_signals['ideal_portfolios']) > 0:
                        #     self.logger.debug({'new_signals':new_signals})
                        #     input('Press Enter to continue...')
                        
                        # Convert signals to orders
                        new_orders = self.rms.convert_signals_to_orders(new_signals)
                        # ## PRINT THE SIGNALS GENERATED IF NEEDED
                        # if len(new_orders) > 0:
                        #     self.logger.debug({'new_orders':new_orders})
                        #     input('Press Enter to continue...')
                            
                        # # Execute orders on the market with the OMS
                        self.oms.execute_orders(new_orders, self.system_timestamp, self.market_data_df)
                        
                        # if len(new_signals['signals']) > 0 or len(new_signals['ideal_portfolios']) > 0:
                        #     symbols = [signal['symbol'] for signal in new_signals['signals']]
                        #     self.logger.debug(f'{self.system_timestamp} | Signals generated: {len(new_signals["signals"])} | Symbols: {symbols}')
                        #     self.logger.debug({'new_orders':self.new_orders})
                        #     raise AssertionError('MANUALLY STOPPING HERE')
                        
                    else:
                        self.logger.info('Backtest completed.')
                        self.logger.debug('This is where the backtest report would be generated.')
                        self.logger.info(f'Final Profit: {self.oms.profit}')
                        self.oms.close_all_open_orders(self.market_data_df)
                        backtest_orders = {'open_orders': self.oms.open_orders, 'closed_orders': self.oms.closed_orders}
                        # save backtest_orders to a json file
                        backtest_folder_path = '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/junk/backtest_reports'
                        # Create folder if it doesn't exist
                        os.makedirs(backtest_folder_path, exist_ok=True)
                        
                        # Save the backtest orders to a pickle fil
                        backtest_orders_path = os.path.join(backtest_folder_path, f'backtest_orders_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
                        with open(backtest_orders_path, 'wb') as file:
                            pickle.dump(backtest_orders, file)
                        break
                    # time.sleep(0.25)
                
                except KeyboardInterrupt:
                    # self.logger.debug({'self.market_data_df':self.market_data_df})
                    self.logger.debug('Exiting...')
                    break
        elif run_mode == 4:
            self.logger.debug({'BEFORE: market_data_df': self.market_data_df.shape})
            stock_symbols_path = '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/db/stock_symbols.json'
            with open(stock_symbols_path) as file:
                list_of_symbols = json.load(file)
            self.config_dict['datafeeder_config']['list_of_symbols'] = ['AAPL', 'MSFT'] # list_of_symbols
            self.market_data_df = self.datafetcher.fetch_updated_price_data(self.market_data_df)
            self.logger.debug({'AFTER: market_data_df': self.market_data_df.shape})
            # self.logger.debug({'market_data_df': self.market_data_df})
            
        else:
            raise AssertionError('Invalid run_mode value: {}'.format(run_mode))
        
if __name__ == '__main__':
    logs_folder = '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/logs'
    # Remove all .log files from logs folder
    for file in os.listdir(logs_folder):
        if file.endswith('.log'):
            os.remove(os.path.join(logs_folder, file))
    
    Mathematricks(config_dict).run()