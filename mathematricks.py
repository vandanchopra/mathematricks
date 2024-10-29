import os, json, time, logging, sys, pickle, warnings
from matplotlib.pylab import f
import pandas as pd
from config import config_dict
from systems.datafetcher import DataFetcher
from systems.datafeeder import DataFeeder
from systems.vault import Vault
from systems.rms import RMS
from systems.oms import OMS
from systems.performance_reporter import PerformanceReporter
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
        self.market_data_df_root = pd.DataFrame()
        self.oms = OMS(self.config_dict)
        # Update the config_dict with the latest values from Vault
        self.config_dict = self.oms.config_dict
        self.vault = Vault(self.config_dict)
        # Update the config_dict with the latest values from Vault
        self.config_dict = self.vault.config_dict
        self.rms = RMS(self.config_dict)
        self.datafeeder = DataFeeder(self.config_dict)
        self.datafetcher = DataFetcher(self.config_dict)
        self.live_bool = False
        self.reporter = PerformanceReporter()
    
    def are_we_live(self, run_mode, system_timestamp, start_date):
        # convert system_timestamp and start_date to date only
        system_timestamp = system_timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

        if run_mode in [1,2] and system_timestamp < start_date:
            live_bool = False
        elif run_mode in [1,2] and system_timestamp >= start_date:
            live_bool = True
        else:
            live_bool = False

        return live_bool
    
    def sync_orders_on_live(self, new_orders):
        # self.logger.debug({'live_bool':live_bool, 'run_mode':run_mode, 'system_timestamp':system_timestamp, 'start_date':start_date, 'prev_live_bool':prev_live_bool})
        if prev_live_bool == False and live_bool == True:
            for count, multi_leg_order in enumerate(self.oms.open_orders):
                for order in multi_leg_order:
                    self.logger.info(f"OPEN Order {count+1}: Symbol: {order['symbol']}, orderDirection: {order['orderDirection']}, orderType: {order['orderType']}, orderQuantity: {order['orderQuantity']}, Status: {order['status']}, Broker: {order['broker']}")
            
            # for count, order in enumerate(self.oms.open_orders):
            #     self.logger.info(f"OPEN Order {count+1}: Symbol: {order[0]['symbol']}, orderDirection: {order[0]['orderDirection']}, orderType: {order[0]['orderType']}, orderQuantity: {order[0]['orderQuantity']}, Status: {order[0]['status']}, Strategy: {order[0]['strategy_name']}, Broker: {order[0]['broker']}")
            
            # print a series of dashes '-' only as wide as the terminal
            print('*'*os.get_terminal_size().columns)
            print('*'*os.get_terminal_size().columns)
            print('*'*os.get_terminal_size().columns)
            print('*'*os.get_terminal_size().columns)
            
            # input('System wants to go live. Press Enter to continue...')
            self.logger.info('System is now live. Syncing OMS with Live Broker.')
            new_orders_from_sync = self.oms.sync_open_orders('oms-to-broker', market_data_df, system_timestamp, brokers=['IBKR'])
            
            # self.logger.debug({'new_orders_from_sync':new_orders_from_sync})
            new_orders.extend(new_orders_from_sync)
            for count, order in enumerate(new_orders):
                price = order[0]['entryPrice'] if order[0]['orderType'].lower() == 'market' else order[0]['exitPrice']
                self.logger.info(f"SYNC New Order {count+1}: Symbol: {order[0]['symbol']}, orderDirection: {order[0]['orderDirection']}, orderType: {order[0]['orderType']}, orderQuantity: {order[0]['orderQuantity']}, Price: {price}, Strategy: {order[0]['strategy_name']}, Broker: {order[0]['broker']}")
            raise AssertionError('MANUALLY STOPPING HERE')
        return new_orders

    def run(self):
        run_mode = config_dict['run_mode']
        
        '''Set the start_date and end_date based on run_mode'''
        if run_mode in [1,2]: # live trading - real money
            # US Eastern Time Zone
            start_date = datetime.datetime.now(pytz.timezone('US/Eastern'))
            start_date = self.datafeeder.previous_market_close(start_date)
            end_date = None
        elif run_mode == 3: # backtesting
            assert 'start_time' in self.config_dict['backtest_inputs'] and 'end_time' in self.config_dict['backtest_inputs'], 'start_time and end_time must be provided in backtest_inputs if run_mode is 3'
            start_date = self.config_dict['backtest_inputs']['start_time']
            end_date = self.config_dict['backtest_inputs']['end_time']
        
        '''Start Running the System'''
        if run_mode in [1,2,3]:
            while True:
                try:
                    next_rows = self.datafeeder.next(market_data_df_root=self.market_data_df_root, run_mode=run_mode, sleep_time=self.sleep_time, start_date=start_date, end_date=end_date)
                    if next_rows is not None:
                        self.system_timestamp = next_rows.index.get_level_values(1)[-1]
                        self.market_data_df_root = pd.concat([self.market_data_df_root, next_rows], axis=0)
                        self.market_data_df_root = self.market_data_df_root[~self.market_data_df_root.index.duplicated(keep='last')]
                        
                        # SYSTEM UPDATE TO CONSOLE
                        # if self.system_timestamp > config_dict['backtest_inputs']['start_time']:
                        for interval, next_datetime in next_rows.index:
                            print('-'*os.get_terminal_size().columns)
                            self.logger.debug(f"Interval: {interval}, Datetime: {next_datetime}, System Timestamp: {self.system_timestamp}, Live Bool: {self.live_bool}")
                        
                        broker = 'IBKR'.lower() if self.live_bool else 'SIM'.lower()
                        base_account_number = list(self.oms.margin_available[broker].keys())[0]
                        trading_currency = self.config_dict['trading_currency']
                        margin_keys_of_interest = ['total_buying_power', 'buying_power_available', 'buying_power_used']
                        log_msg = ''
                        for key in margin_keys_of_interest:
                            current_value = self.oms.margin_available[broker][base_account_number]['combined'][trading_currency][key]
                            log_msg += f"{key}: {current_value} | "
                            # self.logger.debug({f"Margin Available: {self.oms.margin_available}"})
                            if round(current_value, 1) < 0:
                                raise AssertionError(f"Negative Margin Available: {log_msg}")
                        self.logger.debug(log_msg)     
                                
                        #     self.logger.debug(next_rows)
                        #     # self.logger.debug({'self.market_data_df':self.market_data_df.shape})
                        #     time.sleep(1)
                        
                        # # Generate Signals from the Strategies (signals)
                        new_signals = self.vault.generate_signals(next_rows, self.market_data_df_root, self.system_timestamp)
                        # PRINT THE SIGNALS GENERATED IF NEEDED
                        if len(new_signals['signals']) > 0 or len(new_signals['ideal_portfolios']) > 0:
                            # self.logger.debug({'Margins':self.oms.margin_available})
                            # input('Press Enter to continue...')
                            pass
                        
                        # Check if we're going live
                        prev_live_bool = self.live_bool
                        self.live_bool = self.are_we_live(run_mode, self.system_timestamp, start_date)
                        
                        # # Convert signals to orders
                        new_orders = self.rms.convert_signals_to_orders(new_signals, self.oms.margin_available, self.oms.portfolio, self.oms.open_orders, self.system_timestamp, self.live_bool)
                        
                        # If the system is going live, sync the orders with the broker
                        if prev_live_bool == False and self.live_bool == True:
                            new_orders = self.sync_orders_on_live(new_orders)
                            
                        # # # Execute orders on the market with the OMS
                        # if self.live_bool:
                            # self.logger.debug({'open_orders':self.oms.open_orders})
                        self.oms.execute_orders(new_orders, self.system_timestamp, self.market_data_df_root, live_bool=self.live_bool)
                        
                        # if len(new_signals['signals']) > 0 or len(new_signals['ideal_portfolios']) > 0:
                        #     symbols = [signal['symbol'] for signal in new_signals['signals']]
                        #     self.logger.debug(f'{self.system_timestamp} | Signals generated: {len(new_signals["signals"])} | Symbols: {symbols}')
                        #     self.logger.debug({'new_orders':self.new_orders})
                        #     raise AssertionError('MANUALLY STOPPING HERE')
                        
                    else:
                        self.logger.info('Backtest completed.')
                        from pprint import pprint
                        self.logger.info(pprint(self.oms.margin_available))
                        self.reporter.calculate_backtest_performance_metrics(self.config_dict, self.oms.open_orders, self.oms.closed_orders, self.market_data_df_root)
                        self.reporter.generate_report()
                        if self.config_dict['backtest_inputs']['save_backtest_results']:
                            self.test_folder_path, self.test_name = self.reporter.save_backtest(self.config_dict, self.oms.open_orders, self.oms.closed_orders)
                            self.logger.info(f'Backtest results saved at: {self.test_folder_path}')
                            self.logger.info(f'Backtest Name: {self.test_name}')
                        break
                
                except KeyboardInterrupt:
                    # self.logger.debug({'self.market_data_df':self.market_data_df})
                    self.logger.debug('Exiting...')
                    break
        elif run_mode == 4:
            self.logger.debug({'BEFORE: market_data_df': self.market_data_df_root.shape})
            stock_symbols_path = '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/db/stock_symbols.json'
            with open(stock_symbols_path) as file:
                list_of_symbols = json.load(file)
            self.config_dict['datafeeder_config']['list_of_symbols'] = list_of_symbols[:10] #['AAPL', 'MSFT'] # list_of_symbols
            self.market_data_df_root = self.datafetcher.fetch_updated_price_data(self.market_data_df_root)
            self.logger.debug({'AFTER: market_data_df': self.market_data_df_root.shape})
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