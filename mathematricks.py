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
from copy import deepcopy
# warnings.filterwarnings("ignore")

'''
write the software for AAPL, MSFT only.
'''

class Mathematricks:
    def __init__(self, config_dict):
        self.logger = create_logger(log_level='DEBUG', logger_name='Mathematricks', print_to_console=True)
        self.config_dict = config_dict
        self.sleep_time = self.config_dict['sleep_time']
        self.current_market_data_df = pd.DataFrame()
        self.oms = OMS(self.config_dict)
        # Update the config_dict with the latest values from Vault
        self.config_dict = self.oms.config_dict
        self.vault = Vault(self.config_dict)
        # Update the config_dict with the latest values from Vault
        self.config_dict = self.vault.config_dict
        self.rms = RMS(self.config_dict)
        self.datafeeder = DataFeeder(self.config_dict)
        self.live_bool = False
        self.reporter = PerformanceReporter()
        self.first_run = True
        self.system_timestamp = None
        self.market_open_bool = self.datafeeder.is_market_open(pd.Timestamp.now(tz='US/Eastern'))
    
    def are_we_live(self, run_mode, system_timestamp, start_date):
        # convert system_timestamp and start_date to date only
        # system_timestamp = system_timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        live_bool = False
        # self.logger.info({'run_mode':run_mode, 'system_timestamp':system_timestamp, 'start_date':start_date})
        # now_tz = datetime.datetime.now() - datetime.timedelta(days=5)
        system_timestamp = system_timestamp + datetime.timedelta(minutes=1)
        start_date_minus_5_days = start_date - datetime.timedelta(days=5)
        if run_mode in [1,2] and system_timestamp >= start_date_minus_5_days:
            # previous_market_close = self.datafeeder.previous_market_close(system_timestamp)
            # self.logger.info({'run_mode':run_mode, 'system_timestamp+1':system_timestamp+datetime.timedelta(minutes=1), 'start_date':start_date})
            
            if '1m' not in self.datafeeder.config_dict['datafeeder_config']['data_inputs']: #and system_timestamp >= previous_market_close:
                # Add 1min timestamp to the datafeeder and datafetcher interval_inputs
                self.datafeeder.config_dict['datafeeder_config']['data_inputs']['1m'] = self.datafeeder.config_dict['datafeeder_config']['data_inputs']['1d']
                self.datafeeder.datafetcher.config_dict['datafeeder_config']['data_inputs']['1m'] = self.datafeeder.datafetcher.config_dict['datafeeder_config']['data_inputs']['1d']
                self.datafeeder.lookback_dict = self.datafeeder.create_lookback_dict()
            
            now_tz = pd.Timestamp.now(tz='US/Eastern')
            if self.market_open_bool:
                sleep_lookup = {"1m":60,"2m":120,"5m":300,"1d":86400}
                interval_inputs = self.config_dict['datafeeder_config']['data_inputs']
                min_granularity_seconds = min([sleep_lookup[interval] for interval in interval_inputs])
                last_timestamp_tz = now_tz - datetime.timedelta(seconds=min_granularity_seconds)
                last_timestamp_tz = last_timestamp_tz.replace(second=0, microsecond=0)
            else:
                last_timestamp_tz = self.datafeeder.previous_market_close(now_tz)
                last_timestamp_tz = last_timestamp_tz.astimezone(pytz.timezone('US/Eastern'))
            system_timestamp_tz = system_timestamp.astimezone(pytz.timezone('US/Eastern'))
            # self.logger.debug({'system_timestamp_tz':system_timestamp_tz, 'last_timestamp_tz':last_timestamp_tz})
            # if next_expected_timestamp <= min_granularity_seconds:
            if system_timestamp_tz >= last_timestamp_tz:
                live_bool = True
                msg = {'start_date':start_date, 'market_open_bool':self.market_open_bool, 'system_timestamp':system_timestamp, 'now_tz':now_tz, 'last_timestamp_tz':last_timestamp_tz, 'live_bool':live_bool}
                if not self.market_open_bool:
                    msg['prev_market_close'] = last_timestamp_tz
                self.logger.info(msg)
                sleeper(10, 'Going LIVE in: ')

        return live_bool
    
    def sync_orders_on_live(self, new_orders, market_data_df, system_timestamp):
        # self.logger.debug({'live_bool':live_bool, 'run_mode':run_mode, 'system_timestamp':system_timestamp, 'start_date':start_date, 'prev_live_bool':prev_live_bool})
        msg = "------------------------ System is going live. Syncing OMS with Live Broker "
        print(msg + '*'*(os.get_terminal_size().columns - len(msg)))
        
        total_orders_value = 0
        for count, multi_leg_order in enumerate(self.oms.open_orders):
            for order in multi_leg_order:
                price = order['entryPrice'] if order['orderType'].lower() == 'market' else order['exitPrice']
                order_value = price * order['orderQuantity'] if order['orderType'].lower() == 'market' else 0
                total_orders_value += order_value
                self.logger.info(f"OPEN Order {count+1}: Symbol: {order['symbol']}, orderDirection: {order['orderDirection']}, order_value: {order_value}, total_orders_value: {total_orders_value}, orderType: {order['orderType']}, orderQuantity: {order['orderQuantity']}, Status: {order['status']}, Broker: {order['broker']}")
        
        # for count, order in enumerate(self.oms.open_orders):
        #     self.logger.info(f"OPEN Order {count+1}: Symbol: {order[0]['symbol']}, orderDirection: {order[0]['orderDirection']}, orderType: {order[0]['orderType']}, orderQuantity: {order[0]['orderQuantity']}, Status: {order[0]['status']}, Strategy: {order[0]['strategy_name']}, Broker: {order[0]['broker']}")
        
        # print a series of dashes '-' only as wide as the terminal
        print('*'*os.get_terminal_size().columns)
        print('*'*os.get_terminal_size().columns)
        print('*'*os.get_terminal_size().columns)
        print('*'*os.get_terminal_size().columns)
        
        # input('System wants to go live. Press Enter to continue...')
        new_orders_from_sync = self.oms.sync_open_orders('oms-to-broker', market_data_df, system_timestamp, brokers=['IBKR'])
        
        # self.logger.debug({'new_orders_from_sync':new_orders_from_sync})
        new_orders.extend(new_orders_from_sync)
        total_orders_value = 0
        for count, order in enumerate(new_orders_from_sync):
            price = order[0]['entryPrice'] if order[0]['orderType'].lower() == 'market' else order[0]['exitPrice']
            order_value = price * order[0]['orderQuantity'] if order[0]['orderType'].lower() == 'market' else 0
            total_orders_value += order_value
            self.logger.info(f"SYNC New Order {count+1}: Symbol: {order[0]['symbol']}, orderDirection: {order[0]['orderDirection']}, orderType: {order[0]['orderType']}, orderQuantity: {order[0]['orderQuantity']}, Price: {price}, order_value:{order_value}, total_orders_value: {total_orders_value}, Strategy: {order[0]['strategy_name']}, Broker: {order[0]['broker']}")
        
        msg = '------------------------ System is now live. Syncing OMS with Live Broker.'
        print(msg + '*'*(os.get_terminal_size().columns - len(msg)))
    
        return new_orders

    def print_update_to_console(self, next_rows):
        '''PRINT 1: Print Details of the next Timestamp'''
        for interval, next_datetime in next_rows.index:
            next_datetime_EST = next_datetime.astimezone(pytz.timezone('US/Eastern'))
            system_timestamp_EST = self.system_timestamp.astimezone(pytz.timezone('US/Eastern'))
            self.logger.info(f"Interval: {interval}, Latest Data Datetime: {next_datetime_EST}, System Datetime: {system_timestamp_EST}, Live Bool: {self.live_bool}")
            # if interval == '1m':
                # sleeper(1, 'Giving you 1 second to read the message above')
        
        '''PRINT 2: Print Margin Available'''
        broker = 'IBKR'.lower() if self.live_bool else 'SIM'.lower()
        base_account_number = list(self.oms.margin_available[broker].keys())[0]
        trading_currency = self.config_dict['trading_currency']
        margin_keys_of_interest = ['total_buying_power', 'buying_power_available', 'buying_power_used']
        log_msg = f'{broker.upper()} :::: '
        for key in margin_keys_of_interest:
            current_value = self.oms.margin_available[broker][base_account_number]['combined'][trading_currency][key]
            log_msg += f"{key}: {round(current_value, 1)} | "
            # self.logger.debug({f"Margin Available: {self.oms.margin_available}"})
            if round(current_value, 1) < 0:
                raise AssertionError(f"Negative Margin Available: {log_msg}")
        self.logger.info(log_msg)
        
        '''PRINT 3: Print Unrealized PnL'''
        if self.live_bool:
            self.oms.unfilled_orders = self.oms.brokers.ib.execute.ib.reqAllOpenOrders()
        unrealized_pnl_abs_dict, unrealized_pnl_pct_dict = self.reporter.calculate_unrealized_pnl(self.oms.open_orders, self.oms.unfilled_orders)
        # Sort dictionary by values
        unrealized_pnl_abs_dict = dict(sorted(unrealized_pnl_abs_dict.items(), key=lambda item: item[1], reverse=True))
        # Sum of all the values of unrealized_pnl_abs_dict
        total_unrealized_pnl_abs = sum(unrealized_pnl_abs_dict.values())
        log_msg = f'Unrealized PnL Abs: TOTAL: ${round(total_unrealized_pnl_abs, 2)} | '
        for key in unrealized_pnl_abs_dict.keys():
            log_msg += f"{key}: {round(unrealized_pnl_abs_dict[key], 2)} | "
        self.logger.info(log_msg)
        
        log_msg = 'Unrealized PnL % : '
        unrealized_pnl_pct_dict = dict(sorted(unrealized_pnl_pct_dict.items(), key=lambda item: item[1], reverse=True))
        for key in unrealized_pnl_pct_dict.keys():
            log_msg += f"{key}: {round((unrealized_pnl_pct_dict[key] * 100), 2)}% | "
        self.logger.info(log_msg)
        
        print('-'*os.get_terminal_size().columns)
    
    def run(self):
        run_mode = config_dict['run_mode']
        
        '''Set the start_date and end_date based on run_mode'''
        if run_mode in [1,2]: # live trading - real money
            # US Eastern Time Zone
            start_date = datetime.datetime.now().astimezone(pytz.timezone('US/Eastern'))
            market_open_bool = self.datafeeder.is_market_open(start_date)
            # self.logger.warning('Manually setting market_open_bool to True')
            # sleeper(4, 'Giving your system 4 seconds to read the warning message above')
            prev_mkt_close = self.datafeeder.previous_market_close(start_date)
            # If market is closed, then take prev market close else take the current time (Eg. On Sunday, min(start_date, prev_mkt_close) on Monday, at 12 noon, max(start_date, prev_mkt_close)
            start_date = prev_mkt_close if not market_open_bool else start_date
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = None
            
        elif run_mode == 3: # backtesting
            assert 'start_time' in self.config_dict['backtest_inputs'] and 'end_time' in self.config_dict['backtest_inputs'], 'start_time and end_time must be provided in backtest_inputs if run_mode is 3'
            start_date = self.config_dict['backtest_inputs']['start_time']
            end_date = self.config_dict['backtest_inputs']['end_time']
        
        '''Start Running the System'''
        if run_mode in [1,2,3]:
            while True:
                try:
                    next_rows = self.datafeeder.next(system_timestamp=self.system_timestamp, run_mode=run_mode, sleep_time=self.sleep_time, start_date=start_date, end_date=end_date)
                    if next_rows is not None:
                        self.system_timestamp = next_rows.index.get_level_values(1)[-1]
                        self.current_market_data_df = pd.concat([self.current_market_data_df, next_rows], axis=0)
                        self.current_market_data_df = self.current_market_data_df[~self.current_market_data_df.index.duplicated(keep='last')]
                        
                        # Generate Signals from the Strategies (signals)
                        # new_signals, self.config_dict = self.vault.generate_signals(next_rows, self.current_market_data_df, self.system_timestamp)
                        # self.datafeeder.config_dict = self.rms.config_dict = self.vault.config_dict
                        
                        # Check if we're going live
                        prev_live_bool = self.live_bool
                        if prev_live_bool == False:
                            self.live_bool = self.are_we_live(run_mode, self.system_timestamp, start_date)
                        
                        # # Convert signals to orders
                        # new_orders = self.rms.convert_signals_to_orders(new_signals, self.oms.margin_available, self.oms.open_orders, self.system_timestamp, self.live_bool)
                        
                        # If the system is going live, sync the orders with the broker
                        # if prev_live_bool == False and self.live_bool == True:
                        # Get a list of orders that'll be sent to the LIVE broker, based on current open orders
                            # new_orders = self.sync_orders_on_live(new_orders, self.current_market_data_df, self.system_timestamp)
                            
                        # Execute orders on the market with the OMS
                        # self.oms.execute_orders(new_orders, self.system_timestamp, self.current_market_data_df, live_bool=self.live_bool)
                        
                        # Print update messages to console
                        self.print_update_to_console(next_rows)

                    else:
                        self.logger.info('Backtest completed.')
                        from pprint import pprint
                        # self.logger.info(pprint(self.config_dict))
                        self.logger.info(pprint(self.oms.margin_available))
                        self.reporter.calculate_backtest_performance_metrics(self.config_dict, self.oms.open_orders, self.oms.closed_orders, self.current_market_data_df, self.oms.unfilled_orders)
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
            self.update_all_historical_price_data()
            
        else:
            raise AssertionError('Invalid run_mode value: {}'.format(run_mode))

if __name__ == '__main__':
    logs_folder = '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/logs'
    # Remove all .log files from logs folder
    for file in os.listdir(logs_folder):
        if file.endswith('.log'):
            os.remove(os.path.join(logs_folder, file))
    Mathematricks(config_dict).run()