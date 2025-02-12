import os, time #, logging, sys, pickle, warnings, json
# from matplotlib.pylab import f
import pandas as pd
from config import config_dict
from systems.datafeeder import DataFeeder
from systems.vault import Vault
from systems.rms import RMS
from systems.oms import OMS
from systems.performance_reporter import PerformanceReporter
from systems.utils import create_logger, sleeper, MarketDataExtractor, project_path
import datetime, pytz
from copy import deepcopy
from systems.telegram import TelegramBot
from colorama import Fore, Style
# warnings.filterwarnings("ignore")

'''
write the software for AAPL, MSFT only.
'''

class Mathematricks:
    def __init__(self, config_dict):
        self.logger = create_logger(log_level='DEBUG', logger_name='Mathematricks', print_to_console=True)
        self.market_data_extractor = MarketDataExtractor()
        self.config_dict = config_dict
        self.sleep_time = self.config_dict['sleep_time']
        # Initialize with empty DataFrame but with correct structure
        self.current_market_data_df = pd.DataFrame(columns=pd.MultiIndex.from_product([[], []], names=['interval', 'datetime']))
        self.oms = OMS(self.config_dict)
        # Update the config_dict with the latest values from Vault
        self.config_dict = self.oms.config_dict
        self.vault = Vault(self.config_dict, self.market_data_extractor)
        # Update the config_dict with the latest values from Vault
        self.config_dict = self.vault.config_dict
        self.rms = RMS(self.config_dict, self.market_data_extractor)
        self.datafeeder = DataFeeder(self.config_dict)
        self.live_bool = False
        self.reporter = PerformanceReporter(self.market_data_extractor)
        self.system_timestamp = pd.Timestamp('1901-01-01 00:00:00+0000', tz='UTC')
        self.market_open_bool = self.datafeeder.is_market_open(pd.Timestamp.now(tz='US/Eastern'))
        self.next_expected_timestamp_on_now = self.datafeeder.get_next_expected_timestamp(pd.Timestamp.now(tz='US/Eastern'))
        self.time_to_load_minute_data = self.datafeeder.get_prev_market_open(pd.Timestamp.now(tz='US/Eastern')) - datetime.timedelta(days=4)
        self.prev_market_close = self.datafeeder.get_previous_market_close(pd.Timestamp.now(tz='US/Eastern'))
        self.telegram_bot = TelegramBot()
        self.first_telegram_msg_sent = False
        self.update_telegram = self.config_dict['update_telegram']
        self.stop_backtest_at_n_signals = 200
    
    def get_unrealized_pnl_dict(self):
        unrealized_pnl_dict = {}
        distance_from_SL_dict = {}
        
        # Get open signals from OMS
        open_signals = self.oms.get_open_signals()
        for signal in open_signals:
            if signal.status == 'closed':
                continue
            
            # Process each order in the signal
            for order in signal.orders:
                symbol = order.symbol
                # Find entry orders that have been filled
                if order.entryOrderBool and order.status == 'closed' and order.filled_price:
                    # Check if we already processed this symbol for this signal
                    if signal.signal_id in unrealized_pnl_dict and symbol in unrealized_pnl_dict[signal.signal_id]:
                        continue
                    
                    # Get entry details
                    entry_price = order.filled_price
                    position_size = order.orderQuantity * (1 if order.orderDirection == "BUY" else -1)
                    
                    # Find matching exit order
                    exit_order = None
                    for o in signal.orders:
                        if (o.symbol == symbol and not o.entryOrderBool and 
                            o.status == 'open' and o.order_type == 'STOPLOSS'):
                            exit_order = o
                            break
                    
                    # Get current price - first try exit order's LTP
                    current_price = None
                    if exit_order and hasattr(exit_order, 'symbol_ltp') and exit_order.symbol_ltp:
                        latest_timestamp = max(exit_order.symbol_ltp.keys())
                        current_price = exit_order.symbol_ltp[latest_timestamp]
                    
                    # If no exit order LTP, try market data
                    if current_price is None and not self.current_market_data_df.empty:
                        for interval in ['1m', '1d']:
                            if interval in self.current_market_data_df.index.get_level_values(0):
                                try:
                                    current_price = self.current_market_data_df.loc[interval].iloc[-1][symbol]['Close']
                                    break
                                except (KeyError, IndexError):
                                    continue
                    
                    # Calculate unrealized PnL if we have all needed values
                    if current_price is not None and exit_order is not None:
                        # self.logger.info(f"Exit Order: Symbol: {symbol}, Order Type: {exit_order.order_type}, Status: {exit_order.status}, LTP: {current_price}")
                        unrealized_pnl = (current_price - entry_price) * abs(position_size)
                        unrealized_pnl_dict[symbol] = unrealized_pnl
                        if exit_order.orderDirection == 'SELL':
                            distance_from_SL_dict[symbol] = ((current_price - exit_order.price) / current_price) * 100
                        else:
                            distance_from_SL_dict[symbol] = ((exit_order.price - current_price) / current_price) * 100
                        
        return unrealized_pnl_dict, distance_from_SL_dict
    
    def print_updates_to_console(self):
        # Print margins from all brokers (only non-empty combined accounts)
        for broker in self.oms.margin_available:
            for account in self.oms.margin_available[broker]:
                margin_info = self.oms.margin_available[broker][account]
                if margin_info.get('combined', {}).get('USD', {}):  # Only print if margin info exists
                    combined_margin = margin_info['combined']['USD']
                    pct_used = (combined_margin['buying_power_used'] / combined_margin['total_buying_power'] * 100) if combined_margin['total_buying_power'] else 0
                    self.logger.info(f"{Fore.CYAN}{broker.upper()} Account {account}: Used=${combined_margin['buying_power_used']:,.2f}, Available=${combined_margin['buying_power_available']:,.2f}, Total=${combined_margin['total_buying_power']:,.2f}, Used%={pct_used:.2f}%{Style.RESET_ALL}")
        
        # # Print unrealized PnL for open signals
        unrealized_pnl_dict, distance_from_SL_dict = self.get_unrealized_pnl_dict()
        # sort unrealized_pnl_dict by values
        unrealized_pnl_dict = dict(sorted(unrealized_pnl_dict.items(), key=lambda item: item[1], reverse=True))
        unrealized_msg = f"Unrealized PnL: Total: ${sum(unrealized_pnl_dict.values()):,.2f} |"
        for symbol, pnl in unrealized_pnl_dict.items():
            unrealized_msg += f" {symbol}: ${pnl:,.2f} |"
        self.logger.info(unrealized_msg)
        
        distance_pct_from_SL_msg = f"Distance from Stop Loss (%): "
        for symbol in unrealized_pnl_dict:
            distance_pct_from_SL_msg += f" {symbol}: {distance_from_SL_dict[symbol]:.2f}% |"
        self.logger.info(distance_pct_from_SL_msg)
            
        # pair_totals = {}
        # for positions in unrealized_pnl_dict.values():
        #     if len(positions) == 2:  # If it's a pair trade (2 symbols)
        #         symbols = sorted(positions.keys())  # Sort to ensure consistent pair naming
        #         pair_name = f"{symbols[0]}_{symbols[1]}"
        #         pair_totals[pair_name] = sum(positions.values())
        # self.logger.info(f"Unrealized PnL: {pair_totals}")
    
    def are_we_live(self, run_mode, system_timestamp, start_date):
        # convert system_timestamp and start_date to date only
        live_bool = False
        # self.logger.info({'run_mode':run_mode, 'system_timestamp':system_timestamp, 'start_date':start_date})
        # now_tz = datetime.datetime.now() - datetime.timedelta(days=5)
        # last_to_last_market_close = self.datafeeder.get_previous_market_close(system_timestamp - datetime.timedelta(days=1))
        # start_date_minus_5_days = start_date - datetime.timedelta(days=5)
        # self.logger.debug({'run_mode':run_mode, 'system_timestamp':system_timestamp, 'time_to_load_minute_data':self.time_to_load_minute_data.astimezone(pytz.timezone('US/Eastern'))})
        if run_mode in [1,2] and system_timestamp >= self.time_to_load_minute_data:
            # previous_market_close = self.datafeeder.previous_market_close(system_timestamp)
            # self.logger.info({'run_mode':run_mode, 'system_timestamp+1':system_timestamp+datetime.timedelta(minutes=1), 'start_date':start_date})
            if '1m' not in self.datafeeder.config_dict['datafeeder_config']['data_inputs']: #and system_timestamp >= previous_market_close:
                # Add 1min timestamp to the datafeeder and datafetcher interval_inputs
                # self.logger.debug(f"Current config_dict: {self.datafeeder.config_dict}, 'Previous Config Dict: {self.datafeeder.previous_config_dict}")
                self.datafeeder.config_dict['datafeeder_config']['data_inputs']['1m'] = self.datafeeder.config_dict['datafeeder_config']['data_inputs']['1d']
                self.datafeeder.datafetcher.config_dict['datafeeder_config']['data_inputs']['1m'] = self.datafeeder.datafetcher.config_dict['datafeeder_config']['data_inputs']['1d']
                
                # self.datafeeder.lookback_dict = self.datafeeder.create_lookback_dict()
                self.datafeeder.lookback_dict = self.datafeeder.reset_lookback_dict()
                self.datafeeder.datafetcher.lookback_dict = self.datafeeder.lookback_dict
                self.next_expected_timestamp_on_now = self.datafeeder.get_next_expected_timestamp(pd.Timestamp.now(tz='US/Eastern'))
                
                # self.logger.debug(f"Added 1m to datafeeder and datafetcher interval_inputs: system_timestamp: {system_timestamp_tz}")
                # self.logger.debug(f"Current config_dict: {self.datafeeder.config_dict}, 'Previous Config Dict: {self.datafeeder.previous_config_dict}")
            system_timestamp_tz = system_timestamp.astimezone(pytz.timezone('US/Eastern'))
            now_tz = pd.Timestamp.now(tz='US/Eastern')
            if self.market_open_bool:
                sleep_lookup = {"1m":60,"2m":120,"5m":300,"1d":86400}
                interval_inputs = self.config_dict['datafeeder_config']['data_inputs']
                min_granularity_seconds = min([sleep_lookup[interval] for interval in interval_inputs])
                last_timestamp_tz = now_tz - datetime.timedelta(seconds=min_granularity_seconds)
                last_timestamp_tz = last_timestamp_tz.replace(second=0, microsecond=0)
            else:
                last_timestamp_tz = self.prev_market_close
                last_timestamp_tz = last_timestamp_tz.astimezone(pytz.timezone('US/Eastern'))
            # self.logger.debug({'system_timestamp_tz':system_timestamp_tz, 'last_timestamp_tz':last_timestamp_tz})
            self.logger.debug({'system_timestamp':system_timestamp_tz, 'now_tz':now_tz, 'prev_market_close':self.prev_market_close, 'market_open_bool':self.market_open_bool, 'next_expected_timestamp_on_now':self.next_expected_timestamp_on_now, 'Go LIVE BOOL':system_timestamp_tz >= last_timestamp_tz - datetime.timedelta(minutes=2)})
            # if next_expected_timestamp <= min_granularity_seconds:
            # self.logger.debug(f"Now: {now_tz}, System Timestamp: {system_timestamp_tz}, Last Timestamp: {last_timestamp_tz}, Next Expected Timestamp: {self.next_expected_timestamp_on_now}, Market Open: {self.market_open_bool}, 'Time to Go Live: {system_timestamp_tz >= last_timestamp_tz - datetime.timedelta(minutes=1)}")
            # if system_timestamp_tz + datetime.timedelta(minutes=1) >= last_timestamp_tz:
            if system_timestamp_tz >= last_timestamp_tz - datetime.timedelta(minutes=2):
                self.next_expected_timestamp_on_now = self.datafeeder.get_next_expected_timestamp(pd.Timestamp.now(tz='US/Eastern'))
                if self.next_expected_timestamp_on_now >= 60:
                    live_bool = True
                # msg = {'start_date':start_date, 'market_open_bool':self.market_open_bool, 'system_timestamp':system_timestamp, 'now_tz':now_tz, 'last_timestamp_tz':last_timestamp_tz, 'live_bool':live_bool}
            #   if not self.market_open_bool:
                    # msg['prev_market_close'] = last_timestamp_tz
            #   self.logger.info(msg)
                sleeper(10, 'Going LIVE in: ')

        return live_bool
    
    def sync_orders_on_live(self, new_orders, market_data_df, system_timestamp):
        # self.logger.debug({'live_bool':live_bool, 'run_mode':run_mode, 'system_timestamp':system_timestamp, 'start_date':start_date, 'prev_live_bool':prev_live_bool})
        msg = "------------------------ System is going live. Syncing OMS with Live Broker "
        print(msg + '*'*(os.get_terminal_size().columns - len(msg)))
        total_orders_value = 0
        for count, signal in enumerate(self.oms.get_open_signals()):
            if signal.status != 'closed':
                for order in signal.orders:
                    if order.status not in ['closed', 'cancelled']:
                        price = order.entryPrice if hasattr(order, 'entryPrice') else order.exitPrice if hasattr(order, 'exitPrice') else None
                        if price is None:
                            continue
                        order_value = price * order.orderQuantity if hasattr(order, 'entryPrice') else 0
                        total_orders_value += order_value
                        self.logger.info(f"SIM OPEN ORDER {count+1}: Symbol: {order.symbol}, orderDirection: {order.orderDirection}, order_value: {order_value}, total_orders_value: {total_orders_value}, orderType: {order.orderType}, orderQuantity: {order.orderQuantity}, Status: {order.status}")
        
        # print a series of dashes '-' only as wide as the terminal
        print('*'*os.get_terminal_size().columns)
        print('*'*os.get_terminal_size().columns)
        print('*'*os.get_terminal_size().columns)
        print('*'*os.get_terminal_size().columns)

        new_orders_from_sync = self.oms.sync_open_orders('oms-to-broker', market_data_df, system_timestamp, brokers=['IBKR'])
        new_orders.extend(new_orders_from_sync)
        total_orders_value = 0
        for count, signal in enumerate(new_orders_from_sync):
            for order in signal.orders:
                price = order.entryPrice if hasattr(order, 'entryPrice') else order.exitPrice if hasattr(order, 'exitPrice') else None
                if price is None:
                    continue
        total_orders_value = 0
        for count, signal in enumerate(new_orders_from_sync):
            for order in signal.orders:
                price = order.entryPrice if hasattr(order, 'entryPrice') else order.exitPrice if hasattr(order, 'exitPrice') else None
                if price is None:
                    continue
                order_value = price * order.orderQuantity if hasattr(order, 'entryPrice') else 0
                if hasattr(order, 'orderType') and order.orderType.lower() == 'market_exit':
                    order_value = order_value * -1
                total_orders_value += order_value
                self.logger.info(f"SYNC New Order {count+1}: Symbol: {order.symbol}, orderDirection: {order.orderDirection}, orderType: {order.orderType}, orderQuantity: {order.orderQuantity}, Price: {price}, order_value:{order_value}, total_orders_value: {total_orders_value}, Strategy: {signal.strategy_name}")
        
        msg = '------------------------ System is now live. Syncing OMS with Live Broker.'
        print(msg + '*'*(os.get_terminal_size().columns - len(msg)))
    
        return new_orders

    def run(self):
        run_mode = config_dict['run_mode']
        
        '''Set the start_date and end_date based on run_mode'''
        if run_mode in [1,2]: # live trading - real money
            # US Eastern Time Zone
            start_date = datetime.datetime.now().astimezone(pytz.timezone('UTC')) #if 'live_start_date' not in self.config_dict else self.config_dict['live_start_date']
            market_open_bool = self.datafeeder.is_market_open(start_date)
            # self.logger.warning('Manually setting market_open_bool to True')
            # sleeper(4, 'Giving your system 4 seconds to read the warning message above')
            prev_mkt_close = self.datafeeder.get_previous_market_close(start_date)
            # If market is closed, then take prev market close else take the current time (Eg. On Sunday, min(start_date, prev_mkt_close) on Monday, at 12 noon, max(start_date, prev_mkt_close)
            start_date = prev_mkt_close if not market_open_bool else start_date
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = None
            
        elif run_mode == 3: # backtesting
            assert 'start_time' in self.config_dict['backtest_inputs'] and 'end_time' in self.config_dict['backtest_inputs'], 'start_time and end_time must be provided in backtest_inputs if run_mode is 3'
            start_date = self.config_dict['backtest_inputs']['start_time']
            end_date = self.config_dict['backtest_inputs']['end_time']
            # self.logger.debug({'start_date':start_date, 'end_date':end_date})
        
        if run_mode in [1,2,3]:
            # if start_date or end_date is not UTC, convert it to UTC
            if start_date and start_date.tzinfo is not pytz.timezone('UTC'):
                start_date = start_date.astimezone(pytz.timezone('UTC'))
            if end_date and end_date is not pytz.timezone('UTC'):
                end_date = end_date.astimezone(pytz.timezone('UTC'))
            
        '''Start Running the System'''
        start = time.time()
        if run_mode in [1,2,3]:
            while True:
                
                # for interval in self.current_market_data_df.index.get_level_values(0).unique():
                #     self.logger.debug(f"Full Loop Time Taken: {time.time() - start}. Market_data_df Shape:{interval}: {self.current_market_data_df.loc[interval].shape}")
                # start = time.time()
                try:
                    next_rows = self.datafeeder.next(system_timestamp=self.system_timestamp, run_mode=run_mode, sleep_time=self.sleep_time, start_date=start_date, end_date=end_date, live_bool=self.live_bool)
                    if run_mode == 3 and self.stop_backtest_at_n_signals and len(self.oms.closed_signals) >= self.stop_backtest_at_n_signals:
                        self.logger.info(f"Backtest stopped at {len(self.oms.closed_signals)} signals. REMOVE THIS FUNCTIONALITY LATER")
                        next_rows = None
                    if next_rows is not None:
                        msg = "------------------------ New Timestamp "
                        print(msg + '-'*(os.get_terminal_size().columns - len(msg)))
                        
                        # Trim existing data before concatenation
                        if not self.current_market_data_df.empty:
                            self.current_market_data_df = self.datafeeder.trim_current_market_data_df(self.current_market_data_df)
                        
                        # Efficient concatenation
                        if not next_rows.empty:
                            if self.current_market_data_df.empty:
                                self.current_market_data_df = next_rows
                            else:
                                self.current_market_data_df = pd.concat([self.current_market_data_df, next_rows], copy=False)
                        
                        if next_rows.index.get_level_values(1)[-1] > self.system_timestamp: #isinstance(self.system_timestamp, type(None)) or 
                            self.system_timestamp = next_rows.index.get_level_values(1)[-1]
                            
                            '''PRINT 1: Print Details of the next Timestamp'''
                            for interval, next_datetime in next_rows.index:
                                time_start = time.time()
                                next_datetime_EST = next_datetime.astimezone(pytz.timezone('US/Eastern'))
                                system_timestamp_EST = self.system_timestamp.astimezone(pytz.timezone('US/Eastern'))
                                self.logger.info(f"Interval: {interval}, Latest Data Datetime: {next_datetime_EST}, System Datetime: {system_timestamp_EST}, Live Bool: {self.live_bool}, Data Source: {next_rows.loc[interval].iloc[-1][('data_source', '')]}")
                                # if interval == '1d':
                                    # self.logger.info(f"next_rows: {next_rows}")
                            
                            # Get margin information for signal generation
                            broker = 'IBKR'.lower() if self.live_bool else 'SIM'.lower()
                            base_account_number = list(self.oms.margin_available[broker].keys())[0]
                            margin_info = self.oms.margin_available[broker][base_account_number]
                            trading_currency = self.config_dict['trading_currency']
                            
                            # Generate Signals with margin information
                            time_start = time.time()
                            new_signals, self.config_dict = self.vault.generate_signals(next_rows, self.current_market_data_df, self.system_timestamp, self.oms.open_signals, self.oms.margin_available)
                            self.datafeeder.config_dict = self.rms.config_dict = self.vault.config_dict
                            
                            # Send updated signals to OMS
                            for signal in new_signals['signals']:
                                # self.logger.info({'signal':signal})
                                if signal.signal_update:
                                    self.oms.update_signal(signal)
                                    new_signals['signals'].remove(signal)
                            
                            # Check if we're going live
                            time_start = time.time()
                            prev_live_bool = self.live_bool
                            if prev_live_bool == False:
                                self.live_bool = self.are_we_live(run_mode, self.system_timestamp, start_date)
                            
                            # Convert signals to orders
                            time_start = time.time()
                            new_signals = self.rms.run_rms(new_signals, self.oms.margin_available, self.oms.open_signals, self.system_timestamp, self.live_bool)
                            # If the system is going live, sync the orders with the broker # Get a list of orders that'll be sent to the LIVE broker, based on current open orders
                            if prev_live_bool == False and self.live_bool == True:
                                time_start = time.time()
                                new_signals = self.sync_orders_on_live(new_signals, self.current_market_data_df, self.system_timestamp)
                                
                            # Execute orders on the market with the OMS
                            time_start = time.time()
                            self.oms.execute_signals(new_signals, self.system_timestamp, self.current_market_data_df, live_bool=self.live_bool)
                            
                            # Print update messages to console
                            self.print_updates_to_console()
                            
                            # Only trim if we've accumulated significant data
                            total_rows = sum(len(self.current_market_data_df.loc[interval]) 
                                           for interval in self.current_market_data_df.index.get_level_values(0).unique())
                            
                            if total_rows > 1000:  # Arbitrary threshold, adjust based on your needs
                                self.current_market_data_df = self.datafeeder.trim_current_market_data_df(self.current_market_data_df)
                            
                        else:
                            self.logger.info('No new data available. Next Rows: {}'.format(next_rows))

                    else:
                        self.logger.info('Backtest completed.')
                        from pprint import pprint
                        # self.logger.info(pprint(self.config_dict))
                        self.logger.info(pprint(self.oms.margin_available))
                        
                        self.reporter.generate_report(
                            self.config_dict,
                            self.oms.get_open_signals(),
                            self.oms.closed_signals,
                            self.current_market_data_df,
                            with_fees_and_slippage=False
                        )
                        
                        self.reporter.generate_report(
                            self.config_dict,
                            self.oms.get_open_signals(),
                            self.oms.closed_signals,
                            self.current_market_data_df,
                            with_fees_and_slippage=True
                        )
                        
                        if self.config_dict['backtest_inputs']['save_backtest_results']:
                            self.test_folder_path, self.test_name = self.reporter.save_backtest(
                                self.config_dict,
                                self.oms.get_open_signals(),
                                self.oms.closed_signals
                            )
                            self.logger.info(f'Backtest results saved at: {self.test_folder_path}')
                            self.logger.info(f'Backtest Name: {self.test_name}')
                        
                        break
                
                except KeyboardInterrupt:
                    # self.logger.debug({'self.market_data_df':self.market_data_df})
                    self.logger.debug('Exiting...')
                    break
                except Exception as e:
                    raise Exception(e)
                    
        elif run_mode == 4:
            self.datafeeder.update_all_historical_price_data(self.live_bool)
            
        else:
            raise AssertionError('Invalid run_mode value: {}'.format(run_mode))

if __name__ == '__main__':
    logs_folder = project_path + '/logs'
    # Remove all .log files from logs folder
    for file in os.listdir(logs_folder):
        if file.endswith('.log'):
            os.remove(os.path.join(logs_folder, file))
    Mathematricks(config_dict).run()