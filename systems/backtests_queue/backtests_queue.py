from copy import deepcopy
import gspread
from google.oauth2.service_account import Credentials
from systems.utils import generate_hash_id, project_path, create_logger
import re, json
import pandas as pd
from datetime import datetime

class BacktestQueue:
    def __init__(self):
        self.spreadsheet_url = 'https://docs.google.com/spreadsheets/d/1FqOpXfHIci2BQ173dwDV35NjkyEn4QmioIlMEH-WiOA/edit?gid=0'
        self.backtests_queue_sheetname = 'Backtest Queue'
        self.backtests_queue_worksheet = self.get_worksheet(self.backtests_queue_sheetname, self.spreadsheet_url)
        self.completed_backtests_sheetname = 'Completed Backtests'
        self.completed_backtests_worksheet = self.get_worksheet(self.completed_backtests_sheetname, self.spreadsheet_url)
        self.path_to_backtests_queue_json = project_path + 'db/vault/backtests_queue.json'
        self.logger = create_logger(log_level='DEBUG', logger_name='BACKTEST_QUEUE', print_to_console=True)
        
    
    def get_worksheet(self, sheet_name, spreadsheet_url):
    # Define the scope
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
        path_to_google_sheets_json = '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/secrets/mathematricks-222505-d8b1ccce2084.json'

        # Load the credentials
        creds = Credentials.from_service_account_file(path_to_google_sheets_json, scopes=SCOPES)

        # Connect to the Google Sheet
        gc = gspread.authorize(creds)
        self.spreadsheet = gc.open_by_url(spreadsheet_url)
        self.worksheets = self.spreadsheet.worksheets()
        requested_worksheet = None
        for worksheet in self.worksheets:
            if worksheet.title == sheet_name:
                requested_worksheet = worksheet
        return requested_worksheet
    
    def get_backtests_queue_from_googlesheets(self):
        backtests_queue = self.backtests_queue_worksheet.get_all_records()
        return backtests_queue
    
    def update_backtests_queue(self):
        self.backtests_queue_worksheet.delete_rows(2)
        # Add the data back to the worksheet
        
    def add_backtest_to_queue(self, new_backtest_dict):
        print({'new_backtest_dict':new_backtest_dict})
        new_backtest_entry = list(new_backtest_dict.values())
        self.backtests_queue_worksheet.append_row(new_backtest_entry)
        
    def create_backtest_entry_from_config_dict(self, config_dict):
        base_currency = 'CAD'
        backtest_entry_dict = {
                        'start_time': config_dict['backtest_inputs']['start_time'],
                        'end_time': config_dict['backtest_inputs']['end_time'],
                        'strategies': config_dict['strategies'],
                        'data_update_inputs': config_dict['data_update_inputs'],
                        'risk_management': config_dict['risk_management'],
                        'account_info': config_dict['account_info']['sim']['sim_1'][base_currency],
                        'data_source': config_dict['data_update_inputs']['data_sources']
                        }
        backtest_name = config_dict['backtest_inputs'].get('backtest_name')
        backtest_name = backtest_name if backtest_name else ''
        
        backtest_entry_dict['backtest_name'] = backtest_name + '_' + str(generate_hash_id(backtest_entry_dict, 0))
        
        backtest_entry = {}
        
        for key in ['backtest_name', 'start_time', 'end_time', 'strategies', 'data_source', 'risk_management', 'account_info']:
            backtest_entry[key] = str(backtest_entry_dict[key])
        
        return backtest_entry
    
    def create_config_dict_from_backtest_entry(self, backtest_entry):
        def convert_str_to_dict(input_str):
            # Convert string to dict
            input_str = input_str.replace('\'', '\"')
            input_str = re.sub(r'(\w+):', r'"\1":', input_str)
            input_dict = json.loads(input_str)
            return input_dict
        start_date = pd.Timestamp(datetime(2021,1,1)).tz_localize('UTC').tz_convert('EST')
        end_date = start_date + pd.Timedelta(days=120)
        sim_account_margin_multiplier = 3
        sim_account_starting_value_base = 100000
        base_currency = 'CAD'
        trading_currency = 'USD'
        base_currency_to_trading_currency_exchange_rate = 1/1.375
        
        config_dict_template = {
                                'run_mode': 3, # 1: live trading - real money, 2: live trading - paper money, 3: backtesting, 4: data update only
                                'base_currency': base_currency,
                                'trading_currency': trading_currency,
                                'base_currency_to_trading_currency_exchange_rate': base_currency_to_trading_currency_exchange_rate,
                                'backtest_inputs': {
                                    'start_time': start_date, 
                                    'end_time': end_date,
                                    'save_backtest_results':True,
                                    # 'Backtest Name':'alsdkjasldkqw923yasjdaskd23328y',
                                    },
                                'strategies':['strategy_dev.strategy_3'], # 'strategy_1', 'strategy_dev.strategy_3'
                                'data_update_inputs': {'data_sources':['yahoo']}, # 'yahoo', 'ibkr'
                                'sleep_time':60,
                                'account_info':{'sim':{'sim_1': {base_currency:{'total_account_value': sim_account_starting_value_base, 
                                                                                'buying_power_available':sim_account_starting_value_base * sim_account_margin_multiplier, 
                                                                                'buying_power_used':0, 'cushion':0, 'pledge_to_margin_used':0, 
                                                                                'pledge_to_margin_availble':sim_account_starting_value_base
                                                                                }
                                                                }
                                                    }
                                                },
                                'base_account_numbers':{'sim':'sim_1', 'ibkr':'U13152795'},
                                'risk_management': {'max_risk_per_bet':0.05, 'maximum_margin_used_pct':0.80}
                                }
        
        config_dict = deepcopy(config_dict_template)
        config_dict['run_mode'] = 2
        config_dict['backtest_inputs']['start_time'] = pd.Timestamp(backtest_entry['Start Date'])
        config_dict['backtest_inputs']['end_time'] = pd.Timestamp(backtest_entry['End Date'])
        config_dict['strategies'] = re.findall(r"'(.*?)'", backtest_entry['Strategies'])
        config_dict['data_update_inputs']['data_sources'] = re.findall(r"'(.*?)'", backtest_entry['Data Source'])
        config_dict['risk_management'] = convert_str_to_dict(backtest_entry['Risk Management'])
        config_dict['account_info']['sim']['sim_1'][base_currency] = convert_str_to_dict(backtest_entry['Account Info'])
        config_dict['backtest_inputs']['Backtest Name'] = backtest_entry['Backtest Name']
        
        return config_dict
        
    def save_backtests_queue_to_file(self, backtests_queue):
        backtests_queue_df = pd.DataFrame(backtests_queue)
        backtests_queue_df.to_json(self.path_to_backtests_queue_json)
    
    def load_backtests_queue_from_file(self):
        backtests_queue_df = pd.read_json(self.path_to_backtests_queue_json)
        backtests_queue = backtests_queue_df.to_dict(orient='records')
        return backtests_queue
    
    def add_completed_backtest_to_google_sheet(self, completed_backtest, backtest_report, test_folder_path):
        del completed_backtest['Data Source']
        completed_backtest_entry = list(completed_backtest.values())
        completed_backtest_entry.append(backtest_report)
        completed_backtest_entry.append(test_folder_path)
        self.completed_backtests_worksheet.append_row(completed_backtest_entry)
    
if __name__ == '__main__':
    from config import config_dict
    import pandas as pd
    from datetime import datetime

    backtest_queue = BacktestQueue()

    # mode = None 
    mode = 'add_backtest'
    if mode == 'add_backtest':
        # Add a backtest to the queue
        new_backtest_dict = config_dict
        new_backtest_dict['backtest_inputs']['start_date'] = pd.Timestamp(datetime(2022,9,25)).tz_localize('UTC').tz_convert('EST')
        new_backtest_dict['backtest_inputs']['end_date'] = pd.Timestamp(datetime(2024,9,27)).tz_localize('UTC').tz_convert('EST')
        new_backtest_dict['strategies'] = ['strategy_dev.strategy_3']
        new_backtest_dict['data_update_inputs'] = {'data_sources':['yahoo']}
        new_backtest_dict['risk_management'] = {'max_risk_per_bet':0.05, 'max_margin_utilized':3, 'margin_reserve_pct':0.2}
        new_backtest_dict['oms'] = {'funds_available':100000, 'margin_available':100000, 'portfolio':{}}
        backtest_entry = backtest_queue.create_backtest_entry_from_config_dict(config_dict)
        backtest_queue.add_backtest_to_queue(backtest_entry)
    else:
       pass
    