from copy import deepcopy
from wsgiref import headers
import gspread
from google.oauth2.service_account import Credentials
from systems.utils import generate_hash_id, project_path
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
    
    def get_backtests_queue(self):
        backtests_queue = self.backtests_queue_worksheet.get_all_records()
        return backtests_queue
    
    def update_backtests_queue(self, new_backtest_queue_dict):
        if len(new_backtest_queue_dict) != 0:
            # Convert the data to a list of lists format
            data_list = [list(row.values()) for row in new_backtest_queue_dict]
            # Add the header back to the worksheet
            header = list(new_backtest_queue_dict[0].keys())
            
        else:
            data_list = []
            header = ['Backtest Name', 'Start Date', 'End Date', 'Strategies', 'Data Source', 'Risk Management', 'OMS']
            
        data_with_header = [header] + data_list
        self.backtests_queue_worksheet.clear()
        # Add the data back to the worksheet
        self.backtests_queue_worksheet.append_rows(data_with_header)
        
    def add_backtest_to_queue(self, new_backtest_dict):
        print({'new_backtest_dict':new_backtest_dict})
        new_backtest_entry = list(new_backtest_dict.values())
        self.backtests_queue_worksheet.append_row(new_backtest_entry)
        
    def create_backtest_entry_from_config_dict(self, config_dict):
        backtest_entry_dict = {
                        'start_time': config_dict['backtest_inputs']['start_time'],
                        'end_time': config_dict['backtest_inputs']['end_time'],
                        'strategies': config_dict['strategies'],
                        'data_update_inputs': config_dict['data_update_inputs'],
                        'risk_management': config_dict['risk_management'],
                        'oms': config_dict['oms'],
                        'data_source': config_dict['data_update_inputs']['data_sources']
                        }
        backtest_entry_dict['backtest_name'] = generate_hash_id(backtest_entry_dict, 0)
        
        backtest_entry = {}
        
        for key in ['backtest_name', 'start_time', 'end_time', 'strategies', 'data_source', 'risk_management', 'oms']:
            backtest_entry[key] = str(backtest_entry_dict[key])
        
        return backtest_entry
    
    def create_config_dict_from_backtest_entry(self, backtest_entry):
        def convert_str_to_dict(input_str):
            # Convert string to dict
            input_str = input_str.replace('\'', '\"')
            input_str = re.sub(r'(\w+):', r'"\1":', input_str)
            input_dict = json.loads(input_str)
            return input_dict

        config_dict_template = {
                            'run_mode': 3, # 1: live trading - real money, 2: live trading - paper money, 3: backtesting, 4: data update only
                            'backtest_inputs': {
                                'start_time': pd.Timestamp(datetime(2021,9,25)).tz_localize('UTC').tz_convert('EST'), 
                                'end_time': pd.Timestamp(datetime(2024,9,27)).tz_localize('UTC').tz_convert('EST'),
                                },
                            'strategies':['strategy_3'], # 'strategy_1'
                            'data_update_inputs': {'data_sources':['yahoo']}, # 'yahoo', 'ibkr'
                            'sleep_time':60,
                            'account_info':[],
                            'risk_management':{'max_risk_per_bet':0.05, 'max_margin_utilized':3, 'margin_reserve_pct':0.2},
                            'oms': {'funds_available':100000, 'margin_available':100000, 'portfolio':{}}
                        }
        config_dict = deepcopy(config_dict_template)
        config_dict['run_mode'] = 2
        config_dict['backtest_inputs']['start_time'] = pd.Timestamp(backtest_entry['Start Date'])
        config_dict['backtest_inputs']['end_time'] = pd.Timestamp(backtest_entry['End Date'])
        config_dict['strategies'] = re.findall(r"'(.*?)'", backtest_entry['Strategies'])
        config_dict['data_update_inputs']['data_sources'] = re.findall(r"'(.*?)'", backtest_entry['Data Source'])
        config_dict['risk_management'] = convert_str_to_dict(backtest_entry['Risk Management'])
        config_dict['oms'] = convert_str_to_dict(backtest_entry['OMS'])
        config_dict['backtest_inputs']['Backtest Name'] = backtest_entry['Backtest Name']
        
        return config_dict
        
    def save_backtests_queue_to_file(self, backtests_queue):
        backtests_queue_df = pd.DataFrame(backtests_queue)
        backtests_queue_df.to_json(self.path_to_backtests_queue_json)
    
    def load_backtests_queue_from_file(self):
        backtests_queue_df = pd.read_json(self.path_to_backtests_queue_json)
        backtests_queue = backtests_queue_df.to_dict(orient='records')
        return backtests_queue
    
    def add_completed_backtest_to_google_sheet(self, completed_backtest, backtest_performance_metrics, test_folder_path):
        del completed_backtest['Data Source']
        completed_backtest_entry = list(completed_backtest.values())
        for key in ['profit', 'win_pct', 'sharpe_ratio', 'long_count', 'short_count']:
            completed_backtest_entry.append(backtest_performance_metrics[key])
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
    