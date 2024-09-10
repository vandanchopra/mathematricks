import json
import os
# read the json file '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/db/stock_symbols.json'

cwd = os.getcwd()
stock_symbols_path = os.path.join(cwd,'db/stock_symbols.json')
with open(stock_symbols_path) as file:
    stock_symbols = json.load(file)

data_inputs_path = os.path.join(cwd,'db/data_inputs.json')
with open(data_inputs_path) as file:
    data_inputs = json.load(file)
    
config_dict = {
    'run_mode': 4, # 1: live trading - real money, 2: live trading - paper money, 3: backtesting, 4: data update only
    'backtest_inputs': {},
    'data_update_inputs': {'data_sources':['yahoo']},
    'list_of_symbols': stock_symbols[:2],
    'sleep_time':10,
    'log_level':'DEBUG',
    'data_inputs':data_inputs,
}


'''
once a week, update stock symbols by 
'''
