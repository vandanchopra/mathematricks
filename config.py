import json
import os
#read the json file '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/db/stock_symbols.json'
import json
import vault
import vault.base_strategy
import vault.strategy_1
import vault.strategy_2

cwd = os.getcwd()
stock_symbols_path = os.path.join(cwd,'E:\mathematricks-main/stock_symbols.json')
with open(stock_symbols_path) as file:
    stock_symbols = json.load(file)
    
config_dict = {
    'run_mode': 1, # 1: live trading - real money, 2: live trading - paper money, 3: backtesting, 4: data update only
    'backtest_inputs': {},
    'broker': ['IBKR', 'Sim'],
    'data_update_inputs': {'data_sources':['yahoo']},
    'list_of_symbols': stock_symbols[:2],
    'sleep_time':60,
    'strategies':[strategy_1, strategy_2],
    'log_level':'DEBUG',
    'data_inputs':{"1m": {"columns": ["open","high","low","close","volume","SMA15","SMA30" ],"lookback": 100},
                   "1d": {"columns": ["open","high","low","close","volume","SMA15","SMA30"],"lookback": 100},
                   "5m": {"columns": ["open","high","low","close","volume","SMA15","SMA30"],"lookback": 100}},
}

'''
once a week, update stock symbols by 
'''
