import json

# read the json file '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/db/stock_symbols.json'
with open('db/stock_symbols.json') as file:
    stock_symbols = json.load(file)
    
config_dict = {
    'run_mode': 4, # 1: live trading - real money, 2: live trading - paper money, 3: backtesting, 4: data update only
    'backtest_inputs': {},
    'data_update_inputs': {'data_sources':['yahoo']},
    'list_of_symbols': stock_symbols[:20],
    'sleep_time':0,
    'log_level':'DEBUG',
    'strategies':['strategy_1', 'strategy_2'],
    'broker':"IBKR",
    "total_portfolio" : 200000,
    "total_funds_s_1" : 100000,
    "avail_funds_s_1" : 100000,
    "total_funds_s_2" : 100000,
    "avail_funds_s_2" : 100000,
    "max_signal_fund" : 25000,
    "max_risk_per" : 5
}


'''
once a week, update stock symbols by 
'''
