# read the txt file in a list and remove the '\n' character
with open('/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/db/stock_symbols.txt', 'r') as f:
    stock_symbols = f.readlines()
    stock_symbols = [x.strip() for x in stock_symbols]

config_dict = {
    'run_mode': 1, # 1: live trading - real money, 2: live trading - paper money, 3: backtesting, 4: data update only
    'backtest_inputs': {},
    'data_update_inputs': {'data_sources':['yahoo', 'ibkr']},
    'list_of_symbols': stock_symbols[:10],
    'sleep_time':0,
}