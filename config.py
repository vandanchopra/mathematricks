import json
import os
import pandas as pd
from datetime import datetime
# read the json file '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/db/stock_symbols.json'
    
config_dict = {
    'run_mode': 3, # 1: live trading - real money, 2: live trading - paper money, 3: backtesting, 4: data update only
    'backtest_inputs': {
        'start_time': pd.Timestamp(datetime(2021,9,25)).tz_localize('UTC').tz_convert('EST'), 
        'end_time': pd.Timestamp(datetime(2024,10,13)).tz_localize('UTC').tz_convert('EST'),
        'save_backtest_results':False,
        'Backtest Name':'alsdkjasldkqw923yasjdaskd23328y'
        },
    'strategies':['strategy_1'], # 'strategy_1', 'strategy_dev.strategy_3'
    'data_update_inputs': {'data_sources':['yahoo']}, # 'yahoo', 'ibkr'
    'sleep_time':60,
    'account_info':[],
    'risk_management':{'max_risk_per_bet':0.05, 'max_margin_utilized':3, 'margin_reserve_pct':0.2},
    'oms': {'funds_available':100000, 'margin_available':100000, 'portfolio':{}}
}