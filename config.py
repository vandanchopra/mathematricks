import json
import os
import pandas as pd
from datetime import datetime
# read the json file '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/db/stock_symbols.json'
    
config_dict = {
    'run_mode': 2, # 1: live trading - real money, 2: live trading - paper money, 3: backtesting, 4: data update only
    'backtest_inputs': {
        'start_time': pd.Timestamp(datetime(2021,9,25)).tz_localize('UTC').tz_convert('EST'), 
        'end_time': pd.Timestamp(datetime(2024,9,27)).tz_localize('UTC').tz_convert('EST'),
        'funds_available':100000, 'margin_available':100000, 'update_data':False
        },
    'data_update_inputs': {'data_sources':['yahoo']}, # 'yahoo', 'ibkr'
    'sleep_time':60,
    'strategies':['strategy_3'], # 'strategy_1'
    'account_info':[],
    'risk_management':{'max_risk_per_bet':0.05, 'max_margin_utilized':3, 'margin_reserve_pct':0.2},
    'oms': {'funds_available':100000, 'margin_available':100000, 'portfolio':{}}
}