import json
import os
import pandas as pd
from datetime import datetime
# read the json file '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/db/stock_symbols.json'


    
config_dict = {
    'run_mode': 1, # 1: live trading - real money, 2: live trading - paper money, 3: backtesting, 4: data update only
    'backtest_inputs': {
        'start_time': pd.Timestamp(datetime(2024,9,15)).tz_localize('UTC').tz_convert('EST'), 
        'end_time': pd.Timestamp(datetime(2024,9,18)).tz_localize('UTC').tz_convert('EST')
        },
    'data_update_inputs': {'data_sources':['ibkr']}, # 'yahoo', 'ibkr'
    'sleep_time':60,
    'log_level':'DEBUG',
    'strategies':['strategy_1'],
    'account_info':[{'account_id': '1234567890', 'account_type': 'client', 'broker':'ibkr', 'account_name':'paper_money_1', 'account_currency':'CAD', 
                     'funding_transactions':[{'transaction_type':'deposit', 'timestamp':pd.Timestamp(datetime(2024,9,12)).tz_localize('UTC').tz_convert('EST'), 'amount':200000, 'currency':'CAD'}], 
                     'current_account_balance':200000}],
    'risk_management':{'max_risk_per_trade':0.02},
    "total_portfolio" : 200000,
    "total_funds_s_1" : 100000,
    "avail_funds_s_1" : 100000,
    "total_funds_s_2" : 100000,
    "avail_funds_s_2" : 100000,
    "max_signal_fund" : 25000,
    "max_risk_per" : 5
}