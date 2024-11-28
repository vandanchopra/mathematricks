import json
import os
import pandas as pd
from datetime import datetime
import pytz
# read the json file '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/db/stock_symbols.json'
start_date = datetime(2010, 3, 1).astimezone(pytz.timezone('US/Eastern'))
end_date = start_date + pd.Timedelta(days=int(365*1))
sim_account_margin_multiplier = 3
sim_account_starting_value_base = 100000
base_currency = 'CAD'
trading_currency = 'USD'
base_currency_to_trading_currency_exchange_rate = 1/1.375
IBKR_base_account_number = 'DU7994930' #'U13152795

config_dict = {
    'run_mode': 4, # 1: live trading - real money, 2: live trading - paper money, 3: backtesting, 4: data update only
    'base_currency': base_currency,
    'trading_currency': trading_currency,
    'base_currency_to_trading_currency_exchange_rate': base_currency_to_trading_currency_exchange_rate,
    'backtest_inputs': {
        'start_time': start_date, 
        'end_time': end_date,
        'save_backtest_results':True,
        # 'Backtest Name':'alsdkjasldkqw923yasjdaskd23328y',
        },
    'strategies':['strategy_dev.strategy_3_1'], # 'strategy_1', 'strategy_dev.strategy_3', 'strategy_3_small_set
    'data_update_inputs': {'data_sources':{'sim':'ibkr', 'live':'ibkr'}}, # 'yahoo', 'ibkr'
    'sleep_time':60,
    'account_info':{
                    'ibkr':{IBKR_base_account_number:{base_currency:{}
                                         }
                            }, 
                    'sim':{'sim_1': {base_currency:{'total_account_value': sim_account_starting_value_base, 
                                                    'buying_power_available':sim_account_starting_value_base * sim_account_margin_multiplier, 
                                                    'buying_power_used':0, 'cushion':0, 'pledge_to_margin_used':0, 
                                                    'pledge_to_margin_availble':sim_account_starting_value_base
                                                    }
                                     }
                           }
                    },
    'base_account_numbers':{'sim':'sim_1', 'ibkr':IBKR_base_account_number},
    'risk_management': {'max_risk_per_bet':0.05, 'maximum_margin_used_pct':0.65}
    }