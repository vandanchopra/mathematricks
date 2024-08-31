from turtle import update
from config import config_dict
from brokers import Brokers

'''
just getting your existing strategy to run live.
----
update the data.....now run your analysis....make an ideal portfolio......check curr portfolio.....relance curr portfolio.

test start date: datatime or datetime.now
data start date: will be based on the strategy.

'''


def run():
    run_mode = config_dict['run_mode']
    
    if run_mode == 1: # live trading - real money
        print ('live trading - real money')
    elif run_mode == 2: # live trading - paper money
        print ('live trading - paper money')
    elif run_mode == 3: # live simulated trading - paper money
        simulated_trading_speed = config_dict['simulated_trading_speed']
        print ({'simulated_trading_speed': simulated_trading_speed})
    elif run_mode == 4: # backtesting
        backtest_inputs = config_dict['backtest_inputs']
        print ({'backtest_inputs': backtest_inputs})
    elif run_mode == 5: # data update only
        data_update_inputs = config_dict['data_update_inputs']
        print ({'data_update_inputs': data_update_inputs})
    else:
        raise ValueError('Invalid run_mode value: {}'.format(run_mode))

if __name__ == '__main__':
    run()