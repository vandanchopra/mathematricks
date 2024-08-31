config_dict = {
    'run_mode': 2, # 1: live trading - real money, 2: live trading - paper money, 3: live simulated trading - paper money, 4: backtesting, 5: data update only
    'simulated_trading_speed': '1x', # 0.25x, 0.5x, 1x, 2x, 4x, 100x, x
    'backtest_inputs': {},
    'data_update_inputs': {'data_source':['yahoo', 'ibkr']},
}