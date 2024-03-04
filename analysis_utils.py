from datetime import datetime, timedelta
import pickle
import numpy as np
from stock_automation_utils import StockAutomationUtils
from strategy_vault import StrategyVault
import pandas as pd
from tqdm import tqdm


class BacktestAnalyzer:
    def __init__(self, strategy_name, symbols, start_date_dt, end_date_dt, rebalance_frequency, long_count, short_count,
                  portfolio_starting_value, risk_pct, reinvest_profits_bool):
        self.strategy_name = strategy_name
        self.symbols = symbols
        self.start_date_dt = start_date_dt
        self.end_date_dt = end_date_dt
        self.rebalance_frequency = rebalance_frequency
        self.long_count = long_count
        self.short_count = short_count
        self.portfolio_starting_value = portfolio_starting_value
        self.risk_pct = risk_pct
        self.reinvest_profits_bool = reinvest_profits_bool

    def load_backtest(self):
        pickle_filename = f'backtests/{self.strategy_name}_{"_".join(self.symbols)}_{self.start_date_dt.date()}_{self.end_date_dt.date()}_{self.rebalance_frequency}_{self.long_count}_{self.short_count}_{self.portfolio_starting_value}_{self.risk_pct}_{self.reinvest_profits_bool}.pkl'
        test = pickle.load(open(pickle_filename, 'rb'))
        print(f'Backtest results loaded from {pickle_filename}')

        return test
