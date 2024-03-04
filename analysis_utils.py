
def load_backtest(self, symbols, start_date_dt, end_date_dt, rebalance_frequency, long_count, short_count, portfolio_starting_value, risk_pct, reinvest_profits_bool):
    pickle_filename = f'backtests/{self.strategy.get_name()}_{"_".join(symbols)}_{start_date_dt.date()}_{end_date_dt.date()}_{rebalance_frequency}_{long_count}_{short_count}_{portfolio_starting_value}_{risk_pct}_{reinvest_profits_bool}.pkl'
    test = pickle.load(open(pickle_filename, 'rb'))
    print(f'Backtest results loaded from {pickle_filename}')
    return test