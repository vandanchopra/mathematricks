from datetime import datetime, timedelta
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import compress
import quantstats as qs

class BacktestAnalyzer:

    def __init__(self, test_number):
        test_table = pd.read_csv('backtests_table.csv', index_col=[0], header=0)
        self.base_test = test_table.loc[test_number]
        self.strategy_name = self.base_test['strategy_name']
        self.symbols = self.base_test['inputs']['symbols']
        self.start_date_dt = self.base_test['inputs']['start_date_dt']
        self.end_date_dt = self.base_test['inputs']['end_date_dt']
        self.rebalance_frequency = self.base_test['inputs']['rebalance_frequency']
        self.long_count = self.base_test['inputs']['long_count']
        self.short_count = self.base_test['inputs']['short_count']
        self.portfolio_starting_value = self.base_test['inputs']['portfolio_starting_value']
        self.risk_pct = self.base_test['inputs']['risk_pct']
        self.reinvest_profits_bool = self.base_test['inputs']['reinvest_profits_bool']
        self.leverage_multiplier = self.base_test['inputs']['leverage_multiplier']

    def __init__(self, strategy_name, symbols, start_date_dt, end_date_dt, rebalance_frequency, long_count, short_count,
                  portfolio_starting_value, risk_pct, reinvest_profits_bool, leverage_multiplier):
        self.base_test = None
        self.strategy_name = strategy_name
        self.symbols = symbols
        self.start_date_dt = start_date_dt.date()
        self.end_date_dt = end_date_dt.date()
        self.rebalance_frequency = rebalance_frequency
        self.long_count = long_count
        self.short_count = short_count
        self.portfolio_starting_value = portfolio_starting_value
        self.risk_pct = risk_pct
        self.reinvest_profits_bool = reinvest_profits_bool
        self.leverage_multiplier = leverage_multiplier

    def __init__(self, test):
        self.base_test = test
        self.strategy_name = test['strategy_name']
        self.symbols = test['inputs']['symbols']
        self.start_date_dt = test['inputs']['start_date_dt']
        self.end_date_dt = test['inputs']['end_date_dt']
        self.rebalance_frequency = test['inputs']['rebalance_frequency']
        self.long_count = test['inputs']['long_count']
        self.short_count = test['inputs']['short_count']
        self.portfolio_starting_value = test['inputs']['portfolio_starting_value']
        self.risk_pct = test['inputs']['risk_pct']
        self.reinvest_profits_bool = test['inputs']['reinvest_profits_bool']
        self.leverage_multiplier = test['inputs']['leverage_multiplier']

    def get_backtest_filename_without_path(self):
        backtests_table = pd.read_csv('backtests_table.csv', index_col=[0], header=0)#, parse_dates=[3, 4])
        backtests_table['test_number'] = backtests_table.index

        this_backtest = backtests_table.loc[[(backtests_table['strategy_name'] == self.strategy_name) &
                                            # (pd.Series([x == y for x, y in zip([eval(backtests_table.loc[z, 'symbols']) for z in backtests_table.index], self.symbols)])) &
                                            (pd.Series([x == str(self.symbols) for x in backtests_table['symbols']], index=backtests_table.index)) &
                                            (backtests_table['start_date_dt'] == str(self.start_date_dt)) &
                                            (backtests_table['end_date_dt'] == str(self.end_date_dt)) &
                                            (backtests_table['rebalance_frequency'] == self.rebalance_frequency) &
                                            (backtests_table['long_count'] == self.long_count) &
                                            (backtests_table['short_count'] == self.short_count) &
                                            (backtests_table['portfolio_starting_value'] == self.portfolio_starting_value) &
                                            (backtests_table['risk_pct'] == self.risk_pct) &
                                            (backtests_table['reinvest_profits_bool'] == self.reinvest_profits_bool) &
                                            (backtests_table['leverage_multiplier'] == self.leverage_multiplier)][0]]

        this_backtest_filename = f'''Test_{this_backtest['test_number'].values[0]}.pkl'''

        return this_backtest_filename

    def get_backtest_filename(self):
        pickle_filename = f'backtests/{self.get_backtest_filename_without_path()}'
        return pickle_filename

    def analyze_backtest(self):
        if self.base_test is None:
            test_filename = self.get_backtest_filename()
            test = pickle.load(open(test_filename, 'rb'))
        else:
            test = self.base_test

        if len(test['backtest_profits']) < len(test['benchmark_returns']):
            test['rebalance_periods'] = test['rebalance_periods'][:len(test['backtest_profits'])]
            test['benchmark_returns'] = test['benchmark_returns'][:len(test['backtest_profits'])]

        period_length_in_days = (test['rebalance_periods'][0][1] - test['rebalance_periods'][0][0]).days
        periods_per_year = round(365/period_length_in_days)

        long_only = test['inputs']['long_count'] > 0 and test['inputs']['short_count'] == 0

        long_profits = [list(compress(x[-1], [y > 0 for y in np.float64(x[1])])) for x in test['backtest_runs']]
        long_profits = pd.Series([sum(np.float64(x)) for x in long_profits], index=[x[1] for x in test['rebalance_periods']])
        short_profits = [list(compress(x[-1], [y < 0 for y in np.float64(x[1])])) for x in test['backtest_runs']]
        short_profits = pd.Series([sum(np.float64(x)) for x in short_profits], index=[x[1] for x in test['rebalance_periods']])

        backtest_profits = pd.Series(test['backtest_profits'], index=[x[1] for x in test['rebalance_periods']])
        benchmark_profits = pd.Series(test['benchmark_returns'], index=[x[1] for x in test['rebalance_periods']])

        if test['inputs']['reinvest_profits_bool']:
            backtest_returns = backtest_profits/(
                    test['inputs']['portfolio_starting_value'] + pd.Series([1] + list(backtest_profits.values[:-1]),
                                                                           index=[x[1] for x in test['rebalance_periods']]))
            benchmark_returns = benchmark_profits/(
                    test['inputs']['portfolio_starting_value'] + pd.Series([1] + list(benchmark_profits.values[:-1]),
                                                                           index=[x[1] for x in test['rebalance_periods']]))
            long_returns = long_profits / (
                        test['inputs']['portfolio_starting_value'] + pd.Series([1] + list(long_profits.values[:-1]),
                                                                               index=[x[1] for x in test['rebalance_periods']]))
            short_returns = short_profits / (
                        test['inputs']['portfolio_starting_value'] + pd.Series([1] + list(short_profits.values[:-1]),
                                                                               index=[x[1] for x in test['rebalance_periods']]))
        else:
            backtest_returns = backtest_profits/test['inputs']['portfolio_starting_value']
            benchmark_returns = benchmark_profits/test['inputs']['portfolio_starting_value']
            long_returns = long_profits/test['inputs']['portfolio_starting_value']
            short_returns = short_profits/test['inputs']['portfolio_starting_value']

        up_bets = np.sum([(np.int64(np.float64(x[-1]) > 0)).sum() for x in test['backtest_runs']])
        down_bets = np.sum([(np.int64(np.float64(x[-1]) < 0)).sum() for x in test['backtest_runs']])
        total_bets = np.sum([len(x[-1]) for x in test['backtest_runs']])

        long_up_bets = np.sum(
            [(np.int64(np.float64(x[-1][np.float64(x[1]) > 0]) > 0)).sum() for x in test['backtest_runs']])
        long_down_bets = np.sum(
            [(np.int64(np.float64(x[-1][np.float64(x[1]) > 0]) < 0)).sum() for x in test['backtest_runs']])
        long_total_bets = np.sum([(len(np.float64(x[-1][np.float64(x[1]) > 0]))) for x in test['backtest_runs']])

        short_up_bets = np.sum(
            [(np.int64(np.float64(x[-1][np.float64(x[1]) < 0]) > 0)).sum() for x in test['backtest_runs']])
        short_down_bets = np.sum(
            [(np.int64(np.float64(x[-1][np.float64(x[1]) < 0]) < 0)).sum() for x in test['backtest_runs']])
        short_total_bets = np.sum([(len(np.float64(x[-1][np.float64(x[1]) < 0]))) for x in test['backtest_runs']])

        ##########################################################################################################################################################################

        total_report = qs.reports.full(backtest_returns, benchmark_returns, compounded=test['inputs']['reinvest_profits_bool'], title='Strategy', benchmark_title='Benchmark')

        ##########################################################################################################################################################################
        cagr = qs.stats.cagr(backtest_returns, compounded=test['inputs']['reinvest_profits_bool'])
        drawdowns = qs.stats.to_drawdown_series(backtest_returns)
        max_drawdown = qs.stats.max_drawdown(backtest_returns)
        win_rate = qs.stats.win_rate(backtest_returns)
        loss_rate = 1 - win_rate

        median_gain = benchmark_profits.loc[benchmark_profits > 0].median()
        average_loss = benchmark_profits.loc[benchmark_profits < 0].mean()
        median_gain_to_average_loss = median_gain / average_loss

        profit_factor = qs.stats.profit_factor(backtest_profits)
        sharpe = qs.stats.sharpe(backtest_returns, periods=periods_per_year)
        beta = qs.stats.greeks(backtest_returns, benchmark_returns).to_dict().get("beta", 0)
        sortino = qs.stats.sortino(backtest_returns, periods=periods_per_year)

        # Same stats as above but for benchmark
        benchmark_cagr = qs.stats.cagr(benchmark_returns, compounded=test['inputs']['reinvest_profits_bool'], periods=periods_per_year)
        benchmark_drawdowns = qs.stats.to_drawdown_series(benchmark_returns)
        benchmark_max_drawdown = qs.stats.max_drawdown(benchmark_returns)
        benchmark_win_rate = qs.stats.win_rate(benchmark_returns)
        benchmark_loss_rate = 1 - benchmark_win_rate

        benchmark_median_gain = benchmark_profits.loc[benchmark_profits > 0].median()
        benchmark_average_loss = benchmark_profits.loc[benchmark_profits < 0].mean()
        benchmark_median_gain_to_average_loss = benchmark_median_gain / benchmark_average_loss

        benchmark_profit_factor = qs.stats.profit_factor(benchmark_profits)
        benchmark_sharpe = qs.stats.sharpe(benchmark_returns, periods=periods_per_year)
        benchmark_beta = 1.0
        benchmark_sortino = qs.stats.sortino(benchmark_returns, periods=periods_per_year)

        # Same stats as above but for longs only
        long_cagr = qs.stats.cagr(long_returns, compounded=test['inputs']['reinvest_profits_bool'], periods=periods_per_year)
        long_drawdowns = qs.stats.to_drawdown_series(long_returns)
        long_max_drawdown = qs.stats.max_drawdown(long_returns)
        long_win_rate = qs.stats.win_rate(long_returns)
        long_loss_rate = 1 - long_win_rate

        long_median_gain = long_profits.loc[long_profits > 0].median()
        long_average_loss = long_profits.loc[long_profits < 0].mean()
        long_median_gain_to_average_loss = long_median_gain / long_average_loss

        long_profit_factor = qs.stats.profit_factor(long_profits)
        long_sharpe = qs.stats.sharpe(long_returns, periods=periods_per_year)
        long_beta = qs.stats.greeks(long_returns, benchmark_returns).to_dict().get("beta", 0)
        long_sortino = qs.stats.sortino(long_returns, periods=periods_per_year)

        # Same stats as above but for shorts only
        short_cagr = qs.stats.cagr(short_returns, compounded=test['inputs']['reinvest_profits_bool'], periods=periods_per_year)
        short_drawdowns = qs.stats.to_drawdown_series(short_returns)
        short_max_drawdown = qs.stats.max_drawdown(short_returns)
        short_win_rate = qs.stats.win_rate(short_returns)
        short_loss_rate = 1 - short_win_rate

        short_median_gain = short_profits.loc[short_profits > 0].median()
        short_average_loss = short_profits.loc[short_profits < 0].mean()
        short_median_gain_to_average_loss = short_median_gain / short_average_loss

        short_profit_factor = qs.stats.profit_factor(short_profits)
        short_sharpe = qs.stats.sharpe(short_returns, periods=periods_per_year)
        short_beta = qs.stats.greeks(short_returns, benchmark_returns).to_dict().get("beta", 0)
        short_sortino = qs.stats.sortino(short_returns, periods=periods_per_year)

        ##########################################################################################################################################################################

        test_score = lambda: np.random.random(1)[0]  # np.sum([y.values() for y in x])

        backtest_analysis = {
            'total': {
                'cagr': cagr,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'loss_rate': loss_rate,
                'pct_up_bets': up_bets / total_bets,
                'pct_down_bets': down_bets / total_bets,
                'median_gain_to_average_loss': median_gain_to_average_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe,
                'beta': beta,
                'sortino_ratio': sortino,
                'test_score': test_score(),
            },
            'long': {
                'cagr': long_cagr,
                'max_drawdown': long_max_drawdown,
                'win_rate': long_win_rate,
                'loss_rate': long_loss_rate,
                'pct_up_bets': long_up_bets / long_total_bets,
                'pct_down_bets': long_down_bets / long_total_bets,
                'median_gain_to_average_loss': long_median_gain_to_average_loss,
                'profit_factor': long_profit_factor,
                'sharpe_ratio': long_sharpe,
                'beta': long_beta,
                'sortino_ratio': long_sortino,
                'test_score': test_score(),
            },
            'short': {
                'cagr': short_cagr,
                'max_drawdown': short_max_drawdown,
                'win_rate': short_win_rate,
                'loss_rate': short_loss_rate,
                'pct_up_bets': short_up_bets / short_total_bets,
                'pct_down_bets': short_down_bets / short_total_bets,
                'median_gain_to_average_loss': short_median_gain_to_average_loss,
                'profit_factor': short_profit_factor,
                'sharpe_ratio': short_sharpe,
                'beta': short_beta,
                'sortino_ratio': short_sortino,
                'test_score': test_score(),
            },
            'benchmark': {
                'cagr': benchmark_cagr,
                'max_drawdown': benchmark_max_drawdown,
                'win_rate': benchmark_win_rate,
                'loss_rate': benchmark_loss_rate,
                'median_gain_to_average_loss': benchmark_median_gain_to_average_loss,
                'profit_factor': benchmark_profit_factor,
                'sharpe_ratio': benchmark_sharpe,
                'beta': benchmark_beta,
                'sortino_ratio': benchmark_sortino,
                'test_score': test_score(),
            }
        }

        # Pickle analysis
        pickle_filename = self.get_backtest_filename()[:-4] + '_analysis.pkl'
        print(pickle_filename)
        pickle.dump(backtest_analysis, open(pickle_filename, 'wb'))
        print(f'Backtest analysis saved to {pickle_filename}')

        return backtest_analysis

    def load_backtest(self):
        pickle_filename = self.get_backtest_filename()
        test = pickle.load(open(pickle_filename, 'rb'))
        print(f'Backtest results loaded from {pickle_filename}')

        return test


def compare_two_backtests(backtest1: BacktestAnalyzer, backtest2: BacktestAnalyzer):
    backtest1_analysis = backtest1.analyze_backtest()
    backtest2_analysis = backtest2.analyze_backtest()

    # Compare the two backtests
    comparison_columns = {'backtest1': {'strategy_name': backtest1.strategy_name, 'symbols': str(backtest1.symbols),
                                        'start_date_dt': backtest1.start_date_dt, 'end_date_dt': backtest1.end_date_dt,
                                        'rebalance_frequency': backtest1.rebalance_frequency,
                                        'long_count': backtest1.long_count, 'short_count': backtest1.short_count,
                                        'portfolio_starting_value': backtest1.portfolio_starting_value,
                                        'risk_pct': backtest1.risk_pct,
                                        'reinvest_profits_bool': backtest1.reinvest_profits_bool,
                                        'leverage_multiplier': backtest1.leverage_multiplier},
                          'backtest2': {'strategy_name': backtest2.strategy_name, 'symbols': str(backtest2.symbols),
                                        'start_date_dt': backtest2.start_date_dt, 'end_date_dt': backtest2.end_date_dt,
                                        'rebalance_frequency': backtest2.rebalance_frequency,
                                        'long_count': backtest2.long_count, 'short_count': backtest2.short_count,
                                        'portfolio_starting_value': backtest2.portfolio_starting_value,
                                        'risk_pct': backtest2.risk_pct,
                                        'reinvest_profits_bool': backtest2.reinvest_profits_bool,
                                        'leverage_multiplier': backtest2.leverage_multiplier}}

    # Compare comparison_columns and leave only the different columns
    backtest1_df = pd.DataFrame(comparison_columns['backtest1'], index=[0]).fillna(0)
    backtest2_df = pd.DataFrame(comparison_columns['backtest2'], index=[0]).fillna(0)
    comparison_columns = backtest1_df != backtest2_df
    comparison_columns = [', '.join([f'{x}={y}' for x, y in zip(comparison_columns.loc[:, comparison_columns.loc[0]].columns, backtest1_df.loc[0, comparison_columns.loc[0]])])] + (
        [', '.join([f'{x}={y}' for x, y in zip(comparison_columns.loc[:, comparison_columns.loc[0]].columns, backtest2_df.loc[0, comparison_columns.loc[0]])])]
    )

    total_comparison = pd.concat([pd.DataFrame(backtest1_analysis)['total'].T, pd.DataFrame(backtest2_analysis)['total'].T], axis=1)
    long_comparison = pd.concat([pd.DataFrame(backtest1_analysis)['long'].T, pd.DataFrame(backtest2_analysis)['long'].T], axis=1)
    short_comparison = pd.concat([pd.DataFrame(backtest1_analysis)['short'].T, pd.DataFrame(backtest2_analysis)['short'].T], axis=1)

    total_comparison.columns = comparison_columns
    long_comparison.columns = comparison_columns
    short_comparison.columns = comparison_columns

    total_comparison.sort_values(by='test_score', axis=1, ascending=False, inplace=True)
    long_comparison.sort_values(by='test_score', axis=1, ascending=False, inplace=True)
    short_comparison.sort_values(by='test_score', axis=1, ascending=False, inplace=True)

    total_comparison = total_comparison.set_axis(pd.MultiIndex.from_arrays([['total', 'total'], [comparison_columns[0], comparison_columns[1]]], names=['lst', 'test']), axis=1)
    long_comparison = long_comparison.set_axis(pd.MultiIndex.from_arrays([['long', 'long'], [comparison_columns[0], comparison_columns[1]]], names=['lst', 'test']), axis=1)
    short_comparison = short_comparison.set_axis(pd.MultiIndex.from_arrays([['short', 'short'], [comparison_columns[0], comparison_columns[1]]], names=['lst', 'test']), axis=1)

    total_comparison.fillna(0, inplace=True)
    long_comparison.fillna(0, inplace=True)
    short_comparison.fillna(0, inplace=True)

    return total_comparison, long_comparison, short_comparison, backtest1_analysis, backtest2_analysis
