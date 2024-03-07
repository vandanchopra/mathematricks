from datetime import datetime, timedelta
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import compress


class BacktestAnalyzer:
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
                                            ([x == str(self.symbols) for x in backtests_table['symbols']]) &
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

        long_only = test['inputs']['long_count'] > 0 and test['inputs']['short_count'] == 0

        long_profits = [list(compress(x[-1], [y > 0 for y in np.float64(x[1])])) for x in test['backtest_runs']]
        long_profits = [sum(np.float64(x)) for x in long_profits]
        short_profits = [list(compress(x[-1], [y < 0 for y in np.float64(x[1])])) for x in test['backtest_runs']]
        short_profits = [sum(np.float64(x)) for x in short_profits]

        backtest_profits = test['backtest_profits']
        benchmark_returns = test['benchmark_returns']
        benchmark_returns = benchmark_returns[:len(backtest_profits)]

        up_bets = np.sum([(np.float64(x[-1]) > 0).sum() for x in test['backtest_runs']])
        down_bets = np.sum([(np.float64(x[-1]) < 0).sum() for x in test['backtest_runs']])
        total_bets = np.sum([len(np.float64(x[-1])) for x in test['backtest_runs']])

        long_up_bets = np.sum([len(np.float64(np.array(x[1])) > 0) for x in test['backtest_runs'] if [y > 0 for y in np.float64(x[1])]])
        long_down_bets = np.sum([len(np.float64(np.array(x[1])) > 0) for x in test['backtest_runs'] if [y > 0 for y in np.float64(x[1])]])
        long_total_bets = np.sum([len(np.float64(np.array(x[1])) > 0) for x in test['backtest_runs']])

        short_up_bets = np.sum([len(np.float64(np.array(x[1])) < 0) for x in test['backtest_runs'] if [y < 0 for y in np.float64(x[1])]])
        short_down_bets = np.sum([len(np.float64(np.array(x[1])) < 0) for x in test['backtest_runs'] if [y < 0 for y in np.float64(x[1])]])
        short_total_bets = np.sum([len(np.float64(np.array(x[1])) < 0) for x in test['backtest_runs']])

        # Calculate the cumulative returns
        cumulative_returns = np.cumsum(backtest_profits)
        benchmark_cumulative_returns = np.cumsum(benchmark_returns)
        # Calculate the CAGR
        cagr = (cumulative_returns[-1] / [x for x in cumulative_returns if x != 0][0]) ** (1 / len(cumulative_returns)) - 1
        # Benchmark CAGR
        benchmark_cagr = (benchmark_cumulative_returns[-1] / [x for x in benchmark_cumulative_returns if x != 0][0]) ** (1 / len(benchmark_cumulative_returns)) - 1
        # Calculate the drawdowns
        drawdowns = np.maximum.accumulate(cumulative_returns) - cumulative_returns
        benchmark_drawdowns = np.maximum.accumulate(benchmark_cumulative_returns) - benchmark_cumulative_returns
        # Calculate the max drawdowns
        max_drawdown = np.max(drawdowns)
        max_benchmark_drawdown = np.max(benchmark_drawdowns)
        # Calculate the Sharpe Ratio
        sharpe_ratio = np.mean(backtest_profits) / np.std(backtest_profits)
        benchmark_sharpe_ratio = np.mean(benchmark_returns) / np.std(benchmark_returns)
        # Calculate the Sortino Ratio
        sortino_ratio = np.mean(backtest_profits) / np.std([x for x in backtest_profits if x < 0])
        benchmark_sortino_ratio = np.mean(benchmark_returns) / np.std([x for x in benchmark_returns if x < 0])
        # Calculate the Profit to Drawdown Ratio
        profit_to_drawdown_ratio = np.sum(backtest_profits) / max_drawdown
        benchmark_profit_to_drawdown_ratio = np.sum(benchmark_returns) / max_benchmark_drawdown
        # Calculate beta
        beta = np.cov(backtest_profits, benchmark_returns)[0,1] / np.var(benchmark_returns)
        # Calculate alpha
        alpha = np.mean(backtest_profits) - beta * np.mean(benchmark_returns)
        # Calculate the total return
        total_return = np.sum(backtest_profits)
        benchmark_total_return = np.sum(benchmark_returns)
        # Calculate the number of positive returns
        positive_returns = sum([x for x in backtest_profits if x > 0])
        benchmark_positive_returns = sum([x for x in benchmark_returns if x > 0])
        # Calculate the number of negative returns
        negative_returns = sum([x for x in backtest_profits if x < 0])
        benchmark_negative_returns = sum([x for x in benchmark_returns if x < 0])
        # Calculate the % of positive returns
        pct_positive_returns = positive_returns / len(backtest_profits)
        benchmark_pct_positive_returns = benchmark_positive_returns / len(benchmark_returns)
        # Calculate the % of negative returns
        pct_negative_returns = negative_returns / len(backtest_profits)
        benchmark_pct_negative_returns = benchmark_negative_returns / len(benchmark_returns)
        # Calculate the % of bets that were profitable
        pct_profitable_bets = up_bets / total_bets
        pct_negative_bets = down_bets / total_bets
        # Calculate median gain/average loss
        median_gain = np.median([x for x in backtest_profits if x > 0])
        average_loss = np.mean([x for x in backtest_profits if x < 0])
        median_gain_to_average_loss = median_gain / average_loss

        # Same metrics as above for longs and shorts
        long_cumulative_returns = np.cumsum(long_profits)
        long_cagr = (long_cumulative_returns[-1] / [x for x in long_cumulative_returns if x != 0][0]) ** (1 / len(long_profits)) - 1
        long_drawdowns = np.maximum.accumulate(long_cumulative_returns) - long_cumulative_returns
        max_long_drawdown = np.max(long_drawdowns)
        long_profit_to_drawdown_ratio = np.sum(long_profits) / max_long_drawdown
        long_beta = np.cov(long_profits, benchmark_returns)[0,1] / np.var(benchmark_returns)
        long_alpha = np.mean(long_profits) - long_beta * np.mean(benchmark_returns)
        long_sharpe_ratio = np.mean(long_profits) / np.std(long_profits)
        long_sortino_ratio = np.mean(long_profits) / np.std([x for x in long_profits if x < 0])
        long_total_return = np.sum(long_profits)
        long_positive_returns = sum([x for x in long_profits if x > 0])
        long_negative_returns = sum([x for x in long_profits if x < 0])
        long_pct_positive_returns = long_positive_returns / len(long_profits)
        long_pct_negative_returns = long_negative_returns / len(long_profits)
        long_pct_profitable_bets = long_up_bets/long_total_bets
        long_median_gain = np.median([x for x in long_profits if x > 0])
        long_average_loss = np.mean([x for x in long_profits if x < 0])
        long_median_gain_to_average_loss = long_median_gain / long_average_loss

        if not long_only:
            short_cumulative_returns = np.cumsum(short_profits)
            try:
                short_cagr = (short_cumulative_returns[-1] / [x for x in short_cumulative_returns if x != 0][0]) ** (1 / len(short_profits)) - 1
            except:
                short_cagr = 0
            short_drawdowns = np.maximum.accumulate(short_cumulative_returns) - short_cumulative_returns
            max_short_drawdown = np.max(short_drawdowns)
            short_profit_to_drawdown_ratio = np.sum(short_profits) / max_short_drawdown
            short_beta = np.cov(short_profits, benchmark_returns)[0,1] / np.var(benchmark_returns)
            short_alpha = np.mean(short_profits) - short_beta * np.mean(benchmark_returns)
            short_sharpe_ratio = np.mean(short_profits) / np.std(short_profits)
            short_sortino_ratio = np.mean(short_profits) / np.std([x for x in short_profits if x < 0])
            short_total_return = np.sum(short_profits)
            short_positive_returns = sum([x for x in short_profits if x > 0])
            short_negative_returns = sum([x for x in short_profits if x < 0])
            short_pct_positive_returns = short_positive_returns / len(short_profits)
            short_pct_negative_returns = short_negative_returns / len(short_profits)
            short_pct_profitable_bets = short_up_bets/short_total_bets
            short_median_gain = np.median([x for x in short_profits if x > 0])
            short_average_loss = np.mean([x for x in short_profits if x < 0])
            short_median_gain_to_average_loss = short_median_gain / short_average_loss
        else:
            short_cumulative_returns = 0
            short_cagr = 0
            short_drawdowns = 0
            max_short_drawdown = 0
            short_profit_to_drawdown_ratio = 0
            short_beta = 0
            short_alpha = 0
            short_sharpe_ratio = 0
            short_sortino_ratio = 0
            short_total_return = 0
            short_positive_returns = 0
            short_negative_returns = 0
            short_pct_positive_returns = 0
            short_pct_negative_returns = 0
            short_pct_profitable_bets = 0
            short_median_gain = 0
            short_average_loss = 0
            short_median_gain_to_average_loss = 0

        test_score = lambda: np.random.random(1)[0]  # np.sum([y.values() for y in x])

        backtest_analysis = {
            'total': {
                'cagr': cagr,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'profit_to_drawdown_ratio': profit_to_drawdown_ratio,
                'beta': beta,
                'alpha': alpha,
                'total_return': total_return,
                'positive_returns': positive_returns,
                'negative_returns': negative_returns,
                'pct_profitable_bets': pct_profitable_bets,
                'pct_negative_bets': pct_negative_bets,
                'median_gain': median_gain,
                'average_loss': average_loss,
                'median_gain_to_average_loss': median_gain_to_average_loss,
                'test_score': test_score()
            },
            'long': {
                'cagr': long_cagr,
                'max_drawdown': max_long_drawdown,
                'sharpe_ratio': long_sharpe_ratio,
                'sortino_ratio': long_sortino_ratio,
                'profit_to_drawdown_ratio': long_profit_to_drawdown_ratio,
                'beta': long_beta,
                'alpha': long_alpha,
                'total_return': long_total_return,
                'pct_positive_returns': long_pct_positive_returns,
                'pct_negative_returns': long_pct_negative_returns,
                'pct_profitable_bets': long_pct_profitable_bets,
                'median_gain': long_median_gain,
                'average_loss': long_average_loss,
                'median_gain_to_average_loss': long_median_gain_to_average_loss,
                'test_score': test_score()
            },
            'short': {
                'cagr': short_cagr,
                'max_drawdown': max_short_drawdown,
                'sharpe_ratio': short_sharpe_ratio,
                'sortino_ratio': short_sortino_ratio,
                'profit_to_drawdown_ratio': short_profit_to_drawdown_ratio,
                'beta': short_beta,
                'alpha': short_alpha,
                'total_return': short_total_return,
                'pct_positive_returns': short_pct_positive_returns,
                'pct_negative_returns': short_pct_negative_returns,
                'pct_profitable_bets': short_pct_profitable_bets,
                'median_gain': short_median_gain,
                'average_loss': short_average_loss,
                'median_gain_to_average_loss': short_median_gain_to_average_loss,
                'test_score': test_score()
            },
            'benchmark': {
                'cagr': benchmark_cagr,
                'max_drawdown': max_benchmark_drawdown,
                'sharpe_ratio': benchmark_sharpe_ratio,
                'sortino_ratio': benchmark_sortino_ratio,
                'profit_to_drawdown_ratio': benchmark_profit_to_drawdown_ratio,
                'beta': 1.0,
                'alpha': 0.0,
                'total_return': benchmark_total_return,
                'pct_positive_returns': benchmark_pct_positive_returns,
                'pct_negative_returns': benchmark_pct_negative_returns,
                'pct_profitable_bets': np.nan,
                'median_gain': np.nan,
                'average_loss': np.nan,
                'median_gain_to_average_loss': np.nan,
                'test_score': test_score()
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
    backtest1_df = pd.DataFrame(comparison_columns['backtest1'], index=[0])
    backtest2_df = pd.DataFrame(comparison_columns['backtest2'], index=[0])
    comparison_columns = backtest1_df != backtest2_df
    print(comparison_columns.loc[0])
    comparison_columns = [', '.join([f'{x}={y}' for x, y in zip(comparison_columns.loc[:, comparison_columns.loc[0]].columns, backtest1_df.loc[0, comparison_columns.loc[0]])])] + (
        [', '.join([f'{x}={y}' for x, y in zip(comparison_columns.loc[:, comparison_columns.loc[0]].columns, backtest2_df.loc[0, comparison_columns.loc[0]])])]
    )
    print(comparison_columns)

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

    return total_comparison, long_comparison, short_comparison, backtest1_analysis, backtest2_analysis
