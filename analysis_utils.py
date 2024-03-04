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

    def get_backtest_filename_without_path(self):
        return f'{self.strategy.get_name()}_{"_".join(self.symbols)}_{"_".join([str(x) for x in [self.start_date_dt, self.end_date_dt, self.rebalance_frequency, self.long_count, self.short_count, self.portfolio_starting_value, self.risk_pct, self.reinvest_profits_bool]])}.pkl'

    def get_backtest_filename(self):
        pickle_filename = f'backtests/{self.get_backtest_filename_without_path()}'
        return pickle_filename

    def analyze_backtest(self):
        test_filename = self.get_backtest_filename()
        test = pickle.load(open(test_filename, 'rb'))
        long_trades = [x * x['test['backtest_runs']


        backtest_profits = test['backtest_profits']
        benchmark_returns = self.get_benchmark_returns(test['rebalance_periods'][0][0], test['rebalance_periods'][-1][1],
                                                       test['rebalance_periods'],
                                                       test['inputs']['portfolio_starting_value'])
        benchmark_returns = benchmark_returns[:len(backtest_profits)]
        up_bets = np.sum([(x['profit'] > 0).count() for x in test['backtest_runs']])
        down_bets = np.sum([(x['profit'] < 0).count() for x in test['backtest_runs']])
        total_bets = [x.count() for x in test['backtest_runs']]

        # Calculate the CAGR
        cagr = (backtest_profits[-1] / backtest_profits[0]) ** (1 / len(backtest_profits)) - 1
        # Calculate the cumulative returns
        cumulative_returns = np.cumsum(backtest_profits)
        benchmark_cumulative_returns = np.cumsum(benchmark_returns)
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
        sortino_ratio = np.mean(backtest_profits) / np.std(backtest_profits[backtest_profits < 0])
        benchmark_sortino_ratio = np.mean(benchmark_returns) / np.std(benchmark_returns[benchmark_returns < 0])
        # Calculate the Profit to Drawdown Ratio
        profit_to_drawdown_ratio = np.sum(backtest_profits) / max_drawdown
        benchmark_profit_to_drawdown_ratio = np.sum(benchmark_returns) / max_benchmark_drawdown
        # Calculate the Profit to Max Drawdown Ratio
        profit_to_max_drawdown_ratio = np.sum(backtest_profits) / max_drawdown
        benchmark_profit_to_max_drawdown_ratio = np.sum(benchmark_returns) / max_benchmark_drawdown
        # Calculate beta
        beta = np.cov(backtest_profits, benchmark_returns) / np.var(benchmark_returns)
        # Calculate alpha
        alpha = np.mean(backtest_profits) - beta * np.mean(benchmark_returns)
        # Calculate the total return
        total_return = np.sum(backtest_profits)
        benchmark_total_return = np.sum(benchmark_returns)
        # Calculate the number of positive returns
        positive_returns = sum(backtest_profits > 0)
        benchmark_positive_returns = sum(benchmark_returns > 0)
        # Calculate the number of negative returns
        negative_returns = sum(backtest_profits < 0)
        benchmark_negative_returns = sum(benchmark_returns < 0)
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
        median_gain = np.median(backtest_profits[backtest_profits > 0])
        average_loss = np.mean(backtest_profits[backtest_profits < 0])
        median_gain_to_average_loss = median_gain / average_loss

        # Same metrics as above for longs


        test_score = lambda x: np.random.random(1)[0]  # np.sum([y.values() for y in x])

        backtest_analysis = {'cagr': cagr,
                             'cumulative_returns': cumulative_returns,
                             'benchmark_cumulative_returns': benchmark_cumulative_returns,
                             'drawdowns': drawdowns,
                             'benchmark_drawdowns': benchmark_drawdowns,
                             'max_drawdown': max_drawdown,
                             'max_benchmark_drawdown': max_benchmark_drawdown,
                             'sharpe_ratio': sharpe_ratio,
                             'benchmark_sharpe_ratio': benchmark_sharpe_ratio,
                             'sortino_ratio': sortino_ratio,
                             'benchmark_sortino_ratio': benchmark_sortino_ratio,
                             'profit_to_drawdown_ratio': profit_to_drawdown_ratio,
                             'benchmark_profit_to_drawdown_ratio': benchmark_profit_to_drawdown_ratio,
                             'profit_to_max_drawdown_ratio': profit_to_max_drawdown_ratio,
                             'benchmark_profit_to_max_drawdown_ratio': benchmark_profit_to_max_drawdown_ratio,
                             'beta': beta,
                             'alpha': alpha,
                             'total_return': total_return,
                             'benchmark_total_return': benchmark_total_return,
                             'positive_returns': positive_returns,
                             'benchmark_positive_returns': benchmark_positive_returns,
                             'negative_returns': negative_returns,
                             'benchmark_negative_returns': benchmark_negative_returns,
                             'pct_positive_returns': pct_positive_returns,
                             'benchmark_pct_positive_returns': benchmark_pct_positive_returns,
                             'pct_negative_returns': pct_negative_returns,
                             'benchmark_pct_negative_returns': benchmark_pct_negative_returns,
                             'pct_profitable_bets': pct_profitable_bets,
                             'pct_negative_bets': pct_negative_bets,
                             'median_gain': median_gain,
                             'average_loss': average_loss,
                             'median_gain_to_average_loss': median_gain_to_average_loss
                             }
        backtest_analysis['test_score'] = test_score(backtest_analysis)

        # Pickle analysis
        pickle_filename = f'backtests/{self.strategy.get_name()}_{"_".join(test["inputs"]["symbols"])}_{"_".join([str(x) for x in list(test["inputs"].values())[1:]])}_analysis.pkl'
        pickle.dump(backtest_analysis, open(pickle_filename, 'wb'))
        print(f'Backtest analysis saved to {pickle_filename}')

        return backtest_analysis


def compare_two_backtests(backtest1: BacktestAnalyzer, backtest2: BacktestAnalyzer):
    backtest1_analysis = pd.DataFrame(backtest1.analyze_backtest()).T
    backtest2_analysis = pd.DataFrame(backtest2.analyze_backtest()).T

    # Compare the two backtests
    comparison = pd.concat([backtest1_analysis, backtest2_analysis], axis=1)
    comparison_columns = [backtest1.get_backtest_filename_without_path()[:-4].split('_'), backtest2.get_backtest_filename_without_path()[:-4].split('_')]
    # Compare comparison_columns and leave only the different columns
    comparison_columns = [list(set(comparison_columns[0]) - set(comparison_columns[1])), list(set(comparison_columns[1]) - set(comparison_columns[0]))]
    comparison_columns = ['_'.join(comparison_columns[0]), '_'.join(comparison_columns[1])]
    comparison.columns = comparison_columns


    return comparison
