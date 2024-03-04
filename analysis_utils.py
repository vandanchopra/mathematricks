from datetime import datetime, timedelta
import pickle
import numpy as np
from stock_automation_utils import StockAutomationUtils
from strategy_vault import StrategyVault
import pandas as pd
from tqdm import tqdm
from itertools import compress


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

    def get_backtest_filename(self):
        pickle_filename = f'backtests/{self.strategy.get_name()}_{"_".join(self.symbols)}_{"_".join([str(x) for x in [self.start_date_dt.date(), self.end_date_dt.date(), self.rebalance_frequency, self.long_count, self.short_count, self.portfolio_starting_value, self.risk_pct, self.reinvest_profits_bool]])}.pkl'
        return pickle_filename

    def analyze_backtest(self):
        test_filename = self.get_backtest_filename()
        test = pickle.load(open(test_filename, 'rb'))
        long_profits = [sum(compress(x['profit'], [(y > 0) for y in x['trading_symbols_multiplier_array']])) for x in
                        test['backtest_runs']]
        short_profits = [sum(compress(x['profit'], [(y < 0) for y in x['trading_symbols_multiplier_array']])) for x in
                         test['backtest_runs']]

        backtest_profits = test['backtest_profits']
        benchmark_returns = self.get_benchmark_returns(test['rebalance_periods'][0][0],
                                                       test['rebalance_periods'][-1][1],
                                                       test['rebalance_periods'],
                                                       test['inputs']['portfolio_starting_value'])
        benchmark_returns = benchmark_returns[:len(backtest_profits)]
        up_bets = np.sum([(x['profit'] > 0).count() for x in test['backtest_runs']])
        down_bets = np.sum([(x['profit'] < 0).count() for x in test['backtest_runs']])
        total_bets = [x.count() for x in test['backtest_runs']]

        # Calculate the CAGR
        cagr = (backtest_profits[-1] / backtest_profits[0]) ** (1 / len(backtest_profits)) - 1
        # Benchmark CAGR
        benchmark_cagr = (benchmark_returns[-1] / benchmark_returns[0]) ** (1 / len(benchmark_returns)) - 1
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

        # Same metrics as above for longs and shorts
        long_cagr = (np.sum(long_profits) / long_profits[0]) ** (1 / len(long_profits)) - 1
        long_cumulative_returns = np.cumsum(long_profits)
        long_drawdowns = np.maximum.accumulate(long_cumulative_returns) - long_cumulative_returns
        max_long_drawdown = np.max(long_drawdowns)
        long_profit_to_drawdown_ratio = np.sum(long_profits) / max_long_drawdown
        long_beta = np.cov(long_profits, benchmark_returns) / np.var(benchmark_returns)
        long_alpha = np.mean(long_profits) - long_beta * np.mean(benchmark_returns)
        long_sharpe_ratio = np.mean(long_profits) / np.std(long_profits)
        long_sortino_ratio = np.mean(long_profits) / np.std(long_profits[long_profits < 0])
        long_total_return = np.sum(long_profits)
        long_positive_returns = sum(long_profits > 0)
        long_negative_returns = sum(long_profits < 0)
        long_pct_positive_returns = long_positive_returns / len(long_profits)
        long_pct_negative_returns = long_negative_returns / len(long_profits)
        long_pct_profitable_bets = np.sum([(x > 0).count() for x in long_profits]) / [x.count() for x in long_profits]
        long_median_gain = np.median(long_profits[long_profits > 0])
        long_average_loss = np.mean(long_profits[long_profits < 0])
        long_median_gain_to_average_loss = long_median_gain / long_average_loss

        short_cagr = (np.sum(short_profits) / short_profits[0]) ** (1 / len(short_profits)) - 1
        short_cumulative_returns = np.cumsum(short_profits)
        short_drawdowns = np.maximum.accumulate(short_cumulative_returns) - short_cumulative_returns
        max_short_drawdown = np.max(short_drawdowns)
        short_profit_to_drawdown_ratio = np.sum(short_profits) / max_short_drawdown
        short_beta = np.cov(short_profits, benchmark_returns) / np.var(benchmark_returns)
        short_alpha = np.mean(short_profits) - short_beta * np.mean(benchmark_returns)
        short_sharpe_ratio = np.mean(short_profits) / np.std(short_profits)
        short_sortino_ratio = np.mean(short_profits) / np.std(short_profits[short_profits < 0])
        short_total_return = np.sum(short_profits)
        short_positive_returns = sum(short_profits > 0)
        short_negative_returns = sum(short_profits < 0)
        short_pct_positive_returns = short_positive_returns / len(short_profits)
        short_pct_negative_returns = short_negative_returns / len(short_profits)
        short_pct_profitable_bets = np.sum([(x > 0).count() for x in short_profits]) / [x.count() for x in
                                                                                        short_profits]
        short_median_gain = np.median(short_profits[short_profits > 0])
        short_average_loss = np.mean(short_profits[short_profits < 0])
        short_median_gain_to_average_loss = short_median_gain / short_average_loss

        test_score = lambda: np.random.random(1)[0]  # np.sum([y.values() for y in x])

        backtest_analysis = {
            'total': {
                'cagr': cagr,
                'cumulative_returns': cumulative_returns,
                'drawdowns': drawdowns,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'profit_to_drawdown_ratio': profit_to_drawdown_ratio,
                'beta': beta,
                'alpha': alpha,
                'total_return': total_return,
                'positive_returns': positive_returns,
                'negative_returns': negative_returns,
                'pct_positive_returns': pct_positive_returns,
                'pct_negative_returns': pct_negative_returns,
                'pct_profitable_bets': pct_profitable_bets,
                'pct_negative_bets': pct_negative_bets,
                'median_gain': median_gain,
                'average_loss': average_loss,
                'median_gain_to_average_loss': median_gain_to_average_loss,
                'test_score': test_score()
            },
            'long': {
                'cagr': long_cagr,
                'cumulative_returns': long_cumulative_returns,
                'drawdowns': long_drawdowns,
                'max_drawdown': max_long_drawdown,
                'sharpe_ratio': long_sharpe_ratio,
                'sortino_ratio': long_sortino_ratio,
                'profit_to_drawdown_ratio': long_profit_to_drawdown_ratio,
                'beta': long_beta,
                'alpha': long_alpha,
                'total_return': long_total_return,
                'positive_returns': long_positive_returns,
                'negative_returns': long_negative_returns,
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
                'cumulative_returns': short_cumulative_returns,
                'drawdowns': short_drawdowns,
                'max_drawdown': max_short_drawdown,
                'sharpe_ratio': short_sharpe_ratio,
                'sortino_ratio': short_sortino_ratio,
                'profit_to_drawdown_ratio': short_profit_to_drawdown_ratio,
                'beta': short_beta,
                'alpha': short_alpha,
                'total_return': short_total_return,
                'positive_returns': short_positive_returns,
                'negative_returns': short_negative_returns,
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
                'cumulative_returns': benchmark_cumulative_returns,
                'drawdowns': benchmark_drawdowns,
                'max_drawdown': max_benchmark_drawdown,
                'sharpe_ratio': benchmark_sharpe_ratio,
                'sortino_ratio': benchmark_sortino_ratio,
                'profit_to_drawdown_ratio': benchmark_profit_to_drawdown_ratio,
                'beta': 1.0,
                'alpha': 0.0,
                'total_return': benchmark_total_return,
                'positive_returns': benchmark_positive_returns,
                'negative_returns': benchmark_negative_returns,
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
        pickle_filename = f'backtests/{self.strategy_name}_{"_".join(test["inputs"]["symbols"])}_{"_".join([str(x) for x in list(test["inputs"].values())[1:]])}_analysis.pkl'
        pickle.dump(backtest_analysis, open(pickle_filename, 'wb'))
        print(f'Backtest analysis saved to {pickle_filename}')

        return backtest_analysis

