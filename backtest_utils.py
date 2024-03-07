import os
from datetime import datetime, timedelta
import pickle
from os import listdir
import numpy as np
from stock_automation_utils import StockAutomationUtils
from strategy_vault import StrategyVault
import pandas as pd
from tqdm import tqdm

class TradingSimulator:
    def __init__(self, strategy_name):
        self.strategy = None
        self.SAU = StockAutomationUtils()
        from vault.base_strategy import BaseStrategy as Strategy
        self.strategy = Strategy() 
        print('NOTICE: self.strategy: Need to use strategy_vault to automatically load the strategy based on strategy name (need to basically do a dynamic import')

    def calculate_rebalance_periods_with_dates(self, start_date_dt, end_date_dt, rebalance_frequency):
        # Initialize the list to hold the start and end dates of each rebalance period
        rebalance_dates = []

        current_start_date = start_date_dt
        while current_start_date < end_date_dt:
            # Calculate the end date of the current rebalance period
            current_end_date = min(current_start_date + timedelta(days=rebalance_frequency - 1), end_date_dt)

            # Append the start and end dates as a tuple to the list
            rebalance_dates.append((current_start_date.date(), current_end_date.date()))

            # Update the start date for the next rebalance period
            current_start_date = current_end_date + timedelta(days=1)

        return rebalance_dates

    def trading_simulation(self, portfolio_value, long_symbols, short_symbols, trading_data_interval, risk_pct, leverage_multiplier):
        def calculate_position_size(portfolio_value, trading_symbols_interval):
            position_size = (np.array(range(len(trading_symbols_interval))) + 1)
            position_size = portfolio_value/position_size
            position_size = (position_size / np.sum(position_size)) * portfolio_value
            position_size = position_size/sum(position_size)
            position_size = (np.sqrt(position_size) + position_size)/2
            position_size = position_size/sum(position_size)
            position_size = position_size * portfolio_value
            
            return position_size
        def get_entry_price(trading_symbols_interval, trading_data_interval):
            entry_price = []
            for symbol in trading_symbols_interval:
                entry_price.append(trading_data_interval[symbol]['Adj Close'].iloc[0])
                # print({'buy':symbol, 'buy_price':trading_data_interval[symbol]['Adj Close'].iloc[0]})
            return np.array(entry_price)
        def get_exit_price(long_symbols, short_symbols, trading_data_interval, stoploss):
            exit_price = []
            stoploss_hit_bool = []
            trading_symbols_interval = list(long_symbols) + list(short_symbols)
            for count, symbol in enumerate(trading_symbols_interval):
                if (symbol in long_symbols and trading_data_interval[symbol]['Adj Close'].min() < stoploss[count])\
                        or (symbol in short_symbols and trading_data_interval[symbol]['Adj Close'].max() > stoploss[count]):
                    exit_price.append(stoploss[count])
                    # print({'sell':symbol, 'sell_price':stoploss[count]})
                    stoploss_hit_bool.append(True)
                else:
                    exit_price.append(trading_data_interval[symbol]['Adj Close'].iloc[-1])
                    # print({'sell':symbol, 'sell_price':trading_data_interval[symbol]['Adj Close'].iloc[0]})
                    stoploss_hit_bool.append(False)
            return np.array(exit_price), stoploss_hit_bool

        trading_symbols_interval = list(long_symbols) + list(short_symbols)
        trading_symbols_multiplier_array = np.float64(np.array([1]*len(long_symbols) + [-1]*len(short_symbols)))

        position_size = calculate_position_size(portfolio_value, trading_symbols_interval)
        position_size *= trading_symbols_multiplier_array     # Short positions are negative

        entry_price = get_entry_price(trading_symbols_interval, trading_data_interval)
        qty = np.array(position_size*leverage_multiplier/entry_price, dtype=int)

        stoploss = np.array(entry_price * (1 - risk_pct * trading_symbols_multiplier_array))
        exit_price, stoploss_hit_bool = get_exit_price(long_symbols, short_symbols, trading_data_interval, stoploss)

        profit = (exit_price - entry_price)*qty     # qty is negative for short positions

        trading_simulation_array = np.array([trading_symbols_interval,
                                             trading_symbols_multiplier_array,
                                             entry_price,
                                             qty,
                                             stoploss,
                                             exit_price,
                                             stoploss_hit_bool,
                                             profit])

        profit_interval = np.sum(profit)
        # create a pandas dataframe with trading_symbols_interval as index, and buy_price, qty, stoploss, sell_price, profit as columns
        # trading_simulation_array = pd.DataFrame(trading_simulation_array.T, columns=['Symbol', 'Buy Price', 'Qty', 'Stoploss', 'Sell Price', 'Profit'])
        # print(trading_simulation_array)
        # raise AssertionError('MS')
        
        return trading_simulation_array, profit_interval

    def get_benchmark_returns(self, start_date_dt, end_date_dt, rebalance_periods, portfolio_starting_value, benchmark_name='^IXIC'):
        # Bring in the Nasdaq data and calculate the returns
        benchmark_input_data = self.SAU.get_stock_data(benchmark_name, start_date_dt, end_date_dt,)
        
        # Only keep the Adjussted Close column
        benchmark_input_data = benchmark_input_data[['Date', 'Adj Close']]
        # Convert 'Date' to datetime
        benchmark_input_data['Date'] = pd.to_datetime(benchmark_input_data['Date'])
        dates = np.array(rebalance_periods)[:, 1]
        # For each date in dates, find a date in nasdaq_data that is the same or the closest
        pruned_benchmark_data = []
        for date in dates:
            idx = benchmark_input_data['Date'].sub(date).abs().idxmin()
            pruned_benchmark_data.append(benchmark_input_data.loc[idx, 'Adj Close'])
        pruned_benchmark_data = np.array(pruned_benchmark_data)
        benchmark_returns = ((1-(pruned_benchmark_data[1:]/pruned_benchmark_data[:-1]))*portfolio_starting_value) * -1
        benchmark_returns = np.insert(benchmark_returns, 0, 0)
        
        return benchmark_returns

    def save_test(self, test):
        backtests_table_filepath = 'backtests_table.csv'
        if os.path.exists(backtests_table_filepath):
            backtests_table = pd.read_csv(backtests_table_filepath, index_col=[0])
            this_backtest_n = backtests_table.index.max() + 1
        else:
            backtests_table = pd.DataFrame(columns=['strategy_name', 'symbols', 'start_date_dt', 'end_date_dt', 'rebalance_frequency', 'long_count', 'short_count', 'portfolio_starting_value', 'risk_pct', 'reinvest_profits_bool', 'leverage_multiplier'])
            this_backtest_n = 1

        backtests_table.loc[this_backtest_n] = [test["strategy_name"], test["inputs"]["symbols"],
                                                test["inputs"]["start_date_dt"],
                                                test["inputs"]["end_date_dt"], test["inputs"]["rebalance_frequency"],
                                                test["inputs"]["long_count"], test["inputs"]["short_count"],
                                                test["inputs"]["portfolio_starting_value"], test["inputs"]["risk_pct"],
                                                test["inputs"]["reinvest_profits_bool"],
                                                test["inputs"]["leverage_multiplier"]]
        backtests_table.to_csv(backtests_table_filepath)

        pickle_filename = f'backtests/Test_{this_backtest_n}.pkl'
        pickle.dump(test, open(pickle_filename, 'wb'))
        print(f'Backtest results saved to {pickle_filename}')

    def run_backtest(self, symbols, start_date_dt, end_date_dt, rebalance_frequency, long_count, short_count, portfolio_starting_value, risk_pct, reinvest_profits_bool, leverage_multiplier, save_test: bool=False):
        # For each rebalance period, run the analysis and get long short stocks.
        rebalance_periods = self.calculate_rebalance_periods_with_dates(start_date_dt, end_date_dt, rebalance_frequency)
        rebalance_periods = [(datetime.combine(rebalance_periods[0], datetime.min.time()), datetime.combine(rebalance_periods[1], datetime.min.time())) for rebalance_periods in rebalance_periods]
        full_stock_data = self.SAU.load_all_stock_data(symbols, start_date_dt-timedelta(days=365+(rebalance_frequency*2)), end_date_dt+timedelta(days=rebalance_frequency*2))
        symbols = list(full_stock_data.keys())

        backtest_runs = []
        backtest_profits = []
        portfolio_value = portfolio_starting_value
        # Run loop with tqdm
        with tqdm(total=len(rebalance_periods)) as pbar:
            # Set size of the progress bar
            for rebal_interval in rebalance_periods:
                start_date_interval = rebal_interval[0]
                end_date_interval = rebal_interval[1]
                # Run the test for one interval
                # Load all stock data
                historical_data_interval = self.SAU.get_data_for_interval(symbols, full_stock_data, start_date_interval-timedelta(days=365), start_date_interval-timedelta(days=1))
                # Get the analysis array
                symbols_interval, data_analysis_interval, data_index_interval, dates_interval = self.strategy.get_analysis_array(symbols, start_date_interval, historical_data_interval, rebalance_frequency)
                # Get long and short stocks
                long_symbols, short_symbols, symbols_array, data_array, data_index = self.strategy.get_long_short_symbols(long_count, short_count, symbols_interval, data_analysis_interval, data_index_interval)
                # Get the trading symbols
                trading_symbols_interval = list(long_symbols) + list(short_symbols)
                # Get the trading data for the interval
                trading_data_interval = self.SAU.get_data_for_interval(trading_symbols_interval, full_stock_data, start_date_interval, end_date_interval)
                # Run the trades
                trading_simulation_array, profit_interval = self.trading_simulation(portfolio_value, long_symbols, short_symbols, trading_data_interval, risk_pct, leverage_multiplier)

                backtest_runs.append(trading_simulation_array)
                backtest_profits.append(profit_interval)
                pbar.set_description(f"Running backtest: {start_date_interval.date()} - {end_date_interval.date()}")
                pbar.set_postfix({'Profit': round(profit_interval, 2)})
                if profit_interval <= -portfolio_value:
                    break
                if reinvest_profits_bool:
                    portfolio_value += profit_interval

                pbar.update(1)
                
        test = {'strategy_name':self.strategy.get_name(), 
                'inputs':{'symbols':symbols, 'start_date_dt':start_date_dt.date(), 'end_date_dt':end_date_dt.date(), 'rebalance_frequency':rebalance_frequency, 'long_count':long_count, 'short_count':short_count, 'portfolio_starting_value':portfolio_starting_value, 'risk_pct':risk_pct, 'reinvest_profits_bool':reinvest_profits_bool, 'leverage_multiplier':leverage_multiplier},
                'rebalance_periods':rebalance_periods, 
                'backtest_runs': backtest_runs, 
                'backtest_profits': backtest_profits,
                'benchmark_returns': self.get_benchmark_returns(start_date_dt, end_date_dt, rebalance_periods, portfolio_starting_value)
                }

        if save_test:
            self.save_test(test)

        return test
