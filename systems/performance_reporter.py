from systems.utils import project_path
import os, json, pickle
from systems.utils import create_logger, sleeper, generate_hash_id
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta, datetime
import numpy as np
from colorama import Fore, Style

class PerformanceReporter:
    def __init__(self, market_data_extractor):
        self.market_data_extractor = market_data_extractor
        self.backtest_folder_path = project_path + 'db/vault/backtest_reports'
        self.backtest_performance_metrics = None
        self.backtest_report = None
        self.logger = create_logger(log_level='DEBUG', logger_name='REPORTER', print_to_console=True)
    
    def create_data_input_for_cumulative_returns_and_indices(self, trade_data, index_data_dict, starting_capital):
        # Extract profit/loss data
        trade_timestamps = []
        profit_loss = []
        
        for trade in trade_data:
            entry_order, exit_order = trade
                
            # Check order direction to determine long or short
            if entry_order['orderDirection'] == 'BUY':  # Long trade
                profit = (exit_order['fill_price'] - entry_order['fill_price']) * entry_order['orderQuantity']
            elif entry_order['orderDirection'] == 'SELL':  # Short trade
                profit = (entry_order['fill_price'] - exit_order['fill_price']) * entry_order['orderQuantity']
            else:
                profit = 0  # Handle any undefined order direction

            trade_timestamps.append(exit_order['filled_timestamp'])
            profit_loss.append(profit)
        
        # Create a DataFrame for sorting
        trade_df = pd.DataFrame({
            "filled_timestamp": trade_timestamps,
            "profit_loss": profit_loss
        })
        
        # Sort the data by exit timestamp
        trade_df = trade_df.sort_values(by="filled_timestamp").reset_index(drop=True)
        daily_profit_loss = trade_df.groupby("filled_timestamp")["profit_loss"].sum().reset_index()
        daily_profit_loss["cumulative_pnl"] = daily_profit_loss["profit_loss"].cumsum()
        daily_profit_loss["account_value"] = daily_profit_loss["cumulative_pnl"] + starting_capital
        daily_profit_loss["cumulative_return"] = ((daily_profit_loss["account_value"] / daily_profit_loss["account_value"].iloc[0]) - 1) * 100
        
        daily_profit_loss = daily_profit_loss.set_index("filled_timestamp")
        daily_profit_loss = daily_profit_loss.asfreq("D", method="ffill").reset_index()
        trade_df_by_date = daily_profit_loss
        trade_df_by_date.rename(columns={'filled_timestamp':'datetime'}, inplace=True)
        trade_df_by_date.set_index('datetime', inplace=True)
        
        return trade_df_by_date

    def calculate_unrealized_pnl(self, open_orders, unfilled_orders):
        unrealized_pnl_abs_dict = {}
        unrealized_pnl_pct_dict = {}
    
        for multi_leg_order in open_orders:
            symbol = multi_leg_order[0]['symbol']
            unrealized_pnl_abs, unrealized_pnl_pct = self.calculate_multi_leg_order_pnl(multi_leg_order, unfilled_orders, force_close=True)
            unrealized_pnl_abs_dict[symbol] = unrealized_pnl_abs
            unrealized_pnl_pct_dict[symbol] = unrealized_pnl_pct
            
        return unrealized_pnl_abs_dict, unrealized_pnl_pct_dict
    
    def calculate_multi_leg_order_pnl(self, multi_leg_order, unfilled_orders, force_close=False):
        entry_orders = []
        exit_orders = []
        entry_qty = 0
        exit_qty = 0
        total_profit = 0
        total_order_value = 0
        symbol = multi_leg_order[0]['symbol']
        for order in multi_leg_order:
            if 'entryPrice' in order and order['status'].lower() != 'cancelled':
                entry_orders.append(order)
                if order['status'] == 'closed' or (force_close and order['status'] == 'open'):
                    entry_qty += order['orderQuantity']
                    entry_price = order.get('fill_price') if 'fill_price' in order else 0
            elif 'exitPrice' in order and order['status'].lower() != 'cancelled':
                exit_orders.append(order)
                if order['status'] == 'closed' or (force_close and order['status'] in ['open', 'pending']):
                    exit_qty += order['orderQuantity']
        
        if force_close:
            for unfilled_order in unfilled_orders:
                if unfilled_order.contract.symbol == multi_leg_order[0]['symbol'] and unfilled_order.order.orderType.lower() == 'mkt':
                    # self.logger.debug((unfilled_order.contract.symbol, multi_leg_order[0]['symbol'], unfilled_order.order.orderType.lower()))
                    entry_order_direction = multi_leg_order[0]['orderDirection']
                    # self.logger.debug((unfilled_order.contract.symbol, unfilled_order.order.action.lower(), entry_order_direction.lower(), unfilled_order.order.totalQuantity))
                    if entry_order_direction.lower() == unfilled_order.order.action.lower():
                        entry_qty += unfilled_order.order.totalQuantity
                    else:
                        exit_qty += unfilled_order.order.totalQuantity
            
        # self.logger.debug({'symbol':order['symbol'], 'entry_qty':entry_qty, 'exit_qty':exit_qty})
        if float(entry_qty) == float(exit_qty):
            exit_price = None
            for exit_order in exit_orders:
                exit_price = exit_order.get('fill_price') if 'fill_price' in exit_order else list(exit_order['symbol_ltp'].values())[-1]
                if exit_price:
                    break
            # self.logger.debug({'Symbol':symbol, 'Exit Price':exit_price, 'Fill Price': exit_order.get('fill_price') if 'fill_price' in exit_order else None, 'symbol_ltp':exit_order['symbol_ltp'].values()})
            
            for entry_order in entry_orders:
                entry_price = entry_order.get('fill_price')
                order_entry_qty = entry_order.get('orderQuantity')
                entry_orderDirection = entry_order['orderDirection']
                entry_orderDirection_multiplier = 1 if entry_orderDirection == 'BUY' else -1
                entry_order_value = entry_price * order_entry_qty * entry_orderDirection_multiplier * -1
                exit_order_value = exit_price * order_entry_qty * entry_orderDirection_multiplier
                
                total_profit += (exit_order_value + entry_order_value)
                total_order_value += abs(entry_order_value)
                # self.logger.debug({'Symbol':entry_order['symbol'], 'Entry Order Direction': entry_orderDirection, 'Profit':total_profit, 'Entry Qty':entry_qty, 'Entry Price':entry_price, 'Exit Price':exit_price, 'Entry Value':entry_order_value, 'Exit Value':exit_order_value, 'Total Order Value':total_order_value})
        
        return round(total_profit, 10), total_profit / total_order_value if total_order_value > 0 else 0
    
    def calculate_cagr(self, trade_df_by_date):
        """Calculate the Compound Annual Growth Rate"""
        try:
            start_date = trade_df_by_date.index[0]
            end_date = trade_df_by_date.index[-1]
            days = (end_date - start_date).days
            if days > 0:
                cagr = (trade_df_by_date['account_value'].iloc[-1] / trade_df_by_date['account_value'].iloc[0]) ** (365.0 / days) - 1
                return cagr
            return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating CAGR: {e}")
            return 0.0

    def calculate_backtest_performance_metrics(self, config_dict, open_orders, closed_orders, market_data_df_root, unfilled_orders):
        self.backtest_performance_metrics = {}
        profit = 0
        profits_list = []
        losses_list = []
        long_list = []
        short_list = []
        win_count = 0
        loss_count = 0
        long_count = 0
        short_count = 0
        
        trade_df_by_date = None
        sharpe_ratio = 'NOT COMPUTED'
        cagr = 0.0
        
        # Process trades and calculate metrics
        for count, multi_leg_order in enumerate(closed_orders):
            signal_open_date = multi_leg_order[0]['timestamp']
            if signal_open_date > config_dict['backtest_inputs']['start_time'] and signal_open_date < config_dict['backtest_inputs']['end_time']:
                signal_profit, signal_profit_pct = self.calculate_multi_leg_order_pnl(multi_leg_order, unfilled_orders, force_close=False)
                profit += signal_profit
                if signal_profit >= 0:
                    win_count += 1
                    profits_list.append(signal_profit)
                else:
                    loss_count += 1
                    losses_list.append(signal_profit)

                if multi_leg_order[0]['orderDirection'] == 'BUY':
                    long_count += 1
                    long_list.append(signal_profit)
                elif multi_leg_order[0]['orderDirection'] == 'SELL':
                    short_count += 1
                    short_list.append(signal_profit)
                    
        # Calculate CAGR and Sharpe Ratio if there's enough data
        if len(closed_orders) > 0:
            starting_capital = 100000  # Example starting capital
            trade_data = [(order[0], order[1]) for order in closed_orders]
            # Get index data safely handling different DataFrame structures
            try:
                # Handle market data based on DataFrame structure
                # Handle different market data structures
                if isinstance(market_data_df_root.columns, pd.MultiIndex):
                    # For multi-index columns (symbol, field)
                    first_symbol = market_data_df_root.columns.get_level_values(0)[0]
                    index_data = {'close': market_data_df_root['close'][first_symbol]}
                else:
                    # For single-level columns
                    first_symbol = market_data_df_root.columns[0]
                    index_data = {'close': market_data_df_root[first_symbol]}
            except Exception as e:
                self.logger.error(f"Error processing market data: {e}")
                self.logger.debug(f"Market data structure: {market_data_df_root.head()}")
                index_data = {'close': pd.Series()}  # Empty series as fallback
            
            trade_df_by_date = self.create_data_input_for_cumulative_returns_and_indices(trade_data, index_data, starting_capital)
            
            if trade_df_by_date is not None and not trade_df_by_date.empty:
                try:
                    cagr = self.calculate_cagr(trade_df_by_date)
                    sharpe_ratio = round(self.calculate_sharpe_ratio(trade_df_by_date, index_data), 2)
                except Exception as e:
                    self.logger.error(f"Error calculating CAGR or Sharpe Ratio: {e}")
                    cagr = 0.0
                    sharpe_ratio = 'NOT COMPUTED'
        
        # Store metrics
        self.backtest_performance_metrics.update({
            'profit': profit,
            'win_pct': round((win_count / (win_count + loss_count)) * 100, 2) if (win_count + loss_count) > 0 else 0,
            'long_count': long_count,
            'short_count': short_count,
            'Average_Profit': round(sum(profits_list) / len(profits_list), 2) if len(profits_list) > 0 else 0,
            'Average_Loss': round(sum(losses_list) / len(losses_list), 2) if len(losses_list) > 0 else 0,
            'long_Average': round(sum(long_list) / len(long_list), 2) if len(long_list) > 0 else 0,
            'short_Average': round(sum(short_list) / len(short_list), 2) if len(short_list) > 0 else 0,
            'sharpe_ratio': sharpe_ratio,
            'cagr': round(cagr * 100, 2)  # Convert to percentage
        })
        
        return self.backtest_performance_metrics

    def generate_report(self):
        if not self.backtest_performance_metrics:
            return
            
        self.backtest_report = '\n'.join([
            f"profit: {self.backtest_performance_metrics['profit']}",
            f"win_pct: {self.backtest_performance_metrics['win_pct']}",
            f"long_count: {self.backtest_performance_metrics['long_count']}",
            f"short_count: {self.backtest_performance_metrics['short_count']}",
            f"Average_Profit: {self.backtest_performance_metrics['Average_Profit']}",
            f"Average_Loss: {self.backtest_performance_metrics['Average_Loss']}",
            f"long_Average: {self.backtest_performance_metrics['long_Average']}",
            f"short_Average: {self.backtest_performance_metrics['short_Average']}",
            f"sharpe_ratio: {self.backtest_performance_metrics['sharpe_ratio']}",
            f"cagr: {self.backtest_performance_metrics['cagr']}%"
        ])
        self.logger.info(f'Backtest Report: \n{self.backtest_report}')
    
    def save_backtest(self, config_dict, open_orders, closed_orders):
        testname = config_dict['backtest_inputs']['backtest_name'] if 'backtest_name' in config_dict['backtest_inputs'] else None
        if not testname:
            testname = generate_hash_id(config_dict['backtest_inputs'], 1)
        if testname:
            # Get all backtest folders that exist
            backtest_folders = os.listdir(self.backtest_folder_path) if os.path.exists(self.backtest_folder_path) else []
            if testname in backtest_folders:
                # If backtest folder already exists, then add a number to the end of the folder name
                i = 1
                while f'{testname}_{i}' in backtest_folders:
                    i += 1
                testname = f'{testname}_{i}'
            
            
            # Create folder if it doesn't exist
            self.test_folder_path = os.path.join(self.backtest_folder_path, testname)
            os.makedirs(self.test_folder_path, exist_ok=True)
            
            # Save the report as txt file for now and HTML file later
            if self.backtest_report:
                # with open(os.path.join(self.backtest_folder_path, testname, 'backtest_report.html'), 'w') as file:
                #     file.write(self.backtest_report)
                # save as txt for now
                with open(os.path.join(self.backtest_folder_path, testname, 'backtest_report.txt'), 'w') as file:
                    file.write(self.backtest_report)
            
            # Save open and closed orders as pickle
            backtest_output = {'open_orders':open_orders, 'closed_orders':closed_orders, 'config_dict':config_dict}
            backtest_output_path = os.path.join(self.backtest_folder_path, testname, 'backtest_output.pkl')
            with open(backtest_output_path, 'wb') as file:
                pickle.dump(backtest_output, file)
            
            # Save performance metrics as json
            if self.backtest_performance_metrics:
                with open(os.path.join(self.backtest_folder_path, testname, 'performance_metrics.json'), 'w') as file:
                    json.dump(self.backtest_performance_metrics, file)
        else:
            self.test_folder_path = 'None'
            
        return self.test_folder_path, testname

    def get_open_orders_print_msg(self, open_orders, total_buying_power, sequence_of_symbols):
        symbol_quantities = {}
        for order_pair in open_orders:
            for order in order_pair:
                if order['status'] == 'closed':
                    symbol = order['symbol']
                    order_quantity = order['orderQuantity']
                    if order['orderDirection'] == 'SELL':
                        order_quantity = -order_quantity
                    if symbol in symbol_quantities:
                        symbol_quantities[symbol]['orderQuantity'] += order_quantity
                    else:
                        if symbol not in symbol_quantities:
                            symbol_quantities[symbol] = {}
                        symbol_quantities[symbol]['orderQuantity'] = order_quantity

        # for order_pair in open_orders:
        #     for order in order_pair:
        #         if order['status'] in ['pending', 'open']:
        #             latest_price = list(order['symbol_ltp'].values())[-1]
        #             current_value = latest_price * symbol_quantities[symbol]['orderQuantity']
        #             symbol_quantities[symbol]['pct_of_total_buying_power'] = round(((current_value / total_buying_power) * 100), 2)
        
        # self.logger.debug({'symbol_quantities':symbol_quantities})
                
        for symbol in symbol_quantities:
            symbol_quantities[symbol]['pct_of_total_buying_power'] = None
            for order_pair in open_orders:
                for order in order_pair:
                    if order['symbol'] == symbol and order['status'] in ['pending', 'open']:
                        latest_price = list(order['symbol_ltp'].values())[-1]
                        current_value = latest_price * symbol_quantities[symbol]['orderQuantity']
                        symbol_quantities[symbol]['pct_of_total_buying_power'] = round(((current_value / total_buying_power) * 100), 2)
                if symbol_quantities[symbol]['pct_of_total_buying_power'] is not None:
                    break
                
        # self.logger.debug({'symbol_quantities':symbol_quantities})
        # Create the message string
        try:
            msg = ' | '.join([f"{Fore.BLUE}{symbol}{Style.RESET_ALL}: QTY: {symbol_quantities[symbol]['orderQuantity']}, % of PF: {symbol_quantities[symbol]['pct_of_total_buying_power']}%" for symbol in  sequence_of_symbols])
            msg = "Current Open Positions: " + msg
        except Exception as e:
            msg = "Current Open Positions: "
            self.logger.error({'symbol_quantities':symbol_quantities, 'sequence_of_symbols':sequence_of_symbols})
            
        return msg
    
    def get_stoploss_orders_print_msg(self, unfilled_orders, open_order, live_bool, sequence_of_symbols, market_data_df):
        symbol_quantities = {}
        if live_bool:
            for trade in unfilled_orders:
                symbol = trade.contract.symbol
                order_quantity = trade.order.totalQuantity
                sl_price = trade.order.auxPrice
                if trade.order.action == 'SELL':
                    order_quantity = -order_quantity

                min_granularity = self.market_data_extractor.get_market_data_df_minimum_granularity(market_data_df)
                close_prices = self.market_data_extractor.get_market_data_df_symbol_prices(market_data_df, min_granularity, symbol, 'close')
                # self.logger.debug({'symbol':symbol, 'close_prices':close_prices})
                close_prices.dropna(inplace=True)
                close_prices = close_prices.tolist()
                latest_price = close_prices[-1] if len(close_prices) > 0 else sl_price
                symbol_quantities[symbol] = (order_quantity, sl_price, latest_price)
        else:
            for order_pair in open_order:
                for order in order_pair:
                    if order['status'] in ['pending', 'open']:
                        symbol = order['symbol']
                        order_quantity = order['orderQuantity']
                        try:
                            sl_price = order['stoploss_abs']
                        except:
                            self.logger.debug({'order':order})
                        if order['orderDirection'] == 'SELL':
                            order_quantity = -order_quantity
                        latest_price = list(order['symbol_ltp'].values())[-1] if 'symbol_ltp' in order else None
                        symbol_quantities[symbol] = (order_quantity, sl_price, latest_price)

        # Create the message string
        msg = ' | '.join([f"{Fore.BLUE}{symbol}{Style.RESET_ALL}: {symbol_quantities[symbol][0]}, SL: {round(symbol_quantities[symbol][1], 2) if type(symbol_quantities[symbol][1]) != type(None) else None}, LTP: {symbol_quantities[symbol][2]}, SL Dist.: {round(((abs(symbol_quantities[symbol][2]-symbol_quantities[symbol][1])/symbol_quantities[symbol][1])* 100), 2)}%" for symbol in sequence_of_symbols])
        msg = "Current Stoploss Orders: " + msg
        
        return msg

    

class StrategyReporter:
    def load_backtest_output(self, test_folder_path):
        backtest_orders_path = os.path.join(test_folder_path, 'backtest_output.pkl')
        with open(backtest_orders_path, 'rb') as f:
            backtest_orders = pickle.load(f)
        return backtest_orders
    
    def calculate_multi_leg_order_profit(self, count, multi_leg_order):
        sell_orders = []
        buy_orders = []
        signal_id = multi_leg_order[0]['signal_id']

        buy_quantity = 0
        sell_quantity = 0
        # Separate out all sell and buy orders
        for order in multi_leg_order:
            symbol = order['symbol']
            if order['orderDirection'] == 'SELL' and order['status'] == 'closed':
                sell_orders.append(order)
                sell_quantity += order['orderQuantity']
            elif order['orderDirection'] == 'BUY' and order['status'] == 'closed':
                buy_orders.append(order)
                buy_quantity += order['orderQuantity']

        if buy_quantity == sell_quantity:
            total_profit = 0
            # Assuming orders are processed in pairs (e.g., first sell with first buy, second sell with second buy)
            for sell_order, buy_order in zip(sell_orders, buy_orders):
                sell_price = sell_order.get('fill_price')
                buy_price = buy_order.get('fill_price')
                quantity = buy_quantity #min(sell_order.get('orderQuantity'), buy_order.get('orderQuantity'))

                # Calculate profit for the matching sell and buy orders
                profit = (sell_price - buy_price) * quantity
                total_profit += profit
            # self.logger.info({'order':multi_leg_order})
            # self.logger.info(f"Total profit for multi-leg order: {total_profit}")
            # raise AssertionError('PnL calculation not implemented yet.') 
            return total_profit
        else:
            return 0

    def clean_up_order_list(self, nested_list, drop_history=False):
        #  We have removed the history key from both dict as they are not required for most calculation
        #  drop_history=False to keep the history
        for sublist in nested_list:
            # Process each dictionary in the sublist
            for i in range(len(sublist) - 1, -1, -1):  # Iterate in reverse to allow in-place removal
                d = sublist[i]
                
                # Remove dict with status 'cancelled'
                if d.get('status') == 'cancelled':
                    sublist.pop(i)
                    continue
                    
                if drop_history:
                    # Remove 'history' key if present
                    d.pop('history', None)
                
                # Update dict2 status from 'open' to 'closed'
                if d.get('status') == 'open':
                    d['status'] = 'closed'
                    
                    # Ensure 'symbol_ltp' is a dictionary or check if it has values()
                    if isinstance(d.get('symbol_ltp'), dict):
                        d['fill_price'] = list(d['symbol_ltp'].values())[-1]  # Safely access values
                        # Set 'filled_timestamp'
                        d['filled_timestamp'] = pd.Timestamp(datetime.strptime(list(d['symbol_ltp'].keys())[-1], '%Y-%m-%d %H:%M:%S%z'))
                    else:
                        # Handle missing or incorrect structure of symbol_ltp
                        d['fill_price'] = None  # or any default value
                        d['filled_timestamp'] = None
        
        # Filter out empty sublists
        return [sublist for sublist in nested_list if sublist]
    
    def load_and_save_index_data(self, indices, start_date, end_date, folder):

        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)
        index_files = {}  # Dictionary to store DataFrames for all indices

        for index_name in indices:
            filename = os.path.join(folder, f"{index_name}.csv")

            # Check if the file exists
            if os.path.exists(filename):
                # Read the CSV and parse dates
                data = pd.read_csv(filename, parse_dates=['datetime'], index_col='datetime')
            else:
                # Download max data from Yahoo Finance
                data = yf.download(index_name, period='max', interval="1d")

                if data.empty:
                    continue
                
                # Process downloaded data
                if isinstance(data.columns, pd.MultiIndex):  # Handle multi-index columns
                    data.columns = data.columns.droplevel(1)

                # Reset index and rename columns
                data.reset_index(inplace=True)
                data.rename(columns={'Date': 'datetime'}, inplace=True)
                data.set_index('datetime', inplace=True)
                data.columns = [col.lower() for col in data.columns]  # Lowercase column names
                
                # Save the processed data to CSV
                data.to_csv(filename)

            # Ensure the index is datetime and filter by date range
            data.index = pd.to_datetime(data.index)  # Ensure datetime index
            data = data[(data.index >= start_date) & (data.index <= end_date)]
            
            if data.empty:
                continue

            # Store the processed DataFrame in the dictionary
            index_files[index_name] = data

        return index_files

    def create_data_input_for_cumulative_returns_and_indices(self, trade_data, index_data_dict, starting_capital):
        # Extract profit/loss data
        trade_timestamps = []
        profit_loss = []
        
        for trade in trade_data:
            entry_order, exit_order = trade
                
            # Check order direction to determine long or short
            if entry_order['orderDirection'] == 'BUY':  # Long trade
                profit = (exit_order['fill_price'] - entry_order['fill_price']) * entry_order['orderQuantity']
            elif entry_order['orderDirection'] == 'SELL':  # Short trade
                profit = (entry_order['fill_price'] - exit_order['fill_price']) * entry_order['orderQuantity']
            else:
                profit = 0  # Handle any undefined order direction

            trade_timestamps.append(exit_order['filled_timestamp'])
            profit_loss.append(profit)
        
        # Create a DataFrame for sorting
        trade_df = pd.DataFrame({
            "filled_timestamp": trade_timestamps,
            "profit_loss": profit_loss
        })
        
        # Sort the data by exit timestamp
        trade_df = trade_df.sort_values(by="filled_timestamp").reset_index(drop=True)
        daily_profit_loss = trade_df.groupby("filled_timestamp")["profit_loss"].sum().reset_index()
        # # Calculate `cumulative_return` by cumulatively summing `profit_loss`
        daily_profit_loss["cumulative_pnl"] = daily_profit_loss["profit_loss"].cumsum()
        
        daily_profit_loss["account_value"] = daily_profit_loss["cumulative_pnl"] + starting_capital
        # pct growth for cumulative pnl
        daily_profit_loss["cumulative_return"] = ((daily_profit_loss["account_value"] / daily_profit_loss["account_value"].iloc[0]) - 1) * 100
        
        # # Forward-fill missing dates if needed
        daily_profit_loss = daily_profit_loss.set_index("filled_timestamp")
        daily_profit_loss = daily_profit_loss.asfreq("D", method="ffill").reset_index()
        # # Calculate cumulative profit/loss and returns
        # trade_df['cumulative_profit_loss'] = trade_df['profit_loss'].cumsum()
        # trade_df['cumulative_return'] = trade_df['cumulative_profit_loss'] 
        trade_df_by_date = daily_profit_loss
        # rename 'filled_timestamp' to 'datetime'
        trade_df_by_date.rename(columns={'filled_timestamp':'datetime'}, inplace=True)
        trade_df_by_date.set_index('datetime', inplace=True)
        
        return trade_df_by_date

    
    def plot_cumulative_returns_and_indices(self, trade_df, index_data_dict):
        min_max = [0 , 0]
        for index_data in index_data_dict.values():
            # Calculate percentage growth
            index_data["cumulative_return"] = ((index_data["close"] / index_data["close"].iloc[0]) - 1) * 100
            min_max[0] = min(min_max[0], index_data["cumulative_return"].min())
            min_max[1] = max(min_max[1], index_data["cumulative_return"].max())
        
        # Create figure
        fig = go.Figure()

        # Plot cumulative returns for trade data
        fig.add_trace(go.Scatter(
            x=trade_df.index,
            y=trade_df['cumulative_return'],
            mode='lines',
            name='Strategy',
            line=dict(color='red'),
            yaxis="y1"  # Use the left y-axis
        ))

        # Define colors for different indices
        index_colors = ['cyan', 'pink', 'orange']
        
        # Plot index growth
        for i, (index_name, df) in enumerate(index_data_dict.items()):
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['cumulative_return'],
                mode='lines',
                name=index_name,
                line=dict(color=index_colors[i % len(index_colors)]),  # Cycle through the colors
                yaxis="y2"  # Use the right y-axis
            ))

        # Update layout for dual y-axis with adjustments to ensure text and lines don't overlap
        cagr = self.calculate_cagr(trade_df)
        fig.update_layout(
            title=f"Backtest CAGR: {cagr * 100:.2f}%",
            xaxis=dict(
                title="Date",
                titlefont=dict(size=12),
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                title="Cumulative Return (Strategy)",
                titlefont=dict(color="white", size=12),
                tickfont=dict(color="white", size=10),
                range=[min(trade_df['cumulative_return']) * 0.99, max(trade_df['cumulative_return']) * 1.01],  # Adjust y-axis range
                side="right",
            ),
            yaxis2=dict(
                title="Cumulative Return (Indices)",
                titlefont=dict(color="green", size=12),
                tickfont=dict(color="green", size=10),
                overlaying="y",
                side="left",
                range=min_max
            ),
            template="plotly_dark",
            autosize=True
        )
        
        # Make the legend horizontal and on the top right, and outside the plot area
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="right", x=1))
        fig.show()
        
    ## Sharpe Ratio Calculation
    def calculate_sharpe_ratio(self, trade_df_cumulative, index_data_dict):
        # Extract strategy and benchmark data
        strategy_values = trade_df_cumulative['account_value']
        benchmark_values = index_data_dict['close']  # Access close prices directly
        
        # Convert index to datetime and align timestamps
        strategy_values.index = pd.to_datetime(strategy_values.index)
        benchmark_values.index = pd.to_datetime(benchmark_values.index)
        strategy_values.index = strategy_values.index + timedelta(seconds=1)  # Prevent exact match issues
        
        # Create DataFrames and merge
        merged_df = pd.merge(
            strategy_values.to_frame('strategy'),
            benchmark_values.to_frame('benchmark'),
            left_index=True,
            right_index=True,
            how='inner'
        ).astype(float)
        
        # Calculate returns
        strategy_returns = merged_df['strategy'].pct_change().fillna(0)
        benchmark_returns = merged_df['benchmark'].pct_change().fillna(0)
        
        # Calculate Sharpe ratio components
        excess_returns = strategy_returns - benchmark_returns
        annualization_factor = np.sqrt(252)  # Trading days in a year
        
        if len(excess_returns) == 0:
            return 0
            
        mean_excess_return = excess_returns.mean() * 252
        volatility = excess_returns.std() * annualization_factor
        
        # Return Sharpe ratio, handling division by zero
        if volatility == 0:
            return 0
            
        return mean_excess_return / volatility

    def calculate_sharpe_ratio(self, trade_df_cumulative, index_data_dict):
        """Calculate Sharpe Ratio for a strategy vs benchmark"""
        df1 = trade_df_cumulative['account_value']
        df2 = next(iter(index_data_dict.values()))['close']  # Get first index's close prices
        
        df1.index = pd.to_datetime(df1.index)
        df2.index = pd.to_datetime(df2.index)
        df1.index = df1.index + timedelta(seconds=1)  # Add 1 second to prevent exact match issues
        
        # Merge data and process
        df1_df = df1.to_frame()
        df2_df = df2.to_frame()
        merged_df = df1_df.merge(df2_df, left_index=True, right_index=True, how='inner')
        merged_df = merged_df.dropna()
        merged_df.columns = ['account_value', 'index']
        merged_df = merged_df.astype(float)
        
        # Calculate returns
        strategy_returns = merged_df['account_value'].pct_change().dropna()
        index_returns = merged_df['index'].pct_change().dropna()
        
        # Calculate Sharpe components
        excess_returns = strategy_returns - index_returns
        periods_per_year = 252  # Trading days in a year
        mean_excess_return = excess_returns.mean() * periods_per_year
        annualized_volatility = excess_returns.std() * np.sqrt(periods_per_year)
        
        # Return Sharpe ratio, handling division by zero
        return 0 if annualized_volatility == 0 else mean_excess_return / annualized_volatility

    def calculate_rolling_sharpe_ratio(self, trade_df_cumulative, index_data_dict, index_name, window):
        rolling_sharpe_data = []  # Initialize an empty list to store rows

        for i in range(window, len(trade_df_cumulative)):
            # Calculate Sharpe Ratio
            prev_i = i - window
            trimmed_trade_df = trade_df_cumulative.iloc[prev_i:i]
            trimmed_index_df = index_data_dict[index_name][str(trimmed_trade_df.index[0]):str(trimmed_trade_df.index[-1])]
            sharpe_ratio = self.calculate_sharpe_ratio(trimmed_trade_df, trimmed_index_df)

            # Append row data as a dictionary
            rolling_sharpe_data.append({
                'datetime': trade_df_cumulative.index[i],
                'rolling_sharpe_ratio': sharpe_ratio
            })

        # Convert list of dictionaries to DataFrame
        rolling_sharpe_df = pd.DataFrame(rolling_sharpe_data)
        rolling_sharpe_df.set_index('datetime', inplace=True)

        return rolling_sharpe_df

    def plot_rolling_sharpe_ratio(self, rolling_sharpe_df, index_data_dict):
        # Create a plotly figure for the rolling sharpe ratio
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling_sharpe_df.index, y=rolling_sharpe_df['rolling_sharpe_ratio'], mode='lines', name='Rolling Sharpe Ratio'))
        fig.update_layout(title='Rolling Sharpe Ratio', xaxis_title='Date', yaxis_title='Sharpe Ratio', showlegend=True, template="plotly_dark", autosize=True)
        fig.update_layout(xaxis=dict(range=[min(index_data_dict['^DJI'].index), max(index_data_dict['^DJI'].index)]))
        # Make the legend horizontal and on the top right, and outside the plot area
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="right", x=1))
        fig.show()
        
    def prepare_trade_data_df_by_order(self, final_list):
        """
        Convert final_list into a DataFrame, calculate profit/loss for each trade,
        and order the data according to exit_timestamp.
        """
        trade_data = []
        for trade_pair in final_list:
            entry_trade = trade_pair[0]
            exit_trade = trade_pair[1]

            try:
                entry_price = entry_trade['fill_price']
                exit_price = exit_trade.get('fill_price', entry_price)
                exit_timestamp = exit_trade['filled_timestamp']
                trade_type = 'long' if entry_trade['orderDirection'] == 'BUY' else 'short'
                profit_loss = (exit_price - entry_price) * entry_trade['orderQuantity'] if trade_type == 'long' else (entry_price - exit_price) * entry_trade['orderQuantity']
                trade_data.append([exit_timestamp, profit_loss, trade_type])
            except KeyError as e:
                continue

        # Convert to DataFrame
        trade_df = pd.DataFrame(trade_data, columns=['filled_timestamp', 'profit_loss', 'trade_type'])
        
        # Convert 'exit_timestamp' to datetime and floor to day
        trade_df['filled_timestamp'] = pd.to_datetime(trade_df['filled_timestamp']).dt.floor('D')
        
        # Sort by 'exit_timestamp'
        trade_df = trade_df.sort_values(by='filled_timestamp', ascending=True)

        # Ensure 'profit_loss' is in float format
        trade_df['profit_loss'] = trade_df['profit_loss'].astype('float32')

        return trade_df
    
    def plot_rolling_win_percentage(self, trade_df, index_data_dict, window=10):
        """
        Plot rolling win percentage using Plotly.

        Parameters:
            trade_df (pd.DataFrame): DataFrame containing trade data.
            window (int): Rolling window size for calculating win percentage.
        """
        # Calculate win/loss column
        trade_df['is_win'] = trade_df['profit_loss'] > 0
        
        # Calculate rolling win percentage
        trade_df['rolling_win_pct'] = trade_df['is_win'].rolling(window=window).mean() * 100

        # Plot using Plotly
        fig = px.line(
            trade_df,
            x='filled_timestamp',
            y='rolling_win_pct',
            title=f'Rolling Win Percentage (Window Size: {window})',
            labels={'rolling_win_pct': 'Rolling Win Percentage (%)', 'filled_timestamp': 'Filled Timestamp'},
            template='plotly_dark'
        )
        fig.update_layout(yaxis=dict(ticksuffix='%'))
        fig.update_layout(xaxis=dict(range=[min(index_data_dict['^DJI'].index), max(index_data_dict['^DJI'].index)]))
        # Make the legend horizontal and on the top right, and outside the plot area
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="right", x=1))
        fig.show()
        
    # calculate CAGR for the backtest
    def calculate_cagr(self, trade_df_by_date):
        # Calculate the CAGR for the backtest
        start_date = trade_df_by_date.index[0]
        end_date = trade_df_by_date.index[-1]
        days = (end_date - start_date).days
        cagr = (trade_df_by_date['account_value'].iloc[-1] / trade_df_by_date['account_value'].iloc[0]) ** (365.0 / days) - 1
        return cagr
    
    def calculate_drawdowns(self, trade_df_by_date):
        # calculate drawdowns in % based on 'Account Value' column
        trade_df_by_date['drawdown'] = trade_df_by_date['account_value'].cummax() - trade_df_by_date['account_value']
        trade_df_by_date['drawdown_pct'] = trade_df_by_date['drawdown'] / trade_df_by_date['account_value'] * 100 * -1
        return trade_df_by_date

    def plot_drawdown(self, trade_df_by_date, index_data_dict):
        # Underwater plot (area shaded for drawdown)
        underwater_fig = go.Figure()
        underwater_fig.add_trace(go.Scatter(
            x=trade_df_by_date.index,
            y=trade_df_by_date['drawdown_pct'],
            fill='tozeroy',  # Fill area to zero
            mode='none',
            name='Underwater',
            fillcolor='rgba(255, 0, 0, 0.3)'
        ))
        underwater_fig.update_layout(
            title= f"Underwater Plot: Max Drawdown: {trade_df_by_date['drawdown_pct'].min():.2f}%",
            xaxis=dict(title='Date'),
            yaxis=dict(title='Drawdown (%)'),
            legend=dict(x=0.01, y=0.99, borderwidth=1),
            yaxis_range=[-100, 0],
            template='plotly_dark'# Drawdown is always negative
        )
        underwater_fig.update_layout(xaxis=dict(range=[min(index_data_dict['^DJI'].index), max(index_data_dict['^DJI'].index)]))
        # Make the legend horizontal and on the top right, and outside the plot area
        underwater_fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="right", x=1))
        underwater_fig.show()
        
    def process_and_plot_orders(self, data, symbol, save_folder):
        # Step 1: Filter orders for the given symbol
        filtered_orders = [entry for entry in data if entry[0]['symbol'] == symbol]
        if not filtered_orders:
            raise ValueError(f"No orders found for symbol: {symbol}")
        
        # Step 2: Sort the filtered orders by exit order `filled_timestamp`
        sorted_orders = sorted(filtered_orders, key=lambda x: x[1]['filled_timestamp'])
        
        # Convert filled_timestamp to datetime for comparison
        first_timestamp = sorted_orders[0][1]['filled_timestamp'] - timedelta(days=365)
        last_timestamp = sorted_orders[-1][1]['filled_timestamp'] + timedelta(days=365)

        # first_timestamp = start_date.replace(tzinfo=None)
        # last_timestamp = end_date.replace(tzinfo=None)
        
        # Step 4: Get index data and load it from the CSV file
        data_dict = self.load_and_save_index_data([symbol], first_timestamp, last_timestamp, save_folder)
        index_df = data_dict[symbol].copy()
        index_df['close'] = index_df['close'][1:].astype(float)
        
        # Step 5: Prepare data for Plotly chart
        fig = go.Figure()
        
        # Add index line
        fig.add_trace(go.Scatter(
            x=index_df.index,
            y=index_df['close'],
            mode='lines',
            name='Index'
        ))
        
        # Add entry and exit markers
        for order in sorted_orders:
            entry_order = order[0]
            exit_order = order[1]
            
            # Determine entry marker
            if entry_order['orderDirection'].upper() == 'BUY':  # Long entry
                entry_marker = dict(color='green', size=10, symbol='triangle-up')
            else:  # Short entry
                entry_marker = dict(color='red', size=10, symbol='triangle-down')
            
            # Determine exit marker based on profit/loss
            if entry_order['orderDirection'].upper() == 'BUY':  # Long trade
                trade_result = exit_order['fill_price'] - entry_order['fill_price']
            else: 
                trade_result = entry_order['fill_price'] - exit_order['fill_price']
            
            if trade_result > 0: 
                exit_marker = dict(color='green', size=10, symbol='square')
            else:  # Loss
                exit_marker = dict(color='red', size=10, symbol='square')

            # Entry marker
            fig.add_trace(go.Scatter(
                x=[entry_order['filled_timestamp']],
                y=[entry_order['fill_price']],
                mode='markers',
                text=f"Entry: {entry_order['order_num']}",
                marker=entry_marker,
                name='Entry Order'
            ))
            
            # Exit marker
            fig.add_trace(go.Scatter(
                x=[exit_order['filled_timestamp']],
                y=[exit_order['fill_price']],
                mode='markers',
                text=f"Exit: {entry_order['order_num']}",
                marker=exit_marker,
                name='Exit Order'
            ))
                    
        # Set chart title and labels
        fig.update_layout(
            title=f'Order Analysis for Symbol: {symbol}',
            xaxis_title='Timestamp',
            yaxis_title='Price',
            template='plotly_dark'
        )
        
        # Remove the ledgend
        fig.update_layout(showlegend=False)
        
        # on the 
        
        # Display the chart
        fig.show()