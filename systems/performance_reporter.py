from systems.utils import project_path
import os, json, pickle
from systems.utils import create_logger, sleeper, generate_hash_id
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import timedelta, datetime
import numpy as np
from scipy import stats
from colorama import Fore, Style

class PerformanceReporter:
    def __init__(self, market_data_extractor):
        self.market_data_extractor = market_data_extractor
        self.backtest_folder_path = project_path + 'db/vault/backtest_reports'
        self.backtest_performance_metrics = None
        self.backtest_report = None
        self.logger = create_logger(log_level='DEBUG', logger_name='REPORTER', print_to_console=True)
    
    def create_data_input_for_cumulative_returns_and_indices(self, trade_data, index_data_dict, starting_capital):
        """
        Create and validate trade data for metrics calculation with enhanced error handling.
        """
        try:
            self.logger.info("Starting trade data preparation...")
            
            # Validate input parameters
            if not trade_data:
                self.logger.error("No trade data provided")
                return None
                
            if not isinstance(starting_capital, (int, float)) or starting_capital <= 0:
                self.logger.error(f"Invalid starting capital: {starting_capital}")
                return None
            
            # Extract profit/loss data with validation
            trade_timestamps = []
            profit_loss = []
            
            for trade in trade_data:
                try:
                    entry_order, exit_order = trade
                    
                    # Ensure required fields exist and are numeric
                    if not all(key in entry_order and isinstance(entry_order[key], (int, float))
                             for key in ['fill_price', 'orderQuantity']):
                        self.logger.warning("Missing or invalid entry order fields")
                        continue
                        
                    if not all(key in exit_order and isinstance(exit_order[key], (int, float))
                             for key in ['fill_price']):
                        self.logger.warning("Missing or invalid exit order fields")
                        continue
                    
                    # Calculate profit based on order direction
                    if entry_order['orderDirection'] == 'BUY':  # Long trade
                        profit = (exit_order['fill_price'] - entry_order['fill_price']) * entry_order['orderQuantity']
                    elif entry_order['orderDirection'] == 'SELL':  # Short trade
                        profit = (entry_order['fill_price'] - exit_order['fill_price']) * entry_order['orderQuantity']
                    else:
                        self.logger.warning(f"Unknown order direction: {entry_order['orderDirection']}")
                        continue

                    trade_timestamps.append(exit_order['filled_timestamp'])
                    profit_loss.append(profit)
                except Exception as e:
                    self.logger.warning(f"Error processing trade: {e}")
                    continue

            if not trade_timestamps:
                self.logger.error("No valid trades found")
                return None

            # Create DataFrame with validated data
            try:
                trade_df = pd.DataFrame({
                    "filled_timestamp": pd.to_datetime(trade_timestamps),
                    "profit_loss": np.array(profit_loss, dtype=float)
                })
                
                # Sort and process the data
                trade_df = trade_df.sort_values(by="filled_timestamp").reset_index(drop=True)
                daily_profit_loss = trade_df.groupby("filled_timestamp")["profit_loss"].sum().reset_index()
                
                # Calculate and validate metrics
                daily_profit_loss["cumulative_pnl"] = daily_profit_loss["profit_loss"].cumsum()
                
                # Calculate account value and validate
                daily_profit_loss["account_value"] = daily_profit_loss["cumulative_pnl"] + starting_capital
                if (daily_profit_loss["account_value"] <= 0).any():
                    self.logger.error("Account value went negative, invalid trading scenario")
                    return None
                
                # Calculate returns for CAGR
                daily_profit_loss["daily_return"] = daily_profit_loss["account_value"].pct_change()
                daily_profit_loss.loc[daily_profit_loss.index[0], "daily_return"] = (daily_profit_loss["account_value"].iloc[0] / starting_capital) - 1
                
                # Calculate cumulative returns using compounded daily returns
                daily_profit_loss["cumulative_return"] = (1 + daily_profit_loss["daily_return"]).cumprod() - 1
                daily_profit_loss["cumulative_return"] *= 100  # Convert to percentage
                
                self.logger.info(f"""Account value validation:
                    Min: ${daily_profit_loss['account_value'].min():,.2f}
                    Max: ${daily_profit_loss['account_value'].max():,.2f}
                    Mean daily return: {daily_profit_loss['daily_return'].mean():.4%}
                """)
                
                # Set up proper datetime index
                daily_profit_loss = daily_profit_loss.set_index("filled_timestamp")
                daily_profit_loss = daily_profit_loss.asfreq("D", method="ffill").reset_index()
                trade_df_by_date = daily_profit_loss
                trade_df_by_date.rename(columns={'filled_timestamp':'datetime'}, inplace=True)
                trade_df_by_date.set_index('datetime', inplace=True)
                
                self.logger.info(f"""Trade data processing completed:
                    Valid trades processed: {len(trade_timestamps)}
                    Date range: {trade_df_by_date.index[0]} to {trade_df_by_date.index[-1]}
                    Starting value: ${starting_capital:,.2f}
                    Final value: ${trade_df_by_date['account_value'].iloc[-1]:,.2f}
                    Total PnL: ${trade_df_by_date['cumulative_pnl'].iloc[-1]:,.2f}
                """)
                
                return trade_df_by_date
                
            except Exception as e:
                self.logger.error(f"Error creating trade DataFrame: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in trade data preparation: {e}")
            return None

    def calculate_unrealized_pnl(self, open_orders, unfilled_orders):
        unrealized_pnl_abs_dict = {}
        unrealized_pnl_pct_dict = {}
    
        for multi_leg_order in open_orders:
            symbol = multi_leg_order[0]['symbol']
            unrealized_pnl_abs, unrealized_pnl_pct = self.calculate_multi_leg_order_pnl(multi_leg_order, unfilled_orders, force_close=True)
            unrealized_pnl_abs_dict[symbol] = unrealized_pnl_abs
            unrealized_pnl_pct_dict[symbol] = unrealized_pnl_pct
            
        return unrealized_pnl_abs_dict, unrealized_pnl_pct_dict
    
    def calculate_sharpe_ratio(self, trade_df_cumulative, index_data_dict):
        """Calculate Sharpe Ratio for a strategy vs benchmark with improved error handling"""
        try:
            # Input validation
            if trade_df_cumulative.empty or not index_data_dict:
                self.logger.warning("Empty data provided for Sharpe ratio calculation")
                return 0.0

            # Get strategy and benchmark data
            df1 = trade_df_cumulative['account_value']
            try:
                df2 = next(iter(index_data_dict.values()))['close']
            except (StopIteration, KeyError) as e:
                self.logger.error(f"Invalid benchmark data structure: {e}")
                return 0.0

            # Normalize both datetime indices to dates only
            df1.index = pd.to_datetime(df1.index).normalize()
            df2.index = pd.to_datetime(df2.index).normalize()
            
            # Prepare DataFrames for merging
            df1_df = df1.to_frame('account_value')
            df2_df = df2.to_frame('index')
            
            self.logger.info(f"Strategy dates after normalization: {df1_df.index.min()} to {df1_df.index.max()}")
            self.logger.info(f"Benchmark dates after normalization: {df2_df.index.min()} to {df2_df.index.max()}")
            
            # Log data ranges
            self.logger.info(f"Strategy data range: {df1_df.index.min()} to {df1_df.index.max()}")
            self.logger.info(f"Benchmark data range: {df2_df.index.min()} to {df2_df.index.max()}")
            
            # Merge on index
            merged_df = pd.merge(df1_df, df2_df, left_index=True, right_index=True, how='inner')
            
            self.logger.info(f"Merged data points: {len(merged_df)}")
            if merged_df.empty:
                self.logger.warning("No overlapping data points between strategy and benchmark")
                return 0.0
            
            # Log merged data range
            self.logger.info(f"Merged data range: {merged_df.index.min()} to {merged_df.index.max()}")
            
            merged_df = merged_df.dropna()
            
            # Ensure numeric values
            try:
                merged_df = merged_df.astype(float)
            except ValueError as e:
                self.logger.error(f"Data type conversion error: {e}")
                return 0.0
            
            # Sort and resample data
            merged_df = merged_df.sort_index()
            merged_df = merged_df.asfreq('D', method='ffill')
            
            self.logger.info(f"Merged data after resampling: {len(merged_df)} points")
            self.logger.info(f"Date range after resampling: {merged_df.index[0]} to {merged_df.index[-1]}")
            
            # Calculate returns on properly aligned data
            strategy_returns = merged_df['account_value'].pct_change()
            index_returns = merged_df['index'].pct_change()
            
            # Remove any NaN values that might occur at the start of the series
            valid_mask = ~(strategy_returns.isna() | index_returns.isna())
            strategy_returns = strategy_returns[valid_mask]
            index_returns = index_returns[valid_mask]
            
            self.logger.info(f"Valid returns data points: {len(strategy_returns)}")
            
            # Remove outliers and invalid values
            mask = ~(np.isnan(strategy_returns) | np.isinf(strategy_returns) |
                    np.isnan(index_returns) | np.isinf(index_returns))
            strategy_returns = strategy_returns[mask]
            index_returns = index_returns[mask]
            
            self.logger.info(f"""Returns Calculation:
                Date Range: {strategy_returns.index.min()} to {strategy_returns.index.max()}
                Strategy returns - Mean: {strategy_returns.mean():.4%}, Std: {strategy_returns.std():.4%}
                Index returns - Mean: {index_returns.mean():.4%}, Std: {index_returns.std():.4%}
                Valid data points: {len(strategy_returns)}
            """)
            
            if len(strategy_returns) < 2:
                self.logger.warning("Insufficient data points for Sharpe ratio calculation")
                return 0.0
            
            # Require minimum data points for statistical significance
            min_days = 30
            if len(strategy_returns) < min_days:
                self.logger.warning(f"Insufficient data points ({len(strategy_returns)}) for reliable metrics")
                return 0.0

            # Calculate annualized metrics with proper scaling
            actual_days = len(strategy_returns)
            ann_factor = np.sqrt(252 / actual_days)  # Adjust for partial year
            
            # Verify we have enough data points for statistical significance
            if len(strategy_returns) < 30:  # Minimum 30 days of data
                self.logger.warning(f"Insufficient data points for Sharpe ratio: {len(strategy_returns)}")
                return 0.0

            # Calculate excess returns over benchmark
            excess_returns = strategy_returns - index_returns
            
            # Compute annualized statistics
            mean_excess_return = excess_returns.mean() * 252
            daily_vol = excess_returns.std()
            
            # Validate volatility
            if daily_vol <= 0 or np.isnan(daily_vol):
                self.logger.warning(f"Invalid daily volatility: {daily_vol}")
                return 0.0
                
            annualized_volatility = daily_vol * np.sqrt(252) * ann_factor
            
            # Calculate and validate Sharpe ratio
            sharpe_ratio = mean_excess_return / annualized_volatility
            
            self.logger.info(f"""Sharpe Ratio Components:
                Date Range: {excess_returns.index[0]} to {excess_returns.index[-1]}
                Trading days: {actual_days}
                Annualization factor: {ann_factor:.2f}
                Daily Returns - Mean: {excess_returns.mean():.4%}, Std: {daily_vol:.4%}
                Annualized - Mean: {mean_excess_return:.4%}, Std: {annualized_volatility:.4%}
                Sharpe ratio: {sharpe_ratio:.4f}
            """)
            
            # Final validation
            if not -10 < sharpe_ratio < 10 or np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
                self.logger.warning(f"Sharpe ratio {sharpe_ratio:.4f} outside reasonable range (-10 to 10)")
                return 0.0
                
            return sharpe_ratio
            
        except Exception as e:
            self.logger.error(f"Error in Sharpe ratio calculation: {e}")
            return 0.0

    def calculate_drawdowns(self, trade_df_by_date):
        # calculate drawdowns in % based on 'Account Value' column
        trade_df_by_date['drawdown'] = trade_df_by_date['account_value'].cummax() - trade_df_by_date['account_value']
        trade_df_by_date['drawdown_pct'] = trade_df_by_date['drawdown'] / trade_df_by_date['account_value'] * 100 * -1
        return trade_df_by_date

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
        """
        Calculate CAGR with strict validation and proper time-based scaling
        """
        try:
            # Validate input data
            if trade_df_by_date is None or trade_df_by_date.empty:
                self.logger.warning("Empty DataFrame provided for CAGR calculation")
                return 0.0
            
            if 'account_value' not in trade_df_by_date.columns:
                self.logger.error("Missing required 'account_value' column")
                return 0.0
            
            # Ensure proper datetime index and sort
            trade_df_by_date.index = pd.to_datetime(trade_df_by_date.index)
            trade_df_by_date = trade_df_by_date.sort_index()
            
            # Get clean start and end values
            start_date = trade_df_by_date.index[0]
            end_date = trade_df_by_date.index[-1]
            start_value = float(trade_df_by_date['account_value'].iloc[0])
            end_value = float(trade_df_by_date['account_value'].iloc[-1])
            
            # Validate time period
            days = (end_date - start_date).days
            
            # Validate minimum trading period and values
            min_required_days = 5  # Require at least 5 days of data
            if days < min_required_days:
                self.logger.warning(f"Insufficient time period for CAGR: {days} days (minimum {min_required_days} required)")
                return 0.0
                
            # Validate values are reasonable
            if start_value <= 0 or end_value <= 0:
                self.logger.error(f"Invalid values: start=${start_value:,.2f}, end=${end_value:,.2f}")
                return 0.0
                
            # Check for unrealistic returns
            total_return = (end_value / start_value) - 1
            if abs(total_return) > 10:  # More than 1000% return or -90% loss
                self.logger.warning(f"Potentially unrealistic total return: {total_return:.2%}")
                
            # Only calculate CAGR for periods longer than a month
            min_days = 30
            if days < min_days:
                self.logger.warning(f"Trading period too short for meaningful CAGR: {days} days < {min_days}")
                return 0.0

            # Calculate CAGR with scaling and validation
            years = days / 365.0
            total_return = (end_value / start_value) - 1
            
            # Validate return is reasonable
            if abs(total_return) > 10:  # More than 1000% return or 90% loss
                self.logger.warning(f"Extreme return detected: {total_return:.2%}")
            
            # Calculate CAGR
            cagr = (end_value / start_value) ** (1.0 / years) - 1
            
            self.logger.info(f"""CAGR Calculation Details:
                Trading Period: {days} days ({years:.2f} years)
                Start: ${start_value:,.2f} ({start_date})
                End: ${end_value:,.2f} ({end_date})
                Total Return: {total_return:.2%}
                Annualized (CAGR): {cagr:.2%}
            """)
            
            if abs(cagr) > 5.0:  # More than 500% annual return
                self.logger.warning(f"Unusually high CAGR detected: {cagr:.2%}")
            
            # Validate result
            if np.isnan(cagr) or np.isinf(cagr):
                self.logger.error(f"Invalid CAGR calculated: {cagr}")
                return 0.0
            
            self.logger.info(f"Final CAGR: {cagr * 100:.2f}%")
            return cagr
            
        except Exception as e:
            self.logger.error(f"Error calculating CAGR: {str(e)}")
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
        sharpe_ratio = 0.0  # Initialize as float instead of string
        cagr = 0.0
        
        self.logger.info("Starting backtest performance calculation...")
        
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
        self.logger.info(f"Calculating CAGR and Sharpe ratio for {len(closed_orders)} orders")
        if len(closed_orders) >= 5:  # Require minimum number of trades
            starting_capital = 100000  # Example starting capital
            trade_data = [(order[0], order[1]) for order in closed_orders
                         if order[0]['status'] == 'closed' and order[1]['status'] == 'closed'
                         and 'fill_price' in order[0] and 'fill_price' in order[1]]
            
            if not trade_data:
                self.logger.warning("No valid closed trades found for metrics calculation")
                return self.backtest_performance_metrics
            
            # Get index data safely handling different DataFrame structures
            try:
                # Create benchmark index from market data with validation
                self.logger.info("Preparing benchmark data...")
                try:
                    if isinstance(market_data_df_root.columns, pd.MultiIndex):
                        # self.logger.info(f"Initial DataFrame structure:\n{market_data_df_root.head()}")
                        
                        # Handle multi-index rows if present
                        if isinstance(market_data_df_root.index, pd.MultiIndex):
                            # self.logger.info(f"Row index levels: {market_data_df_root.index.names}")
                            # Convert to single-level datetime index
                            market_data_df_root = market_data_df_root.reset_index()
                            market_data_df_root = market_data_df_root.set_index('datetime')
                            # self.logger.info(f"After index reset:\n{market_data_df_root.head()}")
                        
                        # Get symbols from the second level where first level is 'close'
                        close_symbols = [sym for col, sym in market_data_df_root.columns if col == 'close' and sym != '']
                        # self.logger.info(f"Found close symbols: {close_symbols}")
                        
                        if not close_symbols:
                            self.logger.error("No close price columns found")
                            return self.backtest_performance_metrics
                        
                        # Use first available symbol's close price as benchmark
                        first_symbol = close_symbols[0]
                        self.logger.info(f"Using close price for symbol: {first_symbol}")
                        # Extract benchmark data
                        benchmark_data = market_data_df_root.loc[:, ('close', first_symbol)].copy()
                        
                        # Convert string values to float if needed
                        benchmark_data = pd.to_numeric(benchmark_data, errors='coerce')
                        
                        # Create data input first to get the trade date range
                        trade_df_by_date = self.create_data_input_for_cumulative_returns_and_indices(trade_data, {'temp': pd.DataFrame()}, starting_capital)
                        if trade_df_by_date is None:
                            self.logger.error("Failed to create trade data")
                            return self.backtest_performance_metrics
                            
                        # Filter benchmark data to match strategy date range
                        trade_start_date = pd.to_datetime(trade_df_by_date.index.min())
                        trade_end_date = pd.to_datetime(trade_df_by_date.index.max())
                        
                        self.logger.info(f"Strategy date range: {trade_start_date} to {trade_end_date}")
                        self.logger.info(f"Initial benchmark date range: {benchmark_data.index.min()} to {benchmark_data.index.max()}")
                        
                        # Filter benchmark data to strategy date range
                        benchmark_data = benchmark_data[
                            (benchmark_data.index >= trade_start_date) &
                            (benchmark_data.index <= trade_end_date)
                        ]
                        
                        self.logger.info(f"Filtered benchmark date range: {benchmark_data.index.min()} to {benchmark_data.index.max()}")
                        self.logger.info(f"Benchmark data type: {benchmark_data.dtype}")
                    else:
                        first_symbol = market_data_df_root.columns[0]
                        self.logger.info(f"Using single-level column: {first_symbol}")
                        benchmark_data = market_data_df_root[first_symbol].copy()

                    # Ensure data is properly formatted
                    benchmark_data = pd.to_numeric(benchmark_data, errors='coerce')
                    benchmark_data = benchmark_data.dropna()
                    
                    # Validate benchmark data
                    if len(benchmark_data) < 2:
                        self.logger.error("Insufficient benchmark data points")
                        return self.backtest_performance_metrics
                        
                    # Check for valid range
                    if (benchmark_data == 0).any() or np.isinf(benchmark_data).any():
                        self.logger.error("Invalid values in benchmark data")
                        return self.backtest_performance_metrics
                except Exception as e:
                    self.logger.error(f"Error processing benchmark data: {e}")
                    return self.backtest_performance_metrics

                if benchmark_data.empty:
                    self.logger.error("No valid benchmark data available after cleaning")
                    return self.backtest_performance_metrics

                # Create index dictionary with cleaned benchmark data and proper time index
                # First create trade DataFrame with temporary index data to get date range
                trade_df_by_date = self.create_data_input_for_cumulative_returns_and_indices(trade_data, {"temp": pd.DataFrame()}, starting_capital)
                if trade_df_by_date is None or trade_df_by_date.empty:
                    self.logger.error("Failed to create valid trade data")
                    return self.backtest_performance_metrics

                # Create benchmark DataFrame with proper alignment to trade data
                benchmark_df = pd.DataFrame({'close': benchmark_data})
                # Normalize index to remove time component
                benchmark_df.index = pd.to_datetime(benchmark_df.index).normalize()
                
                # Filter benchmark data to match trade data range
                trade_start = pd.to_datetime(trade_df_by_date.index.min()).normalize()
                trade_end = pd.to_datetime(trade_df_by_date.index.max()).normalize()
                benchmark_df = benchmark_df[
                    (benchmark_df.index >= trade_start) &
                    (benchmark_df.index <= trade_end)
                ]
                
                # Ensure at least daily frequency
                benchmark_df = benchmark_df.asfreq('D', method='ffill')
                
                self.logger.info(f"""Benchmark Data Prepared:
                    Symbol: {first_symbol}
                    Start: {benchmark_df.index[0]}
                    End: {benchmark_df.index[-1]}
                    Points: {len(benchmark_df)}
                    First: {benchmark_df['close'].iloc[0]:.2f}
                    Last: {benchmark_df['close'].iloc[-1]:.2f}
                    Trade Date Range: {trade_start} to {trade_end}
                """)
                
                # Create final index data dictionary with aligned benchmark
                index_data = {first_symbol: benchmark_df}
                
                # Recreate trade data with aligned benchmark
                trade_df_by_date = self.create_data_input_for_cumulative_returns_and_indices(trade_data, index_data, starting_capital)
                
                if trade_df_by_date is not None and not trade_df_by_date.empty:
                    # Verify minimum data requirements
                    if len(trade_df_by_date) < 30:
                        self.logger.warning(f"Insufficient data points for metrics: {len(trade_df_by_date)}")
                        return self.backtest_performance_metrics

                    try:
                        # Calculate and validate CAGR
                        self.logger.info("Starting CAGR calculation...")
                        cagr = self.calculate_cagr(trade_df_by_date)
                        self.logger.info(f"Initial CAGR calculation result: {cagr}")
                        
                        # Verify CAGR is reasonable
                        if not isinstance(cagr, float) or np.isnan(cagr) or np.isinf(cagr):
                            self.logger.warning(f"Invalid CAGR value: {cagr}, resetting to 0.0")
                            cagr = 0.0
                        
                        # Calculate and validate Sharpe ratio
                        self.logger.info("Starting Sharpe ratio calculation...")
                        self.logger.info(f"Trade data points: {len(trade_df_by_date)}")
                        self.logger.info(f"Trade date range: {trade_df_by_date.index[0]} to {trade_df_by_date.index[-1]}")
                        self.logger.info(f"Benchmark data points: {len(index_data[next(iter(index_data))])}")
                        
                        sharpe_ratio = self.calculate_sharpe_ratio(trade_df_by_date, index_data)
                        self.logger.info(f"Initial Sharpe ratio calculation result: {sharpe_ratio}")
                        
                        if not isinstance(sharpe_ratio, float) or np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
                            self.logger.warning(f"Invalid Sharpe ratio value: {sharpe_ratio}, resetting to 0.0")
                            sharpe_ratio = 0.0
                            
                        # Store raw CAGR value (will be formatted as percentage in generate_report)
                        cagr = float(cagr)
                        sharpe_ratio = float(sharpe_ratio)
                        
                        self.logger.info(f"Calculated metrics - CAGR: {cagr:.4f}, Sharpe Ratio: {sharpe_ratio:.4f}")
                            
                    except Exception as e:
                        self.logger.error(f"Error calculating CAGR or Sharpe Ratio: {e}")
                        cagr = 0.0
                        sharpe_ratio = 0.0
                else:
                    cagr = 0.0
                    sharpe_ratio = 0.0
            except Exception as e:
                self.logger.error(f"Error processing market data: {e}")
                self.logger.debug(f"Market data structure: {market_data_df_root.head()}")
                cagr = 0.0
                sharpe_ratio = 0.0
        
        # Calculate and store metrics with proper handling of edge cases
        total_trades = win_count + loss_count
        win_percentage = (win_count / total_trades * 100) if total_trades > 0 else 0.0
        
        metrics_update = {
            'profit': round(profit, 2),
            'win_pct': round(win_percentage, 2),
            'long_count': long_count,
            'short_count': short_count,
            'Average_Profit': round(sum(profits_list) / len(profits_list), 2) if profits_list else 0.0,
            'Average_Loss': round(sum(losses_list) / len(losses_list), 2) if losses_list else 0.0,
            'long_Average': round(sum(long_list) / len(long_list), 2) if long_list else 0.0,
            'short_Average': round(sum(short_list) / len(short_list), 2) if short_list else 0.0,
            'sharpe_ratio': round(float(sharpe_ratio), 4),  # Store raw value, format in report
            'cagr': round(float(cagr), 4)  # Store raw value, format in report
        }
        
        # Log the calculated metrics for debugging
        self.logger.debug(f"Calculated metrics: {metrics_update}")
        
        # Update the performance metrics
        self.backtest_performance_metrics.update(metrics_update)
        
        return self.backtest_performance_metrics

    def generate_report(self):
        if not self.backtest_performance_metrics:
            self.logger.warning("No performance metrics available to generate report")
            return
        
        metrics = self.backtest_performance_metrics
        
        # Format numeric values with proper precision and currency symbols
        formatted_metrics = {
            'profit': f"${metrics['profit']:,.2f}",
            'win_pct': f"{metrics['win_pct']:.2f}%",
            'long_count': str(metrics['long_count']),
            'short_count': str(metrics['short_count']),
            'Average_Profit': f"${metrics['Average_Profit']:,.2f}",
            'Average_Loss': f"${metrics['Average_Loss']:,.2f}",
            'long_Average': f"${metrics['long_Average']:,.2f}",
            'short_Average': f"${metrics['short_Average']:,.2f}",
            'sharpe_ratio': f"{float(metrics['sharpe_ratio']):.2f}" if isinstance(metrics['sharpe_ratio'], (int, float)) else "N/A",
            'cagr': f"{float(metrics['cagr']):.2f}%" if isinstance(metrics['cagr'], (int, float)) else "N/A"
        }
        
        self.backtest_report = '\n'.join([
            "=== Backtest Performance Report ===",
            f"Total Profit/Loss: {formatted_metrics['profit']}",
            f"Win Rate: {formatted_metrics['win_pct']}",
            f"Number of Long Trades: {formatted_metrics['long_count']}",
            f"Number of Short Trades: {formatted_metrics['short_count']}",
            f"Average Winning Trade: {formatted_metrics['Average_Profit']}",
            f"Average Losing Trade: {formatted_metrics['Average_Loss']}",
            f"Average Long Trade: {formatted_metrics['long_Average']}",
            f"Average Short Trade: {formatted_metrics['short_Average']}",
            f"Sharpe Ratio: {formatted_metrics['sharpe_ratio']}",
            f"CAGR: {formatted_metrics['cagr']}"
        ])
        
        self.logger.info(f"Backtest Report: \n{self.backtest_report}")
    
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

def remove_all_content_after_class_performancereporter():
    """Remove all content after the PerformanceReporter class to clean up the file."""
    pass

class StrategyReporter(PerformanceReporter):
    """Analyze pairs trading strategy performance."""
    
    def __init__(self):
        super().__init__(type('MockExtractor', (), {}))
    
    def analyze_pairs_trading(self, trades):
        """Calculate performance metrics for pairs trading strategy."""
        df = self._process_trades(trades)
        if df.empty:
            return None
            
        metrics = self._calculate_metrics(df)
        self.logger.info(f"Pairs Trading Metrics: {metrics}")
        return metrics
        
    def _process_trades(self, trades):
        """Convert trades to DataFrame for analysis."""
        records = []
        for entry, exit in trades:
            if not (entry.get('fill_price') and exit.get('fill_price')):
                continue
            pnl = (exit['fill_price'] - entry['fill_price']) * entry['orderQuantity']
            if entry['orderDirection'] == 'SELL':
                pnl *= -1
            records.append({
                'timestamp': pd.to_datetime(exit['filled_timestamp']),
                'pnl': pnl,
                'direction': entry['orderDirection']
            })
            
        if not records:
            return pd.DataFrame()
            
        df = pd.DataFrame(records)
        df = df.set_index('timestamp').sort_index()
        df['cumulative_pnl'] = df['pnl'].cumsum()
        return df
        
    def _calculate_metrics(self, df):
        """Calculate key performance metrics."""
        return {
            'total_trades': len(df),
            'win_rate': (df['pnl'] > 0).mean() * 100,
            'total_pnl': df['pnl'].sum(),
            'avg_win': df[df['pnl'] > 0]['pnl'].mean() if len(df[df['pnl'] > 0]) else 0,
            'avg_loss': df[df['pnl'] < 0]['pnl'].mean() if len(df[df['pnl'] < 0]) else 0
        }

    def prepare_trade_data(self, orders):
        """Convert orders into analysis-ready DataFrame."""
        trades = []
        for entry, exit in orders:
            try:
                # Extract trade details
                timestamp = exit['filled_timestamp']
                direction = 'long' if entry['orderDirection'] == 'BUY' else 'short'
                quantity = entry['orderQuantity']
                entry_price = entry['fill_price']
                exit_price = exit['fill_price']
                
                # Calculate PnL
                pnl = self._calculate_trade_pnl(quantity, entry_price, exit_price, direction)
                
                trades.append({
                    'timestamp': timestamp,
                    'direction': direction,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl
                })
            except (KeyError, TypeError) as e:
                self.logger.warning(f"Skipping invalid trade: {str(e)}")
                continue
                
        # Convert to DataFrame
        if not trades:
            return pd.DataFrame()
            
        df = pd.DataFrame(trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Add derived metrics
        df['is_win'] = df['pnl'] > 0
        df['cumulative_pnl'] = df['pnl'].cumsum()
        
        return df

    def _calculate_trade_pnl(self, quantity, entry_price, exit_price, direction):
        """Calculate trade PnL based on direction."""
        if direction == 'long':
            return (exit_price - entry_price) * quantity
        return (entry_price - exit_price) * quantity

    def plot_performance_metrics(self, df, window=20):
        """Plot key performance metrics."""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Cumulative PnL', 'Rolling Win Rate', 'Trade PnL Distribution'],
            vertical_spacing=0.1,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Cumulative PnL
        fig.add_trace(
            go.Scatter(x=df.index, y=df['cumulative_pnl'],
                      name='Cumulative PnL',
                      line=dict(color='cyan')),
            row=1, col=1
        )
        
        # Rolling win rate
        rolling_win_pct = df['is_win'].rolling(window).mean() * 100
        fig.add_trace(
            go.Scatter(x=df.index, y=rolling_win_pct,
                      name=f'{window}-Trade Win Rate',
                      line=dict(color='green')),
            row=2, col=1
        )
        
        # PnL distribution
        fig.add_trace(
            go.Histogram(x=df['pnl'], name='PnL Distribution',
                        nbinsx=50, marker_color='blue'),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            title_text='Strategy Performance Analysis',
            showlegend=True,
            template='plotly_dark'
        )
        
        # Update axes
        fig.update_yaxes(title_text='PnL ($)', row=1, col=1)
        fig.update_yaxes(title_text='Win Rate %', row=2, col=1)
        fig.update_yaxes(title_text='Count', row=3, col=1)
        fig.update_xaxes(title_text='Trade PnL ($)', row=3, col=1)
        
        fig.show()
    def __init__(self):
        # Create a minimal market_data_extractor mock since we don't need it
        mock_extractor = type('MockExtractor', (), {})()
        super().__init__(mock_extractor)

    def load_backtest_output(self, test_folder_path):
        """Load backtest output from pickle file"""
        backtest_orders_path = os.path.join(test_folder_path, 'backtest_output.pkl')
        with open(backtest_orders_path, 'rb') as f:
            backtest_orders = pickle.load(f)
        return backtest_orders
    
    def clean_up_order_list(self, nested_list, drop_history=False):
        """Clean up order list by handling cancelled orders and updating statuses"""
        for sublist in nested_list:
            for i in range(len(sublist) - 1, -1, -1):
                d = sublist[i]
                if d.get('status') == 'cancelled':
                    sublist.pop(i)
                    continue
                if drop_history:
                    d.pop('history', None)
                if d.get('status') == 'open':
                    d['status'] = 'closed'
                    if isinstance(d.get('symbol_ltp'), dict):
                        d['fill_price'] = list(d['symbol_ltp'].values())[-1]
                        d['filled_timestamp'] = pd.Timestamp(datetime.strptime(list(d['symbol_ltp'].keys())[-1], '%Y-%m-%d %H:%M:%S%z'))
                    else:
                        d['fill_price'] = None
                        d['filled_timestamp'] = None
        return [sublist for sublist in nested_list if sublist]

    def process_and_plot_orders(self, data, symbol, save_folder):
        """Process and plot order data for a given symbol"""
        filtered_orders = [entry for entry in data if entry[0]['symbol'] == symbol]
        if not filtered_orders:
            raise ValueError(f"No orders found for symbol: {symbol}")
        sorted_orders = sorted(filtered_orders, key=lambda x: x[1]['filled_timestamp'])
        first_timestamp = sorted_orders[0][1]['filled_timestamp'] - timedelta(days=365)
        last_timestamp = sorted_orders[-1][1]['filled_timestamp'] + timedelta(days=365)
        data_dict = self.load_and_save_index_data([symbol], first_timestamp, last_timestamp, save_folder)
        index_df = data_dict[symbol].copy()
        index_df['close'] = index_df['close'][1:].astype(float)
        
        fig = self._create_order_plot(index_df, sorted_orders, symbol)
        fig.show()

    def _create_order_plot(self, index_df, sorted_orders, symbol):
        """Create plotly figure for order visualization"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=index_df.index, y=index_df['close'], mode='lines', name='Index'))
        
        for order in sorted_orders:
            entry_order, exit_order = order[0], order[1]
            entry_marker = self._get_entry_marker(entry_order)
            trade_result = self._calculate_trade_result(entry_order, exit_order)
            exit_marker = dict(color='green' if trade_result > 0 else 'red', size=10, symbol='square')
            
            fig.add_trace(go.Scatter(
                x=[entry_order['filled_timestamp']], y=[entry_order['fill_price']],
                mode='markers', text=f"Entry: {entry_order['order_num']}", marker=entry_marker, name='Entry Order'
            ))
            fig.add_trace(go.Scatter(
                x=[exit_order['filled_timestamp']], y=[exit_order['fill_price']],
                mode='markers', text=f"Exit: {entry_order['order_num']}", marker=exit_marker, name='Exit Order'
            ))
        
        fig.update_layout(
            title=f'Order Analysis for Symbol: {symbol}',
            xaxis_title='Timestamp', yaxis_title='Price',
            template='plotly_dark', showlegend=False
        )
        return fig

    def _get_entry_marker(self, entry_order):
        """Get marker style for entry order"""
        return dict(
            color='green' if entry_order['orderDirection'].upper() == 'BUY' else 'red',
            size=10,
            symbol='triangle-up' if entry_order['orderDirection'].upper() == 'BUY' else 'triangle-down'
        )

    def _calculate_trade_result(self, entry_order, exit_order):
        """Calculate trade result"""
        if entry_order['orderDirection'].upper() == 'BUY':
            return exit_order['fill_price'] - entry_order['fill_price']
        return entry_order['fill_price'] - exit_order['fill_price']

    def calculate_rolling_sharpe_ratio(self, trade_df_cumulative, index_data_dict, index_name, window):
        """Calculate rolling Sharpe ratio"""
        rolling_sharpe_data = []
        for i in range(window, len(trade_df_cumulative)):
            prev_i = i - window
            trimmed_trade_df = trade_df_cumulative.iloc[prev_i:i]
            trimmed_index_df = index_data_dict[index_name][str(trimmed_trade_df.index[0]):str(trimmed_trade_df.index[-1])]
            sharpe_ratio = self.calculate_sharpe_ratio(trimmed_trade_df, trimmed_index_df)
            rolling_sharpe_data.append({
                'datetime': trade_df_cumulative.index[i],
                'rolling_sharpe_ratio': sharpe_ratio
            })
        return pd.DataFrame(rolling_sharpe_data).set_index('datetime')

    def plot_rolling_win_percentage(self, trade_df, index_data_dict, window=10):
        """Plot rolling win percentage"""
        trade_df['is_win'] = trade_df['profit_loss'] > 0
        trade_df['rolling_win_pct'] = trade_df['is_win'].rolling(window=window).mean() * 100
        fig = px.line(
            trade_df,
            x='filled_timestamp', y='rolling_win_pct',
            title=f'Rolling Win Percentage (Window Size: {window})',
            labels={'rolling_win_pct': 'Rolling Win Percentage (%)', 'filled_timestamp': 'Filled Timestamp'},
            template='plotly_dark'
        )
        fig.update_layout(
            yaxis=dict(ticksuffix='%'),
            xaxis=dict(range=[min(index_data_dict['^DJI'].index), max(index_data_dict['^DJI'].index)]),
            legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="right", x=1)
        )
        fig.show()

    def prepare_trade_data_df_by_order(self, final_list):
        """Convert order list into DataFrame with profit/loss calculations"""
        trade_data = []
        for trade_pair in final_list:
            try:
                entry_trade, exit_trade = trade_pair[0], trade_pair[1]
                entry_price = entry_trade['fill_price']
                exit_price = exit_trade.get('fill_price', entry_price)
                exit_timestamp = exit_trade['filled_timestamp']
                trade_type = 'long' if entry_trade['orderDirection'] == 'BUY' else 'short'
                profit_loss = self._calculate_profit_loss(entry_trade, exit_trade)
                trade_data.append([exit_timestamp, profit_loss, trade_type])
            except KeyError:
                continue
        
        trade_df = pd.DataFrame(trade_data, columns=['filled_timestamp', 'profit_loss', 'trade_type'])
        trade_df['filled_timestamp'] = pd.to_datetime(trade_df['filled_timestamp']).dt.floor('D')
        return trade_df.sort_values('filled_timestamp', ascending=True).astype({'profit_loss': 'float32'})

    def load_and_save_index_data(self, indices, start_date, end_date, folder):
        """Load and save index data from Yahoo Finance"""
        os.makedirs(folder, exist_ok=True)
        index_files = {}
        
        for index_name in indices:
            filename = os.path.join(folder, f"{index_name}.csv")
            
            try:
                if os.path.exists(filename):
                    data = pd.read_csv(filename, parse_dates=['datetime'], index_col='datetime')
                else:
                    data = yf.download(index_name, period='max', interval="1d")
                    if data.empty:
                        continue
                    
                    # Process downloaded data
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.droplevel(1)
                    
                    data.reset_index(inplace=True)
                    data.rename(columns={'Date': 'datetime'}, inplace=True)
                    data.set_index('datetime', inplace=True)
                    data.columns = [col.lower() for col in data.columns]
                    data.to_csv(filename)
                
                # Filter data to requested date range
                data.index = pd.to_datetime(data.index)
                data = data[(data.index >= start_date) & (data.index <= end_date)]
                
                if not data.empty:
                    index_files[index_name] = data
                
            except Exception as e:
                self.logger.error(f"Error processing index {index_name}: {str(e)}")
                continue
                
        return index_files

    # Removed duplicate data input creation method - using parent class implementation


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
        
    # Removed duplicate Sharpe ratio calculation - using parent class implementation

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
    # Removed duplicate CAGR calculation - using parent class implementation
    
    def calculate_drawdowns(self, trade_df_by_date):
        """Calculate drawdowns with additional metrics and error handling"""
        try:
            if trade_df_by_date.empty or 'account_value' not in trade_df_by_date.columns:
                return trade_df_by_date
            
            # Calculate running maximum
            running_max = trade_df_by_date['account_value'].cummax()
            
            # Calculate drawdown in absolute terms
            trade_df_by_date['drawdown'] = running_max - trade_df_by_date['account_value']
            
            # Calculate drawdown percentage
            trade_df_by_date['drawdown_pct'] = (trade_df_by_date['drawdown'] / running_max) * 100 * -1
            
            # Calculate additional drawdown metrics
            trade_df_by_date['days_in_drawdown'] = 0
            trade_df_by_date['recovery_days'] = 0
            
            # Track drawdown periods
            in_drawdown = False
            drawdown_start = None
            
            for idx in range(len(trade_df_by_date)):
                if trade_df_by_date['drawdown'].iloc[idx] > 0:
                    if not in_drawdown:
                        drawdown_start = idx
                        in_drawdown = True
                    trade_df_by_date.iloc[idx, trade_df_by_date.columns.get_loc('days_in_drawdown')] = idx - drawdown_start
                else:
                    if in_drawdown:
                        recovery_days = idx - drawdown_start
                        trade_df_by_date.iloc[drawdown_start:idx, trade_df_by_date.columns.get_loc('recovery_days')] = recovery_days
                        in_drawdown = False
                        
            return trade_df_by_date
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdowns: {e}")
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