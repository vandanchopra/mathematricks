from systems.utils import project_path
from typing import List, Dict, Optional
import os, json, pickle
from systems.utils import create_logger, sleeper, generate_hash_id
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import timedelta, datetime
import numpy as np
from scipy import stats
from vault.base_strategy import Signal, Order
from colorama import Fore, Style

class Metrics:
    def __init__(self):
        self.logger = create_logger(log_level='DEBUG', logger_name='Metrics', print_to_console=True)
        
    def calculate_signal_pnl(self, signal, force_close=False):
        """Calculate realized or unrealized PnL for a signal
        Args:
            signal: Signal object containing orders
            force_close: If True, calculate unrealized PnL at current market price
        Returns:
            tuple (pnl_abs, pnl_pct): Absolute and percentage PnL
        """
        total_investment = 0
        total_pnl = 0

        for order in signal.orders:
            if order.status == 'filled' or (force_close and order.status in ['pending', 'open']):
                # Calculate investment and return for each order
                quantity = order.orderQuantity
                entry_price = order.filled_price if order.filled_price else order.price
                
                if force_close and order.status in ['pending', 'open']:
                    # Use latest price from symbol_ltp for unrealized PnL
                    exit_price = list(order.symbol_ltp.values())[-1] if order.symbol_ltp else entry_price
                else:
                    exit_price = order.filled_price
                
                if order.orderDirection == 'BUY':
                    order_pnl = quantity * (exit_price - entry_price)
                else:  # SELL
                    order_pnl = quantity * (entry_price - exit_price)
                
                total_pnl += order_pnl
                total_investment += abs(quantity * entry_price)

        pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
        return total_pnl, pnl_pct
    
    def calculate_signal_unrealized_pnl(self, signal):
        """Calculate unrealized PnL for a signal considering unfilled orders
        Args:
            signal: Signal object to calculate unrealized PnL for
            unfilled_orders: List of unfilled orders to consider
        Returns:
            float: Unrealized PnL for the signal
        """
        return self.calculate_signal_pnl(signal, force_close=True)
    
    def calculate_backtest_winning_signals_count(self, closed_signals):
        """Count number of winning signals (positive PnL) from backtest
        Args:
            closed_signals: List of closed Signal objects
        Returns:
            int: Number of winning signals
        """
        return sum(1 for signal in closed_signals if signal.pnl and signal.pnl > 0)
    
    def calculate_backtest_losing_signals_count(self, closed_signals):
        """Count number of losing signals (negative PnL) from backtest"""
        return sum(1 for signal in closed_signals if signal.pnl and signal.pnl < 0)
    
    def calculate_backtest_average_signal_duration(self, closed_signals):
        """Calculate average duration of signals in days"""
        durations = []
        for signal in closed_signals:
            if signal.orders:
                # Find earliest entry and latest exit
                entry_time = min(order.filled_timestamp for order in signal.orders if order.filled_timestamp)
                exit_time = max(order.filled_timestamp for order in signal.orders if order.filled_timestamp)
                if entry_time and exit_time:
                    duration = (exit_time - entry_time).total_seconds() / 3600  # Convert to hours
                    durations.append(duration)
        return np.mean(durations) if durations else 0
    
    def calculate_backtest_average_signal_duration_profitable(self, closed_signals):
        profitable_signals = [signal for signal in closed_signals if signal.pnl and signal.pnl > 0]
        return self.calculate_backtest_average_signal_duration(profitable_signals)
    
    def calculate_backtest_average_signal_duration_loss(self, closed_signals):
        loss_signals = [signal for signal in closed_signals if signal.pnl and signal.pnl < 0]
        return self.calculate_backtest_average_signal_duration(loss_signals)
    
    def calculate_backtest_average_signal_return(self, closed_signals):
        """Calculate average absolute return across all signals"""
        returns = [signal.pnl for signal in closed_signals if signal.pnl is not None]
        return np.mean(returns) if returns else 0
    
    def calculate_backtest_average_signal_return_pct(self, closed_signals):
        """Calculate average percentage return across all signals"""
        pct_returns = []
        for signal in closed_signals:
            total_investment = sum(abs(order.orderQuantity * order.filled_price) 
                                 for order in signal.orders 
                                 if order.filled_price)
            if total_investment > 0 and signal.pnl is not None:
                pct_return = (signal.pnl / total_investment) * 100
                pct_returns.append(pct_return)
        return np.mean(pct_returns) if pct_returns else 0
    
    def calculate_backtest_average_signal_return_pct_per_day(self, closed_signals):
        """Calculate average daily percentage return"""
        daily_returns = []
        for signal in closed_signals:
            if signal.orders and signal.pnl is not None:
                entry_time = min(order.filled_timestamp for order in signal.orders if order.filled_timestamp)
                exit_time = max(order.filled_timestamp for order in signal.orders if order.filled_timestamp)
                if entry_time and exit_time:
                    duration_days = (exit_time - entry_time).total_seconds() / (24 * 3600)
                    if duration_days > 0:
                        total_investment = sum(abs(order.orderQuantity * order.filled_price) 
                                            for order in signal.orders 
                                            if order.filled_price)
                        if total_investment > 0:
                            pct_return = (signal.pnl / total_investment) * 100
                            daily_return = pct_return / duration_days
                            daily_returns.append(daily_return)
        return np.mean(daily_returns) if daily_returns else 0
    
    def calculate_backtest_average_win_signals_return(self, closed_signals):
        """Calculate average return of winning signals"""
        win_returns = [signal.pnl for signal in closed_signals if signal.pnl and signal.pnl > 0]
        return np.mean(win_returns) if win_returns else 0
    
    def calculate_backtest_average_loss_signals_return_pct(self, closed_signals):
        """Calculate average percentage return of losing signals"""
        loss_pct_returns = []
        for signal in closed_signals:
            if signal.pnl and signal.pnl < 0:
                total_investment = sum(abs(order.orderQuantity * order.filled_price) 
                                     for order in signal.orders 
                                     if order.filled_price)
                if total_investment > 0:
                    pct_return = (signal.pnl / total_investment) * 100
                    loss_pct_returns.append(pct_return)
        return np.mean(loss_pct_returns) if loss_pct_returns else 0
    
    def calculate_backtest_sharpe_ratio(self, closed_signals):
        """Calculate Sharpe ratio (assuming risk-free rate of 0)"""
        daily_returns = []
        for signal in closed_signals:
            if signal.orders and signal.pnl is not None:
                entry_time = min(order.filled_timestamp for order in signal.orders if order.filled_timestamp)
                exit_time = max(order.filled_timestamp for order in signal.orders if order.filled_timestamp)
                if entry_time and exit_time:
                    duration_days = (exit_time - entry_time).total_seconds() / (24 * 3600)
                    if duration_days > 0:
                        total_investment = sum(abs(order.orderQuantity * order.filled_price) 
                                            for order in signal.orders 
                                            if order.filled_price)
                        if total_investment > 0:
                            daily_return = (signal.pnl / total_investment) / duration_days
                            daily_returns.append(daily_return)
        
        if not daily_returns:
            return 0
            
        returns_array = np.array(daily_returns)
        excess_returns = returns_array - 0  # Assuming 0 risk-free rate
        if len(excess_returns) > 1:
            sharpe_ratio = np.sqrt(252) * (np.mean(excess_returns) / np.std(excess_returns, ddof=1))
            return sharpe_ratio
        return 0
    
    def calculate_backtest_sortino_ratio(self, closed_signals):
        """Calculate Sortino ratio (considering only downside volatility)"""
        daily_returns = []
        for signal in closed_signals:
            if signal.orders and signal.pnl is not None:
                entry_time = min(order.filled_timestamp for order in signal.orders if order.filled_timestamp)
                exit_time = max(order.filled_timestamp for order in signal.orders if order.filled_timestamp)
                if entry_time and exit_time:
                    duration_days = (exit_time - entry_time).total_seconds() / (24 * 3600)
                    if duration_days > 0:
                        total_investment = sum(abs(order.orderQuantity * order.filled_price) 
                                            for order in signal.orders 
                                            if order.filled_price)
                        if total_investment > 0:
                            daily_return = (signal.pnl / total_investment) / duration_days
                            daily_returns.append(daily_return)
        
        if not daily_returns:
            return 0
            
        returns_array = np.array(daily_returns)
        excess_returns = returns_array - 0  # Assuming 0 risk-free rate
        
        if len(excess_returns) > 1:
            # Calculate downside deviation (standard deviation of negative returns only)
            negative_returns = excess_returns[excess_returns < 0]
            if len(negative_returns) > 0:
                downside_std = np.std(negative_returns, ddof=1)
                if downside_std > 0:
                    sortino_ratio = np.sqrt(252) * (np.mean(excess_returns) / downside_std)
                    return sortino_ratio
        return 0
    
    def calculate_backtest_calmar_ratio(self, closed_signals):
        """Calculate Calmar ratio (average return / maximum drawdown)"""
        max_dd = self.calculate_backtest_max_drawdown(closed_signals)
        avg_return = self.calculate_backtest_average_signal_return_pct_per_day(closed_signals)
        return abs(avg_return / max_dd) if max_dd != 0 else 0
    
    def calculate_backtest_max_drawdown(self, closed_signals):
        """Calculate maximum drawdown percentage"""
        # Sort signals by entry time
        sorted_signals = sorted(closed_signals, 
                              key=lambda x: min(order.filled_timestamp for order in x.orders if order.filled_timestamp))
        
        cumulative_returns = np.cumsum([signal.pnl for signal in sorted_signals if signal.pnl])
        if len(cumulative_returns) > 0:
            rolling_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (rolling_max - cumulative_returns) / rolling_max
            max_drawdown = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0
            return max_drawdown
        return 0
    
    def calculate_backtest_max_drawdown_duration(self, closed_signals):
        """Calculate maximum drawdown duration in days"""
        # Sort signals by entry time
        sorted_signals = sorted(closed_signals, 
                              key=lambda x: min(order.filled_timestamp for order in x.orders if order.filled_timestamp))
        
        cumulative_returns = np.cumsum([signal.pnl for signal in sorted_signals if signal.pnl])
        if len(cumulative_returns) > 0:
            rolling_max = np.maximum.accumulate(cumulative_returns)
            drawdown_starts = np.where(rolling_max != cumulative_returns)[0]
            
            if len(drawdown_starts) > 0:
                max_duration = 0
                for start_idx in drawdown_starts:
                    # Find where cumulative returns get back to the previous peak
                    peak_value = rolling_max[start_idx]
                    recovery_idx = np.where(cumulative_returns[start_idx:] >= peak_value)[0]
                    if len(recovery_idx) > 0:
                        duration = (sorted_signals[start_idx + recovery_idx[0]].orders[-1].filled_timestamp - 
                                  sorted_signals[start_idx].orders[0].filled_timestamp).days
                        max_duration = max(max_duration, duration)
                return max_duration
        return 0
    
    def calculate_backtest_profit_factor(self, closed_signals):
        """Calculate ratio of gross profits to gross losses"""
        gross_profits = sum(signal.pnl for signal in closed_signals if signal.pnl and signal.pnl > 0)
        gross_losses = abs(sum(signal.pnl for signal in closed_signals if signal.pnl and signal.pnl < 0))
        return gross_profits / gross_losses if gross_losses != 0 else float('inf')
    
    def calculate_backtest_win_loss_ratio(self, closed_signals):
        """Calculate ratio of winning trades to losing trades"""
        winning_count = self.calculate_backtest_winning_signals_count(closed_signals)
        losing_count = self.calculate_backtest_losing_signals_count(closed_signals)
        return winning_count / losing_count if losing_count > 0 else float('inf')
    
class PerformanceReporter(Metrics):
    def __init__(self, market_data_extractor):
        self.market_data_extractor = market_data_extractor
        self.backtest_folder_path = project_path + 'db/vault/backtest_reports'
        self.backtest_performance_metrics = None
        self.backtest_report = None
        self.logger = create_logger(log_level='DEBUG', logger_name='REPORTER', print_to_console=True)
    
    def calculate_backtest_performance_metrics(self, config_dict: Dict, open_signals: List[Signal], closed_signals: List[Signal], market_data_df: pd.DataFrame, unfilled_orders: List[Order]) -> Dict:
        """Calculate various performance metrics for the backtest"""
        try:
            # Calculate performance metrics
            metrics = {}
            metrics['total_signals'] = len(open_signals) + len(closed_signals)
            metrics['open_signals'] = len(open_signals)
            metrics['closed_signals'] = len(closed_signals)
            metrics['winning_signals'] = self.calculate_backtest_winning_signals_count(closed_signals)
            metrics['losing_signals'] = self.calculate_backtest_losing_signals_count(closed_signals)
            metrics['average_signal_duration'] = self.calculate_backtest_average_signal_duration(closed_signals)
            metrics['average_signal_duration_profitable'] = self.calculate_backtest_average_signal_duration_profitable(closed_signals)
            metrics['average_signal_duration_loss'] = self.calculate_backtest_average_signal_duration_loss(closed_signals)
            metrics['average_signal_return'] = self.calculate_backtest_average_signal_return(closed_signals)
            metrics['average_signal_return_pct'] = self.calculate_backtest_average_signal_return_pct(closed_signals)
            metrics['average_signal_return_pct_per_day'] = self.calculate_backtest_average_signal_return_pct_per_day(closed_signals)
            metrics['average_win_signals_return'] = self.calculate_backtest_average_win_signals_return(closed_signals)
            metrics['average_loss_signals_return_pct'] = self.calculate_backtest_average_loss_signals_return_pct(closed_signals)
            metrics['sharpe_ratio'] = self.calculate_backtest_sharpe_ratio(closed_signals)
            metrics['sortino_ratio'] = self.calculate_backtest_sortino_ratio(closed_signals)
            metrics['calmar_ratio'] = self.calculate_backtest_calmar_ratio(closed_signals)
            metrics['max_drawdown'] = self.calculate_backtest_max_drawdown(closed_signals)
            metrics['max_drawdown_duration'] = self.calculate_backtest_max_drawdown_duration(closed_signals)
            metrics['profit_factor'] = self.calculate_backtest_profit_factor(closed_signals)
            metrics['win_loss_ratio'] = self.calculate_backtest_win_loss_ratio(closed_signals)
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating backtest performance metrics: {e}")
            raise
    
    def get_stoploss_orders_print_msg(self, signals, live_bool, sequence_of_symbols, market_data_df):
        """Generate message about stoploss orders for signals"""
        symbol_quantities = {}

        if live_bool:
            for signal in signals:
                if signal.status != 'closed':
                    for order in signal.orders:
                        symbol = order.symbol
                        order_quantity = order.orderQuantity
                        sl_price = getattr(order, 'auxPrice', None)
                        if sl_price:
                            if order.orderDirection == 'SELL':
                                order_quantity = -order_quantity
                            min_granularity = self.market_data_extractor.get_market_data_df_minimum_granularity(market_data_df)
                            close_prices = self.market_data_extractor.get_market_data_df_symbol_prices(market_data_df, min_granularity, symbol, 'close')
                            close_prices.dropna(inplace=True)
                            close_prices = close_prices.tolist()
                            latest_price = close_prices[-1] if len(close_prices) > 0 else sl_price
                            symbol_quantities[symbol] = (order_quantity, sl_price, latest_price)
        else:
            for signal in signals:
                if signal.status != 'closed':
                    for order in signal.orders:
                        if order.status in ['pending', 'open']:
                            symbol = order.symbol
                            order_quantity = order.orderQuantity
                            try:
                                sl_price = order.stoploss_abs
                                latest_price = list(order.symbol_ltp.values())[-1] if hasattr(order, 'symbol_ltp') else None
                                
                                if order.orderDirection == 'SELL':
                                    order_quantity = -order_quantity
                                
                                if symbol not in symbol_quantities:  # Only add if not already present
                                    symbol_quantities[symbol] = (order_quantity, sl_price, latest_price)
                            except:
                                # self.logger.debug({'order': order.dict()})
                                continue

        # Create the message string
        msg_parts = []
        for symbol in sequence_of_symbols:
            if symbol in symbol_quantities:
                qty, sl, ltp = symbol_quantities[symbol]
                sl_dist = round(((abs(ltp - sl) / sl) * 100), 2) if sl and ltp else None
                formatted_sl = round(sl, 2) if sl is not None else None
                msg_parts.append(f"{Fore.BLUE}{symbol}{Style.RESET_ALL}: {qty}, SL: {formatted_sl}, LTP: {ltp}, SL Dist.: {sl_dist}%")
        
        msg = "Current Stoploss Signals: " + ' | '.join(msg_parts) if msg_parts else "Current Stoploss Signals: "
        return msg

    def save_backtest(self, config_dict: Dict, open_signals: List[Signal], closed_signals: List[Signal]):
        """Save backtest results to a file"""
        try:
            # Create backtest folder if it doesn't exist
            os.makedirs(self.backtest_folder_path, exist_ok=True)
            
            # Generate a unique test name using timestamp and strategy
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            strategy_name = config_dict.get('strategy_name', 'unknown_strategy')
            test_name = f"{strategy_name}_{timestamp}"
            
            # Create test-specific folder
            test_folder_path = os.path.join(self.backtest_folder_path, test_name)
            os.makedirs(test_folder_path, exist_ok=True)
            
            # Save backtest results
            results = {
                'config': config_dict,
                'metrics': self.backtest_performance_metrics,
                'report': self.backtest_report
            }
            
            with open(os.path.join(test_folder_path, 'results.json'), 'w') as f:
                json.dump(results, f, indent=4, default=str)
            
            return test_folder_path, test_name
        except Exception as e:
            self.logger.error(f"Error saving backtest results: {e}")
            raise
    
    def generate_report(self, config_dict: Dict, open_signals: List[Signal], closed_signals: List[Signal], market_data_df: pd.DataFrame, unfilled_orders: List[Order]):
        """Generate a performance report for the trading system"""
        try:
            self.backtest_performance_metrics = self.calculate_backtest_performance_metrics(config_dict, open_signals, closed_signals, market_data_df, unfilled_orders)
            self.backtest_report = {'metrics': self.backtest_performance_metrics}
            self.logger.info("Performance report generated successfully")
            # create a nicely formatted and printed report using logger from backtest_report
            self.logger.info(f"{Fore.BLUE}Total Signals{Style.RESET_ALL}: {self.backtest_performance_metrics['total_signals']}")
            self.logger.info(f"{Fore.BLUE}Open Signals{Style.RESET_ALL}: {self.backtest_performance_metrics['open_signals']}")
            self.logger.info(f"{Fore.BLUE}Closed Signals{Style.RESET_ALL}: {self.backtest_performance_metrics['closed_signals']}")
            self.logger.info(f"{Fore.BLUE}Winning Signals{Style.RESET_ALL}: {self.backtest_performance_metrics['winning_signals']}")
            self.logger.info(f"{Fore.BLUE}Losing Signals{Style.RESET_ALL}: {self.backtest_performance_metrics['losing_signals']}")
            self.logger.info(f"{Fore.BLUE}Win Loss Ratio{Style.RESET_ALL}: {self.backtest_performance_metrics['win_loss_ratio']}")
            self.logger.info(f"{Fore.BLUE}Average Signal Duration{Style.RESET_ALL}: {self.backtest_performance_metrics['average_signal_duration']} hours")
            self.logger.info(f"{Fore.BLUE}Average Signal Duration (Profitable){Style.RESET_ALL}: {self.backtest_performance_metrics['average_signal_duration_profitable']} hours")
            self.logger.info(f"{Fore.BLUE}Average Signal Duration (Loss){Style.RESET_ALL}: {self.backtest_performance_metrics['average_signal_duration_loss']} hours")
            self.logger.info(f"{Fore.BLUE}Average Signal Return{Style.RESET_ALL}: {self.backtest_performance_metrics['average_signal_return']}")
            self.logger.info(f"{Fore.BLUE}Average Signal Return (%) {Style.RESET_ALL}: {self.backtest_performance_metrics['average_signal_return_pct']}%")
            self.logger.info(f"{Fore.BLUE}Average Signal Return (%/day){Style.RESET_ALL}: {self.backtest_performance_metrics['average_signal_return_pct_per_day']}%")
            self.logger.info(f"{Fore.BLUE}Average Win Signals Return{Style.RESET_ALL}: {self.backtest_performance_metrics['average_win_signals_return']}")
            self.logger.info(f"{Fore.BLUE}Average Loss Signals Return (%) {Style.RESET_ALL}: {self.backtest_performance_metrics['average_loss_signals_return_pct']}%")
            self.logger.info(f"{Fore.BLUE}Sharpe Ratio{Style.RESET_ALL}: {self.backtest_performance_metrics['sharpe_ratio']}")
            self.logger.info(f"{Fore.BLUE}Sortino Ratio{Style.RESET_ALL}: {self.backtest_performance_metrics['sortino_ratio']}")
            self.logger.info(f"{Fore.BLUE}Calmar Ratio{Style.RESET_ALL}: {self.backtest_performance_metrics['calmar_ratio']}")
            self.logger.info(f"{Fore.BLUE}Max Drawdown{Style.RESET_ALL}: {self.backtest_performance_metrics['max_drawdown']}%")
            self.logger.info(f"{Fore.BLUE}Max Drawdown Duration{Style.RESET_ALL}: {self.backtest_performance_metrics['max_drawdown_duration']} days")
            self.logger.info(f"{Fore.BLUE}Profit Factor{Style.RESET_ALL}: {self.backtest_performance_metrics['profit_factor']}")
            
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            raise
