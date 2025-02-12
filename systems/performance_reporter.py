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
    
    def calculate_backtest_winning_signals_count(self, closed_signals, with_fees_and_slippage: bool = True):
        """Count number of winning signals (positive PnL) from backtest
        Args:
            closed_signals: List of closed Signal objects
        Returns:
            int: Number of winning signals
        """
        return sum(1 for signal in closed_signals if getattr(signal, 'pnl_with_fee_and_slippage' if with_fees_and_slippage else 'pnl', 0) >= 0)
    
    def calculate_backtest_losing_signals_count(self, closed_signals, with_fees_and_slippage: bool = True):
        """Count number of losing signals (negative PnL) from backtest"""
        return sum(1 for signal in closed_signals if getattr(signal, 'pnl_with_fee_and_slippage' if with_fees_and_slippage else 'pnl', 0) < 0)
    
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
    
    def calculate_backtest_average_signal_duration_profitable(self, closed_signals, with_fees_and_slippage: bool = True):
        profitable_signals = [signal for signal in closed_signals if getattr(signal, 'pnl_with_fee_and_slippage' if with_fees_and_slippage else 'pnl', 0) >= 0]
        return self.calculate_backtest_average_signal_duration(profitable_signals)
    
    def calculate_backtest_average_signal_duration_loss(self, closed_signals, with_fees_and_slippage: bool = True):
        loss_signals = [signal for signal in closed_signals if getattr(signal, 'pnl_with_fee_and_slippage' if with_fees_and_slippage else 'pnl', 0) < 0]
        return self.calculate_backtest_average_signal_duration(loss_signals)
    
    def calculate_backtest_average_signal_return(self, closed_signals, with_fees_and_slippage: bool = True):
        """Calculate average absolute return across all signals"""
        returns = [getattr(signal, 'pnl_with_fee_and_slippage' if with_fees_and_slippage else 'pnl', 0) for signal in closed_signals if getattr(signal, 'pnl_with_fee_and_slippage' if with_fees_and_slippage else 'pnl') is not None]
        return np.mean(returns) if returns else 0
    
    def calculate_backtest_average_signal_return_pct(self, closed_signals, with_fees_and_slippage: bool = True):
        """Calculate average percentage return across all signals"""
        pct_returns = []
        for signal in closed_signals:
            total_investment = sum(abs(order.orderQuantity * order.filled_price) 
                                 for order in signal.orders 
                                 if order.filled_price and order.entryOrderBool)
            pnl = getattr(signal, 'pnl_with_fee_and_slippage' if with_fees_and_slippage else 'pnl')
            if total_investment > 0 and pnl is not None:
                pct_return = (pnl / total_investment) * 100
                pct_returns.append(pct_return)
        return np.mean(pct_returns) if pct_returns else 0
    
    def calculate_backtest_average_signal_return_pct_per_day(self, closed_signals, with_fees_and_slippage: bool = True):
        """Calculate average daily percentage return"""
        daily_returns = []
        for signal in closed_signals:
            pnl = getattr(signal, 'pnl_with_fee_and_slippage' if with_fees_and_slippage else 'pnl')
            if signal.orders and pnl is not None:
                entry_time = min(order.filled_timestamp for order in signal.orders if order.filled_timestamp)
                exit_time = max(order.filled_timestamp for order in signal.orders if order.filled_timestamp)
                if entry_time and exit_time:
                    duration_days = (exit_time - entry_time).total_seconds() / (24 * 3600)
                    if duration_days > 0:
                        total_investment = sum(abs(order.orderQuantity * order.filled_price) 
                                            for order in signal.orders 
                                            if order.filled_price)
                        if total_investment > 0:
                            pnl = getattr(signal, 'pnl_with_fee_and_slippage' if with_fees_and_slippage else 'pnl')
                            pct_return = (pnl / total_investment) * 100
                            daily_return = pct_return / duration_days
                            daily_returns.append(daily_return)
        return np.mean(daily_returns) if daily_returns else 0
    
    def calculate_backtest_average_win_signals_return(self, closed_signals, with_fees_and_slippage: bool = True):
        """Calculate average return of winning signals"""
        pnl_attr = 'pnl_with_fee_and_slippage' if with_fees_and_slippage else 'pnl'
        win_returns = [getattr(signal, pnl_attr) for signal in closed_signals if getattr(signal, pnl_attr) and getattr(signal, pnl_attr) >= 0]
        return np.mean(win_returns) if win_returns else 0
    
    def calculate_backtest_average_loss_signals_return(self, closed_signals, with_fees_and_slippage: bool = True):
        """Calculate average return of winning signals"""
        pnl_attr = 'pnl_with_fee_and_slippage' if with_fees_and_slippage else 'pnl'
        win_returns = [getattr(signal, pnl_attr) for signal in closed_signals if getattr(signal, pnl_attr) and getattr(signal, pnl_attr) < 0]
        return np.mean(win_returns) if win_returns else 0
    
    def calculate_backtest_average_win_signals_return_pct(self, closed_signals, with_fees_and_slippage: bool = True):
        """Calculate average percentage return of losing signals"""
        loss_pct_returns = []
        for signal in closed_signals:
            pnl_attr = 'pnl_with_fee_and_slippage' if with_fees_and_slippage else 'pnl'
            pnl = getattr(signal, pnl_attr)
            if pnl and pnl >= 0:
                total_investment = sum(abs(order.orderQuantity * order.filled_price)
                                      for order in signal.orders
                                      if order.filled_price)
                if total_investment > 0:
                    pct_return = (pnl / total_investment) * 100
                    loss_pct_returns.append(pct_return)
        return np.mean(loss_pct_returns) if loss_pct_returns else 0
    
    def calculate_backtest_average_loss_signals_return_pct(self, closed_signals, with_fees_and_slippage: bool = True):
        """Calculate average percentage return of losing signals"""
        loss_pct_returns = []
        for signal in closed_signals:
            pnl_attr = 'pnl_with_fee_and_slippage' if with_fees_and_slippage else 'pnl'
            pnl = getattr(signal, pnl_attr)
            if pnl and pnl < 0:
                total_investment = sum(abs(order.orderQuantity * order.filled_price)
                                      for order in signal.orders
                                      if order.filled_price)
                if total_investment > 0:
                    pct_return = (pnl / total_investment) * 100
                    loss_pct_returns.append(pct_return)
        return np.mean(loss_pct_returns) if loss_pct_returns else 0
    
    def generate_equity_curve(self, signals: List[Signal], output_granularity: str = '1d', starting_capital: float = 100000) -> pd.DataFrame:
        """Generate equity curve from signals at specified granularity.
        
        Args:
            signals: List of Signal objects
            output_granularity: Time granularity for equity curve ('1m', '1h', '1d', '1w')
            starting_capital: Initial capital for calculating returns
            
        Returns:
            DataFrame with datetime index and columns: ['equity', 'returns']
        """
        self.logger.info(f"Generating equity curve with {output_granularity} granularity")
        
        # Convert granularity string to timedelta
        granularity_map = {
            '1m': pd.Timedelta(minutes=1),
            '1h': pd.Timedelta(hours=1),
            '1d': pd.Timedelta(days=1),
            '1w': pd.Timedelta(weeks=1)
        }
        if output_granularity not in granularity_map:
            raise ValueError(f"Invalid granularity: {output_granularity}. Must be one of {list(granularity_map.keys())}")
        
        interval = granularity_map[output_granularity]
        
        # Get overall timerange from signals
        all_timestamps = [
            timestamp
            for signal in signals
            for order in signal.orders
            if order.symbol_ltp  # Only consider orders with price history
            for timestamp in pd.to_datetime(list(order.symbol_ltp.keys()))
        ]
        
        if not all_timestamps:
            return pd.DataFrame()
            
        start_time = min(all_timestamps)
        end_time = max(all_timestamps)
        
        # Create datetime range at specified granularity
        timestamps = pd.date_range(start_time, end_time, freq=output_granularity)
        equity_curve = pd.DataFrame(index=timestamps)
        equity_curve['equity'] = starting_capital
        
        for timestamp in timestamps:
            total_position_value = 0
            
            # Calculate value of all open positions at this timestamp
            for signal in signals:
                for order in signal.orders:
                    if not order.symbol_ltp:
                        continue
                        
                    # Convert order.symbol_ltp timestamps to datetime for comparison
                    ltp_times = pd.to_datetime(list(order.symbol_ltp.keys()))
                    ltp_prices = list(order.symbol_ltp.values())
                    
                    # Find the latest price before current timestamp
                    valid_prices = [(t, p) for t, p in zip(ltp_times, ltp_prices) if t <= timestamp]
                    if not valid_prices:
                        continue
                        
                    current_price = valid_prices[-1][1]
                    
                    # Check if order is active at this timestamp
                    if order.filled_timestamp and order.filled_timestamp <= timestamp:
                        position_value = order.orderQuantity * current_price
                        if order.orderDirection == 'SELL':
                            position_value = -position_value
                        total_position_value += position_value
            
            equity_curve.at[timestamp, 'equity'] = starting_capital + total_position_value
            
        # Calculate returns
        equity_curve['returns'] = equity_curve['equity'].pct_change()
        
        self.logger.info({'equity_curve':equity_curve})
        
        return equity_curve
    
    def calculate_backtest_sharpe_ratio(self, closed_signals, with_fees_and_slippage: bool = True):
        """Calculate Sharpe ratio using equity curve returns.
        
        The Sharpe ratio is calculated as: (R_p - R_f) / σ_p
        where:
        R_p = Portfolio return (annualized)
        R_f = Risk-free rate (assumed 0)
        σ_p = Portfolio standard deviation (annualized)
        """
        self.logger.info("Starting Sharpe ratio calculation")
        
        # Determine appropriate granularity based on signals timespan
        all_timestamps = [
            order.filled_timestamp
            for signal in closed_signals
            for order in signal.orders
            if order.filled_timestamp
        ]
        if not all_timestamps:
            # self.logger.warning("No valid timestamps found")
            return 0
            
        timespan = max(all_timestamps) - min(all_timestamps)
        if timespan.days < 1:
            granularity = '1m'
        elif timespan.days < 7:
            granularity = '1h'
        elif timespan.days < 30:
            granularity = '1d'
        else:
            granularity = '1d'
            
        # self.logger.info(f"Using {granularity} granularity based on {timespan.days} days timespan")
        
        # Generate equity curve
        equity_curve = self.generate_equity_curve(closed_signals, granularity)
        if equity_curve.empty:
            self.logger.warning("Empty equity curve")
            return 0
            
        # Calculate annualization factor
        granularity_factors = {
            '1m': 252 * 6.5 * 60,  # Trading minutes per year
            '1h': 252 * 6.5,       # Trading hours per year
            '1d': 252,             # Trading days per year
            '1w': 52               # Weeks per year
        }
        annualization_factor = granularity_factors[granularity]
        # self.logger.info(f"Using annualization factor: {annualization_factor}")
        
        # Calculate Sharpe ratio components
        returns = equity_curve['returns'].dropna()
        if len(returns) < 2:
            # self.logger.warning("Insufficient return data points")
            return 0
            
        mean_return = returns.mean()
        returns_std = returns.std(ddof=1)
        
        # self.logger.debug(f"Mean return per period: {mean_return:.6f}")
        # self.logger.debug(f"Standard deviation per period: {returns_std:.6f}")
        
        if returns_std == 0:
            # self.logger.warning("Zero standard deviation")
            return 0
            
        # Calculate annualized Sharpe ratio
        sharpe_ratio = np.sqrt(annualization_factor) * mean_return / returns_std
        
        # self.logger.info(f"Final Sharpe ratio: {sharpe_ratio:.4f}")
        return sharpe_ratio
    
    def calculate_backtest_sortino_ratio(self, closed_signals, with_fees_and_slippage: bool = True):
        """Calculate Sortino ratio (considering only downside volatility)"""
        daily_returns = []
        for signal in closed_signals:
            pnl = getattr(signal, 'pnl_with_fee_and_slippage' if with_fees_and_slippage else 'pnl')
            if signal.orders and pnl is not None:
                entry_time = min(order.filled_timestamp for order in signal.orders if order.filled_timestamp)
                exit_time = max(order.filled_timestamp for order in signal.orders if order.filled_timestamp)
                if entry_time and exit_time:
                    duration_days = (exit_time - entry_time).total_seconds() / (24 * 3600)
                    if duration_days > 0:
                        total_investment = sum(abs(order.orderQuantity * order.filled_price) 
                                            for order in signal.orders 
                                            if order.filled_price)
                        if total_investment > 0:
                            daily_return = (pnl / total_investment) / duration_days
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
    
    def calculate_backtest_calmar_ratio(self, closed_signals, with_fees_and_slippage: bool = True):
        """Calculate Calmar ratio (average return / maximum drawdown)"""
        max_dd = self.calculate_backtest_max_drawdown(closed_signals, with_fees_and_slippage)
        avg_return = self.calculate_backtest_average_signal_return_pct_per_day(closed_signals, with_fees_and_slippage)
        return abs(avg_return / max_dd) if max_dd != 0 else 0
    
    def calculate_backtest_max_drawdown(self, closed_signals, with_fees_and_slippage: bool = True):
        """Calculate maximum drawdown percentage"""
        # Sort signals by entry time
        sorted_signals = sorted(closed_signals, 
                              key=lambda x: min(order.filled_timestamp for order in x.orders if order.filled_timestamp))
        
        pnl_attr = 'pnl_with_fee_and_slippage' if with_fees_and_slippage else 'pnl'
        cumulative_returns = np.cumsum([getattr(signal, pnl_attr) for signal in sorted_signals if getattr(signal, pnl_attr)])
        if len(cumulative_returns) > 0:
            rolling_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (rolling_max - cumulative_returns) / rolling_max
            max_drawdown = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0
            return max_drawdown
        return 0
    
    def calculate_backtest_max_drawdown_duration(self, closed_signals, with_fees_and_slippage: bool = True):
        """Calculate maximum drawdown duration in days"""
        # Sort signals by entry time
        sorted_signals = sorted(closed_signals, 
                              key=lambda x: min(order.filled_timestamp for order in x.orders if order.filled_timestamp))
        
        pnl_attr = 'pnl_with_fee_and_slippage' if with_fees_and_slippage else 'pnl'
        cumulative_returns = np.cumsum([getattr(signal, pnl_attr) for signal in sorted_signals if getattr(signal, pnl_attr)])
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
    
    def calculate_backtest_profit_factor(self, closed_signals, with_fees_and_slippage: bool = True):
        """Calculate ratio of gross profits to gross losses"""
        pnl_attr = 'pnl_with_fee_and_slippage' if with_fees_and_slippage else 'pnl'
        gross_profits = sum(getattr(signal, pnl_attr) for signal in closed_signals if getattr(signal, pnl_attr) and getattr(signal, pnl_attr) > 0)
        gross_losses = abs(sum(getattr(signal, pnl_attr) for signal in closed_signals if getattr(signal, pnl_attr) and getattr(signal, pnl_attr) < 0))
        return gross_profits / gross_losses if gross_losses != 0 else float('inf')
    
    def calculate_backtest_win_loss_ratio(self, closed_signals, with_fees_and_slippage: bool = True):
        """Calculate ratio of winning trades to losing trades"""
        winning_count = self.calculate_backtest_winning_signals_count(closed_signals, with_fees_and_slippage)
        losing_count = self.calculate_backtest_losing_signals_count(closed_signals, with_fees_and_slippage)
        return winning_count / (losing_count+winning_count) if winning_count+losing_count > 0 else float('inf')
    
    def calculate_backtest_cagr(self, closed_signals, with_fees_and_slippage: bool = True):
        """Calculate Compound Annual Growth Rate (CAGR)"""
        if not closed_signals:
            return 0
            
        # Sort signals by timestamp
        sorted_signals = sorted(closed_signals,
                            key=lambda x: min(order.filled_timestamp for order in x.orders if order.filled_timestamp))
                            
        start_date = min(order.filled_timestamp for signal in sorted_signals for order in signal.orders if order.filled_timestamp)
        end_date = max(order.filled_timestamp for signal in sorted_signals for order in signal.orders if order.filled_timestamp)
        
        # Calculate total PnL
        pnl_attr = 'pnl_with_fee_and_slippage' if with_fees_and_slippage else 'pnl'
        total_pnl = sum(getattr(signal, pnl_attr, 0) for signal in closed_signals)
        
        # Calculate years between first and last trade
        days = (end_date - start_date).days
        if days == 0:
            return 0
            
        # Assume starting capital of 100,000 if not specified
        starting_capital = 100000
        ending_capital = starting_capital + total_pnl
        
        # Calculate CAGR
        cagr = (pow(ending_capital / starting_capital, 365.0 / days) - 1) * 100
        return cagr
    
    def calculate_backtest_cagr(self, closed_signals, with_fees_and_slippage: bool = True):
       """Calculate Compound Annual Growth Rate (CAGR) for the backtest period
       
       CAGR = (Ending Value / Starting Value) ^ (365 / Days) - 1
       
       For a backtest that could be days, weeks, or years long.
       """
       if not closed_signals:
           return 0
           
       # Sort signals by entry time to get first and last dates
       sorted_signals = sorted(closed_signals,
                             key=lambda x: min(order.filled_timestamp for order in x.orders if order.filled_timestamp))
       
       start_date = min(order.filled_timestamp for order in sorted_signals[0].orders if order.filled_timestamp)
       end_date = max(order.filled_timestamp for order in sorted_signals[-1].orders if order.filled_timestamp)
       
       days = (end_date - start_date).days
       if days == 0:  # For intraday backtests
           days = 1

       pnl_attr = 'pnl_with_fee_and_slippage' if with_fees_and_slippage else 'pnl'
       total_pnl = sum(getattr(signal, pnl_attr) for signal in closed_signals if getattr(signal, pnl_attr))
       
       # Calculate initial investment from first signal's entry orders
       initial_investment = sum(abs(order.orderQuantity * order.filled_price)
                              for order in sorted_signals[0].orders
                              if order.filled_price and order.entryOrderBool)
       
       if initial_investment == 0:  # Avoid division by zero
           return 0
           
       ending_value = initial_investment + total_pnl
       cagr = (ending_value / initial_investment) ** (365.0 / days) - 1
       
       return cagr
    
class PerformanceReporter(Metrics):
    def __init__(self, market_data_extractor):
        super().__init__()
        self.market_data_extractor = market_data_extractor
        self.backtest_folder_path = project_path + 'db/vault/backtest_reports'
        self.backtest_performance_metrics = None
        self.backtest_report = None
        self.logger = create_logger(log_level='DEBUG', logger_name='REPORTER', print_to_console=True)

    def filter_signals_by_type(self, signals: List[Signal], signal_type: str) -> List[Signal]:
        """Filter signals by type (long, short, neutral)
        
        A signal is:
        - long if all entry orders are BUY
        - short if all entry orders are SELL
        - neutral if there's a mix of BUY and SELL entry orders
        """
        filtered_signals = []
        for signal in signals:
            # Get list of directions for all entry orders
            entry_directions = [o.orderDirection for o in signal.orders if o.entryOrderBool]
            
            if not entry_directions:  # No entry orders
                if signal_type == "neutral":
                    filtered_signals.append(signal)
            else:
                # Check if all directions are the same
                all_buys = all(d == 'BUY' for d in entry_directions)
                all_sells = all(d == 'SELL' for d in entry_directions)
                
                if signal_type == "long" and all_buys:
                    filtered_signals.append(signal)
                elif signal_type == "short" and all_sells:
                    filtered_signals.append(signal)
                elif signal_type == "neutral" and not (all_buys or all_sells):
                    filtered_signals.append(signal)
                elif signal_type not in ["long", "short", "neutral"]:
                    filtered_signals.append(signal)
                    
        return filtered_signals
    
    def calculate_metrics_by_type(self, config_dict: Dict, open_signals: List[Signal],
                                closed_signals: List[Signal], signal_type: str,
                                with_fees_and_slippage: bool = True) -> Dict:
        """Calculate metrics for a specific signal type"""
        filtered_open = self.filter_signals_by_type(open_signals, signal_type)
        filtered_closed = self.filter_signals_by_type(closed_signals, signal_type)
        return self.calculate_backtest_performance_metrics(config_dict, filtered_open, filtered_closed, with_fees_and_slippage)
    
    def calculate_backtest_performance_metrics(self, config_dict: Dict, open_signals: List[Signal], closed_signals: List[Signal], with_fees_and_slippage: bool = True) -> Dict:
        """Calculate various performance metrics for the backtest"""
        try:
            # Calculate performance metrics
            metrics = {}
            metrics['total_signals'] = len(open_signals) + len(closed_signals)
            metrics['open_signals'] = len(open_signals)
            metrics['closed_signals'] = len(closed_signals)
            metrics['winning_signals'] = self.calculate_backtest_winning_signals_count(closed_signals, with_fees_and_slippage)
            metrics['losing_signals'] = self.calculate_backtest_losing_signals_count(closed_signals, with_fees_and_slippage)
            metrics['average_signal_duration'] = self.calculate_backtest_average_signal_duration(closed_signals)
            metrics['average_signal_duration_profitable'] = self.calculate_backtest_average_signal_duration_profitable(closed_signals, with_fees_and_slippage)
            metrics['average_signal_duration_loss'] = self.calculate_backtest_average_signal_duration_loss(closed_signals, with_fees_and_slippage)
            metrics['average_signal_return'] = self.calculate_backtest_average_signal_return(closed_signals, with_fees_and_slippage)
            metrics['average_signal_return_pct'] = self.calculate_backtest_average_signal_return_pct(closed_signals, with_fees_and_slippage)
            metrics['average_signal_return_pct_per_day'] = self.calculate_backtest_average_signal_return_pct_per_day(closed_signals, with_fees_and_slippage)
            metrics['average_win_signals_return'] = self.calculate_backtest_average_win_signals_return(closed_signals, with_fees_and_slippage)
            metrics['average_loss_signals_return'] = self.calculate_backtest_average_loss_signals_return(closed_signals, with_fees_and_slippage)
            metrics['average_win_signals_return_pct'] = self.calculate_backtest_average_win_signals_return_pct(closed_signals, with_fees_and_slippage)
            metrics['average_loss_signals_return_pct'] = self.calculate_backtest_average_loss_signals_return_pct(closed_signals, with_fees_and_slippage)
            metrics['sharpe_ratio'] = self.calculate_backtest_sharpe_ratio(closed_signals, with_fees_and_slippage)
            metrics['sortino_ratio'] = self.calculate_backtest_sortino_ratio(closed_signals, with_fees_and_slippage)
            metrics['calmar_ratio'] = self.calculate_backtest_calmar_ratio(closed_signals, with_fees_and_slippage)
            metrics['max_drawdown'] = self.calculate_backtest_max_drawdown(closed_signals, with_fees_and_slippage)
            metrics['max_drawdown_duration'] = self.calculate_backtest_max_drawdown_duration(closed_signals, with_fees_and_slippage)
            metrics['profit_factor'] = self.calculate_backtest_profit_factor(closed_signals, with_fees_and_slippage)
            metrics['win_loss_ratio'] = self.calculate_backtest_win_loss_ratio(closed_signals, with_fees_and_slippage)
            metrics['cagr'] = self.calculate_backtest_cagr(closed_signals, with_fees_and_slippage)
            
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
            
            # Save open and closed orders as pickle
            backtest_output = {'open_signals':open_signals, 'closed_signals':closed_signals, 'config_dict':config_dict}
            backtest_output_path = os.path.join(self.backtest_folder_path, test_name, 'backtest_output.pkl')
            with open(backtest_output_path, 'wb') as file:
                pickle.dump(backtest_output, file)
            
            return test_folder_path, test_name
        except Exception as e:
            self.logger.error(f"Error saving backtest results: {e}")
            raise
    
    def generate_report(self, config_dict: Dict, open_signals: List[Signal], closed_signals: List[Signal], market_data_df: pd.DataFrame, with_fees_and_slippage: bool = True):
        """Generate a performance report for the trading system"""
        try:
            # Calculate metrics for all signal types
            signal_types = ["all", "long", "short", "neutral"]
            metrics_by_type = {}
            
            for signal_type in signal_types:
                if signal_type == "all":
                    metrics = self.calculate_backtest_performance_metrics(config_dict, open_signals, closed_signals, with_fees_and_slippage)
                else:
                    metrics = self.calculate_metrics_by_type(config_dict, open_signals, closed_signals, signal_type, with_fees_and_slippage)
                metrics_by_type[signal_type] = metrics
            
            self.backtest_performance_metrics = metrics_by_type["all"]
            self.backtest_report = {'metrics': self.backtest_performance_metrics}
            
            # Generate and save equity curve plot
            self.logger.info("Generating equity curve plot...")
            equity_curve = self.generate_equity_curve(closed_signals)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve['equity'],
                mode='lines',
                name='Portfolio Value'
            ))
            
            fig.update_layout(
                title='Portfolio Equity Curve',
                xaxis_title='Date',
                yaxis_title='Portfolio Value',
                showlegend=True,
                template='plotly_dark'
            )
            
            # Save plot as interactive HTML
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            strategy_name = config_dict.get('strategy_name', 'unknown_strategy')
            plot_filename = f"{strategy_name}_{timestamp}_equity_curve.html"
            plot_path = os.path.join(self.backtest_folder_path, plot_filename)
            
            self.logger.info(f"Saving interactive equity curve plot to: {plot_path}")
            fig.write_html(plot_path)
            
            # Create table headers and rows
            headers = [f"Metric ({'With Costs' if with_fees_and_slippage else 'Without Costs'})", "Total", "Long", "Short", "Neutral"]
            
            # Define metrics to display in order
            metric_rows = [
                ("Total Signals", "total_signals"),
                ("Open Signals", "open_signals"),
                ("Closed Signals", "closed_signals"),
                ("Winning Signals", "winning_signals"),
                ("Losing Signals", "losing_signals"),
                ("Win/Loss Ratio", "win_loss_ratio", "{:0.2f}"),
                ("Average Signal Return", "average_signal_return", "{:0.2f}"),
                ("Average Win Return", "average_win_signals_return", "{:0.2f}"),
                ("Average Loss Return", "average_loss_signals_return", "{:0.2f}"),
                ("Average Return (%)", "average_signal_return_pct", "{:0.2f}%"),
                ("Average Win Return (%)", "average_win_signals_return_pct", "{:0.2f}%"),
                ("Average Loss Return (%)", "average_loss_signals_return_pct", "{:0.2f}%"),
                ("Avg Signal Duration (hours)", "average_signal_duration", "{:0.2f}"),
                ("Avg Win Duration (hours)", "average_signal_duration_profitable", "{:0.2f}"),
                ("Avg Loss Duration (hours)", "average_signal_duration_loss", "{:0.2f}"),
                ("Profit Factor", "profit_factor", "{:0.2f}"),
                ("Average Return (%/day)", "average_signal_return_pct_per_day", "{:0.2f}%"),
                ("CAGR (%)", "cagr", "{:0.2f}%"),
                ("Sharpe Ratio", "sharpe_ratio", "{:0.2f}"),
                ("Sortino Ratio", "sortino_ratio", "{:0.2f}"),
                ("Calmar Ratio", "calmar_ratio", "{:0.2f}"),
                # ("Max Drawdown (%)", "max_drawdown", "{:0.2f}%"),
                # ("Max Drawdown Duration (days)", "max_drawdown_duration", "{:0.2f}")
            ]
            
            # Build table rows
            table_data = []
            for metric_name, metric_key, *format_spec in metric_rows:
                fmt = format_spec[0] if format_spec else "{}"
                row = [metric_name]
                for signal_type in ["all", "long", "short", "neutral"]:
                    value = metrics_by_type[signal_type].get(metric_key, 0)
                    row.append(fmt.format(value))
                table_data.append(row)
            
            from tabulate import tabulate
            
            # Print the report
            self.logger.info("Performance report generated successfully")
            self.logger.info("\n" + tabulate(table_data, headers=headers, tablefmt="grid", stralign="left") + "\n")
            
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            raise

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
        df1 = trade_df_cumulative['account_value']
        df2 = index_data_dict['close']
        df1.index = pd.to_datetime(df1.index)
        df2.index = pd.to_datetime(df2.index)
        # Add 1 minute to df1 index
        df1.index = df1.index + timedelta(seconds=1)
        df1_df = df1.to_frame()
        df2_df = df2.to_frame()
        
        merged_df = df1_df.merge(df2_df, left_index=True, right_index=True, how='inner')
        # Drop rows with NaN values (if any)
        merged_df = merged_df.dropna()
        # rename 'Close' to DJI
        merged_df.rename(columns={'close': 'index'}, inplace=True)
        # make the values of the columns float
        merged_df['account_value'] = merged_df['account_value'].astype(float)
        merged_df['index'] = merged_df['index'].astype(float)
        
        # Calculate sharpe ratio for the strategy
        strategy_cum_returns = merged_df['account_value']
        index_values = merged_df['index']
        # Calculate periodic returns from cumulative returns
        if strategy_cum_returns.iloc[0] == 0:
            strategy_cum_returns.iloc[0] = 0.000000001
        strategy_returns = strategy_cum_returns.pct_change().dropna()
        index_returns = index_values.pct_change().dropna()
        periods_per_year = 252
        # Calculate excess returns
        excess_returns = strategy_returns - index_returns
        # print(excess_returns)

        # Annualized return
        mean_excess_return = excess_returns.mean() * periods_per_year

        # Annualized volatility
        annualized_volatility = excess_returns.std() * np.sqrt(periods_per_year)
        
        # Sharpe Ratio
        sharpe_ratio = mean_excess_return / annualized_volatility
        
        return sharpe_ratio

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
