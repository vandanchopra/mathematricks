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

class PerformanceReporter:
    def __init__(self, market_data_extractor):
        self.market_data_extractor = market_data_extractor
        self.backtest_folder_path = project_path + 'db/vault/backtest_reports'
        self.backtest_performance_metrics = None
        self.backtest_report = None
        self.logger = create_logger(log_level='DEBUG', logger_name='REPORTER', print_to_console=True)

    def calculate_unrealized_pnl(self, signals, unfilled_orders):
        """Calculate unrealized PnL for active signals"""
        unrealized_pnl_abs_dict = {}
        unrealized_pnl_pct_dict = {}
    
        for signal in signals:
            if signal.status != 'closed' and signal.orders:  # Only calculate for active signals with orders
                # For pairs trading, a signal can have multiple symbols
                # Calculate PnL for each symbol in the signal
                symbols = {order.symbol for order in signal.orders}
                unrealized_pnl_abs, unrealized_pnl_pct = self.calculate_signal_pnl(signal, unfilled_orders, force_close=True)
                # Attribute the PnL to each symbol in the signal
                for symbol in symbols:
                    unrealized_pnl_abs_dict[symbol] = unrealized_pnl_abs
                    unrealized_pnl_pct_dict[symbol] = unrealized_pnl_pct
            
        return unrealized_pnl_abs_dict, unrealized_pnl_pct_dict
    
    def calculate_signal_pnl(self, signal, unfilled_orders, force_close=False):
        """Calculate PnL for a signal including unrealized PnL from unfilled orders"""
        total_profit = 0
        total_value = 0
#         self.logger.info(f"""
# PnL Calculation for Signal {signal.signal_id}:
# - Strategy: {signal.strategy_name}
# - Status: {signal.status}
# - Order Count: {len(signal.orders)}
# """)
        
        # First process filled orders
        for order in signal.orders:
            if order.status == 'cancelled':
                continue
            
            try:
                current_price = (
                    order.filled_price if order.status == 'closed'
                    else list(order.symbol_ltp.values())[-1]
                )
                
                direction_multiplier = 1 if order.orderDirection == 'BUY' else -1
                position_value = current_price * order.orderQuantity * direction_multiplier
                
                # For closed orders, use filled price
                if order.status == 'closed' and order.filled_price is not None:
                    total_profit += position_value
                    total_value += abs(position_value)
#                     self.logger.info(f"""
# Order PnL:
# - Symbol: {order.symbol}
# - Direction: {order.orderDirection}
# - Quantity: {order.orderQuantity}
# - Fill Price: ${order.filled_price:.2f}""")
                
                # For unfilled orders in force_close mode, use current price
                elif force_close and order in unfilled_orders:
                    total_profit += position_value
                    total_value += abs(position_value)
#                     self.logger.info(f"""
# Unrealized PnL:
# - Symbol: {order.symbol}
# - Direction: {order.orderDirection}
# - Quantity: {order.orderQuantity}
# - Current Price: ${current_price:.2f}""")
            except Exception as e:
                self.logger.error(f"Error calculating PnL for order {order.symbol}: {e}")
                continue

        # Calculate return percentage
        return_pct = (total_profit / total_value) if total_value > 0 else 0
#         self.logger.info(f"""
# Signal PnL Summary:
# - Total Profit: ${total_profit:.2f}
# - Total Value: ${total_value:.2f}
# - Return: {return_pct*100:.2f}%
# """)
        
        return round(total_profit, 10), return_pct

        # Calculate return percentage
        return_pct = (total_profit / total_value) if total_value > 0 else 0
        self.logger.info(f"""
Signal Total PnL:
- Total Profit: ${total_profit:.2f}
- Total Value: ${total_value:.2f}
- Return %: {return_pct * 100:.2f}%
""")
        
        return round(total_profit, 10), return_pct

    def get_open_signals_print_msg(self, open_signals, total_buying_power, sequence_of_symbols):
        """Generate message about open signals and their positions"""
        signal_info = {}
        
        # Process each signal to gather information
        for signal in open_signals:
            if signal.status != 'closed':
                for order in signal.orders:
                    if order.status not in ['closed', 'cancelled']:
                        symbol = order.symbol
                        if symbol not in signal_info:
                            signal_info[symbol] = {
                                'quantity': 0,
                                'strategy': signal.strategy_name
                            }
                        
                        quantity = order.orderQuantity
                        if order.orderDirection == 'SELL':
                            quantity = -quantity
                        signal_info[symbol]['quantity'] += quantity
        
        # Format the message
        try:
            msg = ' | '.join([f"{Fore.BLUE}{symbol}{Style.RESET_ALL}: QTY: {signal_info[symbol]['quantity']}, Strategy: {signal_info[symbol]['strategy']}" for symbol in sequence_of_symbols if symbol in signal_info])
            msg = "Current Open Signals: " + msg
        except Exception as e:
            msg = "Current Open Signals: "
            self.logger.error({'signal_info': signal_info, 'sequence_of_symbols': sequence_of_symbols, 'error': str(e)})
        
        return msg

    def calculate_backtest_performance_metrics(self, config_dict: Dict, open_signals: List[Signal], closed_signals: List[Signal], market_data_df: pd.DataFrame, unfilled_orders: List[Order]) -> Dict:
        """Calculate performance metrics for backtest results"""
        metrics = {
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'unrealized_pnl': 0.0,
            'sharpe_ratio': 0.0,
            'trades_per_day': 0.0,
            'trades_by_symbol': {},
            'pnl_by_symbol': {},
            'winning_signals': 0,
            'losing_signals': 0,
            'total_signals': 0
        }

        if not closed_signals:
            return metrics

        # Ensure we're working with Signal objects only
        if closed_signals and isinstance(closed_signals[0], tuple):
            self.logger.error(f"Invalid type in closed_signals: expected Signal, got {type(closed_signals[0])}")
            raise AssertionError(f"Invalid type in closed_signals: expected Signal, got {type(closed_signals[0])}")

        self.logger.debug(f"Processing {len(closed_signals)} closed signals")
        # Track PnL and trades
        winning_pnls = []
        losing_pnls = []
        cumulative_pnl = 0.0
        peaks = [0.0]
        drawdowns = [0.0]

        daily_returns = []
        self.logger.info("Calculating backtest performance metrics...")
        
        for signal in closed_signals:
            if not isinstance(signal, Signal):
                self.logger.error(f"Invalid signal type: {type(signal)}, expected Signal")
                raise AssertionError(f"Invalid signal type: {type(signal)}, expected Signal")
                continue

            # Calculate signal's PnL
            signal_pnl, signal_return = self.calculate_signal_pnl(signal, unfilled_orders)
            
            # Track symbols for this signal
            for order in signal.orders:
                if order.status == 'closed':
                    if order.symbol not in metrics['trades_by_symbol']:
                        metrics['trades_by_symbol'][order.symbol] = 0
                        metrics['pnl_by_symbol'][order.symbol] = 0.0
                    metrics['trades_by_symbol'][order.symbol] += 1
                    metrics['pnl_by_symbol'][order.symbol] += signal_pnl

            # Track signal performance
            cumulative_pnl += signal_pnl
            if signal_pnl > 0:
                winning_pnls.append(signal_pnl)
            else:
                losing_pnls.append(signal_pnl)
            
            daily_returns.append(signal_return)
            peaks.append(max(peaks[-1], cumulative_pnl))
            drawdowns.append(peaks[-1] - cumulative_pnl)

        # Calculate final metrics
        metrics['total_pnl'] = cumulative_pnl
        metrics['max_drawdown'] = max(drawdowns)
        
        metrics['winning_signals'] = len(winning_pnls)
        metrics['losing_signals'] = len(losing_pnls)
        metrics['total_signals'] = metrics['winning_signals'] + metrics['losing_signals']
        
        if metrics['total_signals'] > 0:
            metrics['win_rate'] = metrics['winning_signals'] / metrics['total_signals']
            metrics['avg_win'] = np.mean(winning_pnls) if winning_pnls else 0
            metrics['avg_loss'] = abs(np.mean(losing_pnls)) if losing_pnls else 0
            
            total_gains = sum(winning_pnls)
            total_losses = abs(sum(losing_pnls))
            metrics['profit_factor'] = total_gains / total_losses if total_losses > 0 else float('inf')
            
            trading_days = max(1, (closed_signals[-1].timestamp - config_dict['backtest_inputs']['start_time']).days)
            metrics['trades_per_day'] = metrics['total_signals'] / trading_days

            # Calculate Sharpe Ratio (assuming 0 risk-free rate for simplicity)
            if len(daily_returns) > 1:
                metrics['sharpe_ratio'] = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)  # Annualized

        # Calculate unrealized PnL from open signals
        unrealized_pnl_abs_dict, _ = self.calculate_unrealized_pnl(open_signals, unfilled_orders)
        metrics['unrealized_pnl'] = sum(unrealized_pnl_abs_dict.values())

        self.logger.info(f"""
Backtest Results:
- Total PnL: ${metrics['total_pnl']:.2f}
- Win Rate: {metrics['win_rate']*100:.1f}%
- Winning Signals: {metrics['winning_signals']}
- Losing Signals: {metrics['losing_signals']}
- Average Win: ${metrics['avg_win']:.2f}
- Average Loss: ${metrics['avg_loss']:.2f}
- Profit Factor: {metrics['profit_factor']:.2f}
- Max Drawdown: ${metrics['max_drawdown']:.2f}
- Trades per Day: {metrics['trades_per_day']:.1f}
- Unrealized PnL: ${metrics['unrealized_pnl']:.2f}
- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
""")

        return metrics
    
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
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            raise
