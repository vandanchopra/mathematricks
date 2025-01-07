import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from .base_strategy import BaseStrategy

class PairsTradingStrategy(BaseStrategy):
    def __init__(self, config_dict=None):
        super().__init__()
        if config_dict:
            self.set_params(config_dict.get('strategy_params', {}))
        self.strategy_name = 'PairsTradingStrategy'
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.capital = 100000
        self.spread_history = {}
        self.datafeeder_inputs = {
            'get_inputs': lambda: {
                'symbols': self.TICKERS,
                'data_types': ['price', 'volume'],
                'timeframe': '1d'
            }
        }
        
        # Configuration parameters
        self.TICKERS = ['AAPL', 'GOOG', 'SPY', 'GLD', 'CL']
        self.TRAINING_PERIOD = '1y'
        self.CORRELATION_THRESHOLD = 0.6
        self.ADF_P_VALUE_THRESHOLD = 0.15
        self.ENTRY_THRESHOLD = 1.5
        self.EXIT_THRESHOLD = 0.5
        self.STOP_LOSS_THRESHOLD = 3
        self.VOLUME_WINDOW = 5
        self.MAX_VOLUME_PERCENTAGE = 0.1
        self.TOTAL_CAPITAL = 100000
        self.MIN_RISK_REWARD_RATIO = 1.5
        self.LOOKBACK_PERIOD = 40
        self.POSITION_SIZE = 0.2
        self.MAX_RISK_PER_BET = 0.05
        self.MIN_OPPORTUNITY_SCORE = 0.25
        self.DYNAMIC_ENTRY_THRESHOLD = True
        self.ENTRY_THRESHOLD_RANGE = (1.2, 2.0)
        self.DYNAMIC_OPPORTUNITY_THRESHOLD = True
        self.OPPORTUNITY_THRESHOLD_RANGE = (0.2, 0.35)
        self.OPPORTUNITY_SCORE_WEIGHTS = {
            'zscore': 0.35,
            'correlation': 0.25,
            'volatility': 0.20,
            'cointegration': 0.20
        }

    def get_signal(self, data):
        """Generate trading signals based on current market data"""
        signals = {}
        pairs = self.identify_pairs(self.TICKERS, data)
        
        for pair_info in pairs:
            pair = pair_info['pair']
            asset1, asset2 = pair
            
            # Calculate spread statistics
            lookback_data = data.iloc[-self.LOOKBACK_PERIOD:]
            spread = lookback_data[asset1] - lookback_data[asset2]
            spread_mean = spread.mean()
            spread_std = spread.std()
            current_spread = data.iloc[-1][asset1] - data.iloc[-1][asset2]
            z_score = (current_spread - spread_mean) / spread_std
            
            # Calculate opportunity score
            opportunity_score = self.calculate_opportunity_score(pair, z_score)
            
            # Generate signal
            if abs(z_score) >= self.ENTRY_THRESHOLD and opportunity_score >= self.MIN_OPPORTUNITY_SCORE:
                signals[pair] = {
                    'asset1': asset1,
                    'asset2': asset2,
                    'z_score': z_score,
                    'direction': -1 if z_score > 0 else 1,
                    'opportunity_score': opportunity_score
                }
                
        return signals

    def get_target(self, data):
        """Calculate target positions based on current signals"""
        signals = self.get_signal(data)
        targets = {}
        
        for pair, signal in signals.items():
            position_size = min(
                self.POSITION_SIZE * self.capital * (signal['opportunity_score'] ** 2),
                self.MAX_RISK_PER_BET * self.capital / max(1, abs(signal['z_score']))
            )
            
            targets[pair] = {
                'asset1': signal['asset1'],
                'asset2': signal['asset2'],
                'position_size': position_size,
                'direction': signal['direction']
            }
            
        return targets

    def get_trades(self, data):
        """Generate trade execution instructions"""
        targets = self.get_target(data)
        trades = []
        
        for pair, target in targets.items():
            trades.append({
                'pair': pair,
                'asset1': target['asset1'],
                'asset2': target['asset2'],
                'position_size': target['position_size'],
                'direction': target['direction']
            })
            
        return trades

    def get_metrics(self, data):
        """Calculate strategy performance metrics"""
        if len(self.equity_curve) < 2:
            return {
                'cagr': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
            
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        
        # CAGR
        try:
            years = (data.index[-1] - data.index[0]).days / 365.25
            cagr = (self.equity_curve[-1] / self.equity_curve[0]) ** (1/years) - 1
        except:
            cagr = 0
            
        # Sharpe Ratio
        try:
            excess_returns = returns - 0.05/252
            sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
            if np.isinf(sharpe) or np.isnan(sharpe):
                sharpe = 0
        except:
            sharpe = 0
            
        # Max Drawdown
        try:
            rolling_max = pd.Series(self.equity_curve).expanding().max()
            drawdowns = pd.Series(self.equity_curve) / rolling_max - 1
            max_drawdown = drawdowns.min()
        except:
            max_drawdown = 0
            
        return {
            'cagr': cagr * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown * 100
        }

    def get_params(self):
        """Get current strategy parameters"""
        return {
            'TICKERS': self.TICKERS,
            'TRAINING_PERIOD': self.TRAINING_PERIOD,
            'CORRELATION_THRESHOLD': self.CORRELATION_THRESHOLD,
            'ADF_P_VALUE_THRESHOLD': self.ADF_P_VALUE_THRESHOLD,
            'ENTRY_THRESHOLD': self.ENTRY_THRESHOLD,
            'EXIT_THRESHOLD': self.EXIT_THRESHOLD,
            'STOP_LOSS_THRESHOLD': self.STOP_LOSS_THRESHOLD,
            'VOLUME_WINDOW': self.VOLUME_WINDOW,
            'MAX_VOLUME_PERCENTAGE': self.MAX_VOLUME_PERCENTAGE,
            'TOTAL_CAPITAL': self.TOTAL_CAPITAL,
            'MIN_RISK_REWARD_RATIO': self.MIN_RISK_REWARD_RATIO,
            'LOOKBACK_PERIOD': self.LOOKBACK_PERIOD,
            'POSITION_SIZE': self.POSITION_SIZE,
            'MAX_RISK_PER_BET': self.MAX_RISK_PER_BET,
            'MIN_OPPORTUNITY_SCORE': self.MIN_OPPORTUNITY_SCORE,
            'DYNAMIC_ENTRY_THRESHOLD': self.DYNAMIC_ENTRY_THRESHOLD,
            'ENTRY_THRESHOLD_RANGE': self.ENTRY_THRESHOLD_RANGE,
            'DYNAMIC_OPPORTUNITY_THRESHOLD': self.DYNAMIC_OPPORTUNITY_THRESHOLD,
            'OPPORTUNITY_THRESHOLD_RANGE': self.OPPORTUNITY_THRESHOLD_RANGE,
            'OPPORTUNITY_SCORE_WEIGHTS': self.OPPORTUNITY_SCORE_WEIGHTS
        }

    def set_datafeeder_inputs(self, inputs):
        """Set datafeeder inputs for the strategy"""
        if not callable(inputs.get('get_inputs')):
            self.logger.error("datafeeder_inputs must have a callable 'get_inputs' method")
            return
            
        self.datafeeder_inputs = inputs
        self.logger.info("Updated datafeeder inputs with symbols: %s",
                        self.datafeeder_inputs['get_inputs']()['symbols'])

    def set_params(self, config):
        """Set strategy parameters"""
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def fetch_data(self, tickers: List[str], period: str) -> pd.DataFrame:
        """Fetch historical adjusted close prices for given tickers from Yahoo Finance."""
        try:
            data = yf.download(tickers, period=period, progress=False)['Adj Close']
            data = data.dropna(axis=1, how='all')
            data = data.ffill().bfill()
            return data
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            raise

    def calculate_correlation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Pearson correlation matrix for all pairs of assets."""
        return data.corr()

    def cointegration_test(self, pair: Tuple[str, str], data: pd.DataFrame) -> float:
        """Perform comprehensive cointegration testing on a pair of assets."""
        try:
            P_i = data[pair[0]]
            P_j = data[pair[1]]
            
            # Augmented Dickey-Fuller test
            X = sm.add_constant(P_j)
            model = OLS(P_i, X).fit()
            residuals = model.resid
            adf_result = adfuller(residuals, maxlag=1)
            return adf_result[1]
        except Exception as e:
            self.logger.error(f"Error in cointegration test for {pair}: {str(e)}")
            return 1.0

    def identify_pairs(self, tickers: List[str], data: pd.DataFrame) -> List[Dict]:
        """Identify potential trading pairs based on correlation and cointegration."""
        pairs = []
        corr_matrix = self.calculate_correlation(data)
        
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                pair = (tickers[i], tickers[j])
                corr = corr_matrix.loc[pair[0], pair[1]]
                
                if corr >= self.CORRELATION_THRESHOLD:
                    adf_p_value = self.cointegration_test(pair, data)
                    if adf_p_value < self.ADF_P_VALUE_THRESHOLD:
                        pairs.append({
                            'pair': pair,
                            'corr': corr,
                            'adf_p_value': adf_p_value
                        })
                        
        return pairs

    def calculate_opportunity_score(self, pair: Tuple[str, str], z_score: float) -> float:
        """Calculate opportunity score based on multiple factors."""
        if not hasattr(self, 'spread_history') or pair not in self.spread_history:
            return 0
            
        spreads = self.spread_history[pair]
        if len(spreads) < self.LOOKBACK_PERIOD:
            return 0
            
        # Calculate correlation component
        try:
            window_sizes = [20, 40, 60]
            correlations = []
            for window in window_sizes:
                if len(spreads) > window:
                    rolling_corr = pd.Series(spreads).rolling(window=window).corr(pd.Series(spreads).shift(1))
                    correlations.append(abs(rolling_corr.iloc[-1]))
            
            correlation = np.mean([c for c in correlations if not np.isnan(c)])
            correlation_stability = 1 - np.std([c for c in correlations if not np.isnan(c)])
            correlation_score = correlation * correlation_stability
        except:
            correlation_score = 0.0
        
        # Volatility component
        try:
            vol_windows = [5, 10, 20]
            volatilities = []
            for window in vol_windows:
                vol = pd.Series(spreads).rolling(window=window).std().iloc[-1]
                volatilities.append(vol)
            
            # Normalize volatility using logarithmic scaling
            min_vol = 0.01
            max_vol = 0.5
            scaled_vol = (np.log(volatility + min_vol) - np.log(min_vol)) / \
                        (np.log(max_vol + min_vol) - np.log(min_vol))
            
            # Apply trend adjustment
            trend_factor = 1.0 if vol_trend <= 1.1 else 0.75
            vol_score = (1 - scaled_vol) * trend_factor
        except:
            vol_score = 0.0
        
        # Z-score component
        try:
            z_score_abs = abs(z_score)
            z_score_abs = 0.0 if np.isnan(z_score_abs) else z_score_abs
            
            # Calculate historical z-scores with weighted average
            lookback_windows = [20, 40, 60]
            z_scores = []
            for window in lookback_windows:
                if len(self.spread_history[pair]) > window:
                    spread = pd.Series(self.spread_history[pair][-window:])
                    z = (spread - spread.mean()) / spread.std()
                    z_scores.append(z.iloc[-1])
            
            # Weight recent z-scores more heavily
            weights = [0.2, 0.3, 0.5]
            weighted_z = sum(z * w for z, w in zip(z_scores, weights)) / sum(weights)
            
            # Calculate z_score_score with smoother scaling
            z_score_score = min(z_score_abs / self.ENTRY_THRESHOLD, 1.0)
            z_score_score *= 1 / (1 + np.exp(-weighted_z))
        except:
            z_score_score = 0.0
        
        # Cointegration score
        try:
            window = min(len(spreads), 120)
            rolling_coint = []
            for i in range(max(window, len(spreads)-3), len(spreads)):
                y = pd.Series(spreads[i-window:i])
                x = pd.Series(range(len(y)))
                model = sm.OLS(y, sm.add_constant(x)).fit()
                residuals = model.resid
                coint_stat = sm.tsa.stattools.adfuller(residuals)[0]
                rolling_coint.append(coint_stat)
            
            coint_score = np.tanh(-np.mean(rolling_coint))
            coint_stability = 1 - np.std(rolling_coint) / abs(np.mean(rolling_coint))
            coint_score *= max(0.2, coint_stability)
        except:
            coint_score = 0.0
        
        # Calculate final score with updated weights
        final_score = (
            self.OPPORTUNITY_SCORE_WEIGHTS['zscore'] * z_score_score +
            self.OPPORTUNITY_SCORE_WEIGHTS['correlation'] * correlation_score +
            self.OPPORTUNITY_SCORE_WEIGHTS['volatility'] * vol_score +
            self.OPPORTUNITY_SCORE_WEIGHTS['cointegration'] * coint_score
        )
        
        return final_score

    def backtest(self, data: pd.DataFrame, pairs: List[Dict]) -> Dict:
        """Run backtest on identified pairs"""
        results = {
            'total_pnl': 0,
            'n_trades': 0,
            'win_rate': 0,
            'equity_curve': [self.capital],
            'metrics': {},
            'trades': []
        }
        
        # Initialize portfolio and spread history
        portfolio_value = self.capital
        active_trades = {}
        
        # Initialize spread history for all pairs
        for pair_info in pairs:
            pair = pair_info['pair']
            self.spread_history[pair] = []
        
        # Iterate through each day
        for i in range(self.LOOKBACK_PERIOD, len(data)):
            current_date = data.index[i]
            prices = data.iloc[i]
            
            # Check for trade exits
            for pair, trade in list(active_trades.items()):
                spread = prices[trade['asset1']] - prices[trade['asset2']]
                z_score = (spread - trade['spread_mean']) / trade['spread_std']
                
                # Exit conditions
                if (abs(z_score) <= self.EXIT_THRESHOLD or
                    (current_date - trade['entry_date']).days >= 30):  # Max holding period
                    
                    # Calculate PnL
                    pnl = self._calculate_trade_pnl(trade, prices)
                    portfolio_value += pnl
                    results['total_pnl'] += pnl
                    results['n_trades'] += 1
                    
                    # Record trade
                    trade.update({
                        'exit_date': current_date,
                        'exit_price1': prices[trade['asset1']],
                        'exit_price2': prices[trade['asset2']],
                        'pnl': pnl
                    })
                    results['trades'].append(trade)
                    del active_trades[pair]
            
            # Check for new trade opportunities
            for pair_info in pairs:
                pair = pair_info['pair']
                asset1, asset2 = pair
                
                # Calculate spread statistics
                lookback_data = data.iloc[i-self.LOOKBACK_PERIOD:i]
                spread = lookback_data[asset1] - lookback_data[asset2]
                spread_mean = spread.mean()
                spread_std = spread.std()
                current_spread = prices[asset1] - prices[asset2]
                z_score = (current_spread - spread_mean) / spread_std
                
                # Check entry conditions
                opportunity_score = self.calculate_opportunity_score(pair, z_score)
                
                if (abs(z_score) >= self.ENTRY_THRESHOLD and
                    opportunity_score >= self.MIN_OPPORTUNITY_SCORE and
                    pair not in active_trades):
                    
                    # Calculate position size
                    position_size = min(
                        self.POSITION_SIZE * portfolio_value * (opportunity_score ** 2),
                        self.MAX_RISK_PER_BET * portfolio_value / max(1, abs(z_score))
                    )
                    
                    # Execute trade
                    trade = {
                        'pair': pair,
                        'asset1': asset1,
                        'asset2': asset2,
                        'entry_date': current_date,
                        'entry_price1': prices[asset1],
                        'entry_price2': prices[asset2],
                        'spread_mean': spread_mean,
                        'spread_std': spread_std,
                        'position_size': position_size,
                        'z_score': z_score,
                        'direction': -1 if z_score > 0 else 1
                    }
                    active_trades[pair] = trade
                    portfolio_value -= position_size * 0.001  # Transaction cost
            
            # Update spread history and equity curve
            for pair_info in pairs:
                pair = pair_info['pair']
                spread = prices[pair[0]] - prices[pair[1]]
                self.spread_history[pair].append(spread)
                
            results['equity_curve'].append(portfolio_value)
        
        # Calculate performance metrics
        results['metrics'] = self.get_metrics(data)
        
        # Calculate win rate
        if results['n_trades'] > 0:
            winning_trades = len([t for t in results['trades'] if t['pnl'] > 0])
            results['win_rate'] = winning_trades / results['n_trades']
        
        return results

    def _calculate_trade_pnl(self, trade: Dict, prices: pd.Series) -> float:
        """Calculate PnL for a single trade"""
        price_change1 = prices[trade['asset1']] - trade['entry_price1']
        price_change2 = prices[trade['asset2']] - trade['entry_price2']
        
        if trade['direction'] == -1:
            pnl = (-price_change1 + price_change2) * trade['position_size']
        else:
            pnl = (price_change1 - price_change2) * trade['position_size']
            
        return pnl

    def plot_performance(self, equity_curve: pd.Series, benchmark: pd.Series):
        """Plot strategy performance vs benchmark"""
        try:
            strategy_norm = equity_curve / equity_curve.iloc[0] * 100
            benchmark_norm = benchmark / benchmark.iloc[0] * 100
            
            plt.figure(figsize=(12, 6))
            plt.plot(strategy_norm.index, strategy_norm.values, label='Strategy')
            plt.plot(benchmark_norm.index, benchmark_norm.values, label='Benchmark')
            plt.title('Strategy Performance')
            plt.xlabel('Date')
            plt.ylabel('Normalized Value')
            plt.legend()
            plt.grid(True)
            plt.savefig('performance.png')
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting performance: {str(e)}")

if __name__ == "__main__":
    strategy = PairsTradingStrategy()
    data = strategy.fetch_data(strategy.TICKERS, strategy.TRAINING_PERIOD)
    results = strategy.backtest(data, strategy.identify_pairs(strategy.TICKERS, data))
    print(results)