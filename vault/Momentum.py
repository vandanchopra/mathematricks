"""
Institutional Momentum Strategy with robust risk management and adaptive portfolio construction
"""

from vault.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.covariance import LedoitWolf  # For robust covariance estimation

class Strategy(BaseStrategy):

    def __init__(self, config_dict):
        super().__init__()
        # Strategy configuration
        self.strategy_name = 'Momentum'
        self.granularity = "1d"
        self.orderType = "MARKET"
        self.timeInForce = "DAY"
        self.orderQuantity = 100
        self.stop_loss_pct = 0.05
        self.exit_order_type = "stoploss_pct"
        
        # Risk floors and thresholds
        self.volatility_window = 21
        self.volatility_floor = 0.10  # 10% annualized vol floor
        self.volume_floor = 100000    # Minimum daily volume
        self.min_sector_size = 3      # Minimum number of stocks per sector
        # Enhanced momentum parameters
        self.momentum_windows = [10, 21, 63]  # Longer windows for more stable signals
        self.momentum_weights = [0.3, 0.4, 0.3]  # More balanced weights
        self.min_momentum_threshold = 0.01  # 1% minimum momentum
        self.trend_windows = [20, 50, 100]  # Reduced trend windows for initial backtest
        self.volume_windows = [5, 21]  # Volume analysis windows
        self.cooldown_periods = {}  # Track cooldown for each symbol
        self.cooldown_days = 5  # Cooldown period after signal
        self.max_holding_days = 21  # Maximum position holding time
        self.volume_confirmation = True
        self.volume_confirmation = True  # Use volume to confirm signals
        
        # Risk parameters
        self.position_sizing = {
            'normal': {'base': 1.0, 'strong': 2.0},
            'high_vol': {'base': 0.75, 'strong': 1.5},
            'stress': {'base': 0.5, 'strong': 1.0}
        }
        self.stop_loss_multipliers = {
            'normal': 1.0,
            'high_vol': 1.5,
            'stress': 2.0
        }
        
        # Enhanced risk parameters with tighter controls
        self.risk_params = {
            'max_leverage': 3.0,  # Increased max leverage
            'sector_cap': 0.15,  # Increased sector limits
            'base_liquidity_impact': 0.001,  # More aggressive slippage estimate
            'vol_scaling_factor': 0.25,  # More aggressive volatility targeting
            'max_position_size': 0.06,  # Larger maximum position
            'min_position_size': 0.005,  # Lower minimum for more opportunities
            'market_impact_power': 0.4,  # Less conservative market impact
            'max_drawdown': 0.20,  # Higher drawdown threshold
            'beta_neutral_threshold': 0.15,  # Increased beta deviation allowance
            'turnover_limit': 0.30,  # Higher portfolio turnover limit
            'correlation_threshold': 0.8,  # Higher correlation tolerance
            'var_confidence': 0.97,  # Slightly lower VaR confidence
            'stress_test_scenarios': ['2008_crisis', '2020_covid', 'rising_rates'],
            'risk_factor_constraints': {
                'momentum': (-3, 3),  # Z-score limits
                'volatility': (0.1, 0.4),  # Annualized vol limits
                'beta': (-0.2, 0.2),  # Market beta limits
                'size': (-2, 2),  # Size factor exposure
                'value': (-2, 2)  # Value factor exposure
            }
        }
        
        # Initialize state variables
        self.current_leverage = 1.0
        self.sector_exposures = {}
        self.crisis_mode = False
        self.market_regime = 'normal'  # normal, high_vol, stress
        self.regime_thresholds = {
            'high_vol': 0.30,  # 30% annualized vol
            'stress': 0.45    # 45% annualized vol
        }
        
        # Load data requirements
        self.data_inputs, self.tickers = self.datafeeder_inputs()

    def get_name(self):
        return self.strategy_name

    def datafeeder_inputs(self):
        """Expanded universe of liquid stocks across sectors"""
        tickers = [
            # Technology
            'AAPL', 'MSFT', 'CSCO', 'ORCL', 'INTC', 'IBM',
            # Consumer
            'WMT', 'PG', 'MCD', 'KO',
            # Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C',
            # Healthcare
            'JNJ', 'PFE', 'MRK', 'ABT', 'BMY',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB',
            # Industrial
            'GE', 'CAT', 'BA', 'MMM',
            # Communication
            'VZ', 'T', 'DIS',
            # Materials
            'APD'
        ]
        
        self.logger.info(f"Total number of tickers: {len(tickers)}")
        
        data_inputs = {
            '1d': {
                'columns': ['open', 'high', 'close', 'volume', 'Market Cap', 'Sector'],
                'lookback': max(self.momentum_windows) + 21  # Lookback based on longest momentum window plus buffer
            }
        }
        
        self.logger.info(f"Data requirements: {data_inputs}")
        return data_inputs, tickers

    def calculate_market_impact(self, price, volume, position_size, is_high_urgency=False):
        """Enhanced market impact model with urgency and regime adjustment"""
        daily_volume_usd = price * volume
        participation_rate = (price * position_size) / daily_volume_usd
        base_impact = self.risk_params['base_liquidity_impact']
        
        # Square root model with volume adjustment
        impact = base_impact * np.power(participation_rate, self.risk_params['market_impact_power'])
        
        # Volume weighted adjustment
        vol_z = (volume - volume.mean()) / volume.std()
        impact *= (1 + 0.2 * np.exp(-vol_z))  # Higher impact for lower volume
        
        # Urgency adjustment
        if is_high_urgency:
            impact *= 1.5
        
        # Regime-based impact scaling
        regime_multipliers = {
            'normal': 1.0,
            'high_vol': 1.75,  # More conservative
            'stress': 2.5      # Even more conservative
        }
        impact *= regime_multipliers[self.market_regime]
        
        # Add non-linear impact component for large trades
        if participation_rate > 0.1:  # 10% of daily volume
            impact *= (1 + np.log(participation_rate * 10))  # Non-linear increase
            
        return impact

    def calculate_optimal_kelly(self, returns, volatility):
        """Calculate optimal position sizes using the Kelly Criterion"""
        # Estimate Sharpe ratio components
        annual_return = returns.mean() * 252
        annual_vol = volatility * np.sqrt(252)
        
        # Risk-free rate assumption
        risk_free = 0.02  # 2% annual risk-free rate
        
        # Calculate half-Kelly fraction for more conservative sizing
        half_kelly = 0.5 * (annual_return - risk_free) / (annual_vol ** 2)
        
        # Apply regime-based scaling
        kelly_scales = {
            'normal': 1.0,
            'high_vol': 0.7,
            'stress': 0.4
        }
        
        return half_kelly * kelly_scales[self.market_regime]
    def calculate_exponential_weights(self, n):
        """Calculate exponentially decaying weights"""
        decay_factor = 0.85  # Slower decay for more stable weights
        weights = np.array([decay_factor ** i for i in range(n)])
        return weights / weights.sum()

    def calculate_risk_metrics(self, returns, vol, drawdowns):
        """Calculate comprehensive risk metrics"""
        metrics = {}
        
        # Volatility metrics
        metrics['rolling_vol'] = returns.rolling(21).std() * np.sqrt(252)
        metrics['vol_of_vol'] = metrics['rolling_vol'].rolling(63).std()
        
        # Drawdown metrics
        metrics['max_drawdown'] = drawdowns.min()
        metrics['avg_drawdown'] = drawdowns.mean()
        metrics['drawdown_duration'] = self.calculate_drawdown_duration(drawdowns)
        
        # Value at Risk
        metrics['var_99'] = np.percentile(returns, 1)
        metrics['cvar_99'] = returns[returns <= metrics['var_99']].mean()
        
        # Higher moments
        metrics['skewness'] = returns.skew()
        metrics['excess_kurtosis'] = returns.kurtosis()
        
        return metrics

    def detect_market_regime(self, df):
        """Enhanced market regime detection using advanced indicators with proper index handling"""
        try:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a DataFrame")

            # Ensure proper date handling
            if 'Date' not in df.columns and not isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            
            # Convert date column to datetime if it isn't already
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Handle data format conversion with robust error handling
            try:
                # Ensure data has proper columns and index
                if isinstance(df.index, pd.MultiIndex):
                    df_reset = df.reset_index()
                else:
                    df_reset = df.copy()

                # Ensure we have Date and Symbol columns
                # Handle various date column names
                date_columns = {'date', 'Date', 'datetime', 'timestamp'}
                found_date_col = next((col for col in df_reset.columns if col.lower() in date_columns), None)
                if found_date_col and found_date_col != 'Date':
                    df_reset = df_reset.rename(columns={found_date_col: 'Date'})
                
                # Handle symbol column
                if 'Symbol' not in df_reset.columns and 'symbol' in df_reset.columns:
                    df_reset = df_reset.rename(columns={'symbol': 'Symbol'})
                
                # Convert to datetime if needed
                if 'Date' not in df_reset.columns:
                    self.logger.error(f"No date column found. Available columns: {df_reset.columns}")
                    raise ValueError("Missing Date column")
                
                df_reset['Date'] = pd.to_datetime(df_reset['Date'])
                
                # Create pivot tables
                prices = df_reset.pivot(index='Date', columns='Symbol', values='close')
                volumes = df_reset.pivot(index='Date', columns='Symbol', values='volume')
                
                # Ensure we have data
                if prices.empty or volumes.empty:
                    raise ValueError("Empty price/volume data after pivoting")
                    
                self.logger.debug(f"Successfully converted data with shape: {prices.shape}")
                
            except Exception as e:
                self.logger.error(f"Data format conversion failed: {str(e)}")
                self.logger.debug(f"DataFrame columns: {df.columns}")
                self.logger.debug(f"DataFrame index: {df.index}")
                raise ValueError(f"Could not convert data to required format: {str(e)}")
            
            # Ensure we have data
            if prices.empty or volumes.empty:
                raise ValueError("No valid price/volume data for regime detection")
            
            # Calculate returns
            returns = prices.pct_change()
            
            # 1. Enhanced volatility metrics
            rolling_vol = returns.rolling(21).std() * np.sqrt(252)
            vol_of_vol = rolling_vol.rolling(63).std()
            current_vol = rolling_vol.mean(axis=1).iloc[-1]
            
            # 2. Advanced correlation analysis
            corr_matrix = returns.tail(63).corr()
            
            # Principal component analysis for systemic risk
            try:
                from sklearn.decomposition import PCA
                pca = PCA()
                pca_explained_var = pca.fit(returns.tail(63).fillna(0)).explained_variance_ratio_
                systemic_risk = pca_explained_var[0]  # First component explained variance
            except Exception as e:
                self.logger.warning(f"PCA calculation failed: {str(e)}. Using default value.")
                systemic_risk = 0.5  # Default if PCA fails
            
            # 3. Volume analysis
            volume_ratio = volumes.rolling(5).mean() / volumes.rolling(63).mean()
            abnormal_volume = (volume_ratio.iloc[-1] > 2).any()
            
            # 4. Drawdown analysis
            rolling_max = prices.expanding().max()
            drawdown = (prices - rolling_max) / rolling_max
            severe_drawdown = (drawdown.iloc[-1] < -0.2).any()
            
            # 5. Cross-sectional dispersion
            daily_returns_std = returns.std(axis=1)
            dispersion_z = (daily_returns_std - daily_returns_std.mean()) / daily_returns_std.std()
            
            # 6. Trend strength
            major_trend = returns.rolling(63).mean() > 0
            trend_consistency = major_trend.rolling(21).mean()
            weak_trends = (trend_consistency.iloc[-1] < 0.6).mean() > 0.5  # More than half symbols showing weak trends
            
            # Composite regime detection with more sophisticated thresholds
            # Safely evaluate numeric conditions
            def safe_compare(value, threshold, default=False):
                try:
                    if isinstance(value, (pd.Series, pd.DataFrame)):
                        value = float(value.iloc[-1])
                    return float(value) > float(threshold)
                except:
                    return default

            # Convert boolean conditions to integers for counting with safe evaluation
            try:
                stress_conditions = [
                    safe_compare(current_vol, self.regime_thresholds['stress']),
                    safe_compare(systemic_risk, 0.6),
                    bool(abnormal_volume),  # Already boolean
                    bool(severe_drawdown),  # Already boolean
                    safe_compare(dispersion_z.iloc[-1], 2),
                    bool(weak_trends)  # Already boolean
                ]
                stress_signals = sum(1 for condition in stress_conditions if condition)
                
                high_vol_conditions = [
                    safe_compare(current_vol, self.regime_thresholds['high_vol']),
                    safe_compare(systemic_risk, 0.45),
                    safe_compare(volume_ratio.mean(), 1.5),
                    safe_compare(dispersion_z.iloc[-1], 1)
                ]
                high_vol_signals = sum(1 for condition in high_vol_conditions if condition)
            except Exception as e:
                self.logger.warning(f"Error calculating regime signals: {str(e)}")
                stress_signals = 0
                high_vol_signals = 0
            
            # Log comprehensive metrics
            self.logger.debug(
                f"Market regime metrics:\n"
                f"Volatility: {current_vol:.4f}\n"
                f"Systemic Risk: {systemic_risk:.4f}\n"
                f"Dispersion Z-score: {dispersion_z.iloc[-1]:.4f}\n"
                f"Stress Signals: {stress_signals}\n"
                f"High Vol Signals: {high_vol_signals}"
            )
            
            # Determine regime with quantitative thresholds
            if stress_signals >= 3:
                self.logger.info("Entering stress regime")
                return 'stress'
            elif high_vol_signals >= 2:
                self.logger.info("Entering high volatility regime")
                return 'high_vol'
            
            self.logger.info("Market in normal regime")
            return 'normal'
            
        except Exception as e:
            self.logger.warning(f"Error in market regime detection: {str(e)}. Defaulting to normal regime.")
            return 'normal'

    def calculate_drawdown_duration(self, drawdowns):
        """Calculate average duration of significant drawdowns"""
        significant_dd = drawdowns[drawdowns < -0.05]  # Focus on 5%+ drawdowns
        if len(significant_dd) == 0:
            return 0
            
        dd_start = significant_dd.index[0]
        dd_duration = 0
        current_dd = 0
        
        for date, value in significant_dd.items():
            if value == 0:  # End of drawdown
                if current_dd > 0:
                    dd_duration += current_dd
                current_dd = 0
            else:
                current_dd += 1
                
        return dd_duration / len(significant_dd) if len(significant_dd) > 0 else 0

    def calculate_robust_scores(self, df):
        """Enhanced scoring system with proper multi-index handling"""
        try:
            # Handle data format conversion with validation
            if isinstance(df.index, pd.MultiIndex):
                prices = df['close'].unstack(level='Symbol')
                volumes = df['volume'].unstack(level='Symbol')
            else:
                # If data is not multi-indexed, pivot it
                df_reset = df.reset_index()
                prices = df_reset.pivot(index='Date', columns='Symbol', values='close')
                volumes = df_reset.pivot(index='Date', columns='Symbol', values='volume')
            
            # 1. Multi-period momentum with dynamic weights
            momentum_scores = pd.DataFrame(0, index=prices.index, columns=prices.columns)
            for lookback, weight in zip(self.momentum_windows, self.momentum_weights):
                returns = prices.pct_change(lookback)
                decay = np.exp(-0.5 * (lookback/21))
                momentum_scores += returns * decay * weight
            
            # 2. Volatility calculation
            rolling_vol = prices.pct_change().ewm(span=self.volatility_window).std()
            rolling_vol = rolling_vol.clip(lower=self.volatility_floor/np.sqrt(252))
            
            # Scale momentum by volatility
            scaled_momentum = momentum_scores.div(rolling_vol)
            
            # 3. Volume signals
            volume_ma = volumes.rolling(21).mean()
            volume_trend = volumes.rolling(5).mean() / volumes.rolling(21).mean()
            volume_z = pd.DataFrame(
                zscore(np.log(volume_ma.fillna(1e-6))),
                index=volume_ma.index,
                columns=volume_ma.columns
            )
            volume_score = volume_z.multiply(1 + (volume_trend - 1))
            
            # 4. Sector relative strength with improved calculation
            sector_scores = pd.DataFrame(0, index=prices.index, columns=prices.columns)
            
            # Group returns by sector
            df_reset = df.reset_index()
            sectors = df_reset['Sector'].unique()
            sector_returns = {}
            sector_vols = {}
            
            for sector in sectors:
                sector_symbols = df_reset[df_reset['Sector'] == sector]['Symbol'].unique()
                if len(sector_symbols) > 0:
                    sector_prices = prices[prices.columns.intersection(sector_symbols)]
                    if not sector_prices.empty:
                        sector_rets = sector_prices.pct_change(21)
                        sector_returns[sector] = sector_rets.mean(axis=1)
                        sector_vols[sector] = sector_prices.pct_change().std(axis=1)
            
            # Calculate sector-relative scores
            for symbol in prices.columns:
                sector = df_reset[df_reset['Symbol'] == symbol]['Sector'].iloc[0]
                if sector in sector_returns:
                    symbol_sector_return = sector_returns[sector]
                    symbol_sector_vol = sector_vols[sector]
                    sector_scores[symbol] = (scaled_momentum[symbol] - symbol_sector_return) / (symbol_sector_vol + 1e-6)
            
            # 5. Trend signals
            trend_signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
            for window in [10, 21, 63]:
                ma = prices.rolling(window).mean()
                trend_signals += np.where(prices > ma, 1, -1)
            trend_strength = trend_signals / 3  # Average across windows
            
            # 6. Market cap score
            market_caps = df['Market Cap'].unstack(level='Symbol')
            mcap_z = pd.DataFrame(
                zscore(np.log(market_caps)),
                index=market_caps.index,
                columns=market_caps.columns
            )
            vol_adj_mcap = mcap_z.multiply(np.sqrt(volume_ma.div(volume_ma.mean())))
            
            # 7. Up/Down volume pressure
            up_moves = prices > prices.shift(1)
            down_moves = prices < prices.shift(1)
            up_vol = volumes.where(up_moves, 0).rolling(5).mean()
            down_vol = volumes.where(down_moves, 0).rolling(5).mean()
            vol_ratio = (down_vol - up_vol) / (down_vol + up_vol + 1e-6)
            short_pressure = pd.DataFrame(
                zscore(vol_ratio),
                index=vol_ratio.index,
                columns=vol_ratio.columns
            )

            # Dynamic factor weights based on market regime
            weights = {
                'normal': {
                    'momentum': 0.45, 'sector': 0.15, 'volume': 0.15,
                    'trend': 0.15, 'size': 0.05, 'short': 0.05
                },
                'high_vol': {
                    'momentum': 0.35, 'sector': 0.20, 'volume': 0.20,
                    'trend': 0.15, 'size': 0.05, 'short': 0.05
                },
                'stress': {
                    'momentum': 0.25, 'sector': 0.25, 'volume': 0.20,
                    'trend': 0.20, 'size': 0.05, 'short': 0.05
                }
            }[self.market_regime]

            # Calculate final composite score
            composite_score = (
                weights['momentum'] * scaled_momentum +
                weights['sector'] * sector_scores +
                weights['volume'] * volume_score +
                weights['trend'] * trend_strength +
                weights['size'] * vol_adj_mcap +
                weights['short'] * short_pressure
            )

            # Convert back to long format and merge with original data
            df['score'] = composite_score.stack()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Score calculation failed: {str(e)}")
            raise ValueError(f"Score calculation failed: {str(e)}")

    def construct_adaptive_portfolio(self, df):
        """Enhanced portfolio construction with dynamic constraints"""
        # 1. Enhanced liquidity and quality filters
        df = df[
            (df['Market Cap'] >= 5e9) &  # More lenient market cap requirement
            (df['volume'] >= self.volume_floor) &  # Volume floor check
            (df['score'].abs() > 0.25)  # Lower score threshold for more signal generation
        ]
        
        # 2. Enhanced sector-neutral cluster formation
        portfolio = pd.DataFrame()
        for sector, sector_df in df.groupby('Sector'):
            if len(sector_df) < self.min_sector_size:
                continue
                
            # Enhanced covariance estimation with longer lookback in stress
            lookback = 252 if self.market_regime == 'stress' else 126
            prices = sector_df.pivot(index='Date', columns='Symbol', values='close')
            returns = prices.pct_change().dropna().tail(lookback)
            cov = LedoitWolf().fit(returns).covariance_
            
            # More aggressive portfolio weights calculation
            inv_vol = 1 / np.sqrt(np.diag(cov))
            mpt_weights = inv_vol / inv_vol.sum()  # Remove risk scaling to maintain position sizes
            
            # Ensure minimum weight threshold
            mpt_weights[mpt_weights < 0.01] = 0  # Filter out tiny positions
            if mpt_weights.sum() > 0:
                mpt_weights = mpt_weights / mpt_weights.sum()  # Renormalize
            
            try:
                # Score-adjusted positioning with dynamic thresholds
                sector_df = sector_df.copy()  # Create copy to avoid SettingWithCopyWarning
                sector_df['weight'] = pd.Series(mpt_weights, index=sector_df.index) * sector_df['score']
                
                # More aggressive position thresholds
                thresholds = {
                    'normal': {'long': 0.5, 'short': 0.5},  # Take top/bottom 50%
                    'high_vol': {'long': 0.6, 'short': 0.4},  # Less aggressive in high vol
                    'stress': {'long': 0.7, 'short': 0.3}  # Most conservative in stress
                }[self.market_regime]
                
                if len(sector_df) > 0 and not sector_df['weight'].isna().all():
                    # Split into long/short clusters based on absolute weight values
                    weights_abs = sector_df['weight'].abs()
                    weight_threshold = weights_abs.quantile(0.3)  # Take positions with significant weights
                    
                    # Select positions that have meaningful weights
                    longs = sector_df[(sector_df['weight'] > 0) & (weights_abs >= weight_threshold)]
                    shorts = sector_df[(sector_df['weight'] < 0) & (weights_abs >= weight_threshold)]
                    
                    if not (longs.empty and shorts.empty):
                        portfolio = pd.concat([portfolio, longs, shorts]) if not portfolio.empty else pd.concat([longs, shorts])
            except Exception as e:
                self.logger.warning(f"Error processing sector weights: {str(e)}")
                continue
        
        # 3. Enhanced market impact adjustment with error handling
        if not portfolio.empty and 'weight' in portfolio.columns:
            try:
                portfolio = portfolio.copy()  # Create a copy for safe modification
                for idx, row in portfolio.iterrows():
                    try:
                        if pd.notna(row['close']) and pd.notna(row['volume']) and pd.notna(row['weight']):
                            impact = self.calculate_market_impact(
                                row['close'],
                                row['volume'],
                                abs(row['weight'])
                            )
                            portfolio.loc[idx, 'weight'] *= (1 - impact)
                    except Exception as e:
                        self.logger.warning(f"Error calculating impact for row {idx}: {str(e)}")
                        continue
            except Exception as e:
                self.logger.warning(f"Error in market impact adjustment: {str(e)}")
        
        # 4. Enhanced dynamic leverage control with regime-based scaling and error handling
        if not portfolio.empty and 'weight' in portfolio.columns:
            try:
                portfolio = portfolio.copy()  # Create a copy for safe modification
                weights = portfolio['weight'].dropna()
                
                if len(weights) > 0:
                    port_vol = weights.std() * np.sqrt(252)
                    max_leverage = self.risk_params['max_leverage'] * {
                        'normal': 1.0, 'high_vol': 0.7, 'stress': 0.5
                    }[self.market_regime]
                    
                    target_vol = self.risk_params['vol_scaling_factor']
                    self.current_leverage = min(
                        max_leverage,
                        target_vol / port_vol if port_vol > 0 else 1.0
                    )
                    portfolio['weight'] *= self.current_leverage
                    self.logger.debug(f"Applied leverage adjustment: {self.current_leverage}")
            except Exception as e:
                self.logger.warning(f"Error in leverage adjustment: {str(e)}")
        
        return portfolio

    def risk_management(self, portfolio):
        """Enhanced risk management with comprehensive controls"""
        try:
            # Ensure we have the required columns
            if 'weight' not in portfolio.columns:
                self.logger.warning("Portfolio missing weight column")
                return portfolio
                
            if portfolio.empty:
                return portfolio
                
            portfolio = portfolio.copy()  # Create a copy to avoid SettingWithCopyWarning
            
            # 1. Enhanced correlation and concentration controls
            try:
                prices = portfolio.pivot(columns='Symbol', values='close')
                returns = prices.pct_change().dropna()
            except Exception as e:
                self.logger.warning(f"Error creating price matrix: {str(e)}")
                return portfolio
            
            # Calculate correlation matrix using exponential weights
            corr_matrix = returns.ewm(span=63).corr().iloc[-1]
            high_corr_pairs = np.where(np.abs(corr_matrix) > self.risk_params['correlation_threshold'])
            
            for i, j in zip(*high_corr_pairs):
                if i != j:  # Skip self-correlations
                    sym1, sym2 = corr_matrix.index[i], corr_matrix.index[j]
                    # Reduce position in less liquid stock
                    if portfolio.loc[sym1, 'volume'] < portfolio.loc[sym2, 'volume']:
                        portfolio.loc[sym1, 'weight'] *= 0.7
                    else:
                        portfolio.loc[sym2, 'weight'] *= 0.7

            # 2. Enhanced sector exposure control with regime adaptation
            max_sector_exposure = self.risk_params['sector_cap'] * {
                'normal': 1.0, 'high_vol': 0.75, 'stress': 0.5
            }[self.market_regime]
            
            sector_exposure = portfolio.groupby('Sector')['weight'].sum().abs()
            overexposed = sector_exposure[sector_exposure > max_sector_exposure]
            
            for sector in overexposed.index:
                correction = max_sector_exposure / sector_exposure[sector]
                portfolio.loc[portfolio['Sector'] == sector, 'weight'] *= correction

            # 3. Value at Risk (VaR) control
            position_values = portfolio['weight'] * portfolio['close']
            var_lookback = {'normal': 252, 'high_vol': 126, 'stress': 63}[self.market_regime]
            
            historical_var = returns.multiply(position_values, axis=1).sum(axis=1).sort_values().iloc[
                int(len(returns) * (1 - self.risk_params['var_confidence']))
            ]
            
            if abs(historical_var) > self.risk_params['vol_scaling_factor'] / np.sqrt(252):
                var_scaling = (self.risk_params['vol_scaling_factor'] / np.sqrt(252)) / abs(historical_var)
                portfolio['weight'] *= var_scaling

            # 4. Stress test based position limits
            for scenario in self.risk_params['stress_test_scenarios']:
                stress_returns = self.get_stress_scenario_returns(scenario, returns)
                stress_pnl = (portfolio['weight'] * stress_returns).sum()
                
                if abs(stress_pnl) > self.risk_params['max_drawdown']:
                    stress_scaling = self.risk_params['max_drawdown'] / abs(stress_pnl)
                    portfolio['weight'] *= stress_scaling

            # 5. Dynamic position sizing with market impact
            for idx in portfolio.index:
                impact = self.calculate_market_impact(
                    portfolio.loc[idx, 'close'],
                    portfolio.loc[idx, 'volume'],
                    abs(portfolio.loc[idx, 'weight']),
                    is_high_urgency=(self.market_regime == 'stress')
                )
                max_position = min(
                    self.risk_params['max_position_size'] * {
                        'normal': 1.0, 'high_vol': 0.7, 'stress': 0.5
                    }[self.market_regime],
                    1 / (impact * 10)  # Limit size based on market impact
                )
                portfolio.loc[idx, 'weight'] = np.clip(
                    portfolio.loc[idx, 'weight'],
                    -max_position,
                    max_position
                )

            # 6. Liquidity-adjusted minimum position filter
            min_position = self.risk_params['min_position_size'] * {
                'normal': 1.0, 'high_vol': 1.5, 'stress': 2.0
            }[self.market_regime]
            
            adv = portfolio['volume'].rolling(21).mean()
            max_participation = {
                'normal': 0.1, 'high_vol': 0.07, 'stress': 0.05
            }[self.market_regime]
            
            portfolio = portfolio[
                (portfolio['weight'].abs() >= min_position) &
                (portfolio['weight'].abs() < max_participation * adv * portfolio['close'])
            ]

            # 7. Dynamic stop-loss adjustment
            volatility = returns.std() * np.sqrt(252)
            base_stop = self.stop_loss_pct * {
                'normal': 1.0, 'high_vol': 1.5, 'stress': 2.0
            }[self.market_regime]
            
            portfolio['stop_loss'] = np.maximum(
                base_stop,
                2 * volatility * np.sqrt(5/252)  # 5-day volatility-based stop
            )

            return portfolio
            
        except Exception as e:
            self.logger.warning(f"Risk management failed: {str(e)}. Returning original portfolio.")
            return portfolio

    def get_stress_scenario_returns(self, scenario, returns):
        """Get historical returns for stress testing"""
        if scenario == '2008_crisis':
            # Approximate 2008 financial crisis pattern
            return returns.min() * 1.5
        elif scenario == '2020_covid':
            # COVID crash pattern
            return returns.sort_values().iloc[:int(len(returns)*0.1)].mean()
        elif scenario == 'rising_rates':
            # Rising rates scenario
            return returns.rolling(5).mean().min() * 1.2
        else:
            return returns.mean()  # Default case

    def validate_data(self, df):
        """Enhanced data validation and cleaning with robust index handling"""
        try:
            # Reset index to start with clean state
            df = df.reset_index()
            
            # Standardize column names
            column_map = {
                'symbol': 'Symbol',
                'date': 'Date',
                'time': 'Date',
                'timestamp': 'Date'
            }
            df.rename(columns=column_map, inplace=True)
            
            # Handle multi-level columns
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten multi-level columns
                flat_df = pd.DataFrame()
                for symbol in df.columns.get_level_values(1).unique():
                    if pd.notna(symbol):  # Skip empty/NA symbols
                        symbol_data = pd.DataFrame({
                            'Symbol': symbol,
                            'Date': df.index,
                            'close': df[('close', symbol)] if ('close', symbol) in df.columns else None,
                            'volume': df[('volume', symbol)] if ('volume', symbol) in df.columns else None
                        }).dropna(subset=['close', 'volume'])
                        flat_df = pd.concat([flat_df, symbol_data], ignore_index=True)
                df = flat_df
            
            # Ensure required columns exist
            required_cols = ['Date', 'Symbol', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Clean and standardize data
            df = df.dropna(subset=['Date', 'Symbol', 'close', 'volume'])
            df = df[(df['close'] > 0) & (df['volume'] > 0)]
            
            # Remove duplicate date-symbol combinations
            df = df.drop_duplicates(subset=['Date', 'Symbol'], keep='last')
            
            # Add placeholder values for required fields
            df['Market Cap'] = df.get('Market Cap', 1e10)  # $10B placeholder
            df['Sector'] = df.get('Sector', 'Technology')  # placeholder
            
            # Filter extreme price movements
            df['ret'] = df.groupby('Symbol')['close'].pct_change()
            df = df[df['ret'].abs() < 0.25]  # Filter out >25% price moves
            df = df.drop('ret', axis=1)  # Remove temporary column
            
            # Sort and set proper multi-index
            df = df.sort_values(['Symbol', 'Date'])
            df = df.set_index(['Date', 'Symbol'])
            
            return df
            
        except Exception as e:
            raise ValueError(f"Data validation failed: {str(e)}\nColumns: {df.columns}")

    def generate_signals(self, next_rows, market_data_df, system_timestamp):
        """Generate enhanced momentum trading signals with adaptive risk management"""
        return_type = 'signals'
        all_signals = []  # List to collect individual signals
        self.system_timestamp = system_timestamp  # Store timestamp for cooldown calculations
        
        try:
            # Initialize return structure
            signals = {
                'signals': all_signals,
                'ideal_portfolios': []
            }
            
            # 1. Data Validation and Preparation
            if self.granularity not in market_data_df.index.levels[0]:
                self.logger.warning(f"Required granularity {self.granularity} not in market data")
                return return_type, signals, self.tickers
            
            try:
                current_data = market_data_df.loc[self.granularity]
            except Exception as e:
                self.logger.error(f"Error accessing granularity data: {str(e)}")
                return return_type, signals, self.tickers
            
            # Initialize storage for valid data
            valid_tickers = []
            ticker_data_dict = {}
            
            # Collect and validate data for each ticker
            for ticker in self.tickers:
                try:
                    # Initialize data holders
                    close_prices = None
                    volume_data = None

                    # Attempt to get data with robust error handling
                    try:
                        if isinstance(current_data.columns, pd.MultiIndex):
                            # Try direct column access
                            if ('close', ticker) in current_data.columns:
                                close_prices = current_data['close', ticker]
                                volume_data = current_data['volume', ticker]
                            # Try lowercase
                            elif ('close', ticker.lower()) in current_data.columns:
                                close_prices = current_data['close', ticker.lower()]
                                volume_data = current_data['volume', ticker.lower()]
                            # Try uppercase
                            elif ('close', ticker.upper()) in current_data.columns:
                                close_prices = current_data['close', ticker.upper()]
                                volume_data = current_data['volume', ticker.upper()]
                        else:
                            # Try cross-section with various case combinations
                            try:
                                close_prices = current_data.xs(ticker, level=1, axis=1)['close']
                                volume_data = current_data.xs(ticker, level=1, axis=1)['volume']
                            except:
                                try:
                                    close_prices = current_data.xs(ticker.lower(), level=1, axis=1)['close']
                                    volume_data = current_data.xs(ticker.lower(), level=1, axis=1)['volume']
                                except:
                                    try:
                                        close_prices = current_data.xs(ticker.upper(), level=1, axis=1)['close']
                                        volume_data = current_data.xs(ticker.upper(), level=1, axis=1)['volume']
                                    except:
                                        self.logger.debug(f"Could not get data for {ticker} with any case variation")

                        if close_prices is None or volume_data is None:
                            self.logger.warning(f"No data found for {ticker} after all attempts")
                            continue

                    except Exception as e:
                        self.logger.debug(f"Could not get data for {ticker}: {str(e)}")
                        self.logger.debug(f"Available columns: {current_data.columns}")
                        continue
                    
                    # Convert to DataFrame and handle missing data
                    ticker_data = pd.DataFrame({'close': close_prices, 'volume': volume_data})
                    # Handle NaN values with proper dtype preservation
                    ticker_data = (ticker_data
                        .infer_objects(copy=False)  # First infer proper types
                        .transform(lambda x: pd.to_numeric(x, errors='coerce'))  # Convert to numeric
                        .ffill()  # Forward fill
                        .bfill()  # Back fill
                    )
                    
                    # Additional data validation
                    if len(ticker_data) < max(self.momentum_windows):
                        self.logger.debug(f"Insufficient data points for {ticker} (need {max(self.momentum_windows)}, got {len(ticker_data)})")
                        continue
                        
                    if ticker_data[['close', 'volume']].isnull().any().any():
                        self.logger.debug(f"Still have null values after filling for {ticker}")
                        continue
                    
                    # Store valid data
                    ticker_data_dict[ticker] = ticker_data
                    valid_tickers.append(ticker)
                    
                except Exception as e:
                    self.logger.warning(f"Error preparing data for {ticker}: {str(e)}")
                    continue
                    
            # Exit if no valid data
            if not valid_tickers:
                self.logger.warning("No valid ticker data available")
                return return_type, signals, self.tickers
                
            # Calculate portfolio-wide metrics
            try:
                # Market regime detection
                market_regime_data = pd.concat(ticker_data_dict.values(), keys=ticker_data_dict.keys(), names=['Symbol'])
                self.market_regime = self.detect_market_regime(market_regime_data.reset_index())
                
                # Portfolio volatility
                all_closes = [data['close'] for data in ticker_data_dict.values()]
                combined_closes = pd.concat(all_closes, axis=1, keys=valid_tickers)
                returns = combined_closes.pct_change().infer_objects(copy=False).dropna()
                portfolio_volatility = returns.std().mean() * np.sqrt(252) if not returns.empty else 0.2
                
                # Risk adjustments based on regime
                if self.market_regime == 'stress':
                    position_multiplier = 0.5
                    stop_loss_multiplier = 2.0
                elif self.market_regime == 'high_vol':
                    position_multiplier = 0.75
                    stop_loss_multiplier = 1.5
                else:
                    position_multiplier = 1.0
                    stop_loss_multiplier = 1.0
                
                # Adjust for portfolio volatility
                volatility_scale = max(0.1, portfolio_volatility)
                position_multiplier *= (0.2 / volatility_scale)  # Target 20% volatility
                
            except Exception as e:
                self.logger.warning(f"Error calculating portfolio metrics: {e}. Using default values.")
                # Use default values if calculation fails
                self.market_regime = 'normal'
                portfolio_volatility = 0.2
                position_multiplier = 1.0
                stop_loss_multiplier = 1.0
                
                # Adjust for portfolio volatility
                volatility_scale = max(0.1, portfolio_volatility)
                position_multiplier *= (0.2 / volatility_scale)  # Target 20% volatility
            
            # Generate signals for each ticker
            for ticker in valid_tickers:
                ticker_data = ticker_data_dict[ticker]
                
                try:
                    # Check cooldown period
                    if ticker in self.cooldown_periods and self.system_timestamp - self.cooldown_periods[ticker] < pd.Timedelta(days=self.cooldown_days):
                        self.logger.debug(f"Skipping {ticker} due to cooldown period: last signal at {self.cooldown_periods[ticker]}")
                        continue

                    # Calculate momentum scores
                    momentum_score = 0
                    volume_score = 0
                    trend_score = 0
                    
                    # Enhanced momentum calculation with volume confirmation
                    for window, weight in zip(self.momentum_windows, self.momentum_weights):
                        returns = ticker_data['close'].pct_change(window).iloc[-1]
                        if pd.notna(returns):
                            momentum_score += returns * weight
                    
                    # Volume analysis using multiple timeframes
                    for window in self.volume_windows:
                        vol_ma = ticker_data['volume'].rolling(window).mean()
                        vol_std = ticker_data['volume'].rolling(window).std()
                        current_vol = ticker_data['volume'].iloc[-1]
                        vol_z = (current_vol - vol_ma.iloc[-1]) / (vol_std.iloc[-1] + 1e-6)
                        volume_score += np.clip(vol_z, -2, 2) / len(self.volume_windows)
                    
                    # Trend confirmation using multiple timeframes
                    for window in self.trend_windows:
                        ma = ticker_data['close'].rolling(window).mean().iloc[-1]
                        current_price = ticker_data['close'].iloc[-1]
                        if pd.isna(ma):  # Handle NaN moving averages
                            trend_score += 0  # Neutral trend if MA is NaN
                        else:
                            trend_score += np.sign(current_price - ma) / len(self.trend_windows)
                    
                    # Skip if no valid scores
                    if pd.isna(momentum_score) or pd.isna(volume_score) or pd.isna(trend_score):
                        self.logger.debug(f"Skipping {ticker} due to invalid scores: momentum={momentum_score}, volume={volume_score}, trend={trend_score}")
                        continue
                    
                    # Adjust thresholds based on regime
                    base_threshold = 0.015  # Increased base threshold
                    regime_multipliers = {
                        'normal': 1.0,
                        'high_vol': 1.5,
                        'stress': 2.0
                    }
                    threshold = base_threshold * regime_multipliers[self.market_regime]
                    
                    # Calculate composite signal quality score
                    signal_quality = abs(momentum_score) * (0.4 + 0.3 * abs(volume_score) + 0.3 * abs(trend_score))
                    
                    # Only generate signals when momentum aligns with trend
                    self.logger.debug(f"Signal evaluation for {ticker}: momentum={momentum_score:.4f}, threshold={threshold:.4f}, trend={trend_score:.4f}")
                    if abs(momentum_score) > threshold and np.sign(momentum_score) == np.sign(trend_score):
                        current_price = float(ticker_data['close'].iloc[-1])
                        self.logger.debug(f"Generating signal for {ticker}: momentum={momentum_score:.4f}, trend={trend_score:.4f}, price={current_price:.2f}, max_holding_days={self.max_holding_days}")
                        # Calculate final signal strength incorporating volume
                        signal_strength = min(signal_quality * (0.5 + 0.5 * abs(volume_score)), 1.0)
                        
                        # Enhanced position sizing based on multiple factors
                        base_size = max(10, int(self.orderQuantity * signal_strength))
                        
                        # Apply regime-based position scaling
                        regime_size_multiplier = {
                            'normal': 1.0,
                            'high_vol': 0.6,
                            'stress': 0.3
                        }[self.market_regime]
                        
                        # Adjust size based on volatility
                        vol_adj = 1.0
                        returns_vol = ticker_data['close'].pct_change().std() * np.sqrt(252)
                        if returns_vol > 0.4:  # High volatility
                            vol_adj = 0.5
                        elif returns_vol > 0.3:  # Moderate volatility
                            vol_adj = 0.7
                            
                        position_size = int(base_size * regime_size_multiplier * vol_adj)
                        
                        # Dynamic stop loss based on volatility
                        volatility = ticker_data['close'].pct_change().std() * np.sqrt(252)
                        base_stop = max(
                            self.stop_loss_pct,
                            volatility * np.sqrt(5/252) * 2  # 2 sigma move over 5 days
                        )
                        regime_stop_multiplier = {
                            'normal': 1.0,
                            'high_vol': 1.5,
                            'stress': 2.0
                        }[self.market_regime]
                        stoploss_pct = base_stop * regime_stop_multiplier
                        stoploss_abs = current_price * (1 - stoploss_pct) if momentum_score > 0 else current_price * (1 + stoploss_pct)
                        
                        # Update cooldown period
                        self.cooldown_periods[ticker] = self.system_timestamp
                        
                        # Generate signal
                        signal = dict(
                            symbol=ticker,
                            signal_strength=float(signal_strength),
                            strategy_name=str(self.strategy_name),
                            timestamp=self.system_timestamp,
                            entry_order_type=str(self.orderType),
                            exit_order_type=str(self.exit_order_type),
                            stoploss_pct=float(stoploss_pct),
                            stoploss_abs=float(stoploss_abs),
                            symbol_ltp={str(self.system_timestamp): float(current_price)},
                            timeInForce=str(self.timeInForce),
                            orderQuantity=int(position_size),
                            orderDirection="BUY" if momentum_score > 0 else "SELL",
                            granularity=str(self.granularity),
                            signal_type="BUY_SELL",
                            market_neutral=False,
                            force_entry=True,
                            status="pending",
                            # Temporarily removing max_holding_time for testing
                            # max_holding_time=self.max_holding_days,
                            risk_params=dict(
                                market_regime=str(self.market_regime),
                                momentum_score=float(momentum_score),
                                volume_score=float(volume_score),
                                trend_score=float(trend_score),
                                signal_quality=float(signal_quality),
                                volatility=float(volatility)
                            )
                        )
                        all_signals.append(signal)
                        self.logger.debug(f"Generated signal for {ticker}")
                        
                except Exception as e:
                    self.logger.error(f"Error generating signal for {ticker}: {str(e)}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
        
        # Validate and package signals
        try:
            validated_signals = []
            
            # Skip validation if no signals
            if not all_signals:
                self.logger.info("No signals to validate")
                return return_type, {'signals': [], 'ideal_portfolios': []}, self.tickers
                
            # Validate each signal
            for signal in all_signals:
                try:
                    if not isinstance(signal, dict):
                        self.logger.warning(f"Skipping non-dict signal: {type(signal)}")
                        self.logger.debug(f"Non-dict signal content: {signal}")  # Add content logging
                        continue
                        
                    # All required fields with explicit type conversion
                    validated_signal = {
                        "symbol": str(signal.get("symbol", "")),
                        "signal_strength": float(signal.get("signal_strength", 0.0)),
                        "strategy_name": str(self.strategy_name),
                        "timestamp": signal.get("timestamp", self.system_timestamp),
                        "entry_order_type": str(signal.get("entry_order_type", self.orderType)),
                        "exit_order_type": str(signal.get("exit_order_type", self.exit_order_type)),
                        "stoploss_pct": float(signal.get("stoploss_pct", self.stop_loss_pct)),
                        "stoploss_abs": float(signal.get("stoploss_abs", 0.0)),
                        "symbol_ltp": dict(signal.get("symbol_ltp", {})),
                        "timeInForce": str(signal.get("timeInForce", self.timeInForce)),
                        "orderQuantity": int(signal.get("orderQuantity", 0)),
                        "orderDirection": str(signal.get("orderDirection", "BUY")),
                        "granularity": str(signal.get("granularity", self.granularity)),
                        "signal_type": "BUY_SELL",
                        "market_neutral": False,
                        "status": "pending"
                    }
                    
                    # Add optional fields if present
                    if "max_holding_time" in signal:
                        validated_signal["max_holding_time"] = int(signal["max_holding_time"])
                    if "risk_params" in signal:
                        validated_signal["risk_params"] = dict(signal["risk_params"])
                        
                    # Verify all required fields are present and non-empty
                    if not all([
                        validated_signal["symbol"],
                        validated_signal["signal_strength"] > 0,
                        validated_signal["orderQuantity"] > 0
                    ]):
                        self.logger.warning(f"Signal missing required fields: {validated_signal}")
                        continue
                        
                    validated_signals.append(validated_signal)
                    self.logger.debug(f"Validated signal for {validated_signal['symbol']}")
                    
                except Exception as e:
                    self.logger.error(f"Error validating signal: {str(e)}")
                    continue
            
            if validated_signals:
                self.logger.info(f"Generated {len(validated_signals)} valid momentum signals")
                self.logger.debug(f"Signals before return: {validated_signals}")
                return "signals", {
                    'signals': validated_signals,
                    'ideal_portfolios': []
                }, self.tickers
            else:
                self.logger.info("No valid signals after validation")
                return "signals", {
                    'signals': [],
                    'ideal_portfolios': []
                }, self.tickers
                
        except Exception as e:
            self.logger.error(f"Error in signal validation: {str(e)}")
            return "signals", {
                'signals': [],
                'ideal_portfolios': []
            }, self.tickers
