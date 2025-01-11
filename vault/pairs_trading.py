import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr
import logging
import datetime
from typing import List, Tuple, Dict, Optional
import os
import pytz
import random
from statsmodels.sandbox.stats.multicomp import multipletests
import json
import optuna
from abc import ABC, abstractmethod
from scipy.stats import linregress
import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from scipy.optimize import minimize
from tabulate import tabulate
from collections import deque


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
BID_ASK_SPREAD_MULTIPLIER = 0.0005
ORDER_BOOK_SLIPPAGE_MULTIPLIER = 0.0001
MIN_DATA_POINTS_FOR_CALCULATION = 2
DEFAULT_ATR_PERIOD = 14
DEFAULT_VOLATILITY_LOOKBACK = 20
MULTIPLE_TESTING_METHOD = 'fdr_bh'
COINTEGRATION_TEST_FREQUENCY_BASE = 10
MAX_DRAWDOWN_CALCULATION_LOOKBACK = 252
DEFAULT_RISK_PER_TRADE = 0.01
TRAILING_STOP_LOSS_OFFSET = 0.01
ROLLING_WINDOW_TEST_SIZE = 50
DEFAULT_INITIAL_CAPITAL = 100000
DEFAULT_WALK_FORWARD_TEST_SIZE = 126 # 6 months roughly
DEFAULT_WALK_FORWARD_TRAINING_SIZE = 252 # 1 year roughly
DEFAULT_OPTUNA_N_TRIALS = 25
DEFAULT_OPTUNA_TIMEOUT = 900
VOLATILITY_LOOKBACK = 20
MIN_COINTEGRATION_PERIOD = 20
TRANSACTION_COST_MULTIPLIER = 0.001
HARDCODED_DATA_DIR = "/mnt/VANDAN_DISK/code_stuff/projects/mathematricks_gagan/db/data/ibkr/1d"
DEFAULT_REBALANCE_THRESHOLD = 0.02
DEFAULT_LOOKBACK_SPREAD = 20
DEFAULT_OUTLIER_THRESHOLD = 3

class ConfigManager:
    """Manages configuration settings for the trading strategy."""
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.config = self._load_config()

    def _load_config(self):
        try:
            with open(self.config_file_path, 'r') as f:
                config = json.load(f)
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Failed to load config file from {self.config_file_path}. Error: {e}. Using defaults.")
            return {}

    def get(self, key, default=None):
        return self.config.get(key, default)

    def get_all(self):
      return self.config

class DataHandler:
    """Handles loading, validating, and preprocessing historical data."""
    def __init__(self, data_dir: str, tickers: List[str], data_frequency: str = "D", timezone: str = "UTC"):
        self.data_dir = data_dir
        self.tickers = tickers
        self.data_frequency = data_frequency
        self.timezone = timezone
        self.historical_data = {}
        self.data_cache = {}  # Cache for rolling window data

    def _validate_data(self, df: pd.DataFrame, ticker: str) -> bool:
        """Validates dataframe for non-empty, numeric values, required columns, and date duplicates"""
        if df.empty:
            logging.warning(f"Empty dataframe for {ticker}")
            return False
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            logging.error(f"Data for {ticker} missing required columns (Open, High, Low, Close). Columns present:{df.columns}")
            return False
        for col in ['Open', 'High', 'Low', 'Close']:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logging.error(f"Column {col} for {ticker} is not numeric")
                return False
        if df.index.duplicated().any():
            logging.error(f"Duplicate dates detected for {ticker}")
            return False
        return True

    def _load_data_from_csv(self, ticker: str) -> Optional[pd.DataFrame]:
        """Loads data from a CSV, handles errors, and returns a dataframe."""
        file_path = os.path.join(self.data_dir, f"{ticker}.csv")
        logging.info(f"Loading data from: {file_path}")
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df.index = df.index.tz_localize(self.timezone)
            if not self._validate_data(df, ticker):
                return None
            logging.info(f"Successfully loaded {ticker} data, shape={df.shape}")
            return df[['Open', 'High', 'Low', 'Close']]
        except FileNotFoundError:
            logging.error(f"Data file not found for {ticker}: {file_path}")
            return None
        except Exception as e:
            logging.error(f"Error loading data for {ticker} at {file_path}: {e}")
            return None

    def load_historical_data(self, training_period: int) -> bool:
        """Loads and stores historical data for all tickers, with error handling."""
        all_data_loaded = True
        for ticker in self.tickers:
            try:
                df = self._load_data_from_csv(ticker)
                if df is None:
                    all_data_loaded = False
                    continue
                self.historical_data[ticker] = df
                logging.info(f"Historical data loaded for {ticker}, shape: {df.shape}")
            except Exception as e:
                logging.error(f"Failed to load data for {ticker}: {e}")
                all_data_loaded = False
        if not all_data_loaded:
            logging.error("Failed to load all historical data.")
        return all_data_loaded

    def preprocess_data(self, training_period: int) -> bool:
        """Preprocesses data: handles missing values, aligns time series."""
        if not self.historical_data:
            logging.error("No historical data, cannot preprocess.")
            return False
        for ticker, df in self.historical_data.items():
            if df.isnull().values.any():
                logging.warning(f"Missing values detected in {ticker}, performing linear interpolation.")
                self.historical_data[ticker] = df.interpolate(method='linear')

        close_prices = []
        for ticker, df in self.historical_data.items():
            close_prices.append(df['Close'])

        try:
            merged_df = pd.concat(close_prices, axis=1, keys=list(self.historical_data.keys()), join="inner")
        except Exception as e:
            logging.error(f"Error aligning price series: {e}")
            return False
        for ticker in self.historical_data.keys():
            self.historical_data[ticker] = merged_df[ticker].to_frame(name="Close")
        logging.info("Data preprocessed.")
        return True

    def get_data(self) -> Dict[str, pd.DataFrame]:
        """Returns the processed historical data."""
        return self.historical_data

    def get_rolling_window_data(self, ticker: str, window_start: datetime.datetime,
                                 window_end: datetime.datetime) -> pd.DataFrame:
        """Retrieves rolling window data for a ticker, uses cache."""
        cache_key = (ticker, window_start, window_end)
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        try:
            df = self.historical_data[ticker].loc[window_start:window_end].copy()
            self.data_cache[cache_key] = df
            return df
        except Exception as e:
            logging.error(f"Error loading rolling data {e}")
            return pd.DataFrame()


class PairSelector:
    """Selects cointegrated pairs based on rolling window."""
    def __init__(self, config_manager: ConfigManager, data_handler: DataHandler):
        self.correlation_threshold = config_manager.get('correlation_threshold', 0.7)
        self.adf_p_value_threshold = config_manager.get('adf_p_value_threshold', 0.05)
        self.min_training_period = config_manager.get('min_training_period', 30)
        self.coint_test_freq_base = config_manager.get('coint_test_freq_base', COINTEGRATION_TEST_FREQUENCY_BASE)
        self.pairs = {}
        self.config_manager = config_manager
        self.data_handler = data_handler

    def _calculate_correlation(self, prices1: np.ndarray, prices2: np.ndarray) -> Optional[float]:
       """Calculates Pearson correlation, with error handling."""
       try:
           if len(prices1) < MIN_DATA_POINTS_FOR_CALCULATION or len(prices2) < MIN_DATA_POINTS_FOR_CALCULATION:
               logging.warning("Insufficient data to calculate correlation.")
               return None
           corr_value, _ = pearsonr(prices1, prices2)
           return corr_value
       except Exception as e:
           logging.error(f"Error calculating correlation: {e}")
           return None

    def _calculate_cointegration(self, prices1: np.ndarray, prices2: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
       """Performs cointegration test and returns p-value and beta, with error handling."""
       try:
           if len(prices1) < MIN_DATA_POINTS_FOR_CALCULATION or len(prices2) < MIN_DATA_POINTS_FOR_CALCULATION:
                logging.warning("Insufficient data to calculate cointegration.")
                return None, None
           X = sm.add_constant(prices2)
           model = sm.OLS(prices1, X)
           results = model.fit()
           residuals = results.resid
           adf_result = adfuller(residuals)
           p_value = adf_result[1]
           beta = results.params[1]
           return p_value, beta
       except Exception as e:
            logging.error(f"Error during cointegration test: {e}")
            return None, None

    def _calculate_cointegration_decay(self, prices1: np.ndarray, prices2: np.ndarray, decay_lookback: int) -> Optional[float]:
        """Calculates the rate of cointegration decay by tracking changes in beta."""
        try:
             if len(prices1) < decay_lookback or len(prices2) < decay_lookback:
                return 0 # No decay if not enough data
             X1 = sm.add_constant(prices2[-decay_lookback:])
             model1 = sm.OLS(prices1[-decay_lookback:], X1)
             beta1 = model1.fit().params[1]
             X2 = sm.add_constant(prices2[:decay_lookback])
             model2 = sm.OLS(prices1[:decay_lookback], X2)
             beta2 = model2.fit().params[1]
             return np.abs(beta1 - beta2)  # Return the absolute change in beta
        except Exception as e:
             logging.error(f"Error during decay test {e}")
             return None

    def select_pairs(self, historical_data: Dict[str, pd.DataFrame], window_start: datetime.datetime,
                     window_end: datetime.datetime, current_pairs: Dict[Tuple[str,str], dict]) -> Dict[Tuple[str,str], Dict[str,float]]:
        """Selects or re-evaluates cointegrated pairs from historical data."""
        pairs = {}
        tickers = list(historical_data.keys())
        p_values = []
        pair_candidates = []
        decay_lookback = self.config_manager.get('cointegration_decay_lookback', 50)
        decay_threshold = self.config_manager.get('decay_threshold', 0.1)
        min_coint_period = self.config_manager.get('min_coint_period', MIN_COINTEGRATION_PERIOD)

        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                ticker1, ticker2 = tickers[i], tickers[j]
                prices1 = self.data_handler.get_rolling_window_data(ticker1, window_start, window_end)['Close'].values
                prices2 = self.data_handler.get_rolling_window_data(ticker2, window_start, window_end)['Close'].values
                pair = (ticker1, ticker2)
                if len(prices1) < self.min_training_period or len(prices2) < self.min_training_period:
                    logging.warning(f"Insufficient data for pair selection of {ticker1}, {ticker2} between {window_start} and {window_end}")
                    continue
                corr_value = self._calculate_correlation(prices1, prices2)
                if corr_value is None or abs(corr_value) < self.correlation_threshold:
                     continue
                p_value, beta = self._calculate_cointegration(prices1, prices2)
                if p_value is not None:
                    p_values.append(p_value)
                    decay_value = self._calculate_cointegration_decay(prices1, prices2, decay_lookback)
                    if len(prices1) >= min_coint_period:
                      pair_candidates.append((ticker1, ticker2, corr_value, p_value, beta, decay_value))
        if not pair_candidates:
           logging.info("No pair candidates found")
           return {}

        # Apply multiple testing correction
        reject, corrected_p_values, _, _ = multipletests(p_values, method = MULTIPLE_TESTING_METHOD, alpha = self.adf_p_value_threshold)

        # Filter by results of multiple testing and decay
        for (k, (ticker1, ticker2, corr_value, p_value, beta, decay_value)) in enumerate(pair_candidates):
            if reject[k] and decay_value is not None and decay_value < decay_threshold:
                pairs[(ticker1, ticker2)] = {
                    'correlation': corr_value,
                    'adf_p_value': corrected_p_values[k],
                    'beta': beta,
                    'decay_value': decay_value
                }
                logging.info(f"Cointegrated pair found: {ticker1}, {ticker2}, correlation: {corr_value}, ADF p-value: {corrected_p_values[k]}, decay_value: {decay_value}")

        # Check for existing pairs and remove any that fail cointegration or decay criteria
        for pair in list(current_pairs.keys()):
            if pair not in pairs and pair in self.pairs:
               del self.pairs[pair]
               logging.info(f"Pair {pair} removed as it no longer cointegrated or fails decay criteria.")
        self.pairs.update(pairs)
        return self.pairs

    def get_pairs(self) -> Dict[Tuple[str,str], Dict[str,float]]:
        """Returns the selected pairs"""
        return self.pairs

class TradingLogic:
    """Implements the core trading logic for the pairs strategy."""
    def __init__(self, config_manager: ConfigManager, data_handler: DataHandler):
        self.entry_threshold = config_manager.get('entry_threshold', 2.0)
        self.exit_threshold = config_manager.get('exit_threshold', 0.5)
        self.stop_loss_atr_multiple = config_manager.get('stop_loss_atr_multiple', 2.0)
        self.volume_window = config_manager.get('volume_window', 20)
        self.max_volume_percentage = config_manager.get('max_volume_percentage', 0.2)
        self.min_risk_reward_ratio = config_manager.get('min_risk_reward_ratio', 1.5)
        self.atr_period = config_manager.get('atr_period', DEFAULT_ATR_PERIOD)
        self.risk_per_trade = config_manager.get('risk_per_trade', DEFAULT_RISK_PER_TRADE)
        self.positions = {}
        self.config_manager = config_manager
        self.data_handler = data_handler

    # ... (rest of TradingLogic class implementation remains the same)

class PairsTradingStrategy:
    """Main strategy class implementing the pairs trading logic."""
    def __init__(self, config_manager: ConfigManager, data_handler: DataHandler):
        self.datafeeder_inputs = {
            'get_inputs': lambda: {
                '1d': {
                    'columns': ['open', 'high', 'low', 'close', 'volume'],
                    'lookback': 365
                }
            }
        }
        self.config_manager = config_manager
        self.data_handler = data_handler
        self.pair_selector = PairSelector(config_manager, data_handler)
        # Define TradingLogic class reference
        from vault.pairs_trading import TradingLogic
        self.trading_logic = TradingLogic(config_manager, data_handler)
        self.trade_log = []
        
    def process_data(self, historical_data: Dict[str, pd.DataFrame],
                    current_prices: Dict[str, float],
                    window_start: datetime.datetime,
                    window_end: datetime.datetime,
                    current_equity: float):
        """Main processing method called by the trading system"""
        # Select pairs
        pairs = self.pair_selector.select_pairs(historical_data, window_start, window_end, {})
        
        # Process trading logic
        self.trading_logic.process_new_data(
            historical_data,
            current_prices,
            self.pair_selector,
            self.trade_log,
            bid_ask_spread_percentage=BID_ASK_SPREAD_MULTIPLIER,
            order_book_slippage=ORDER_BOOK_SLIPPAGE_MULTIPLIER,
            window_start=window_start,
            window_end=window_end,
            current_equity=current_equity
        )
        
        return self.trade_log
