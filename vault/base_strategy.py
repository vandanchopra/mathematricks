from ast import Or
import numpy as np
from datetime import timedelta, datetime
import pandas as pd
from systems.utils import create_logger, sleeper
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Literal, Any, Tuple

class Order(BaseModel):
    symbol: str
    orderQuantity: int
    orderDirection: Literal["BUY", "SELL"]
    order_type: Literal["MARKET", "LIMIT", "STOPLOSS"]
    price: Optional[float] = None  # For LIMIT and STOPLOSS orders, None for MARKET
    symbol_ltp: Dict[datetime, float]
    timeInForce: Literal["DAY", "Expiry", "IoC", "TTL", "GTC"]
    status: Literal["pending", "open", "closed", 'cancel', 'cancelled', 'modify']
    filled_price: Optional[float] = None
    filled_timestamp: Optional[datetime] = None
    order_id: Optional[str] = None
    entryOrderBool: Optional[bool] = False
    broker_order_id: Optional[str] = None
    history: List[Dict[str, Any]] = []
    message: Optional[str] = None
    brokerage_fee_abs: Optional[float] = None
    slippage_abs: Optional[float] = None
    fresh_update: Optional[bool] = False
    pnl: Optional[float] = None
    pnl_with_fee_and_slippage: Optional[float] = None
    model_config = {
        "arbitrary_types_allowed": True
    }

class Signal(BaseModel):
    strategy_name: str
    signal_id: Optional[str] = None
    timestamp: datetime
    orders: List[Order] = []  # Multiple orders per signal
    signal_strength: float = 1.0  # Signal conviction level (0.0 to 1.0)
    signal_correlation: Optional[float] = None  # For correlated asset signals
    granularity: str
    signal_type: Literal["BUY_SELL"]
    market_neutral: bool
    total_buying_power: float = 0.0
    buying_power_used: float = 0.0
    strategy_inputs: Optional[Dict[str, Any]] = None
    status: str = "pending"
    rejection_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_with_fee_and_slippage: Optional[float] = None
    signal_update: Optional[bool] = False
    model_config = {
        "arbitrary_types_allowed": True
    }

class SignalResponse(BaseModel):
    return_type: Optional[Literal["signals"]] = None
    signals: List[Signal] = Field(default_factory=list)
    tickers: List[str]

    model_config = {
        "arbitrary_types_allowed": True
    }

class BaseStrategy:
    def __init__(self):
        self.strategy_name = 'BaseStrategy'
        self.logger = create_logger(log_level='DEBUG', logger_name='Vault-Strategy', print_to_console=True)
        self.sleeper = sleeper
        
    def get_name(self):
        return self.strategy_name
    
    def generate_signals(
        self, 
        next_rows, 
        market_data_df, 
        system_timestamp, 
        total_buying_power: float = 0.0,
        buying_power_used: float = 0.0,
        open_signals: Optional[List[Signal]] = None
    ) -> Tuple[Optional[str], List[Signal], List[str]]:
        """
        Generate trading signals with dynamic risk management
        Parameters:
            total_buying_power: Total available buying power for the strategy
            buying_power_used: Current buying power being used 
            open_signals: List of currently active signals
        """
        raise NotImplementedError("Subclasses must implement generate_signals")

    def set_params(self, config: Dict[str, Any]):
        raise NotImplementedError
    
    def create_analysis_array_symbol_efficient(self, df, start_date_dt, days_ago_list):
        # Convert 'Date' column to datetime type
        df['Date'] = pd.to_datetime(df['Date'])

        # Filter the dataframe to include only the data needed
        df_pruned = df.iloc[-days_ago_list[-1]:] if len(df) >= days_ago_list[-1] else df

        # Ensure we have data to process
        if not df_pruned.empty:
            # Calculate the target dates for all 'days_ago' at once
            target_dates = start_date_dt - np.array(days_ago_list) * timedelta(days=1)

            # Find the closest date indices for all target dates
            closest_date_indices = (df_pruned['Date'] - target_dates[:, np.newaxis]).abs().idxmin()

            # Get the prices and dates for the closest date indices
            prices = df_pruned.loc[closest_date_indices, 'Adj Close'].tolist()
            dates = df_pruned.loc[closest_date_indices, 'Date'].tolist()

            # Generate the data index
            data_index = ['{}d_ago_price'.format(days_ago) for days_ago in days_ago_list]

            # Convert the list of prices into a numpy array
            analysis_array = np.array(prices)

            return analysis_array, dates, data_index
        else:
            return None, None, None
    
    def create_analysis_array_symbol(self, df, start_date_dt, days_ago_list):
        # Filter the dataframe to include only the data needed
        df_pruned = df.iloc[-days_ago_list[-1]:] if len(df) >= days_ago_list[-1] else df

        # Ensure we have data to process
        if not df_pruned.empty:
            # Define the days for which we want to get the prices
            # Initialize an empty list to hold the prices for the specified days ago
            prices = []
            dates = []
            data_index = []
            
            for days_ago in days_ago_list:
                # Calculate the target date for each 'days_ago'
                target_date = start_date_dt - timedelta(days=days_ago)
                
                # Find the row in df_last_380 that has the closest date to the target date
                closest_date_index = (df_pruned['Date'] - target_date).abs().idxmin()
                
                # Get the price for the closest date
                price_on_closest_date = df_pruned.loc[closest_date_index, 'Adj Close']
                prices.append(price_on_closest_date)
                date_on_closest_date = df_pruned.loc[closest_date_index, 'Date']
                dates.append(date_on_closest_date)
                data_index.append('{}d_ago_price'.format(days_ago))
                        
            # Convert the list of prices into a numpy array
            analysis_array = np.array(prices)

            return analysis_array, dates, data_index
        else:
            return None, None, None
        
    def get_analysis_array(self, symbols, start_date_dt, historical_data_interval, rebalance_frequency):
        symbols_array = []  # To hold symbols
        data_array = []  # To hold analysis arrays from each symbol
        dates_final = {}
        update_data_index_final = True
        days_ago_list = [180, 270, 365]
        data_index_final = []
        
        for count, symbol in enumerate(symbols):
            analysis_array, dates, data_index = self.create_analysis_array_symbol(historical_data_interval[symbol], start_date_dt, days_ago_list)
            analysis_np = np.array(analysis_array)
            if analysis_np.size > 2:
                if update_data_index_final:
                    data_index_final = data_index
                    for days_ago in days_ago_list[:-1]:
                        data_index_final.append('{}d_pct_change'.format(days_ago))
                    for days_ago in days_ago_list[:-1]:
                        data_index_final.append('{}d_pct_growth'.format(days_ago))
                    update_data_index_final = False

                # Append symbol to the index array
                symbols_array.append(symbol)
                
                # Append analysis array to the data array
                data_array.append(analysis_np)
                
                # Append dates to the dates dictionary
                dates_final[symbol] = dates

        return symbols_array, np.array(data_array), data_index_final, dates_final
    
    def get_long_short_symbols(self, long_count, short_count, symbols_array, data_array, data_index):
        # Get the Stocks to Short
        short_symbols = np.random.choice(symbols_array, short_count)

        # Get the Stocks for LONG
        long_symbols = np.random.choice(symbols_array, long_count)

        return long_symbols, short_symbols, symbols_array, data_array, data_index

class Strategy(BaseStrategy):
    pass

if __name__ == '__main__':
    bs = Strategy()
    bs.logger.debug('STRATEGY object created.')