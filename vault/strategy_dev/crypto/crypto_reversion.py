from vault.base_strategy import BaseStrategy, Signal, Order
import numpy as np
import pandas as pd
from systems.utils import create_logger, MarketDataExtractor

class Strategy(BaseStrategy):
    def __init__(self, config_dict):
        super().__init__()
        self.logger = create_logger(logger_name='pairs_trading', log_level='INFO')
        self.strategy_name = 'crypto_reversion'
        self.granularity = "1m"
        self.orderType = "MARKET"
        self.stoploss_pct = 0.01
        self.timeInForce = "DAY"
        self.orderQuantity = 1
        self.data_inputs, self.tickers = self.datafeeder_inputs()
        self.market_data_extractor = MarketDataExtractor()

    def get_name(self):
        return self.strategy_name

    def datafeeder_inputs(self):
        tickers = [
            'BTCUSD', 'ETHUSD', 'XRPUSD', 'SOLUSD', 'DOTUSD', 'ADAUSD', 'LINKUSD',
            'MATICUSD', 'ATOMUSD', 'AVAXUSD', 'FILUSD', 'LTCUSD'
        ]
        data_inputs = {'1m': {'columns': ['close'], 'lookback': 50}}
        return data_inputs, tickers

    def generate_signals(self, next_rows, market_data_df, system_timestamp, open_signals=None):
        signals = []
        return_type = None
        open_signals = open_signals or []

        if self.granularity not in market_data_df.index.levels[0]:
            return return_type, signals, self.tickers

        market_df = market_data_df.loc[self.granularity]
        if len(market_df) <= self.data_inputs[self.granularity]['lookback']:
            return return_type, signals, self.tickers

        # Calculate index value based on BTC, ETH, and XRP (equal-weighted index)
        top_assets = ['BTCUSD', 'ETHUSD', 'XRPUSD']
        index_df = pd.DataFrame([self.market_data_extractor.get_market_data_df_symbol_prices(market_data_df, self.granularity, symbol=top_assets[0], column='close'), self.market_data_extractor.get_market_data_df_symbol_prices(market_data_df, self.granularity, symbol=top_assets[1], column='close'), self.market_data_extractor.get_market_data_df_symbol_prices(market_data_df, self.granularity, symbol=top_assets[2], column='close')]
        ).mean().dropna()
        
        for symbol in self.tickers:
            if symbol in top_assets:
                continue
            
            asset_data_df = market_data_df.loc[self.granularity].xs(symbol, level=1, axis=1)
            spread = asset_data_df['close'] - index_df
            mean_spread = spread.rolling(window=20).mean() 
            std_spread = spread.rolling(window=20).std()

            z_score = (spread.iloc[-1] - mean_spread.iloc[-1]) / std_spread.iloc[-1]
            current_price = asset_data_df.iloc[-1]['close']
            
            if abs(z_score) > 2:
                orderDirection = "BUY" if z_score < -2 else "SELL"
                signal_strength = 1
            else:
                signal_strength = 0

            if signal_strength:
                order = Order(
                    symbol=symbol,
                    orderQuantity=self.orderQuantity,
                    orderDirection=orderDirection,
                    order_type=self.orderType,
                    symbol_ltp={system_timestamp: current_price},
                    timeInForce=self.timeInForce,
                    entryOrderBool=True,
                    status="pending"
                )

                stoploss_price = current_price * (1 - self.stoploss_pct) if orderDirection == "BUY" else current_price * (1 + self.stoploss_pct)
                stoploss_order = Order(
                    symbol=symbol,
                    orderQuantity=self.orderQuantity,
                    orderDirection="SELL" if orderDirection == "BUY" else "BUY",
                    order_type="STOPLOSS",
                    price=stoploss_price,
                    symbol_ltp={system_timestamp: current_price},
                    timeInForce=self.timeInForce,
                    entryOrderBool=False,
                    status="pending"
                )

                signal = Signal(
                    strategy_name=self.strategy_name,
                    timestamp=system_timestamp,
                    orders=[order, stoploss_order],
                    signal_strength=signal_strength,
                    granularity=self.granularity,
                    signal_type="BUY_SELL",
                    market_neutral=True,
                    status="pending"
                    
                )
                signals.append(signal)
                self.logger.info(f'SIGNAL GENERATED: {symbol}')
                return_type = 'signals'
        
        return return_type, signals, self.tickers
