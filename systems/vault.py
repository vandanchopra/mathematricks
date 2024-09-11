class Vault:
    def __init__(self, config_dict):
        self.strategies = self.load_strategies(config_dict['strategies'])
        self.datafeeder_config = self.create_datafeeder_config()

    def load_strategies(self, strategies):
        strategies_dict = {}
        for strategy in strategies:
            # import strategy module and get the class
            strategies_dict[strategy] = getattr(__import__('vault.{}'.format(strategy), fromlist=[strategy]), 'Strategy')()
        return strategies_dict
    
    def create_datafeeder_config(self):
        '''
        # Datafeeder starting parameters (right now it'll be hardcoded, and later, it will be fetched from a datafeeder_config variable)
        data_inputs = {'1min':{'columns':['open', 'high', 'low', 'close', 'volume,' 'SMA15', 'SMA30'], 'lookback':100}, '1d':{'columns':['open', 'high', 'low', 
        'close', 'volume,' 'SMA15', 'SMA30'], 'lookback':100}}
        tickers = ['AAPL', 'MSFT', 'NVDA']

        datafeeder_config = {'data_inputs':data_inputs, 'tickers':tickers}
        '''
        # for each strategy in self.strategies, get the granularity, lookback period, raw data columns and indicators needed and create a dict like above.
        # we also need to get the tickers from each strategy and create a unified list of tickers.
        # This information should get added to the config_dict file, which can then be passed to the DataFeeder class.
        data_inputs = {}
        tickers = []
        for strategy in self.strategies:
            data_input_temp , ticker_temp = strategy.datafeeder_inputs()
            data_inputs[strategy.get_name()] = data_input_temp
            tickers += ticker_temp
        tickers = list(set(tickers))
        
        datafeeder_config = {'data_inputs':data_inputs, 'tickers':tickers}
        return datafeeder_config
        
    def generate_signals(self, market_data_df):
        signals_output = {'signals':[], 'ideal_portfolios':[]}
        ''' 
        for each strategy in self.strategies, get the signals and ideal portfolio.
        combine the signals and ideal portfolio from all strategies and return the combined signals.
        '''
        for strategy in self.strategies:
            signal , ideal_portfolio = strategy.generate_signals()
            if(signal):
                signals_output["signals"].append(signal)
            if(ideal_portfolio):
                signals_output["ideal_portfolios"].append(ideal_portfolio)
                
        
        # run reach strategy and get either the signals or ideal portforlio from each strategy, based on current data.
        # combine the signals from all strategies and return the combined signals.
        return signals_output

class RMS:
    def __init__(self):
        pass
    
    def convert_signals_to_orders(self, signals):
        '''
        calculate total value of portfolio - 200k
        calculate total cash available - 45k
        for each strategy, calculate the amount of cash to be invested - 100k
        for each signal, calculate the amount of cash to be invested - 25k
        max_risk = 0.05
        for each signal, check if the total risk is above max risk, if yes, then reduce the amount of cash to be invested.
        if signal is 'ideal portfolio', then calculate the amount of cash to be invested. Then compare it to the current portfolio, and generate the orders to adjust the portfolio.
        '''
        orders = []
        return orders

if __name__ == '__main__':
    from config import config_dict
    import pandas as pd
    import logging
    from utils import create_logger
    logger = create_logger(log_level=logging.DEBUG, logger_name='datafetcher', print_to_console=True)
    
    vault = Vault(config_dict)
    logger.debug({'datafeeder_config':vault.datafeeder_config})
    # Load dummy data and check if the signals are generated.
    market_data_df = pd.DataFrame()
    signals_output = vault.generate_signals(market_data_df)
    logger.debug({'signals_output':signals_output})
    rms = RMS()
    orders = rms.convert_signals_to_orders(signals_output)
    logger.debug({'orders':orders})
    
    '''
    After this, we need to integrate the Vault class and RMS class with the Mathematricks class in mathematricks.py
    (This integration will only happen after datafeeder and datafetcher classes are developed and tested.)
    '''