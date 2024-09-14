class Vault:
    def __init__(self, config_dict):
        self.config_dict = config_dict
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
        for strategy in self.strategies.values():
            data_input_temp , ticker_temp = strategy.datafeeder_inputs()
            data_inputs = data_inputs | data_input_temp
            tickers += ticker_temp
        tickers = list(set(tickers))
        
        datafeeder_config = {'data_inputs':data_inputs, 'tickers':tickers}
        self.config_dict["datafeeder_config"] = datafeeder_config
        
        return self.config_dict
        
    def generate_signals(self, market_data_df):
        signals_output = {'signals':[], 'ideal_portfolios':[]}
        ''' 
        for each strategy in self.strategies, get the signals and ideal portfolio.
        combine the signals and ideal portfolio from all strategies and return the combined signals.
        '''
        for strategy in self.strategies.values():
            signal , ideal_portfolio = strategy.generate_signals(market_data_df)
            if(signal):
                signals_output["signals"] += signal
            if(ideal_portfolio):
                signals_output["ideal_portfolios"].append(ideal_portfolio)
                
        
        # run reach strategy and get either the signals or ideal portforlio from each strategy, based on current data.
        # combine the signals from all strategies and return the combined signals.
        return signals_output

class RMS:
    def __init__(self):
        self.total_portfolio = 200000
        self.total_funds_s_1 = 100000
        self.avail_funds_s_1 = 100000
        self.total_funds_s_2 = 100000
        self.avail_funds_s_2 = 100000
        self.max_signal_fund = 25000
        self.max_risk_per = 5
        self.current_portfolio = {}
        self.orders = []
                
        
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
        for signal in signals["signals"]:
            #{'symbol': 'AAPL','signal_strength': -1, 'strategy_name': 'strategy_1', 'timestamp': Timestamp('2023-01-01 01:39:00'), 'entry_order_type': 'MARKET', 'exit_order_type': 'stoploss_pct', 'sl_pct': 0.2, 'sl_abs': 0.3, 'symbol_ltp': 99.58440971032188, 'timeInForce': 'DAY', 'orderQuantity': 50}
            cash_req = signal["symbol_ltp"] * signal["orderQuantity"]
            if(cash_req > self.max_signal_fund):
                signal["orderQuantity"] = int(self.max_signal_fund/signal["symbol_ltp"])
                print("RMS Message: required margin exceeded 25K, reducing order quantity to",signal["orderQuantity"])
                cash_req = signal["symbol_ltp"] * signal["orderQuantity"]
                
            if(cash_req > self.avail_funds_s_1):
                print("RMS Failure: not enough funds")
                continue
                
            if(signal["exit_order_type"] == "stoploss_pct"):
                sl_points = signal["sl_pct"]*0.01*signal["symbol_ltp"]
            elif(signal["exit_order_type"] == "stoploss_abs"):
                sl_points = signal["sl_abs"]
                
            max_risk = self.max_risk_per* 0.01 * self.total_funds_s_1
            order_risk = sl_points*signal["orderQuantity"]
            
            if(order_risk > max_risk):
                signal["orderQuantity"] = int(max_risk/sl_points)
                print("RMS Message: exceeded max risk of 5 percent, reducing order quantity to",signal["orderQuantity"])

            if(signal["signal_strength"] > 0):
                orderSide = "BUY"
                entryPrice = signal["symbol_ltp"]
                slPrice = entryPrice - sl_points
                targetPrice = entryPrice + sl_points
                sign = 1
            elif(signal["signal_strength"] < 0):
                orderSide = "SELL"
                entryPrice = signal["symbol_ltp"]
                slPrice = entryPrice + sl_points
                targetPrice = entryPrice - sl_points
                sign = -1
                
            self.avail_funds_s_1 -= cash_req
            order = {'symbol': signal["symbol"], 'timestamp': signal["timestamp"], "orderSide":orderSide, "entryPrice":entryPrice, "slPrice":slPrice, "targetPrice":targetPrice, "orderType":signal["entry_order_type"], "timeInForce":signal["timeInForce"], "orderQuantity":signal["orderQuantity"]}
            self.current_portfolio[signal["symbol"]] = [sign*signal["orderQuantity"] , cash_req]
            orders.append(order)
        """
        {'symbols': {'V': [0.09, 198.00943623239164],
        'CRM': [0.09, 187.38256532849798],
        'NKE': [0.09, 128.40017716412416],
        'AAPL': [0.09, 144.0302258910366],
        'AMD': [0.09, 131.3198501045325],
        'UNH': [0.1, 187.90253479135262],
        'GOOGL': [0.1, 179.4439295516151],
        'INTC': [0.11, 154.3252555271092],
        'PG': [0.11, 177.19804766115195],
        'NVDA': [0.13, 187.7101330926931],
        'XOM': [-0.07, 101.32675122322887],
        'FB': [-0.09, 120.48480353543567],
        'MSFT': [-0.09, 124.40910425648656],
        'HD': [-0.09, 133.91860299251468],
        'JPM': [-0.1, 133.70959995181218],
        'DIS': [-0.1, 149.05802123807155],
        'AMZN': [-0.1, 110.07858687426095],
        'TSLA': [-0.1, 117.11536211049835],
        'NFLX': [-0.12, 154.64842303201655],
        'JNJ': [-0.13, 171.53675705782933]},
       'strategy_name': 'strategy_2',
       'timestamp': Timestamp('2023-07-19 00:00:00'),
       'entry_order_type': 'MARKET',
       'exit_order_type': 'stoploss_pct',
       'sl_pct': 0.2,
       'sl_abs': 0.3,
       'timeInForce': 'DAY',
       'orderQuantity': 100}
        """
        for portfolio in signals["ideal_portfolios"]:
            weighted_ltp = sum([abs(i[0])*i[1] for i in portfolio["symbols"].values()])
            cash_req = weighted_ltp * portfolio["orderQuantity"]
            if(cash_req > self.avail_funds_s_2):
                portfolio["orderQuantity"] = int(self.avail_funds_s_2/weighted_ltp)
                print("RMS Message: required margin exceeded 100K, reducing order quantity to",portfolio["orderQuantity"])
                cash_req = weighted_ltp * portfolio["orderQuantity"]
            
            for symbol in portfolio["symbols"]:
                temp = portfolio["symbols"][symbol]
                weight = temp[0]
                price = temp[1]     
                funds_s_1 = 0
                funds_s_2 = 0
                order_qty = weight*portfolio["orderQuantity"]
                if(symbol in self.current_portfolio):
                    temp = self.current_portfolio[symbol]
                    final_qty = order_qty - temp[0]
                    if(final_qty == 0):
                        continue
                    if(((order_qty)/temp[0] > 0)):
                        if(abs(order_qty) >= abs(temp[0])):
                            funds_s_2 -= abs(final_qty)*price
                        else:
                            funds_s_1 += abs(final_qty)*price
                    elif(((order_qty)/temp[0] < 0)):
                        funds_s_1 += abs(temp[0])*price
                        funds_s_2 -= abs(order_qty)*price
                else:
                    final_qty = order_qty
                    funds_s_2 -= abs(order_qty)*price

                cash_req = price * abs(final_qty)

                if(cash_req > self.avail_funds_s_1):
                    print("RMS Failure: not enough funds")
                    continue
                
                if(portfolio["exit_order_type"] == "stoploss_pct"):
                    sl_points = portfolio["sl_pct"]*0.01*price
                elif(portfolio["exit_order_type"] == "stoploss_abs"):
                    sl_points = portfolio["sl_abs"]
                    
                max_risk = self.max_risk_per* 0.01 * self.total_funds_s_2
                order_risk = sl_points*abs(final_qty)
                
                if(order_risk > max_risk):
                    final_qty = int((max_risk/sl_points)*(final_qty/abs(final_qty)))
                    print("RMS Message: exceeded max risk of 5 percent, reducing order quantity to",final_qty)
    
                if(final_qty > 0):
                    orderSide = "BUY"
                    entryPrice = price
                    slPrice = entryPrice - sl_points
                    targetPrice = entryPrice + sl_points
                elif(final_qty < 0):
                    orderSide = "SELL"
                    entryPrice = price
                    slPrice = entryPrice + sl_points
                    targetPrice = entryPrice - sl_points

                self.avail_funds_s_1 += funds_s_1
                self.avail_funds_s_2 += funds_s_2
                order = {'symbol': symbol, 'timestamp': portfolio["timestamp"], "orderSide":orderSide, "entryPrice":entryPrice, "slPrice":slPrice, "targetPrice":targetPrice, "orderType":portfolio["entry_order_type"], "timeInForce":portfolio["timeInForce"], "orderQuantity":abs(final_qty)}
                self.current_portfolio[symbol] = [order_qty , abs(order_qty)*price]
                orders.append(order)
        
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