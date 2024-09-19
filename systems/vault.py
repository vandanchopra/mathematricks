import json

class Vault:
    def __init__(self, config_dict):
        self.strategies = self.load_strategies(config_dict['strategies'])
        self.config_dict = self.create_datafeeder_config(config_dict, self.strategies)

    def load_strategies(self, strategy_names):
        strategies_dict = {}
        for strategy in strategy_names:
            # import strategy module and get the class
            strategies_dict[strategy] = getattr(__import__('vault.{}'.format(strategy), fromlist=[strategy]), 'Strategy')()
        return strategies_dict
    
    # def create_datafeeder_config(self, config_dict, strategies):
    #     '''
    #     # Datafeeder starting parameters (right now it'll be hardcoded, and later, it will be fetched from a datafeeder_config variable)
    #     data_inputs = {'1min':{'columns':['open', 'high', 'low', 'close', 'volume,' 'SMA15', 'SMA30'], 'lookback':100}, '1d':{'columns':['open', 'high', 'low', 
    #     'close', 'volume,' 'SMA15', 'SMA30'], 'lookback':100}}
    #     tickers = ['AAPL', 'MSFT', 'NVDA']

    #     datafeeder_config = {'data_inputs':data_inputs, 'tickers':tickers}
    #     '''
    #     # for each strategy in self.strategies, get the granularity, lookback period, raw data columns and indicators needed and create a dict like above.
    #     # we also need to get the tickers from each strategy and create a unified list of tickers.
    #     # This information should get added to the config_dict file, which can then be passed to the DataFeeder class.
    #     data_inputs = {}
    #     list_of_symbols = []
    #     for strategy in strategies.values():
    #         data_input_temp , list_of_symbols_temp = strategy.datafeeder_inputs()
    #         print({'data_input_temp':data_input_temp})
    #         raise AssertionError('MANUALLY STOPPING THE CODE')
    #         data_inputs = data_inputs | data_input_temp
    #         list_of_symbols += list_of_symbols_temp
    #     list_of_symbols = list(set(list_of_symbols))
        
    #     datafeeder_config = {'data_inputs':data_inputs, 'list_of_symbols':list_of_symbols}
    #     config_dict["datafeeder_config"] = datafeeder_config
        
    #     return config_dict
    
    def create_datafeeder_config(self, config_dict, strategies):
        def to_lowercase(d):
            if isinstance(d, dict):
                return {k.lower(): to_lowercase(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [to_lowercase(i) for i in d]
            elif isinstance(d, str):
                return d.lower()
            else:
                return d

        data_inputs = {}
        list_of_symbols = []
        for strategy in strategies.values():
            data_input_temp , list_of_symbols_temp = strategy.datafeeder_inputs()
            data_inputs = data_inputs | data_input_temp
            list_of_symbols += list_of_symbols_temp
        list_of_symbols = list(set(list_of_symbols))
        
        # Convert all the columns to lowercase before returning the datafeeder_config
        datafeeder_config = {'data_inputs':to_lowercase(data_inputs), 'list_of_symbols':list_of_symbols}
        config_dict["datafeeder_config"] = datafeeder_config
        
        return config_dict
        
    def generate_signals(self, market_data_df):
        signals_output = {'signals':[], 'ideal_portfolios':[]}
        ''' 
        for each strategy in self.strategies, get the signals and ideal portfolio.
        combine the signals and ideal portfolio from all strategies and return the combined signals.
        '''
        for strategy in self.strategies.values():
            signal, ideal_portfolio = strategy.generate_signals(market_data_df)
            if(signal):
                signals_output["signals"] += signal
            if(ideal_portfolio):
                signals_output["ideal_portfolios"].append(ideal_portfolio)
                
        with open('db/vault/signals.json', 'w') as file:
            json.dump(signals_output, file)
        # run reach strategy and get either the signals or ideal portforlio from each strategy, based on current data.
        # combine the signals from all strategies and return the combined signals.
        return signals_output

class RMS:
    def __init__(self,config_dict):
        #initializing constants from config dict
        self.config_dict = config_dict
        # self.total_portfolio = config_dict["total_portfolio"]
        # self.total_funds_s_1 = config_dict["total_funds_s_1"]
        # self.avail_funds_s_1 = config_dict["avail_funds_s_1"]
        # self.total_funds_s_2 = config_dict["total_funds_s_2"]
        # self.avail_funds_s_2 = config_dict["avail_funds_s_2"]
        # self.max_signal_fund = config_dict["max_signal_fund"]
        # self.max_risk_per = config_dict["max_risk_per"]
        # with open('db/vault/current_portfolio.json') as file:
        #     self.current_portfolio = json.load(file)
        # with open('db/vault/vault.json') as file:
        #     self.orders = json.load(file)
        
                
    def RMS_check_signal(self, signal,symbol,avail_funds,total_funds,price,weight):
        order_qty = int(weight * signal["orderQuantity"])
        
        cash_req = price*abs(order_qty)
        if(cash_req > self.max_signal_fund):
            order_qty = int((self.max_signal_fund/price)*(order_qty/abs(order_qty)))
            print("RMS Message: required margin exceeded 25K, reducing order quantity to",order_qty)
            cash_req = price*abs(order_qty)
            
        if(cash_req > avail_funds):
            print("RMS Failure: not enough funds")
            return 0,0,0,True
            
        if(signal["exit_order_type"] == "stoploss_pct"):
            sl_points = signal["sl_pct"]*0.01*price
        elif(signal["exit_order_type"] == "stoploss_abs"):
            sl_points = signal["sl_abs"]
            
        max_risk = self.max_risk_per* 0.01 * total_funds
        order_risk = sl_points*abs(order_qty)
        
        if(order_risk > max_risk):
            order_qty = int((max_risk/sl_points)*(order_qty/abs(order_qty)))
            print("RMS Message: exceeded max risk of 5 percent, reducing order quantity to",order_qty)
        
        if(order_qty > 0):
            orderSide = "BUY"
            SLorderSide = "SELL"
            slPrice = price - sl_points
            
        elif(order_qty < 0):
            orderSide = "SELL"
            SLorderSide = "BUY"
            slPrice = price + sl_points
        
        order = [{'symbol': symbol, 
                 'timestamp': signal["timestamp"], 
                 "orderSide":orderSide, 
                 "entryPrice":price, 
                 "orderType":signal["entry_order_type"], 
                 "timeInForce":signal["timeInForce"], 
                 "orderQuantity":abs(order_qty),
                 "strategy_name":signal["strategy_name"],
                 "broker":self.config_dict["broker"]
                },
                 {"symbol": symbol, 
                 "timestamp": signal["timestamp"], 
                 "orderSide":SLorderSide, 
                 "exitPrice":slPrice, 
                 "orderType":signal["exit_order_type"],
                 "timeInForce":signal["timeInForce"], 
                 "orderQuantity":abs(order_qty),
                 "strategy_name":signal["strategy_name"],
                 "broker":self.config_dict["broker"]
                }]
        return cash_req, order_qty, order, False

        
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

        #signals to orders
        for signal in signals["signals"]:
            symbol = signal["symbol"]
            price = signal["symbol_ltp"]
            weight = signal["signal_strength"]
            avail_funds = self.avail_funds_s_1
            total_funds = self.total_funds_s_1

            cash_req, order_qty, order, check = self.RMS_check_signal(signal,symbol,avail_funds,total_funds,price,weight)
            if(check):
                check = False
                continue
            
            self.avail_funds_s_1 -= cash_req
            if(symbol in self.current_portfolio):
                self.current_portfolio[symbol] += order_qty
            else:
                self.current_portfolio[symbol] = order_qty
            self.orders.append(order)
        
        #ideal portfolio to orders
        for portfolio in signals["ideal_portfolios"]:
            weighted_ltp = sum([abs(i[0])*i[1] for i in portfolio["symbols"].values()])
            cash_req = weighted_ltp * portfolio["orderQuantity"]
            #checking to see if the margin req for ideal portfolio is under the available margin
            if(cash_req > self.avail_funds_s_2):
                portfolio["orderQuantity"] = int(self.avail_funds_s_2/weighted_ltp)
                print("RMS Message: required margin exceeded 100K, reducing order quantity to",portfolio["orderQuantity"])
                cash_req = weighted_ltp * portfolio["orderQuantity"]
            
            for symbol in portfolio["symbols"]:
                temp = portfolio["symbols"][symbol]
                weight = temp[0]
                price = temp[1]     
                avail_funds = self.avail_funds_s_2
                total_funds = self.total_funds_s_2
                
                cash_req, order_qty, order, check = self.RMS_check_signal(portfolio,symbol,avail_funds,total_funds,price,weight)
                if(check):
                    check = False
                    continue
                
                self.avail_funds_s_2 -= cash_req
                if(symbol in self.current_portfolio):
                    self.current_portfolio[symbol] += order_qty
                else:
                    self.current_portfolio[symbol] = order_qty
                self.orders.append(order)

        with open('db/vault/current_portfolio.json', 'w') as file:
            json.dump(self.current_portfolio, file)
        with open('db/vault/vault.json', 'w') as file:
            json.dump(self.orders, file)
        return self.orders

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
    rms = RMS(config_dict)
    orders = rms.convert_signals_to_orders(signals_output)
    logger.debug({'orders':orders})
    
    '''
    After this, we need to integrate the Vault class and RMS class with the Mathematricks class in mathematricks.py
    (This integration will only happen after datafeeder and datafetcher classes are developed and tested.)
    '''