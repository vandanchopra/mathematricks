from brokers.brokers import Brokers
import pandas as pd

class OMS:
    def __init__(self):
        self.brokers = Brokers()
    
    def execute_orders(self, orders):
        for order in orders:
            if order['broker'] == 'IBKR':
                self.brokers.ib.execute_order(order)
            elif order['broker'] == 'SIM':
                self.brokers.sim.execute_order(order)

            else:
                raise Exception("Broker not supported. Please check broker is 'IBKR'.")
            
    ''' 
    for each order in orders:
        send the order to the exchange
        save the trasaction details in a csv (including the broker's orderID)
    
    Now, after all orders have been sent to the exchange, check the status of each order one time, and keep updating the csv with the status of each order
    # Open the CSV file
    # get a list of all open orders
    # update the status.
    # check if any order or position needs to be updated (eg. stoploss on any order needs to be updated.)
    # if there has been an update, update the trasaction to the CSV file.
    '''
    
if __name__ == '__main__':
    oms = OMS()
    # load the orders from the json file
    orders = orders = [[
        {'symbol': 'INTC',
        'timestamp': pd.Timestamp('2023-01-01 01:39:00'),
        'orderSide': 'BUY',
        'entryPrice': 151.4395341959,
        'orderType': 'MARKET',
        'timeInForce': 'DAY',
        'orderQuantity': 10}, 

        {'symbol': 'INTC',
        'timestamp': pd.Timestamp('2023-01-01 01:39:00'),
        'orderSide': 'SELL',
        'exitPrice': 149.4395341959,
        'orderType': 'STOPLOSS-MARKET',
        'timeInForce': 'DAY',
        'orderQuantity': 10}]] # Orders might have 1 leg or multiple legs. Each leg is a dictionary. and the OMS needs to implement the logic to handle multiple legs.
    oms.execute_orders(orders)