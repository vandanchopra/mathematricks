from brokers.brokers import Brokers
import pandas as pd

class OMS:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.brokers = Brokers()
        self.open_orders = self.load_open_orders()
        sync_direction = 'broker-to-oms' if config_dict['run_mode'] == 1 else 'oms-to-broker'
        self.sync_open_orders(sync_direction)
        
    def load_open_orders(self):
        ''' 
        Open the json and load all the open orders into oms.orders'''
        open_orders = [] # Remove this line once u have written the code for this function.
        return open_orders
    
    def save_open_orders(self):
        '''
        Update the open orders in the json file.
        return new list of open orders.
        '''
        self.open_orders = self.load_open_orders()
    
    def sync_open_orders(self, sync_direction):
        if sync_direction == 'broker-to-oms':
            ''' 
            Get all the open orders from IBKR and update the self.open_orders with the new orders.
            '''
            pass
        elif sync_direction == 'oms-to-broker':
            ''' 
            Get all the open orders from self.open_orders and update the IBKR with the new orders.
            '''
            pass
        else:
            raise Exception("sync_direction not supported. Please check sync_direction is 'ib-to-oms' or 'oms-to-ib'.")
        
        # Now that the orders have been synced, update the open orders in the json file.
        self.save_open_orders()
        
    def run_oms(self, new_orders):
        for multi_leg_order in new_orders:
            for order in multi_leg_order:
                run_mode = self.config_dict['run_mode']
                # Override the broker to SIM if the run_mode is 3
                if run_mode == 3:
                    order['broker']='SIM'
                    
                # Push the order to the broker    
                if order['broker'] == 'IBKR':
                    order = self.brokers.ib.execute_order(order)
                elif order['broker'] == 'SIM':
                    order = self.brokers.sim.execute_order(order)
                else:
                    raise Exception("Broker not supported. Please check broker is 'IBKR'.")
            # Update the multi_leg_order to oms.orders with the orderID from the broker.
        
        # Now that all the orders have been sent to the broker, check the status of each order one time, and keep updating the oms.orders with the status of each order.
        sync_direction = 'broker-to-oms'
        self.sync_open_orders(sync_direction)
            
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
    from config import config_dict
    oms = OMS(config_dict)
    # load the orders from the json file
    orders = [[
        # Entry order
        {'symbol': 'INTC',
        'timestamp': pd.Timestamp('2023-01-01 01:39:00'),
        'orderSide': 'BUY',
        'entryPrice': 151.4395341959,
        'orderType': 'MARKET',
        'timeInForce': 'DAY',
        'orderQuantity': 10, 
        'strategy_name': 'SMA15-SMA30', 
        'broker': 'IBKR'}, 
        # Exit Order / Trail Stop Order
        {'symbol': 'INTC',
        'timestamp': pd.Timestamp('2023-01-01 01:39:00'),
        'orderSide': 'SELL',
        'exitPrice': 149.4395341959,
        'orderType': 'STOPLOSS-TRAILING-MARKET',
        'timeInForce': 'DAY',
        'orderQuantity': 10,
        'strategy_name': 'SMA15-SMA30', 
        'broker': 'IBKR'}, 
        # Exit Order / Trail Stop Order
        {'symbol': 'INTC',
        'timestamp': pd.Timestamp('2023-01-01 01:39:00'),
        'orderSide': 'SELL',
        'exitPrice': 151.43,
        'orderType': 'STOPLOSS-TRAILING-MARKET',
        'timeInForce': 'DAY',
        'orderQuantity': 10,
        'strategy_name': 'SMA15-SMA30', 
        'broker': 'IBKR'}
        ]] # Orders might have 1 leg or multiple legs. Each leg is a dictionary. and the OMS needs to implement the logic to handle multiple legs.
    oms.execute_orders(orders)