from brokers.brokers import Brokers

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
    orders = []
    oms.execute_orders(orders)