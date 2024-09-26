#from ib_insync import *
from copy import deepcopy
import pandas as pd
#from brokers import Brokers
import json
from config import config_dict
import pandas as pd  # Assuming you are using pandas Timestamps
from brokers.brokers import Brokers
from systems.utils import create_logger


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle non-serializable types like pd.Timestamp."""
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()  # Convert Timestamp to ISO format
        return super().default(obj)

class OMS:
    def __init__(self, config):
        self.config_dict = config
        self.logger = create_logger(log_level='DEBUG', logger_name='OMS', print_to_console=True)
        self.brokers = Brokers()
        self.open_orders = self.load_json('db/oms/backtests/open_orders.json')
        self.closed_orders = self.load_json('db/oms/backtests/closed_orders.json')
        self.portfolio = self.load_json('db/oms/backtests/portfolio.json')
        # sync_direction = 'broker-to-oms' if config_dict['run_mode'] == 1 else 'oms-to-broker'
        # self.sync_open_orders(sync_direction)

    def load_json(self, file_name):
        """Load JSON data from the file, or return an empty list if not found."""
        data = []
        try:
            with open(file_name, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            self.logger.debug(f"No {file_name} file found, starting with an empty list.")
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding the {file_name}, please check the format.")
        return data

    def save_json(self, file_name, data):
        """Save the data to a JSON file using the custom encoder."""
        with open(file_name, 'w') as file:
            json.dump(data, file, cls=CustomJSONEncoder)

    def sync_open_orders(self, sync_direction, brokers):
        if sync_direction == 'broker-to-oms':
            for broker in brokers:
                # Fetch open orders from IBKR or SIM broker and update self.open_orders
                open_orders = self.brokers.ib.get_open_orders() if broker == 'IBKR' else self.brokers.sim.get_open_orders()

                if open_orders:
                    for order in open_orders:
                        print(f"Order ID: {order.orderId}, Action: {order.action}, Status: {order.status}")
                        self.open_orders.append(order)
                else:
                    print("No open orders found.")
            
        elif sync_direction == 'oms-to-broker':
            # # Send all open orders from self.open_orders to the broker
            # for order in self.open_orders:
            #     self.ib.update_order(order) if order['broker'] == 'IBKR' else self.brokers.sim.update_order(order)
            raise NotImplementedError("OMS to broker sync is not implemented yet.")
        
        self.save_json('open_orders.json', self.open_orders)

    def process_open_orders(self, open_orders, closed_orders, system_timestamp, market_data_df):
        """Execute a list of multi-leg orders."""
        updated_open_orders = deepcopy(open_orders)
        
        
        for level_1_count, multi_leg_order in enumerate(open_orders):
            for level_2_count, order in enumerate(multi_leg_order):
                order_status = order['status']
                if order_status not in ['closed', 'rejected', 'cancelled']: # Basically if status is 'open' or 'pending'
                    order_open = False
                    if self.config_dict['run_mode'] == 3:
                        order['broker'] = 'SIM'

                    if order['broker'] == 'IBKR':
                        # self.brokers.ib.place_order(
                        #     ticker=order['symbol'],
                        #     exchange='SMART',
                        #     currency='USD',
                        #     orderSide=order['orderSide'],
                        #     orderQuantity=order['orderQuantity'],
                        #     orderType=order['orderType'],
                        #     limit_price=order.get('entryPrice', 0),
                        #     stop_price=order.get('exitPrice', 0)
                        # )
                        # self.open_orders.append(order)
                        # self.sync_open_orders('broker-to-oms', order['broker'])
                        # response_order = self.brokers.ib.execute.execute_order(order, market_data_df=market_data_df)
                        response_order = self.brokers.ib.execute.execute_order(order, market_data_df=market_data_df)
                    
                    elif order['broker'] == 'SIM':
                        response_order = self.brokers.sim.execute.execute_order(order, market_data_df=market_data_df)
                    else:
                        raise Exception(f"Broker {order['broker']} not supported.")
                else:
                    order['fresh_update'] = False
                    response_order = order
                
                '''Update the response order in oms order lists.'''
                if 'fresh_update' in response_order and response_order['fresh_update'] == True:
                    updated_order = response_order
                    if 'history' not in updated_order:
                        updated_order['history'] = []
                    updated_order['history'].append(order)
                    updated_open_orders[level_1_count][level_2_count] = updated_order
                    self.logger.info(f"System Timestamp: {system_timestamp} | Symbol: {updated_order['symbol']} | order_id: {updated_order['order_id']} | Status: {updated_order['status']} | Message: {updated_order['message']}")
                    # self.logger.debug({'updated_open_orders':updated_open_orders})
                
                '''Check if the order is closed and move it to closed orders list.'''
                order_open = False
                for leg in updated_open_orders[level_1_count]:
                    if leg['status'] not in ['closed', 'rejected', 'cancelled']: # Basically if status is 'open' or 'pending'
                        order_open = True
                        break
                if order_open == False:
                    # remove updated_open_orders[level_1_count][level_2_count] from open_orders
                    closed_order = updated_open_orders[level_1_count].pop(level_2_count)
                    closed_orders.append(closed_order)
        
        return updated_open_orders, closed_orders
    
    def execute_orders(self, new_orders, system_timestamp, market_data_df):
        """ add the list of new orders to open_orders list """
        if len(new_orders) > 0:
            self.open_orders.extend(new_orders)
            # self.logger.debug({'new_orders':new_orders})
        
        # Process all open orders
        self.open_orders, self.closed_orders = self.process_open_orders(self.open_orders, self.closed_orders, system_timestamp, market_data_df)

        if len(new_orders) > 0:
            self.logger.warning(f'NOTE NOTE: This is where you save the open orders to a file. len(closed_orders): {len(self.closed_orders)}')
            self.logger.warning('NOTE NOTE: This is where you save the closed orders to a file.')
            self.logger.warning('NOTE NOTE: This is where you save the open portfolio to a file.')
        # self.save_json('open_orders.json', self.open_orders)
        # self.update_order_status()


# Usage example
if __name__ == '__main__':    
    # ibkr_trader = IBKR(None)
    # sim_trader = Sim()
    # instance1 = ibkr.IBKR(ib)
    # sim_instance = sim.Execute()
    # SIM = sim.Sim()
    
    
    # orders = retOrders()  # Example orders to execute

    # instance = OMS(config_dict)
    # instance.execute_orders(orders)
    pass
