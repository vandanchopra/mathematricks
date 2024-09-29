#from ib_insync import *
from copy import deepcopy
from hmac import new
import pandas as pd
#from brokers import Brokers
import json
from config import config_dict
import pandas as pd  # Assuming you are using pandas Timestamps
from brokers.brokers import Brokers
from systems.utils import create_logger, sleeper


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle non-serializable types like pd.Timestamp."""
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()  # Convert Timestamp to ISO format
        return super().default(obj)

class OMS:
    def __init__(self, config):
        self.config_dict = config
        self.logger = create_logger(log_level='INFO', logger_name='OMS', print_to_console=True)
        self.brokers = Brokers()
        self.open_orders = self.load_json('db/oms/backtests/open_orders.json')
        self.closed_orders = self.load_json('db/oms/backtests/closed_orders.json')
        self.portfolio = self.load_json('db/oms/backtests/portfolio.json')
        # sync_direction = 'broker-to-oms' if config_dict['run_mode'] == 1 else 'oms-to-broker'
        # self.sync_open_orders(sync_direction)
        self.profit = 0

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

    def update_trailing_stop_losses(self, order, market_data_df):
        if order['status'] != 'pending' and order['orderType'] == 'stoploss_pct':
            symbol = order['symbol']
            granularity = order['granularity'] # Get minimum granularity from market_data_df
            current_stoploss = order['exitPrice']
            system_timestamp = market_data_df.index.get_level_values(1)[-1]
            current_price = market_data_df.loc[granularity].xs(symbol, axis=1, level='symbol')['close'][-1]
            stoploss_pct = order['stoploss_pct']
            stoploss_abs = order['stoploss_abs']
            orderDirection = order['orderDirection']
            ideal_stoploss = current_price * (1 - stoploss_pct) if orderDirection == 'SELL' else current_price * (1 + stoploss_pct)
            acceptable_loss_pct_deviation = stoploss_pct/5
            current_stoploss_deviation = (ideal_stoploss/current_stoploss)
            update_stoploss = ((ideal_stoploss/current_stoploss) > (1 + acceptable_loss_pct_deviation)) if orderDirection == 'SELL' else ((ideal_stoploss/current_stoploss) < (1 - acceptable_loss_pct_deviation)) 
            
            if update_stoploss:
                order['exitPrice'] = ideal_stoploss
                order['stoploss_abs'] = ideal_stoploss
                order['fresh_update'] = True
                order['status'] = 'modify'
                order['modify_reason'] = 'stoploss_update'
                self.logger.warning('Ideally the stoploss should be updated on minimul granularity available in the market_data_df')
    
        return order
            
    def check_if_all_legs_of_the_order_are_closed(self, multi_leg_order):
        '''Check if the order is closed and move it to closed orders list.'''
        order_open = False
        for leg in multi_leg_order:
            if leg['status'] not in ['closed', 'rejected', 'cancelled']: # Basically if status is 'open' or 'pending'
                order_open = True
                break
        return order_open
    
    def calculate_multi_leg_order_profit(self, multi_leg_order):
        sell_orders = []
        buy_orders = []

        # Separate out all sell and buy orders
        for order in multi_leg_order:
            if order['orderDirection'] == 'SELL' and order['status'] == 'closed':
                sell_orders.append(order)
            elif order['orderDirection'] == 'BUY' and order['status'] == 'closed':
                buy_orders.append(order)

        # Check if there are both buy and sell orders
        if not sell_orders or not buy_orders:
            return "No matching BUY or SELL orders found."

        total_profit = 0
        # Assuming orders are processed in pairs (e.g., first sell with first buy, second sell with second buy)
        for sell_order, buy_order in zip(sell_orders, buy_orders):
            sell_price = sell_order.get('fill_price')
            buy_price = buy_order.get('fill_price')
            quantity = min(sell_order.get('orderQuantity'), buy_order.get('orderQuantity'))

            # Calculate profit for the matching sell and buy orders
            profit = (sell_price - buy_price) * quantity
            total_profit += profit
            
        # self.logger.info({'order':multi_leg_order})
        # self.logger.info(f"Total profit for multi-leg order: {total_profit}")
        # raise AssertionError('PnL calculation not implemented yet.')        
        return total_profit
    
    def remove_closed_orders_from_open_orders_list(self, open_orders, closed_orders):
        updated_open_orders = []
        # check if order is closed
        for level_1_count, multi_leg_order in enumerate(open_orders):
            order_open = self.check_if_all_legs_of_the_order_are_closed(multi_leg_order)
            if order_open == False:
                # remove updated_open_orders[level_1_count][level_2_count] from open_orders
                closed_orders.append(multi_leg_order)
                total_profit = self.calculate_multi_leg_order_profit(multi_leg_order)
                self.profit += total_profit
                multi_leg_order.append({'total_profit': total_profit})
            else:
                # Remove multi_leg_order from self.open_orders
                updated_open_orders.append(multi_leg_order)
        return updated_open_orders, closed_orders
    
    def process_open_orders(self, open_orders, closed_orders, system_timestamp, market_data_df):
        """Execute a list of multi-leg orders."""
        
        for level_1_count, multi_leg_order in enumerate(open_orders):
            for level_2_count, order in enumerate(multi_leg_order):
                updated_order = None
                order_status = order['status']
                if order_status not in ['closed', 'rejected', 'cancelled']: # Basically if status is 'open' or 'pending'
                    order_open = False
                    
                    # First check if Stoploss needs to be udpated.
                    order = response_order = self.update_trailing_stop_losses(order, market_data_df)
                    
                    # Make the order broker SIM if run_mode is Backtest
                    if self.config_dict['run_mode'] == 3:
                        order['broker'] = 'SIM'

                    if order['broker'] == 'IBKR':
                        response_order = self.brokers.ib.execute.execute_order(order, market_data_df=market_data_df, system_timestamp=system_timestamp)
                    
                    elif order['broker'] == 'SIM':
                        response_order = self.brokers.sim.execute.execute_order(order, market_data_df=market_data_df, system_timestamp=system_timestamp)
                    else:
                        raise Exception(f"Broker {order['broker']} not supported.")
                else:
                    order['fresh_update'] = False
                    response_order = order
                
                '''Update the response order in oms order lists.'''
                if order['orderType'] == 'stoploss_pct' and 'order_id' not in order.keys() and order['status'] != 'pending':
                    self.logger.error({'status':order['status']})
                    raise AssertionError('Stoploss order should have an order_id.')
                
                if 'fresh_update' in response_order and response_order['fresh_update'] == True:
                    updated_order = response_order.copy()
                    if 'history' not in updated_order:
                        updated_order['history'] = []
                    # remove history from order
                    order.pop('history', None)
                    updated_order['history'].append(order)
                    open_orders[level_1_count][level_2_count] = updated_order
                    self.logger.info(f"System Timestamp: {system_timestamp} | Symbol: {updated_order['symbol']} | orderType: {updated_order['orderType']} | partial order_id: {updated_order['order_id'][-5:]} | Status: {updated_order['status']} | Message: {updated_order['message']}")
                    # self.logger.debug({'updated_open_orders':updated_open_orders})
                
        updated_open_orders, closed_orders = self.remove_closed_orders_from_open_orders_list(open_orders, closed_orders)
        
        return updated_open_orders, closed_orders
    
    def execute_orders(self, new_orders, system_timestamp, market_data_df):
        """ add the list of new orders to open_orders list """
        if len(new_orders) > 0:
            self.open_orders.extend(new_orders)
            # self.logger.debug({'new_orders':new_orders})
        
        # Process all open orders
        self.open_orders, self.closed_orders = self.process_open_orders(self.open_orders, self.closed_orders, system_timestamp, market_data_df)

        if len(new_orders) > 0:
            self.logger.debug(f'NOTE NOTE: This is where you save the open orders to a file. len(closed_orders): {len(self.closed_orders)}')
            self.logger.debug('NOTE NOTE: This is where you save the closed orders to a file.')
            self.logger.debug('NOTE NOTE: This is where you save the open portfolio to a file.')
        # self.save_json('open_orders.json', self.open_orders)
        # self.update_order_status()

    def close_all_open_orders(self, market_data_df):
        '''Close all open orders.'''
        for multi_leg_order in self.open_orders:
            for order in multi_leg_order:
                if order['status'] in ['open', 'pending']:
                    symbol = order['symbol']
                    order['status'] = 'closed'
                    order['message'] = 'Closed all open orders function called.'
                    granularity = order['granularity']
                    current_price = market_data_df.loc[granularity].xs(symbol, axis=1, level='symbol')['close'][-1]
                    order['fill_price'] = current_price
        
        self.open_orders, self.closed_orders = self.remove_closed_orders_from_open_orders_list(self.open_orders, self.closed_orders)
        
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
