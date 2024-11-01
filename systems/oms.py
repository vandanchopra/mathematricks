#from ib_insync import *
from copy import deepcopy
import sys
from urllib import response
import pandas as pd
#from brokers import Brokers
import json
from config import config_dict
import pandas as pd  # Assuming you are using pandas Timestamps
from brokers.brokers import Brokers
from systems.utils import create_logger, generate_hash_id
from pprint import pprint
from systems.performance_reporter import PerformanceReporter

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
        self.profit = 0
        self.margin_available = self.update_all_margin_available()
        self.open_orders = self.load_json('db/oms/backtests/open_orders.json')
        self.closed_orders = self.load_json('db/oms/backtests/closed_orders.json')
        self.portfolio = {}
        self.reporter = PerformanceReporter()
        self.granularity_lookup_dict = {"1m":60,"2m":120,"5m":300,"1d":86400}
    
    def get_strategy_margin_available(self, strategy_name):
        num_of_strategy_count = len(self.config_dict["strategies"])
        strategy_margin = self.margin_available['all']['total_margin_available'] / num_of_strategy_count
        
        return strategy_margin
    
    def update_all_margin_available(self):
        trading_currency = self.config_dict['trading_currency']
        base_currency = self.config_dict['base_currency']
        base_currency_to_trading_currency_exchange_rate = self.config_dict['base_currency_to_trading_currency_exchange_rate']
        margin_dict = {}
        for broker in self.config_dict['account_info']:
            if broker not in margin_dict:
                margin_dict[broker] = {}
            for account_number in self.config_dict['account_info'][broker]:
                if account_number not in margin_dict[broker]:
                    margin_dict[broker][account_number] = {'combined':{}}
                if broker == 'sim':
                    margin_dict[broker][account_number]['combined'] = self.brokers.sim.execute.create_account_summary(trading_currency, base_currency, base_currency_to_trading_currency_exchange_rate, self.config_dict['account_info'][broker][account_number])
                elif broker == 'ibkr':
                    if self.config_dict['run_mode'] in [1,2]:
                        margin_dict[broker][account_number]['combined'] = self.brokers.ib.execute.get_account_summary(trading_currency, base_currency, account_number)
                else:
                    raise AssertionError(f"Broker {broker} not supported.")
                
        '''STEP 2: Calculate the total margin available for all strategies'''
        for broker in margin_dict:
            for account_number in margin_dict[broker]:
                for strategy_name in self.config_dict["strategies"]:
                    strategy_name = strategy_name.split('.')[-1] if '.' in strategy_name else strategy_name
                    if strategy_name not in margin_dict[broker][account_number]:
                        margin_dict[broker][account_number][strategy_name] = {}
                    for currency in margin_dict[broker][account_number]['combined']:
                        if currency not in margin_dict[broker][account_number][strategy_name]:
                            margin_dict[broker][account_number][strategy_name][currency] = {}
                        for key in margin_dict[broker][account_number]['combined'][currency]:
                            if key not in ['cushion', 'margin_multiplier', 'pct_of_margin_used']:
                                margin_dict[broker][account_number][strategy_name][currency][key] = margin_dict[broker][account_number]['combined'][currency][key] / len(self.config_dict["strategies"])
                            else:
                                margin_dict[broker][account_number][strategy_name][currency][key] = margin_dict[broker][account_number]['combined'][currency][key]

        return margin_dict
    
    def update_all_margin_available_old(self):
        self.margin_available = {}
        if 'all' not in self.margin_available:
            self.margin_available['all'] = {}
        self.margin_available['all']['total_margin_available'] = self.config_dict["oms"]["funds_available"] * (1 - self.config_dict["risk_management"]["margin_reserve_pct"])
        self.margin_available['all']['current_margin_available'] = self.config_dict["oms"]["funds_available"] * (1 - self.config_dict["risk_management"]["margin_reserve_pct"])
        
        for strategy_name in self.config_dict["strategies"]:
            strategy_name = strategy_name.split('.')[-1] if '.' in strategy_name else strategy_name
            if strategy_name not in self.margin_available:
                self.margin_available[strategy_name] = {}
            self.margin_available[strategy_name]['total_margin_available'] = self.get_strategy_margin_available(strategy_name)
            self.margin_available[strategy_name]['current_margin_available'] = self.get_strategy_margin_available(strategy_name)
        return self.margin_available
    
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

    def create_postions_from_open_orders(self, open_orders):
        positions = {'all':{}}
        pending_orders_sim = {'all':{}}
        for open_order in open_orders:
            for order in open_order:
                symbol = order['symbol']
                strategy_name = order['strategy_name']
                
                if order['status'] not in ['pending', 'open']:
                    # If stragegy_name not in current_positions, then add it
                    if strategy_name not in positions:
                        positions[strategy_name] = {}
                    # if symbol not in strategy_name, then add it
                    if symbol not in positions[strategy_name]:
                        positions[strategy_name][symbol] = 0
                    if symbol not in positions['all']:
                        positions['all'][symbol] = 0
                    order_direction_multiplier = 1 if order['orderDirection'].lower() == 'buy' else -1
                    positions['all'][symbol] += order['orderQuantity'] * order_direction_multiplier
                    positions[strategy_name][symbol] += order['orderQuantity'] * order_direction_multiplier
                    # print(f"Order ADDED: symbol: {symbol}, status: {order['status']}, strategy_name: {strategy_name}, orderDirection: {order['orderDirection'].upper()}, orderQuantity: {order['orderQuantity']}")
                else:
                    if strategy_name not in pending_orders_sim:
                        pending_orders_sim[strategy_name] = {}
                    # if symbol not in strategy_name, then add it
                    if symbol not in pending_orders_sim[strategy_name]:
                        pending_orders_sim[strategy_name][symbol] = 0
                    if symbol not in pending_orders_sim['all']:
                        pending_orders_sim['all'][symbol] = 0
                    # print(f"Order is still pending: symbol: {symbol}, status: {order['status']}, strategy_name: {strategy_name}, orderDirection: {order['orderDirection'].upper()}, orderQuantity: {order['orderQuantity']}")
                    order_direction_multiplier = 1 if order['orderDirection'].lower() == 'buy' else -1
                    pending_orders_sim['all'][symbol] += order['orderQuantity'] * order_direction_multiplier
                    pending_orders_sim[strategy_name][symbol] += order['orderQuantity'] * order_direction_multiplier
            
        return positions, pending_orders_sim
    
    def get_account_balances(self):
        account_summary = self.brokers.ib.ib.accountSummary()
        account_balances_dict = {}
        tags_of_interest = ['TotalCashValue', 'AvailableFunds', 'NetLiquidation', 'BuyingPower', 'MaintMarginReq',
                            'GrossPositionValue'
                            ]
        # Find and display relevant fields for account balance and margin
        for count, item in enumerate(account_summary):
            if item.account in ['DU7994930', 'U13152795']:
                if item.tag in tags_of_interest:
                    account_balances_dict[item.tag] = item.value
                    
        for count, item in enumerate(account_summary):
            if item.tag == 'ExchangeRate' and item.currency == 'USD':
                account_balances_dict[item.tag] = item.value
        return account_balances_dict
    
    def sync_open_orders(self, sync_direction, market_data_df, system_timestamp, brokers=['IBKR']):
        self.brokers.ib.execute.check_ib_connection()

        def get_positions_to_buy_for_sync(market_data_df, sim_open_orders, sim_net_liquidation_value, ibkr_available_funds, open_positions_ibkr):
            # calculate % of balance being used for sim.
            total_ratio_of_each_order_to_net_liquidation_value = {}
            actual_quantity_to_buy = {}
            ideal_quantity_to_buy = {}
            for sim_open_order in sim_open_orders:
                for order in sim_open_order:
                    if order['status'] == 'closed':
                        symbol = order['symbol']
                        if symbol not in total_ratio_of_each_order_to_net_liquidation_value:
                            total_ratio_of_each_order_to_net_liquidation_value[symbol] = {'open_positions':[]}
                        total_ratio_of_each_order_to_net_liquidation_value[symbol]['open_positions'].append({'entryPrice': order['fill_price'], 'orderQuantity': order['orderQuantity']})
            # ratio_of_funds_utilized_sim = total_value_of_open_positions_sim / sim_net_liquidation_value

            for symbol in total_ratio_of_each_order_to_net_liquidation_value:
                # First get the value of all orders.
                symbol_invested_value = 0
                total_quantity_open_symbol = 0
                for order_entry in total_ratio_of_each_order_to_net_liquidation_value[symbol]['open_positions']:
                    symbol_invested_value += order_entry['entryPrice'] * order_entry['orderQuantity']
                    total_quantity_open_symbol += order_entry['orderQuantity']
                # Then get the ratio to sim_net_liquidation_value
                total_ratio_of_each_order_to_net_liquidation_value[symbol]['ratio_of_invested_value_to_sim_net_liquidation_value'] = symbol_invested_value / sim_net_liquidation_value
                # Then calculate the ratio of funds to be alloted to symbol on IBKR
                total_ratio_of_each_order_to_net_liquidation_value[symbol]['availble_funds_for_asset_from_ibkr'] = total_ratio_of_each_order_to_net_liquidation_value[symbol]['ratio_of_invested_value_to_sim_net_liquidation_value'] * ibkr_available_funds
                
                # self.logger.debug({'symbol':symbol, 'symbol_invested_value':symbol_invested_value, 'availble_funds_for_asset_from_ibkr':total_ratio_of_each_order_to_net_liquidation_value[symbol]['availble_funds_for_asset_from_ibkr'], 'ratio_of_invested_value_to_sim_net_liquidation_value': total_ratio_of_each_order_to_net_liquidation_value[symbol]['ratio_of_invested_value_to_sim_net_liquidation_value'], 'sim_net_liquidation_value':sim_net_liquidation_value, 'ibkr_available_funds':ibkr_available_funds})
                # Then get the current value of the stock.
                #### Get the minimum granularity from market_data_df
                self.available_granularities = market_data_df.index.get_level_values(0).unique()
                self.min_granularity_val = min([self.granularity_lookup_dict[granularity] for granularity in self.available_granularities])
                self.min_granularity = list(self.granularity_lookup_dict.keys())[list(self.granularity_lookup_dict.values()).index(self.min_granularity_val)]
                total_ratio_of_each_order_to_net_liquidation_value[symbol]['current_price'] = market_data_df.loc[self.min_granularity].xs(symbol, axis=1, level='symbol')['close'][-1] # THIS VALUE IS HYPOTHETICAL
                # Then calculate the closest integer number of stocks that can be bought with the funds available for IBKR
                total_ratio_of_each_order_to_net_liquidation_value[symbol]['ideal_quantity_to_buy'] = int(total_ratio_of_each_order_to_net_liquidation_value[symbol]['availble_funds_for_asset_from_ibkr'] / total_ratio_of_each_order_to_net_liquidation_value[symbol]['current_price'])
                # self.logger.debug({'current_price':total_ratio_of_each_order_to_net_liquidation_value[symbol]['current_price'], 'fill_price':symbol_invested_value/total_quantity_open_symbol})
                # Find out how much of the quantity is already bought
                for ibkr_open_position in open_positions_ibkr:
                    if ibkr_open_position.contract.symbol == symbol:
                        total_ratio_of_each_order_to_net_liquidation_value[symbol]['quantity_bought'] = ibkr_open_position.position
                        break
                    else:
                        total_ratio_of_each_order_to_net_liquidation_value[symbol]['quantity_bought'] = 0
                # Find out actual quantity to buy
                total_ratio_of_each_order_to_net_liquidation_value[symbol]['actual_quantity_to_buy'] = total_ratio_of_each_order_to_net_liquidation_value[symbol]['ideal_quantity_to_buy'] - (total_ratio_of_each_order_to_net_liquidation_value[symbol]['quantity_bought'] if 'quantity_bought' in total_ratio_of_each_order_to_net_liquidation_value[symbol] else 0)
                if total_ratio_of_each_order_to_net_liquidation_value[symbol]['actual_quantity_to_buy'] > 0:
                    actual_quantity_to_buy[symbol] = total_ratio_of_each_order_to_net_liquidation_value[symbol]['actual_quantity_to_buy']
                if total_ratio_of_each_order_to_net_liquidation_value[symbol]['ideal_quantity_to_buy'] > 0:
                    ideal_quantity_to_buy[symbol] = total_ratio_of_each_order_to_net_liquidation_value[symbol]['ideal_quantity_to_buy']
            return actual_quantity_to_buy, ideal_quantity_to_buy
        
        def create_entry_orders_for_sync(actual_quantity_to_buy, market_data_df, sim_open_orders, system_timestamp):
            new_entry_orders = []
            
            for symbol, actual_symbol_quantity in actual_quantity_to_buy.items():
                run_once = 0
                for multi_leg_order in sim_open_orders:
                    if run_once == 0:
                        for order in multi_leg_order:
                            if order['symbol'] == symbol and 'entryPrice' in order:
                                new_order = deepcopy(order)
                                new_order['orderQuantity'] = actual_symbol_quantity
                                # fix entry price
                                new_order['entryPrice'] = market_data_df.loc['1d'].xs(symbol, axis=1, level='symbol')['close'][-1]
                                # fix broker
                                new_order['broker'] = 'IBKR'
                                # status
                                new_order['status'] = 'pending'
                                # make fill_price = 0
                                new_order['fill_price'] = 0
                                # filled_timestamp = None
                                new_order['filled_timestamp'] = None
                                # broker_order_id = None
                                new_order['broker_order_id'] = None
                                # Remove fresh_update
                                if 'fresh_update' in new_order:
                                    new_order.pop('fresh_update')
                                # Remove message
                                if 'message' in new_order:
                                    new_order.pop('message')
                                # Remove history
                                if 'history' in new_order:
                                    new_order.pop('history')
                                
                                # order_id = create new order_id
                                new_order['original_order_id'] = new_order['order_id']
                                new_order['order_id'] = generate_hash_id(new_order, system_timestamp)
                                new_entry_orders.append([new_order]) # Order is being sent as a single-leg multi-leg order, that's why it's in a list.
                                # self.logger.debug({'symbol':symbol, 'new_order':new_order})
                                run_once += 1
                            
            return new_entry_orders
        
        def create_exit_orders_for_sync(sim_open_orders, new_entry_orders, system_timestamp, ideal_quantity_to_buy):
            unfilled_orders_ibkr = self.brokers.ib.ib.reqOpenOrders()
            new_exit_orders = []
            for order in new_entry_orders:
                # self.logger.info({'order':order})
                symbol = order[0]['symbol']
                run_once = 0
                for multi_leg_order in sim_open_orders:
                    if run_once == 0:
                        for order_2 in multi_leg_order:
                            if order_2['symbol'] == symbol and 'exitPrice' in order_2:
                                new_order = deepcopy(order_2)
                                
                                # Get the unfilled order quantity for the symbol that is already in IBKR
                                unfilled_order_quantity_ibkr_for_symbol = 0
                                for unfilled_order in unfilled_orders_ibkr:
                                    # self.logger.debug({'unfilled_order':unfilled_order})
                                    if unfilled_order.contract.symbol == symbol:
                                        unfilled_order_quantity_ibkr_for_symbol += unfilled_order.order.totalQuantity
                                
                                # Get the ideal quantity to buy for the symbol that we are trying to achieve with the new entry orders
                                ideal_quantity_to_buy_for_symbol = ideal_quantity_to_buy[symbol]
                                # Calculate the order quantity as the difference between the ideal quantity and the unfilled quantity already on ibkr
                                orderQuantity = ideal_quantity_to_buy_for_symbol - unfilled_order_quantity_ibkr_for_symbol
                                new_order['orderQuantity'] = orderQuantity
                                # fix entry price
                                new_order['exitPrice'] = order_2['stoploss_abs']
                                # fix broker
                                new_order['broker'] = 'IBKR'
                                # status
                                new_order['status'] = 'pending'
                                # make fill_price = 0
                                new_order['fill_price'] = 0
                                # filled_timestamp = None
                                new_order['filled_timestamp'] = None
                                # broker_order_id = None
                                new_order['broker_order_id'] = None
                                # Remove fresh_update
                                if 'fresh_update' in new_order:
                                    new_order.pop('fresh_update')
                                # Remove message
                                if 'message' in new_order:
                                    new_order.pop('message')
                                # Remove history
                                if 'history' in new_order:
                                    new_order.pop('history')
                                
                                # order_id = create new order_id
                                if 'order_id' in new_order:
                                    new_order['original_order_id'] = new_order['order_id']
                                new_order['order_id'] = generate_hash_id(new_order, system_timestamp)
                                new_exit_orders.append([new_order]) # Order is being sent as a single-leg multi-leg order, that's why it's in a list.
                                run_once += 1
            return new_exit_orders
        
        new_orders = []
        if sync_direction == 'broker-to-oms':
            raise NotImplementedError(f'This {sync_direction} direction is not supported yet.')
        elif sync_direction == 'oms-to-broker':
            sim_open_orders = deepcopy(self.open_orders)
            sim_net_liquidation_value = self.margin_available['all']['total_margin_available']
            open_positions_ibkr = self.brokers.ib.ib.positions()
            # get open positions from SIM
            open_positions_sim, pending_orders_sim = self.create_postions_from_open_orders(sim_open_orders)
            # get account balance for IBKR
            account_balances_dict = self.get_account_balances()
            ibkr_available_funds = float(account_balances_dict['AvailableFunds']) / float(account_balances_dict['ExchangeRate'])
            # get account balance for SIM
            sim_net_liquidation_value = sim_net_liquidation_value
            # get positions to buy
            actual_quantity_to_buy, ideal_quantity_to_buy = get_positions_to_buy_for_sync(market_data_df, sim_open_orders, sim_net_liquidation_value, ibkr_available_funds, open_positions_ibkr)
            
            self.logger.info({'actual_quantity_to_buy':actual_quantity_to_buy, 'ideal_quantity_to_buy':ideal_quantity_to_buy})
            # Create new orders
            new_entry_orders = create_entry_orders_for_sync(actual_quantity_to_buy, market_data_df, sim_open_orders, system_timestamp)
            new_orders.extend(new_entry_orders)
            new_exit_orders = create_exit_orders_for_sync(sim_open_orders, new_entry_orders, system_timestamp, ideal_quantity_to_buy)
            new_orders.extend(new_exit_orders)
            self.open_orders = []
        return new_orders

    def get_open_orders(self, broker):
        if broker == 'IBKR':
            return self.brokers.ib.get_open_orders()
        elif broker == 'SIM':
            return self.open_orders['SIM']
        else:
            raise AssertionError(f"Broker {broker} not supported.")

    def update_trailing_stop_losses(self, order, market_data_df):
        if order['status'] != 'pending' and order['orderType'] == 'stoploss_pct':
            symbol = order['symbol']
            granularity = order['granularity'] # Get minimum granularity from market_data_df
            system_timestamp = market_data_df.index.get_level_values(1)[-1]
            stoploss_pct = order['stoploss_pct']
            current_price = market_data_df.loc[granularity].xs(symbol, axis=1, level='symbol')['close'][-1]
            stoploss_abs = order['stoploss_abs']
            orderDirection = order['orderDirection']

            current_stoploss = order['exitPrice']
            ideal_stoploss = current_price * (1 - stoploss_pct) if orderDirection == 'SELL' else current_price * (1 + stoploss_pct)
            acceptable_loss_pct_deviation = stoploss_pct/5
            acceptable_spotloss = current_price * (1 - (stoploss_pct+acceptable_loss_pct_deviation)) if orderDirection == 'SELL' else current_price * (1 + (stoploss_pct-acceptable_loss_pct_deviation))
            update_stoploss = current_stoploss < acceptable_spotloss and ideal_stoploss > current_stoploss if orderDirection == 'SELL' else current_stoploss > acceptable_spotloss and ideal_stoploss < current_stoploss
            
            if update_stoploss:
                # self.logger.debug({'system_timestamp':system_timestamp, 'update_stoploss':update_stoploss, 'symbol':symbol, 'orderDirection':orderDirection, 'current_price':current_price, 'stoploss_abs':stoploss_abs, 'current_stoploss':current_stoploss, 'ideal_stoploss':ideal_stoploss, 'acceptable_spotloss':acceptable_spotloss, 'stoploss_pct':stoploss_pct, 'acceptable_loss_pct_deviation':acceptable_loss_pct_deviation})
                # if orderDirection == 'BUY':
                    # raise AssertionError({'symbol':symbol, 'orderDirection':orderDirection, 'stoploss_abs':stoploss_abs, 'ideal_stoploss':ideal_stoploss})
                order['exitPrice'] = ideal_stoploss
                order['stoploss_abs'] = ideal_stoploss
                order['status'] = 'modify'
                # order['modify_reason'] = 'stoploss_update'
                order['message'] = f'Stoploss updated due to trailing stoploss update from: Current SL: {current_stoploss} to New SL: {ideal_stoploss}'
                # self.logger.debug(f'System Timestamp: {system_timestamp}, Symbol: {symbol}, orderDirection: {orderDirection}, message: {order["message"]}')
                # self.logger.debug('Ideally the stoploss should be updated on minimul granularity available in the market_data_df')
    
        return order
            
    def check_if_all_legs_of_the_order_are_closed(self, multi_leg_order):
        '''Check if the order is closed and move it to closed orders list.'''
        order_open = False
        for leg in multi_leg_order:
            if leg['status'] not in ['closed', 'rejected', 'cancelled']: # Basically if status is 'open' or 'pending'
                order_open = True
                break
        return order_open
    
    def remove_closed_orders_from_open_orders_list(self, open_orders, closed_orders):
        updated_open_orders = []
        # check if order is closed
        for level_1_count, multi_leg_order in enumerate(open_orders):
            order_open = self.check_if_all_legs_of_the_order_are_closed(multi_leg_order)
            # self.logger.debug({f'Symbol: {multi_leg_order[0]["symbol"]}, order_open':order_open})
            if not order_open:
                # remove updated_open_orders[level_1_count][level_2_count] from open_orders
                if len(multi_leg_order) == 0:
                    raise AssertionError(f"Multi-leg order is empty: {multi_leg_order}")
                closed_orders.append(multi_leg_order)
            else:
                # Remove multi_leg_order from self.open_orders
                updated_open_orders.append(multi_leg_order)
            
        return updated_open_orders, closed_orders
    
    def update_porfolio(self, response_order):
        if response_order['status'] == 'closed':
            '''Update the portfolio with the response_order.'''
            # self.logger.debug(f"Updating portfolio with response_order: {response_order}")
            strategy_name = response_order['strategy_name']
            symbol = response_order['symbol']
            orderDirection = response_order['orderDirection']
            orderDirection_multiplier = 1 if orderDirection == 'BUY' else -1
            orderQuantity = response_order['orderQuantity']
            price = response_order['fill_price']
            
            # Strategy Level Update
            if strategy_name not in self.portfolio:
                self.portfolio[strategy_name] = {}
            if symbol not in self.portfolio[strategy_name]:
                self.portfolio[strategy_name][symbol] = {}
            if 'position' not in self.portfolio[strategy_name][symbol]:
                self.portfolio[strategy_name][symbol]['position'] = 0
            if 'average_price' not in self.portfolio[strategy_name][symbol]:
                self.portfolio[strategy_name][symbol]['average_price'] = 0
            if 'total_value' not in self.portfolio[strategy_name][symbol]:
                self.portfolio[strategy_name][symbol]['total_value'] = 0

            # Overall Portfolio Update
            if 'all' not in self.portfolio:
                self.portfolio['all'] = {}
            if symbol not in self.portfolio['all']:
                self.portfolio['all'][symbol] = {}
            if 'position' not in self.portfolio['all'][symbol]:
                self.portfolio['all'][symbol]['position'] = 0
            if 'average_price' not in self.portfolio['all'][symbol]:
                self.portfolio['all'][symbol]['average_price'] = 0
            if 'total_value' not in self.portfolio['all'][symbol]:
                self.portfolio['all'][symbol]['total_value'] = 0

            # Now Update the order to the Portfolio
            if response_order['orderType'].lower() == 'market':
                self.portfolio[strategy_name][symbol]['position'] += orderQuantity * orderDirection_multiplier
                self.portfolio[strategy_name][symbol]['total_value'] += (price * orderQuantity) * orderDirection_multiplier
                self.portfolio[strategy_name][symbol]['average_price'] = (self.portfolio[strategy_name][symbol]['total_value'] / self.portfolio[strategy_name][symbol]['position']) if self.portfolio[strategy_name][symbol]['position'] > 0 else 0
                self.portfolio['all'][symbol]['position'] += orderQuantity * orderDirection_multiplier
                self.portfolio['all'][symbol]['total_value'] += (price * orderQuantity) * orderDirection_multiplier
                self.portfolio['all'][symbol]['average_price'] = (self.portfolio['all'][symbol]['total_value'] / self.portfolio['all'][symbol]['position']) if self.portfolio['all'][symbol]['position'] > 0 else 0
            elif response_order['orderType'].lower() in ['stoploss_pct', 'stoploss_abs', 'market_exit']:
                self.portfolio[strategy_name][symbol]['position'] += orderQuantity * orderDirection_multiplier
                self.portfolio[strategy_name][symbol]['total_value'] += (price * orderQuantity) * orderDirection_multiplier
                self.portfolio[strategy_name][symbol]['average_price'] = (self.portfolio[strategy_name][symbol]['total_value'] / self.portfolio[strategy_name][symbol]['position']) if self.portfolio[strategy_name][symbol]['position'] > 0 else 0
                self.portfolio['all'][symbol]['position'] += orderQuantity * orderDirection_multiplier
                self.portfolio['all'][symbol]['total_value'] += (price * orderQuantity) * orderDirection_multiplier
                self.portfolio['all'][symbol]['average_price'] = (self.portfolio['all'][symbol]['total_value'] / self.portfolio['all'][symbol]['position']) if self.portfolio['all'][symbol]['position'] > 0 else 0
            else:
                raise AssertionError('OrderType not supported: {}'.format(response_order['orderType']))
            # self.logger.debug({'self.portfolio':self.portfolio})
        
        # Remove all symbols with 0 position
        portfolio_copy = deepcopy(self.portfolio)
        for strategy_name in portfolio_copy:
            for symbol in portfolio_copy[strategy_name]:
                if portfolio_copy[strategy_name][symbol]['position'] == 0:
                    del self.portfolio[strategy_name][symbol]
    
    def update_available_margin(self, response_order, multi_leg_order, system_timestamp):
        '''Update the available margin after executing the order.'''
        if response_order['status'] == 'closed':
            # self.logger.debug({f'system_timestamp':system_timestamp, 'response_order':response_order})
            strategy_name = response_order['strategy_name']
            symbol = response_order['symbol']
            orderDirection = response_order['orderDirection']
            orderDirection_multiplier = 1 if orderDirection == 'BUY' else -1
            orderQuantity = response_order['orderQuantity']
            fill_price = response_order['fill_price']
            broker = response_order['broker'].lower()
            margin_used_by_order = (fill_price * orderQuantity)
            self.logger.debug({f"Symbol: {symbol} | Margin Used: {margin_used_by_order}, Fill Price: {fill_price}, Order Quantity: {orderQuantity}, Order Type: {response_order['orderType']}"})
            base_account_number = list(self.config_dict['account_info'][broker].keys())[0]
            trading_currency = self.config_dict['trading_currency']
            # self.logger.debug({f"Symbol: {symbol} | Margin Used: {margin_used_by_order}, Order Type: {response_order['orderType']}"})
            # msg = f"BEFORE: Symbol: {symbol} | Total Margin: {self.margin_available[broker][base_account_number]['combined'][trading_currency]['total_buying_power']} | Margin Used: {self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used']} | Margin Available: {self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available']}"
            # prev_margin_available = deepcopy(self.margin_available)
            # self.logger.debug(msg)
            if response_order['orderType'].lower() == 'market':
                self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available'] -= abs(margin_used_by_order)
                self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_available'] -= abs(margin_used_by_order)
                self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used'] += abs(margin_used_by_order)
                self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_used'] += abs(margin_used_by_order)
                profit = None
                # msg = f"Margin MATH: Margin Used Diff: {self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used'] - prev_margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used']} | Margin Available Diff: {self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available'] - prev_margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available']}"
                # self.logger.debug(msg)
                
            elif response_order['orderType'].lower() in ['stoploss_pct', 'stoploss_abs', 'market_exit']:
                order_open = self.check_if_all_legs_of_the_order_are_closed(multi_leg_order)
                if not order_open:
                    profit = self.reporter.calculate_multi_leg_order_profit(multi_leg_order)
                    # self.logger.info(f"Order Closed | Symbol: {symbol} | Profit: {profit} | Order Quantiy: {orderQuantity}")
                else:
                    profit = 0
                # self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available'] += abs(margin_used_by_order)
                # self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_available'] += abs(margin_used_by_order)
                if orderDirection_multiplier == 1:
                    self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used'] -= (abs(margin_used_by_order) + (profit * orderDirection_multiplier))
                    self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_used'] -= (abs(margin_used_by_order) + (profit * orderDirection_multiplier))
                    
                    self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available'] += (abs(margin_used_by_order) + (profit * orderDirection_multiplier) + (profit * orderDirection_multiplier))
                    self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_available'] += (abs(margin_used_by_order) + (profit * orderDirection_multiplier) + (profit * orderDirection_multiplier))
                else:
                    self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used'] -= (abs(margin_used_by_order) + (profit * orderDirection_multiplier))
                    self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_used'] -= (abs(margin_used_by_order) + (profit * orderDirection_multiplier))
                    self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available'] += abs(margin_used_by_order) 
                    self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_available'] += abs(margin_used_by_order) 
                # self.logger.debug({f"Symbol: {symbol} | Margin Used by Order: {margin_used_by_order}, Profit: {profit}, Order Type: {response_order['orderType']}"})
                    
            else:
                profit = None
                raise AssertionError('OrderType not supported: {}'.format(response_order['orderType']))
            
            self.margin_available[broker][base_account_number]['combined'][trading_currency]['total_buying_power'] = self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available'] + self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used']
            self.margin_available[broker][base_account_number][strategy_name][trading_currency]['total_buying_power'] = self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_available'] + self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_used']
            self.margin_available[broker][base_account_number]['combined'][trading_currency]['pct_of_margin_used'] = self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_used'] / self.margin_available[broker][base_account_number][strategy_name][trading_currency]['total_buying_power']
            self.margin_available[broker][base_account_number][strategy_name][trading_currency]['pct_of_margin_used'] = self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_used'] / self.margin_available[broker][base_account_number][strategy_name][trading_currency]['total_buying_power']
            self.margin_available[broker][base_account_number]['combined'][trading_currency]['cushion'] = 1 - self.margin_available[broker][base_account_number]['combined'][trading_currency]['pct_of_margin_used']
            self.margin_available[broker][base_account_number][strategy_name][trading_currency]['cushion'] = 1 - self.margin_available[broker][base_account_number][strategy_name][trading_currency]['pct_of_margin_used']
            self.margin_available[broker][base_account_number]['combined'][trading_currency]['pledge_to_margin_used'] = round(self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used'] / self.margin_available[broker][base_account_number]['combined'][trading_currency]['margin_multiplier'], 4)
            self.margin_available[broker][base_account_number][strategy_name][trading_currency]['pledge_to_margin_used'] = round(self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_used'] / self.margin_available[broker][base_account_number][strategy_name][trading_currency]['margin_multiplier'], 4)
            self.margin_available[broker][base_account_number]['combined'][trading_currency]['pledge_to_margin_availble'] = round(self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available'] / self.margin_available[broker][base_account_number]['combined'][trading_currency]['margin_multiplier'], 4)
            self.margin_available[broker][base_account_number][strategy_name][trading_currency]['pledge_to_margin_availble'] = round(self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_available'] / self.margin_available[broker][base_account_number][strategy_name][trading_currency]['margin_multiplier'], 4)
            # msg = f"AFTER: Symbol: {symbol} | Total Margin: {self.margin_available[broker][base_account_number]['combined'][trading_currency]['total_buying_power']} | Margin Used: {self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used']} | Margin Available: {self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available']}"
            # self.logger.debug(msg)
            # if profit:
                # msg = f"Margin MATH: Margin Used Diff: {self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used'] - prev_margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used']} | Margin Available Diff: {self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available'] - prev_margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available']} | Margin Used + Profit + Profit: {abs(margin_used_by_order) + profit + profit}"
                # self.logger.debug(msg)
            
            # self.logger.debug(f"Margin Available - AFTER: {self.margin_available} | Symbol: {symbol} | Order Direction: {orderDirection} | Order Type: {response_order['orderType']} | Order Quantity: {orderQuantity} | Fill Price: {fill_price} | Margin Used: {margin_used_by_order}")
    
    def update_order_history(self, order, response_order):
        updated_order = response_order.copy()
        # remove 'fresh_update' from order
        del response_order['fresh_update']
        # remove 'history' from order and response_order
        if 'history' in order:
            order_history = order['history']
            del order['history']
        else:
            order_history = []
        order_history.append(order)
        updated_order['history'] = order_history
        
        return updated_order
        
    def process_open_orders(self, open_orders, closed_orders, system_timestamp, market_data_df, live_bool):
        """Execute a list of multi-leg orders."""
        
        for level_1_count, multi_leg_order in enumerate(open_orders):
            for level_2_count, order in enumerate(multi_leg_order):
                updated_order = None
                response_order = None
                order_status = order['status']
                
                # Update symbol_ltp to the order
                if order_status not in ['closed', 'cancelled']:
                    if 'symbol_ltp' in order:
                        order['symbol_ltp'] = []
                        order['symbol_ltp'].append(market_data_df.loc['1d'].xs(order['symbol'], axis=1, level='symbol')['close'][-1])
                
                if order_status not in ['closed', 'cancelled']: # Basically if status is 'open' or 'pending'
                    # First check if Stoploss needs to be udpated.
                    order = self.update_trailing_stop_losses(order, market_data_df)
                    
                    # Make the order broker SIM if run_mode is Backtest or if live_bool is False
                    if self.config_dict['run_mode'] == 3 or not live_bool:
                        order['broker'] = 'SIM'

                    if order['broker'] == 'IBKR':
                        # self.logger.debug(f'Order being sent to IBKR: Symbol: {order["symbol"]} | orderType: {order["orderType"]} | Order Direction: {order["orderDirection"]} | orderQuantity: {order["orderQuantity"]} |  Status: {order["status"]}')
                        response_order = self.brokers.ib.execute.execute_order(order, market_data_df=market_data_df, system_timestamp=system_timestamp)
                    
                    elif order['broker'] == 'SIM':
                        response_order = self.brokers.sim.execute.execute_order(order, market_data_df=market_data_df, system_timestamp=system_timestamp)
                    else:
                        raise Exception(f"Broker {order['broker']} not supported.")
                    
                else:
                    order['fresh_update'] = False
                    response_order = order
                    
                '''Update the response order in oms order lists.'''
                if response_order and ('fresh_update' in response_order) and (response_order['fresh_update'] == True):
                    updated_order = self.update_order_history(order, response_order)
                    # Update the updated order to open orders list
                    open_orders[level_1_count][level_2_count] = updated_order
                    # self.update_porfolio(updated_order)
                    self.update_available_margin(response_order, open_orders[level_1_count], system_timestamp)
                    
                    # PRINT UPDATE TO LOG
                    if 'entryPrice' in updated_order:
                        price = round(updated_order['entryPrice'], 3)
                    elif 'exitPrice' in updated_order:
                        price = round(updated_order['exitPrice'], 3)
                    else:
                        price = None
                    fill_price = round(updated_order['fill_price'], 3) if 'fill_price' in updated_order else 'Not Filled'
                    orderValue = round(updated_order['orderQuantity'] * fill_price, 3) if isinstance(fill_price, float) else None
                    self.logger.info(f"OPEN ORDER UPDATE: Symbol:{updated_order['symbol']}|Type:{updated_order['orderType']}|Dir:{updated_order['orderDirection']}|Qty:{updated_order['orderQuantity']}|Price:{price}|Fill Price:{fill_price}|orderValue:{orderValue}|_id:{updated_order['order_id'][-5:]}|Status:{updated_order['status']}|Msg:{updated_order['message']}")
        
        updated_open_orders, closed_orders = self.remove_closed_orders_from_open_orders_list(open_orders, closed_orders)
        # self.logger.debug({'updated_open_orders':len(updated_open_orders), 'closed_orders':len(closed_orders)})
        
        return updated_open_orders, closed_orders
    
    def execute_orders(self, new_orders, system_timestamp, market_data_df, live_bool):
        """ add the list of new orders to open_orders list """
        if len(new_orders) > 0:
            new_orders_bifurcated = {'new_orders':[], 'update_orders':[]}
            for multi_leg_order_new_order in new_orders:
                update_order_bool = False
                for new_order in multi_leg_order_new_order:
                    if new_order['status'] == 'cancel_pending':
                        update_order_bool = True
                
                if not update_order_bool:
                    new_orders_bifurcated['new_orders'].append(multi_leg_order_new_order)
                else:
                    new_orders_bifurcated['update_orders'].append(multi_leg_order_new_order)
                    
            # self.logger.debug({'len-new_order_bifurcated-new_orders':len(new_orders_bifurcated['new_orders']), 'len-new_order_bifurcated-update_orders':len(new_orders_bifurcated['update_orders'])})
                    
            for multi_leg_order in new_orders_bifurcated['new_orders']:
                self.open_orders.append(multi_leg_order)
            
            for multi_leg_update_order in new_orders_bifurcated['update_orders']:
                # find the open order to update
                open_order_to_update = None
                open_order_to_update_count = None
                for update_order in multi_leg_update_order:
                    for open_order_count, multi_leg_open_order in enumerate(self.open_orders):
                        for open_order in multi_leg_open_order:
                            if not open_order_to_update and 'order_id' in open_order and 'order_id' in update_order and open_order['order_id'] == update_order['order_id']:
                                open_order['status'] = 'cancelled'
                                open_order['message'] = 'Order Cancelled due to new order update.'
                                open_order_to_update = multi_leg_open_order
                                open_order_to_update_count = open_order_count
                
                for update_order in multi_leg_update_order:
                    status = update_order['status']
                    if status == 'cancel_pending':
                        for open_order in open_order_to_update:
                            if open_order['status'] != 'pending' and open_order['order_id'] == update_order['order_id']:
                                open_order['status'] = 'cancelled'
                                open_order['message'] = 'Order Cancelled and replaced by new MARKET ORDER.'
                    else:
                        open_order_to_update.append(update_order)
                
                self.open_orders[open_order_to_update_count] = open_order_to_update
                # self.logger.debug('UPDATED ORDER: {}'.format(pprint(self.open_orders[open_order_to_update_count])))

        # Process all open orders
        self.open_orders, self.closed_orders = self.process_open_orders(self.open_orders, self.closed_orders, system_timestamp, market_data_df, live_bool)

        # if len(new_orders) > 0:
        #     self.logger.debug(f'NOTE NOTE: This is where you save the open orders to a file. len(closed_orders): {len(self.closed_orders)}')
        #     self.logger.debug('NOTE NOTE: This is where you save the closed orders to a file.')
        #     self.logger.debug('NOTE NOTE: This is where you save the open portfolio to a file.')
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
