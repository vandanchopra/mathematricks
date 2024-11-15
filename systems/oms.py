#from ib_insync import *
from copy import deepcopy
from operator import le
import sys
from urllib import response
import pandas as pd
#from brokers import Brokers
import json
from config import config_dict
import pandas as pd  # Assuming you are using pandas Timestamps
from brokers.brokers import Brokers
from systems.utils import create_logger, generate_hash_id, sleeper
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
        self.unfilled_orders = []
    
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

        def get_positions_to_buy_for_sync(market_data_df, sim_open_orders, sim_net_liquidation_value, ibkr_available_funds, open_orders_ibkr, unfilled_orders_ibkr):
            # self.logger.debug({'open_orders_ibkr':open_orders_ibkr})
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
                        total_ratio_of_each_order_to_net_liquidation_value[symbol]['open_positions'].append({'entryPrice': order['fill_price'], 'orderQuantity': order['orderQuantity'], 'orderDirection': order['orderDirection']})
            # ratio_of_funds_utilized_sim = total_value_of_open_positions_sim / sim_net_liquidation_value

            for symbol in total_ratio_of_each_order_to_net_liquidation_value:
                # First get the value of all orders.
                symbol_invested_value = 0
                total_quantity_open_symbol = 0
                
                for order_entry in total_ratio_of_each_order_to_net_liquidation_value[symbol]['open_positions']:
                    orderDirection_multiplier = 1 if order_entry['orderDirection'].lower() == 'buy' else -1
                    symbol_invested_value += order_entry['entryPrice'] * order_entry['orderQuantity'] * orderDirection_multiplier
                    total_quantity_open_symbol += order_entry['orderQuantity'] * orderDirection_multiplier
                    # self.logger.debug({'symbol':symbol, 'orderQuantity':order_entry['orderQuantity'], 'symbol_invested_value':symbol_invested_value, 'total_quantity_open_symbol':total_quantity_open_symbol})
                    
                # Then get the ratio to sim_net_liquidation_value
                total_ratio_of_each_order_to_net_liquidation_value[symbol]['ratio_of_invested_value_to_sim_net_liquidation_value'] = symbol_invested_value / sim_net_liquidation_value
                # self.logger.debug({'symbol':symbol, 'ratio_of_invested_value_to_sim_net_liquidation_value': total_ratio_of_each_order_to_net_liquidation_value[symbol]['ratio_of_invested_value_to_sim_net_liquidation_value']})
                # Then calculate the ratio of funds to be alloted to symbol on IBKR
                total_ratio_of_each_order_to_net_liquidation_value[symbol]['availble_funds_for_asset_from_ibkr'] = total_ratio_of_each_order_to_net_liquidation_value[symbol]['ratio_of_invested_value_to_sim_net_liquidation_value'] * ibkr_available_funds
                # self.logger.debug({'symbol':symbol, 'availble_funds_for_asset_from_ibkr':total_ratio_of_each_order_to_net_liquidation_value[symbol]['availble_funds_for_asset_from_ibkr']})
                
                # self.logger.debug({'symbol':symbol, 'symbol_invested_value':symbol_invested_value, 'availble_funds_for_asset_from_ibkr':total_ratio_of_each_order_to_net_liquidation_value[symbol]['availble_funds_for_asset_from_ibkr'], 'ratio_of_invested_value_to_sim_net_liquidation_value': total_ratio_of_each_order_to_net_liquidation_value[symbol]['ratio_of_invested_value_to_sim_net_liquidation_value'], 'sim_net_liquidation_value':sim_net_liquidation_value, 'ibkr_available_funds':ibkr_available_funds})
                # Then get the current value of the stock.
                try:
                    close_prices = market_data_df.loc[self.min_granularity].xs(symbol, axis=1, level='symbol')['close']
                    # drop nan values
                    close_prices = close_prices.dropna()
                    total_ratio_of_each_order_to_net_liquidation_value[symbol]['current_price'] = close_prices.loc[-1] # THIS VALUE IS HYPOTHETICAL
                    # Then calculate the closest integer number of stocks that can be bought with the funds available for IBKR
                    total_ratio_of_each_order_to_net_liquidation_value[symbol]['ideal_quantity_to_buy'] = int(total_ratio_of_each_order_to_net_liquidation_value[symbol]['availble_funds_for_asset_from_ibkr'] / total_ratio_of_each_order_to_net_liquidation_value[symbol]['current_price'])
                    # self.logger.info({'symbol':symbol, 'ideal_quantity_to_buy':total_ratio_of_each_order_to_net_liquidation_value[symbol]['ideal_quantity_to_buy'], 'availble_funds_for_asset_from_ibkr':total_ratio_of_each_order_to_net_liquidation_value[symbol]['availble_funds_for_asset_from_ibkr'], 'current_price':total_ratio_of_each_order_to_net_liquidation_value[symbol]['current_price']})
                except Exception as e:
                    raise Exception(f"Symbol: {symbol}, availble_funds_for_asset_from_ibkr: {total_ratio_of_each_order_to_net_liquidation_value[symbol]['availble_funds_for_asset_from_ibkr']}, current_price: {total_ratio_of_each_order_to_net_liquidation_value[symbol]['current_price']}, Tail Prices: {market_data_df.loc[self.min_granularity].xs(symbol, axis=1, level='symbol')['close'].tail()}")
                # self.logger.debug({'symbol':symbol, 'ideal_quantity_to_buy':total_ratio_of_each_order_to_net_liquidation_value[symbol]['ideal_quantity_to_buy']})
                # self.logger.debug({'current_price':total_ratio_of_each_order_to_net_liquidation_value[symbol]['current_price'], 'fill_price':symbol_invested_value/total_quantity_open_symbol})
                # Find out how much of the quantity is already bought
                for open_order_ibkr in open_orders_ibkr:
                    if open_order_ibkr[0]['symbol'] == symbol:
                        ibkr_orderDirection_multiplier = 1 if open_order_ibkr[0]['orderDirection'].lower() == 'buy' else -1
                        total_ratio_of_each_order_to_net_liquidation_value[symbol]['quantity_bought'] = open_order_ibkr[0]['orderQuantity'] * ibkr_orderDirection_multiplier
                        break
                    else:
                        total_ratio_of_each_order_to_net_liquidation_value[symbol]['quantity_bought'] = 0
                
                for unfilled_order in unfilled_orders_ibkr:
                    if unfilled_order.contract.symbol == symbol and unfilled_order.order.orderType.lower() not in ['stp']:
                        ibkr_orderDirection_multiplier = 1 if unfilled_order.order.action == 'BUY' else -1
                        total_ratio_of_each_order_to_net_liquidation_value[symbol]['quantity_bought'] += unfilled_order.order.totalQuantity * ibkr_orderDirection_multiplier
                        # self.logger.debug({'symbol':symbol, 'Quantity bought':unfilled_order.order.totalQuantity * ibkr_orderDirection_multiplier})
                        # self.logger.debug({'unfilled_order':unfilled_order, 'quantity_bought':total_ratio_of_each_
                        # order_to_net_liquidation_value[symbol]['quantity_bought']})
                        
                # self.logger.debug({'symbol':symbol, 'quantity_bought':total_ratio_of_each_order_to_net_liquidation_value[symbol]['quantity_bought']})
                # Find out actual quantity to buy
                total_ratio_of_each_order_to_net_liquidation_value[symbol]['actual_quantity_to_buy'] = total_ratio_of_each_order_to_net_liquidation_value[symbol]['ideal_quantity_to_buy'] - (total_ratio_of_each_order_to_net_liquidation_value[symbol]['quantity_bought'] if 'quantity_bought' in total_ratio_of_each_order_to_net_liquidation_value[symbol] else 0)
                # self.logger.debug({'symbol':symbol, 'actual_quantity_to_buy':total_ratio_of_each_order_to_net_liquidation_value[symbol]['actual_quantity_to_buy']})
                if total_ratio_of_each_order_to_net_liquidation_value[symbol]['actual_quantity_to_buy'] != 0:
                    actual_quantity_to_buy[symbol] = total_ratio_of_each_order_to_net_liquidation_value[symbol]['actual_quantity_to_buy']
                if total_ratio_of_each_order_to_net_liquidation_value[symbol]['ideal_quantity_to_buy'] != 0:
                    ideal_quantity_to_buy[symbol] = total_ratio_of_each_order_to_net_liquidation_value[symbol]['ideal_quantity_to_buy']
            return actual_quantity_to_buy, ideal_quantity_to_buy
        
        def create_entry_orders_for_sync(actual_quantity_to_buy, market_data_df, sim_open_orders, system_timestamp, unfilled_orders_ibkr):
            new_entry_orders = []
            
            for symbol, actual_symbol_quantity in actual_quantity_to_buy.items():
                run_once = 0
                for multi_leg_order in sim_open_orders:
                    if run_once == 0:
                        for order in multi_leg_order:
                            if order['symbol'] == symbol and 'entryPrice' in order:
                                new_order = deepcopy(order)
                                new_order['orderQuantity'] = abs(actual_symbol_quantity)
                                # fix entry price
                                new_order['entryPrice'] = market_data_df.loc['1d'].xs(symbol, axis=1, level='symbol')['close'].iloc[-1]
                                # fix broker
                                new_order['broker'] = 'IBKR'
                                # status
                                new_order['status'] = 'pending'
                                # make fill_price = 0
                                # new_order['fill_price'] = 0
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
                                new_order['orderDirection'] = 'BUY' if actual_symbol_quantity > 0 else 'SELL'
                                
                                # order_id = create new order_id
                                new_order['original_order_id'] = new_order['order_id']
                                new_order['order_id'] = generate_hash_id(new_order, system_timestamp)
                                new_entry_orders.append([new_order]) # Order is being sent as a single-leg multi-leg order, that's why it's in a list.
                                # self.logger.debug({'symbol':symbol, 'new_order':new_order})
                                run_once += 1
                            
            return new_entry_orders
        
        def create_exit_orders_for_sync(sim_open_orders, new_entry_orders, system_timestamp, ideal_quantity_to_buy, unfilled_orders_ibkr, open_orders_ibkr):
            # self.logger.debug({'ideal_quantity_to_buy':ideal_quantity_to_buy})
            '''
            For each symbol that has ideal quantity to buy, we need to place a equal stoploss
            
            for symbol in ideal_quantity_to_buy:
                first find the entry order. It's either in open_orders_ibkr or in new_entry_orders.
                once you've found it, then find the stoploss order in unfilled_orders. If it's there, then modify it.
                if it's not there, then find the stoploss order in sim_open_orders and use that and add it to the order
            '''
            for symbol, ideal_stoploss_quantity in ideal_quantity_to_buy.items():
                # First find the entry order.
                entry_order = None
                entry_order_location = {}
                for count, multi_leg_order in enumerate(open_orders_ibkr):
                    for order in multi_leg_order:
                        if order['symbol'] == symbol:
                            entry_order = multi_leg_order
                            entry_order_location['open_orders_ibkr'] = count
                            break
                    if entry_order:
                        break
                if not entry_order:
                    for count, multi_leg_order in enumerate(new_entry_orders):
                        for order in multi_leg_order:
                            if order['symbol'] == symbol:
                                entry_order = multi_leg_order
                                entry_order_location['new_entry_orders'] = count
                                break
                        if entry_order:
                            break
                
                if entry_order:
                    # Find the stoploss order
                    stoploss_order_location = None
                    current_stoploss_quantity = 0
                    current_stoploss_list = []
                    stoploss_order_temp = None
                    for unfilled_order in unfilled_orders_ibkr:
                        if unfilled_order.contract.symbol == symbol and unfilled_order.order.orderType.lower() in ['stp']:
                            stoploss_order_temp = unfilled_order
                            current_stoploss_quantity += unfilled_order.order.totalQuantity
                            current_stoploss_list.append(unfilled_order)
                            stoploss_order_location = 'unfilled_orders_ibkr'
                            # self.logger.debug('Found in unfilled_orders_ibkr')
                            break
                        
                    if not stoploss_order_temp:
                        for multi_leg_order in sim_open_orders:
                            for order in multi_leg_order:
                                if order['symbol'] == symbol and 'exitPrice' in order:
                                    stoploss_order_temp = order
                                    # self.logger.debug('Found in sim_open_orders')
                                    break
                            if stoploss_order_temp:
                                break
                            
                    if stoploss_order_temp:
                        if current_stoploss_quantity == 0:
                            stoploss_order = deepcopy(stoploss_order_temp)
                            stoploss_order['orderQuantity'] = abs(ideal_stoploss_quantity)
                            # fix broker
                            stoploss_order['broker'] = 'IBKR'
                            # status
                            stoploss_order['status'] = 'pending'
                            # make fill_price = 0
                            stoploss_order['fill_price'] = 0
                            # filled_timestamp = None
                            stoploss_order['filled_timestamp'] = None
                            # broker_order_id = None
                            stoploss_order['broker_order_id'] = None
                            # Remove fresh_update
                            if 'fresh_update' in stoploss_order:
                                stoploss_order.pop('fresh_update')
                            # Remove message
                            if 'message' in stoploss_order:
                                stoploss_order.pop('message')
                            # Remove history
                            if 'history' in stoploss_order:
                                stoploss_order.pop('history')
                            # order_id = create new order_id
                            if 'order_id' in stoploss_order:
                                stoploss_order['original_order_id'] = stoploss_order['order_id']
                            stoploss_order['order_id'] = generate_hash_id(stoploss_order, system_timestamp)
                            if 'open_orders_ibkr' in entry_order_location:
                                open_orders_ibkr[entry_order_location['open_orders_ibkr']].append(stoploss_order)
                            else:
                                new_entry_orders[entry_order_location['new_entry_orders']].append(stoploss_order)
                            # self.logger.debug({'symbol':symbol, 'stoploss_order':stoploss_order})
                            self.logger.debug(f"SYNC STOPLOSS ORDER: Symbol: {symbol}, OrderQuantity: {stoploss_order['orderQuantity']}, OrderDirection: {stoploss_order['orderDirection']}, Stoploss Price: {stoploss_order['exitPrice']}")
                            if 'entryPrice' in stoploss_order:
                                raise AssertionError(f"STOPLOSS ORDER HAS ENTRY PRICE")
                        else:
                            # self.logger.debug({'stoploss_order_temp':stoploss_order_temp})
                            new_system_stoploss_order = self.brokers.ib.execute.create_system_order_from_ibkr_order(stoploss_order_temp)
                            orderdirection_multiplier = 1 if new_system_stoploss_order['orderDirection'].lower() == 'buy' else -1
                            if current_stoploss_quantity * orderdirection_multiplier * -1 != ideal_stoploss_quantity:
                                # self.logger.debug({'symbol':symbol, 'current_stoploss_quantity':current_stoploss_quantity, 'ideal_stoploss_quantity':ideal_stoploss_quantity})
                                modified_order = self.modify_order(new_system_stoploss_order, new_quantity=abs(ideal_stoploss_quantity))
                                if 'open_orders_ibkr' in entry_order_location:
                                    open_orders_ibkr[entry_order_location['open_orders_ibkr']].append(modified_order)
                                    self.logger.debug(f"SYNC STOPLOSS ORDER: Symbol: {symbol}, OrderQuantity: {modified_order['orderQuantity']}, OrderDirection: {modified_order['orderDirection']}, Stoploss Price: {modified_order['exitPrice']}")
                                    if 'entryPrice' in modified_order:
                                        raise AssertionError(f"modified_order ORDER HAS ENTRY PRICE")
                                else:
                                    new_entry_orders[entry_order_location['new_entry_orders']].append(modified_order)
                                    self.logger.debug(f"SYNC STOPLOSS ORDER: Symbol: {symbol}, OrderQuantity: {modified_order['orderQuantity']}, OrderDirection: {modified_order['orderDirection']}, Stoploss Price: {modified_order['exitPrice']}")
                                    if 'entryPrice' in modified_order:
                                        raise AssertionError(f"modified_order ORDER HAS ENTRY PRICE")

            return new_entry_orders, open_orders_ibkr
        
        new_orders = []
        if sync_direction == 'broker-to-oms':
            raise NotImplementedError(f'This {sync_direction} direction is not supported yet.')
        elif sync_direction == 'oms-to-broker':
            sim_open_orders = deepcopy(self.open_orders)
            
            ''' First Get Account balances from the broker and update the margin_available dict'''
            self.margin_available = self.update_all_margin_available()
            # self.logger.debug({'self.margin_available':self.margin_available})
            
            #### Get the minimum granularity from market_data_df
            self.available_granularities = market_data_df.index.get_level_values(0).unique()
            self.min_granularity_val = min([self.granularity_lookup_dict[granularity] for granularity in self.available_granularities])
            self.min_granularity = list(self.granularity_lookup_dict.keys())[list(self.granularity_lookup_dict.values()).index(self.min_granularity_val)]
            
            open_orders_ibkr, unfilled_orders_ibkr = self.brokers.ib.execute.get_open_orders(market_data_df, system_timestamp, self.min_granularity)
            
            # # get open positions from SIM
            # open_positions_sim, pending_orders_sim = self.create_postions_from_open_orders(sim_open_orders)
            # get account balance for IBKR
            # account_balances_dict = self.get_account_balances()
            self.update_all_margin_available()
            # self.logger.debug({'account_balances_dict':account_balances_dict})
            ibkr_available_funds = self.margin_available['ibkr'][self.config_dict['base_account_numbers']['ibkr']]['combined'][self.config_dict['trading_currency']]['total_buying_power']
            
            # self.logger.debug({'ibkr_available_funds':ibkr_available_funds})
            # get account balance for SIM
            sim_net_liquidation_value = self.margin_available['sim']['sim_1']['combined'][self.config_dict['trading_currency']]['buying_power_available']
            # self.logger.debug({'sim_net_liquidation_value':sim_net_liquidation_value})
            # get positions to buy
            actual_quantity_to_buy, ideal_quantity_to_buy = get_positions_to_buy_for_sync(market_data_df, sim_open_orders, sim_net_liquidation_value, ibkr_available_funds, open_orders_ibkr, unfilled_orders_ibkr)
            # self.logger.debug({'ideal_quantity_to_buy':ideal_quantity_to_buy})
            # Create a log msg with the keys and values of the ideal_quantity_to_buy seperated by a ' | '
            self.logger.info("Ideal Position Size: " + ' | '.join(f"{key}: {value}" for key, value in ideal_quantity_to_buy.items()))
            self.logger.info("Actual Qty to Buy:   " + ' | '.join(f"{key}: {value}" for key, value in actual_quantity_to_buy.items()))
            # Create new orders
            new_entry_orders = create_entry_orders_for_sync(actual_quantity_to_buy, market_data_df, sim_open_orders, system_timestamp, unfilled_orders_ibkr)
            new_entry_orders, open_orders_ibkr = create_exit_orders_for_sync(sim_open_orders, new_entry_orders, system_timestamp, ideal_quantity_to_buy, unfilled_orders_ibkr, open_orders_ibkr)
            
            '''LAST STEP: Fix the order in which the orders are placed in the new_orders list. Right now we're going above margin'''
            self.logger.warning('FIX THE ORDER IN WHICH THE ORDERS ARE PLACED IN THE NEW_ORDERS LIST. RIGHT NOW WE ARE GOING ABOVE MARGIN')
            new_orders.extend(new_entry_orders)
            self.open_orders = open_orders_ibkr
            # self.logger.debug({'self.open_orders':self.open_orders})
        
        return new_orders

    def get_open_orders(self, broker):
        if broker == 'IBKR':
            return self.brokers.ib.get_open_orders()
        elif broker == 'SIM':
            return self.open_orders['SIM']
        else:
            raise AssertionError(f"Broker {broker} not supported.")

    def modify_order(self, order, new_price=None, new_quantity=None):
        if new_price or new_quantity:
            modify_reason = []
            modify_message = ''
            
            if new_price:
                modify_reason.append('new_price')
                modify_message += f'| New Price: {new_price}, old price: {order["stoploss_abs"]}'
                order['exitPrice'] = new_price
                order['stoploss_abs'] = new_price
            if new_quantity:
                modify_reason.append('new_quantity')
                modify_message += f'| New Quantity: {new_quantity}, old quantity: {order["orderQuantity"]}'
                order['orderQuantity'] = new_quantity
                
            order['status'] = 'modify'
            order['modify_reason'] = modify_reason
            order['message'] = modify_message
            
        return order    
    
    def update_trailing_stop_losses(self, order, market_data_df):
        if order['status'] != 'pending' and order['orderType'] == 'stoploss_pct':
            symbol = order['symbol']
            granularity = order['granularity'] # Get minimum granularity from market_data_df
            # system_timestamp = market_data_df.index.get_level_values(1)[-1]
            stoploss_pct = order['stoploss_pct']
            current_price = market_data_df.loc[granularity].xs(symbol, axis=1, level='symbol')['close'].iloc[-1]
            # stoploss_abs = order['stoploss_abs']
            orderDirection = order['orderDirection']

            current_stoploss = order['exitPrice']
            ideal_stoploss = current_price * (1 - stoploss_pct) if orderDirection == 'SELL' else current_price * (1 + stoploss_pct)
            acceptable_loss_pct_deviation = stoploss_pct/5
            acceptable_spotloss = current_price * (1 - (stoploss_pct+acceptable_loss_pct_deviation)) if orderDirection == 'SELL' else current_price * (1 + (stoploss_pct-acceptable_loss_pct_deviation))
            update_stoploss = current_stoploss < acceptable_spotloss and ideal_stoploss > current_stoploss if orderDirection == 'SELL' else current_stoploss > acceptable_spotloss and ideal_stoploss < current_stoploss
            
            if update_stoploss:
                order = self.modify_order(order, new_price=ideal_stoploss)
    
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
            try:
                fill_price = response_order['fill_price']
            except Exception as e:
                self.logger.debug({'response_order':response_order})
                raise Exception(e)
            broker = response_order['broker'].lower()
            margin_used_by_order = round((fill_price * orderQuantity), 10)
            # self.logger.debug({f"Symbol: {symbol} | Margin Used: {margin_used_by_order}, Fill Price: {fill_price}, Order Quantity: {orderQuantity}, Order Type: {response_order['orderType']}"})
            base_account_number = list(self.config_dict['account_info'][broker].keys())[0]
            trading_currency = self.config_dict['trading_currency']
            # self.logger.debug({'broker':broker, 'base_account_number':base_account_number, 'strategy_name':strategy_name, 'trading_currency':trading_currency})
            # self.logger.debug({f"Symbol: {symbol} | Margin Used: {margin_used_by_order}, Order Type: {response_order['orderType']}"})
            # msg = f"BEFORE: Symbol: {symbol} | Total Margin: {self.margin_available[broker][base_account_number]['combined'][trading_currency]['total_buying_power']} | Margin Used: {self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used']} | Margin Available: {self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available']}"
            prev_margin_available = deepcopy(self.margin_available)
            profit = 0
            
            # self.logger.debug(msg)
            if response_order['orderType'].lower() == 'market':
                self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available'] -= abs(margin_used_by_order)
                self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_available'] -= abs(margin_used_by_order)
                self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used'] += abs(margin_used_by_order)
                self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_used'] += abs(margin_used_by_order)
                profit = 0
                
            elif response_order['orderType'].lower() in ['stoploss_pct', 'stoploss_abs', 'market_exit']:
                order_open = self.check_if_all_legs_of_the_order_are_closed(multi_leg_order)
                if not order_open:
                    profit, profit_pct = self.reporter.calculate_multi_leg_order_pnl(multi_leg_order, self.unfilled_orders, force_close=False)
                    # self.logger.info(f"Order Closed | Symbol: {symbol} | Profit: {round(profit, 2)} | Order Quantiy: {orderQuantity}")
                # self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available'] += abs(margin_used_by_order)
                # self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_available'] += abs(margin_used_by_order)
                margin_used_by_entry_order = round((margin_used_by_order - (profit * (orderDirection_multiplier * -1))), 10)
                self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used'] -= margin_used_by_entry_order
                self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_used'] -= margin_used_by_entry_order
                self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available'] += margin_used_by_entry_order + profit
                self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_available'] += margin_used_by_entry_order + profit
                # self.logger.debug({f"Symbol: {symbol} | Margin Used by Order: {margin_used_by_order}, Profit: {profit}, Order Type: {response_order['orderType']}"})
                    
            else:
                profit = 0
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
            # msg = f"AFTER: Broker: {broker} | Symbol: {symbol} | Total Margin: {self.margin_available[broker][base_account_number]['combined'][trading_currency]['total_buying_power']} | Margin Used: {self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used']} | Margin Available: {self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available']}"
            # self.logger.debug(msg)
            # if profit:
                # msg = f"Margin MATH: Margin Used Diff: {self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used'] - prev_margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used']} | Margin Available Diff: {self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available'] - prev_margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available']} | Margin Used + Profit + Profit: {abs(margin_used_by_order) + profit + profit}"
                # self.logger.debug(msg)
            
            # self.logger.debug(f"Margin Available - AFTER: {self.margin_available} | Symbol: {symbol} | Order Direction: {orderDirection} | Order Type: {response_order['orderType']} | Order Quantity: {orderQuantity} | Fill Price: {fill_price} | Margin Used: {margin_used_by_order}")
            
            # MARGIN CALCULATION MATH:
            opening_buying_power_available = prev_margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available']
            opening_buying_power_used = prev_margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used']
            closing_buying_power_available = self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available']
            closing_buying_power_used = self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used']
            change_in_buying_power_available = closing_buying_power_available-opening_buying_power_available
            change_in_buying_power_used = closing_buying_power_used-opening_buying_power_used
            margin_math = (change_in_buying_power_available - profit) + change_in_buying_power_used
            # if margin_math != 0:
            # self.logger.debug(f"Margin Math: {margin_math} | Profit: {profit} | Opening Buying Power Available: {opening_buying_power_available} | Opening Buying Power Used: {opening_buying_power_used} | Closing Buying Power Available: {closing_buying_power_available} | Closing Buying Power Used: {closing_buying_power_used} | Change in Buying Power Available: {change_in_buying_power_available} | Change in Buying Power Used: {change_in_buying_power_used}")
            
            return profit
    
    def update_order_history(self, order, response_order):
        # remove 'fresh_update' from order
        del response_order['fresh_update']
        updated_order = response_order.copy()
        
        # remove 'history' from order and response_order
        order_history = order['history'] if 'history' in order else []
        if 'history' in updated_order:
            del order['history']
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
                    if 'symbol_ltp' not in order:
                        order['symbol_ltp'] = {}
                    self.available_granularities = market_data_df.index.get_level_values(0).unique()
                    self.min_granularity_val = min([self.granularity_lookup_dict[granularity] for granularity in self.available_granularities])
                    self.min_granularity = list(self.granularity_lookup_dict.keys())[list(self.granularity_lookup_dict.values()).index(self.min_granularity_val)]
                    close_prices = market_data_df.loc[self.min_granularity].xs(order['symbol'], axis=1, level='symbol')['close']
                    # Drop nan values
                    close_prices = close_prices.dropna()
                    if len(close_prices) > 0:
                        try:
                            order['symbol_ltp'][str(system_timestamp)] = close_prices[-1]
                        except Exception as e:
                            raise Exception(f"Error in updating symbol_ltp: {e}, Symbol: {order['symbol']}, Close Prices: {close_prices}")
                
                if order_status not in ['closed', 'cancelled']: # Basically if status is 'open' or 'pending'
                    # First check if Stoploss needs to be udpated.
                    order = self.update_trailing_stop_losses(order, market_data_df)
                    
                    # Make the order broker SIM if run_mode is Backtest or if live_bool is False
                    if self.config_dict['run_mode'] == 3 or not live_bool:
                        order['broker'] = 'SIM'
                        
                    if order['broker'] == 'IBKR':
                        # self.logger.debug(f'Order being sent to IBKR: Symbol: {order["symbol"]} | orderType: {order["orderType"]} | Order Direction: {order["orderDirection"]} | orderQuantity: {order["orderQuantity"]} |  Status: {order["status"]}')
                        response_order = self.brokers.ib.execute.execute_order(order, multi_leg_order, market_data_df=market_data_df, system_timestamp=system_timestamp)
                    
                    elif order['broker'] == 'SIM':
                        response_order = self.brokers.sim.execute.execute_order(order, market_data_df=market_data_df, system_timestamp=system_timestamp)
                    else:
                        raise Exception(f"Broker {order['broker']} not supported.")
                    
                else:
                    order['fresh_update'] = False
                    response_order = order
                    
                '''Update the response order in oms order lists.'''
                
                if response_order and ('fresh_update' in response_order) and (response_order['fresh_update'] == True):
                    # self.logger.debug({'response_order':response_order})
                    updated_order = self.update_order_history(order, response_order)
                    # Update the updated order to open orders list
                    open_orders[level_1_count][level_2_count] = updated_order
                    multi_leg_order[level_2_count] = updated_order
                    # self.update_porfolio(updated_order)
                    profit = self.update_available_margin(response_order, open_orders[level_1_count], system_timestamp)
                    
                    # PRINT UPDATE TO LOG
                    if 'entryPrice' in updated_order:
                        price = round(updated_order['entryPrice'], 3)
                    elif 'exitPrice' in updated_order:
                        price = round(updated_order['exitPrice'], 3)
                    else:
                        price = None
                    fill_price = updated_order['fill_price'] if 'fill_price' in updated_order else 'Not Filled'
                    orderValue = updated_order['orderQuantity'] * fill_price if isinstance(fill_price, float) else None
                    if 'fresh_update' in updated_order:
                        self.logger.info(f"Symbol: {updated_order['symbol']} | Fresh Update: {updated_order['fresh_update']} | Type: {updated_order['orderType']} | Dir: {updated_order['orderDirection']} | Qty: {updated_order['orderQuantity']} | Symbol LTP: {list(updated_order['symbol_ltp'].values())[-5:]}")
                    msg = f"ORDER UPDATED: Symbol:{updated_order['symbol']} | Type:{updated_order['orderType']} | Dir:{updated_order['orderDirection']} | Qty:{updated_order['orderQuantity']} | Price:{price} | Fill Price:{fill_price} | orderValue: {orderValue} | Status:{updated_order['status']} | Msg:{updated_order['message']}"
                    if isinstance(profit, float) or isinstance(profit, int):
                        msg += f" | Profit: {round(profit, 2)}"
                    else:
                        msg += f" | Profit: {profit}"
                    self.logger.info(msg)
        
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
                    # self.logger.debug({'new_order':new_order})
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
