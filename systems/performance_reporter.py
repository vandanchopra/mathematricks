import signal
from systems.utils import project_path
import os, json, pickle
from systems.utils import create_logger, sleeper, generate_hash_id

class PerformanceReporter:
    def __init__(self):
        self.backtest_folder_path = project_path + 'db/vault/backtest_reports'
        self.backtest_performance_metrics = None
        self.backtest_report = None
        self.logger = create_logger(log_level='DEBUG', logger_name='REPORTER', print_to_console=True)
    
    def calculate_multi_leg_order_profit(self, multi_leg_order):
        sell_orders = []
        buy_orders = []
        signal_id = multi_leg_order[0]['signal_id']

        buy_quantity = 0
        sell_quantity = 0
        # Separate out all sell and buy orders
        for order in multi_leg_order:
            if order['orderDirection'] == 'SELL' and order['status'] == 'closed':
                sell_orders.append(order)
                sell_quantity += order['orderQuantity']
            elif order['orderDirection'] == 'BUY' and order['status'] == 'closed':
                buy_orders.append(order)
                buy_quantity += order['orderQuantity']

        if buy_quantity == sell_quantity:
            total_profit = 0
            # Assuming orders are processed in pairs (e.g., first sell with first buy, second sell with second buy)
            for sell_order, buy_order in zip(sell_orders, buy_orders):
                sell_price = sell_order.get('fill_price')
                buy_price = buy_order.get('fill_price')
                quantity = buy_quantity #min(sell_order.get('orderQuantity'), buy_order.get('orderQuantity'))

                # Calculate profit for the matching sell and buy orders
                profit = (sell_price - buy_price) * quantity
                total_profit += profit
                # self.logger.debug({f"Signal ID: {signal_id}, Buy Price: {buy_price}, Sell Price: {sell_price}, Quantity: {quantity}, Profit: {profit}"})
            # self.logger.debug({f"Signal ID: {signal_id}, Total Profit: {total_profit}"})
            # print('- ' * 50)
            # self.logger.info({'order':multi_leg_order})
            # self.logger.info(f"Total profit for multi-leg order: {total_profit}")
            # raise AssertionError('PnL calculation not implemented yet.') 
            return total_profit
        else:
            # self.logger.debug({f"Signal ID: {signal_id}, Buy Quantity: {buy_quantity}, Sell Quantity: {sell_quantity}"})
            # print('- ' * 50)
            return 0
    
    def calculate_backtest_performance_metrics(self, config_dict, open_orders, closed_orders, market_data_df_root):
        # Implementation for calculating performance metrics
        self.backtest_performance_metrics = {}
        profit = 0
        profits_list = []
        losses_list = []
        long_list = []
        short_list = []
        win_count = 0
        loss_count = 0
        long_count = 0
        short_count = 0
        sharpe = 'NOT COMPUTED'
        for count, multi_leg_order in enumerate(closed_orders):
            signal_id = multi_leg_order[0]['signal_id']
            symbol = multi_leg_order[0]['symbol']
            # Implementation for calculating performance metrics
            signal_open_date = multi_leg_order[0]['timestamp']
            if signal_open_date > config_dict['backtest_inputs']['start_time'] and signal_open_date < config_dict['backtest_inputs']['end_time']:
                signal_profit = self.calculate_multi_leg_order_profit(multi_leg_order)
                profit += signal_profit
                if signal_profit >= 0:
                    win_count += 1
                    profits_list.append(signal_profit)
                else:
                    loss_count += 1
                    losses_list.append(signal_profit)
                # self.logger.debug({'multi_leg_order':multi_leg_order})
                if multi_leg_order[0]['orderDirection'] == 'BUY':
                    long_count += 1
                    long_list.append(signal_profit)
                elif multi_leg_order[0]['orderDirection'] == 'SELL':
                    short_count += 1
                    short_list.append(signal_profit)
        self.backtest_performance_metrics['profit'] = profit
        self.backtest_performance_metrics['win_pct'] = round((win_count / (win_count + loss_count)) * 100, 2) if (win_count + loss_count) > 0 else 0
        self.backtest_performance_metrics['long_count'] = long_count
        self.backtest_performance_metrics['short_count'] = short_count
        self.backtest_performance_metrics['Average_Profit'] = round(sum(profits_list) / len(profits_list), 2) if len(profits_list) > 0 else 0
        self.backtest_performance_metrics['Average_Loss'] = round(sum(losses_list) / len(losses_list), 2) if len(losses_list) > 0 else 0
        self.backtest_performance_metrics['long_Average'] = round(sum(long_list) / len(long_list), 2) if len(long_list) > 0 else 0
        self.backtest_performance_metrics['short_Average'] = round(sum(short_list) / len(short_list), 2) if len(short_list) > 0 else 0
        self.backtest_performance_metrics['sharpe_ratio'] = sharpe
        
        return self.backtest_performance_metrics

    def generate_report(self):
        # Create a txt file with Key: Value followed by a new line from self.backtest_performance_metrics
        self.backtest_report = ''
        for key, value in self.backtest_performance_metrics.items():
            self.backtest_report += f'{key}: {value}\n'
        self.logger.info(f'Backtest Report: \n{self.backtest_report}')
    
    def save_backtest(self, config_dict, open_orders, closed_orders):
        testname = config_dict['backtest_inputs']['backtest_name'] if 'backtest_name' in config_dict['backtest_inputs'] else None
        if not testname:
            testname = generate_hash_id(config_dict['backtest_inputs'], 1)
        if testname:
            # Get all backtest folders that exist
            backtest_folders = os.listdir(self.backtest_folder_path) if os.path.exists(self.backtest_folder_path) else []
            if testname in backtest_folders:
                # If backtest folder already exists, then add a number to the end of the folder name
                i = 1
                while f'{testname}_{i}' in backtest_folders:
                    i += 1
                testname = f'{testname}_{i}'
            
            
            # Create folder if it doesn't exist
            self.test_folder_path = os.path.join(self.backtest_folder_path, testname)
            os.makedirs(self.test_folder_path, exist_ok=True)
            
            # Save the report as txt file for now and HTML file later
            if self.backtest_report:
                # with open(os.path.join(self.backtest_folder_path, testname, 'backtest_report.html'), 'w') as file:
                #     file.write(self.backtest_report)
                # save as txt for now
                with open(os.path.join(self.backtest_folder_path, testname, 'backtest_report.txt'), 'w') as file:
                    file.write(self.backtest_report)
            
            # Save open and closed orders as pickle
            backtest_output = {'open_orders':open_orders, 'closed_orders':closed_orders, 'config_dict':config_dict}
            backtest_output_path = os.path.join(self.backtest_folder_path, testname, 'backtest_output.pkl')
            with open(backtest_output_path, 'wb') as file:
                pickle.dump(backtest_output, file)
            
            # Save performance metrics as json
            if self.backtest_performance_metrics:
                with open(os.path.join(self.backtest_folder_path, testname, 'performance_metrics.json'), 'w') as file:
                    json.dump(self.backtest_performance_metrics, file)
        else:
            self.test_folder_path = 'None'
            
        return self.test_folder_path, testname
        