from systems.utils import project_path
import os, json, pickle

class PerformanceReporter:
    def __init__(self, config_dict, open_orders, closed_orders, market_data_df_root):
        self.config_dict = config_dict
        self.open_orders = open_orders
        self.closed_orders = closed_orders
        self.market_data_df_root = market_data_df_root
        self.backtest_folder_path = project_path + 'db/vault/backtest_reports'
        self.backtest_performance_metrics = None
        self.backtest_report = None
    
    def close_all_open_orders(self, market_data_df):
        self.logger.warning('THIS FUNCTION IS WRONG. IT NEEDS TO CLOSE ORDERS BY SENDING CLOSE ORDERS TO THE OMS. OR RENAME THIS FUNCTION TO ONLY USE WITH BACKTEST ENDING.')
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
            self.closed_orders.append(multi_leg_order)
    
    def calculate_multi_leg_order_profit(self, multi_leg_order):
        sell_orders = []
        buy_orders = []
        quantity = 0

        # Separate out all sell and buy orders
        for order in multi_leg_order:
            if order['orderDirection'] == 'SELL' and order['status'] == 'closed':
                sell_orders.append(order)
                quantity += order['orderQuantity'] * -1
            elif order['orderDirection'] == 'BUY' and order['status'] == 'closed':
                buy_orders.append(order)
                quantity += order['orderQuantity']

        if quantity == 0:
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
        else:
            return 0
    def calculate_backtest_performance_metrics(self):
        # Implementation for calculating performance metrics
        self.backtest_performance_metrics = {}
        profit = 0
        win_count = 0
        loss_count = 0
        long_count = 0
        short_count = 0
        sharpe = 0
        for multi_leg_order in self.closed_orders:
            # Implementation for calculating performance metrics
            profit += self.calculate_multi_leg_order_profit(multi_leg_order)
            if profit > 0:
                win_count += 1
            elif profit < 0:
                loss_count += 1
            if multi_leg_order[0]['orderDirection'] == 'BUY':
                long_count += 1
            elif multi_leg_order[0]['orderDirection'] == 'SELL':
                short_count += 1
        self.backtest_performance_metrics['profit'] = profit
        self.backtest_performance_metrics['win_pct'] = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else 0
        self.backtest_performance_metrics['long_count'] = long_count
        self.backtest_performance_metrics['short_count'] = short_count
        self.backtest_performance_metrics['sharpe_ratio'] = sharpe
        
        return self.backtest_performance_metrics

    def generate_report(self):
        # Create a txt file with Key: Value followed by a new line from self.backtest_performance_metrics
        self.backtest_report = ''
        for key, value in self.backtest_performance_metrics.items():
            self.backtest_report += f'{key}: {value}\n'
    
    def save_backtest(self):
        testname = self.config_dict['backtest_inputs']['Backtest Name'] if 'Backtest Name' in self.config_dict['backtest_inputs'] else None
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
            backtest_orders = {'open_orders':self.open_orders, 'closed_orders':self.closed_orders}
            backtest_orders_path = os.path.join(self.backtest_folder_path, testname, 'backtest_orders.pkl')
            with open(backtest_orders_path, 'wb') as file:
                pickle.dump(backtest_orders, file)
            
            # Save performance metrics as json
            if self.backtest_performance_metrics:
                with open(os.path.join(self.backtest_folder_path, testname, 'performance_metrics.json'), 'w') as file:
                    json.dump(self.backtest_performance_metrics, file)
        else:
            self.test_folder_path = 'None'
            
        return self.test_folder_path
        