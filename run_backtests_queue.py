from cgi import test
from turtle import back
from systems.backtests_queue import BacktestQueue
from mathematricks import Mathematricks

if __name__ == '__main__':
    bq = BacktestQueue()
    run_bool = True
    
    while run_bool:
        '''Try: First load all the backtests queue'''
        try:
            backtests_queue = bq.get_backtests_queue()
            ''' Then save it to the local file.'''
            bq.save_backtests_queue_to_file(backtests_queue)
        except:
            pass
            
        backtests_queue = bq.load_backtests_queue_from_file()
        run_bool = True if len(backtests_queue) > 0 else False
        
        if run_bool:
            ''' Now, take the first backtest from the queue, and create a config_dict from it. and run the mathematricks.py with this config_dict.'''
            backtest = backtests_queue.pop(0)
            config_dict = bq.create_config_dict_from_backtest_entry(backtest)
            config_dict['backtest_inputs']['save_backtest_results'] = True
            try:
                mathematricks = Mathematricks(config_dict)
                mathematricks.run()
                    
                '''After the backtest is run, remove it from backtests_queue and update the google sheet'''
                # remove backtest[0] from backtests_queue
                bq.update_backtests_queue(backtests_queue)
                # Update the local file as well.
                bq.save_backtests_queue_to_file(backtests_queue)
                
                # Update the performance metrics of the backtest
                backtest_performance_metrics = mathematricks.reporter.backtest_performance_metrics
                test_folder_path = mathematricks.reporter.test_folder_path
            except Exception as e:
                backtest_performance_metrics = {}
                backtest_performance_metrics['profit'] = 'ERROR'
                backtest_performance_metrics['win_pct'] = 'ERROR'
                backtest_performance_metrics['long_count'] = 'ERROR'
                backtest_performance_metrics['short_count'] = 'ERROR'
                backtest_performance_metrics['sharpe_ratio'] = 'ERROR'
                test_folder_path = 'ERROR'
            # Now update the Completed Backtest to the Sheet
            bq.add_completed_backtest_to_google_sheet(backtest, backtest_performance_metrics, test_folder_path)

            if len(backtests_queue) == 0:
                run_bool = False
            