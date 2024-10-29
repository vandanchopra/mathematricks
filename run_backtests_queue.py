from systems.backtests_queue.backtests_queue import BacktestQueue
from mathematricks import Mathematricks

if __name__ == '__main__':
    bq = BacktestQueue()
    run_bool = True
    
    while run_bool:
        '''Try: First load all the backtests queue'''
        try:
            backtests_queue = bq.get_backtests_queue_from_googlesheets()
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
                bq.update_backtests_queue()
                # Update the local file as well.
                bq.save_backtests_queue_to_file(backtests_queue)
                
                # Update the performance metrics of the backtest
                backtest_report = mathematricks.reporter.backtest_report
                test_folder_path = mathematricks.reporter.test_folder_path
                bq.add_completed_backtest_to_google_sheet(backtest, backtest_report, test_folder_path)
            except Exception as e:
                # raise Exception(e)
                bq.logger.debug({f"{backtest['backtest_name']}: {str(Exception)}: {str(e)}"})
                backtest_report = 'ERROR'
                test_folder_path = 'ERROR'
                bq.add_completed_backtest_to_google_sheet(backtest, backtest_report, test_folder_path)
            # Now update the Completed Backtest to the Sheet

            if len(backtests_queue) == 0:
                run_bool = False
            