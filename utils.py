# python: Create a logger for my system

import logging
from os import name

def create_logger(log_level, logger_name='mathematricks'):
    # add the handlers to the logger
    logfile_path = f'./logs/{logger_name}.log'
    logging.basicConfig(filename=logfile_path, encoding='utf-8', level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(logger_name)
    return logger

if __name__ == '__main__':
    logger = create_logger(log_level=logging.DEBUG, logger_name='mathematricks2')
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')
    pass