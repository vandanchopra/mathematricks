# python: Create a logger for my system

import logging
import os

# def create_logger(log_level, logger_name='mathematricks', print_to_console=True):
#     logger = logging.getLogger(logger_name)
#     if not logger.hasHandlers():
#         # add the handlers to the logger
#         logfile_path = f'./logs/{logger_name}.log'
#         logging.basicConfig(filename=logfile_path, encoding='utf-8', level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         logger = logging.getLogger(logger_name)
        
#         # Create console handler and set level to debug
#         if print_to_console:
#             ch = logging.StreamHandler()
#             ch.setLevel(log_level)
#             # Add ch to logger
#             logger.addHandler(ch)
    
#     return logger


def create_logger(log_level, logger_name='mathematricks', print_to_console=True):
    logger = logging.getLogger(logger_name)
    if not logger.hasHandlers():
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Ensure the logs directory exists
        os.makedirs('./logs', exist_ok=True)
        
        # Create file handler and set level to log_level
        logfile_path = f'./logs/{logger_name}.log'
        fh = logging.FileHandler(logfile_path, encoding='utf-8')
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Create console handler and set level to log_level
        if print_to_console:
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        # Set the logger level
        logger.setLevel(log_level)
    
    return logger

if __name__ == '__main__':
    logger = create_logger(log_level=logging.DEBUG, logger_name='mathematricks2', print_to_console=True)
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')