#!/usr/bin/env python
import sys
import os
import argparse
import signal
from tqdm import tqdm

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from systems.utils import create_logger, load_symbols_universe_df, project_path
from brokers.brokers import Brokers
logger = create_logger(log_level='INFO', logger_name='manual_downloader')

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nDownload interrupted by user. Cleaning up...")
    # Try to disconnect from brokers gracefully
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

def download_data(granularity, broker_name=None, forward_update=True, backward_update=False, throttle_secs=1):
    logger.info(f"Starting data download for granularity: {granularity}")
    logger.info(f"Forward update: {forward_update}, Backward update: {backward_update}")
    
    # Load symbols
    try:
        symbols_universe_df = load_symbols_universe_df(broker_name=broker_name)
        symbols = symbols_universe_df['symbol'].tolist()
    except Exception as e:
        logger.error(f"Error loading symbols: {e}")
        return
    
    # Initialize brokers
    brokers = Brokers()
    
    # Set up the interval inputs format required by brokers
    interval_inputs = {
        granularity: {
            'columns': ['open', 'high', 'low', 'close', 'volume'],
            'lookback': 0  # No lookback needed for manual download
        }
    }
    
    sources = {
        'yahoo': ('Yahoo Finance', brokers.sim.data),
        'ibkr': ('IBKR', brokers.ib),
        'kraken': ('Kraken', brokers.kraken)
    }

    # Filter sources based on broker_name if provided
    if broker_name:
        if broker_name not in sources:
            logger.error(f"Invalid broker: {broker_name}")
            return
        sources_to_process = {broker_name: sources[broker_name]}
    else:
        sources_to_process = sources

    for source_id, (source_name, broker) in sources_to_process.items():
        # try:
            logger.info(f"Starting download from {source_name}")
            
            # Handle IBKR connection
            if source_name == 'IBKR':
                if not broker.connected:
                    logger.info("Connecting to IBKR...")
                    broker.connect_to_IBKR()
                if not broker.connected:
                    logger.error("Failed to connect to IBKR. Skipping IBKR download.")
                    continue
                broker_data = broker.data
            # Handle Kraken connection
            elif source_name == 'Kraken':
                if not broker.connected:
                    logger.info("Connecting to Kraken...")
                    broker.connect_to_kraken()
                if not broker.connected:
                    logger.error("Failed to connect to Kraken. Skipping Kraken download.")
                    continue
                broker_data = broker.data
            else:
                broker_data = broker
            
            # Download data
            logger.info(f"{source_name}: {'Backward update enabled' if backward_update else 'Backward update disabled'}, "
                       f"{'Forward update enabled' if forward_update else 'Forward update disabled'}")
            
            data_config = dict(stock_symbols=symbols, interval_inputs=interval_inputs,
                             throttle_secs=throttle_secs, start_date=None, end_date=None,
                             update_data=True, forward_update=forward_update, backward_update=backward_update)
            market_data_df = broker_data.update_price_data(**data_config)
            
            logger.info(f"Completed {source_name} download:")
            logger.info(f"- Granularity: {granularity}")
            logger.info(f"- Throttle: {throttle_secs} seconds between requests")
            
            if market_data_df is not None and not market_data_df.empty:
                logger.info(f"Successfully downloaded data for {len(market_data_df.columns.unique(level=1))} symbols from {source_name}")
            else:
                logger.warning(f"No data downloaded from {source_name}")
                
        # except Exception as e:
        #     logger.error(f"Error downloading from {source_name}: {e}")
        #     continue

def main():
    parser = argparse.ArgumentParser(description='Download market data for all assets')
    parser.add_argument('granularity', choices=['1m', '2m', '5m', '1d'],
                      help='Data granularity to download')
    parser.add_argument('--broker', choices=['yahoo', 'ibkr', 'kraken'],
                      help='Specific broker to download from (default: all brokers)')
    parser.add_argument('--no-forward', action='store_true',
                      help='Disable forward (new) data updates')
    parser.add_argument('--backward', action='store_true',
                      help='Enable backward data updates')
    parser.add_argument('--throttle', type=int, default=1,
                      help='Throttle seconds between requests (default: 1)')
    
    args = parser.parse_args()
    
    print(f"\nStarting manual data download:")
    print(f"- Granularity: {args.granularity}")
    print(f"- Forward updates: {'disabled' if args.no_forward else 'enabled'}")
    print(f"- Broker: {args.broker if args.broker else 'all'}")
    print(f"- Backward updates: {'enabled' if args.backward else 'disabled'}")
    print(f"- Throttle: {args.throttle} seconds")
    
    # Set up signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    
    download_data(
        granularity=args.granularity,
        broker_name=args.broker,
        forward_update=not args.no_forward,
        backward_update=args.backward,
        throttle_secs=args.throttle
    )
    print("\nData download completed successfully!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user")
        sys.exit(0)