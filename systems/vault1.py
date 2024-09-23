import pandas as pd


def retOrders():
    orders = [[
        # Entry order
        {
            'symbol': "INTC",
            'timestamp': pd.Timestamp('2023-01-01 01:39:00'),
            'orderSide': "BUY",
            'entryPrice': 151.43,
            'orderType': "MKT",
            'timeInForce': 'DAY',
            'orderQuantity': 10,
            'strategy_name': "SMA15-SMA30",
            'broker': 'IBKR'
        },
        # Exit Order / Trail Stop Order
        {
            'symbol': "INTC",
            'timestamp': pd.Timestamp('2023-01-01 01:39:00'),
            'orderSide': 'SELL',
            'exitPrice': 149.43,
            'orderType': 'STP',
            'timeInForce': 'DAY',
            'orderQuantity': 10,
            'strategy_name': 'SMA15-SMA30',
            'broker': 'SIM'
        }]]
    return orders
