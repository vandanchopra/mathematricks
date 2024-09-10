from ib_insync import IB, Stock
import nest_asyncio
nest_asyncio.apply()

class IBKR():
    def __init__(self, ib):
        if ib is None:
            self.ib = IB()
        else:
            self.ib = ib

    def connect_to_IBKR(self):
        '''NOTE: First start the TWS or Gateway software from IBKR'''

        ib = self.ib

        # Connect to the IBKR TWS (Trader Workstation) or Gateway
        ib.connect('127.0.0.1', 7496, clientId=0)

        # Check if the connection is successful
        if ib.isConnected():
            print('Connected to IBKR')
        else:
            print('Failed to connect to IBKR')
        return ib

    def get_current_portfolio(self):
        # Request the current portfolio from IBKR
        portfolio = self.ib.portfolio()

        # Return the portfolio
        return portfolio

    def get_current_price(self, ticker: str, exchange: str, currency: str):
        # Request the current market data for AAPL
        contract = self.ib.qualifyContracts(Stock(ticker, exchange, currency))
        ticker = self.ib.reqTickers(contract)[0]

        # Get the current price
        current_price = ticker.marketPrice()

        # Return the current price
        return current_price

    def get_current_stop_loss_orders(self):
        # Request the current stop loss orders from IBKR
        all_open_orders = self.ib.openOrders()
        stop_loss_orders = []
        for open_order in all_open_orders:
            if open_order.orderType == 'STP':
                stop_loss_orders.append(open_order)

        # Return the stop loss orders
        return stop_loss_orders

    def execute_order(self, ticker: str, exchange: str, currency: str, action: str, quantity: int, order_type: str, limit_price: float = 0, stop_price: float = 0):
        # Create a contract for the stock
        ib = self.ib
        contract = ib.qualifyContracts(Stock(ticker, exchange, currency))

        # Create an order for the stock
        if order_type == 'MKT':
            order = ib.marketOrder(action, quantity)
        elif order_type == 'LMT':
            order = ib.LimitOrder(action, quantity, limit_price)
        elif order_type == 'STP':
            order = ib.StopOrder(action, quantity, stop_price)

        # Place the order
        trade = ib.placeOrder(contract, order)

        # Return the trade
        return trade

    def cancel_order(self, trade):
        # Cancel the order
        self.ib.cancelOrder(trade)

        # Return the trade
        return trade

    def modify_order(self, trade, new_quantity: int, new_limit_price: float, new_stop_price: float):
        # Modify the order
        self.ib.modifyOrder(trade, new_quantity, new_limit_price, new_stop_price)

        # Return the trade
        return trade
    
    
if __name__ == '__main__':
    ibkr = IBKR()
    ibkr.connect_to_IBKR()
    print(ibkr.get_current_portfolio())
    # get a stream of market data for AAPL and MSFT and print the streaming data
    