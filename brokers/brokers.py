from brokers.ibkr import IBKR
from brokers.yahoo import Yahoo

class Brokers():
    def __init__(self):
        # self.ib = IBKR()
        self.yahoo = Yahoo()