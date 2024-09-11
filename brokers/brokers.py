from brokers.ibkr import IBKR
from brokers.yahoo import Yahoo

class Brokers():
    def __init__(self):
        self.ib = IBKR(None)
        self.yahoo = Yahoo()