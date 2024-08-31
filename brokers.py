from brokers.ib import IB
from brokers.yahoo import Yahoo

class Brokers():
    def __init__(self):
        self.ib = IB()
        self.yahoo = Yahoo()