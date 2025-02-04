from brokers.ibkr import IBKR
from brokers.sim import Sim
from brokers.kraken import Kraken

class Brokers():
    def __init__(self):
        self.ib = IBKR()
        self.sim = Sim()
        self.kraken = Kraken()