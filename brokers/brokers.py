from brokers.ibkr import IBKR
from brokers.sim import Sim

class Brokers():
    def __init__(self):
        self.ib = IBKR(None)
        self.sim = Sim()