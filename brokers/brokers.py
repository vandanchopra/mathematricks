from brokers.ibkr import IBKR
from brokers.sim import Sim

class Brokers():
    def __init__(self):
        ib = None
        self.ib = IBKR(ib)
        self.sim = Sim()