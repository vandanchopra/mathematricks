# strategy_1.py

from systems.base_strategy import BaseStrategy
import numpy as np

class Strategy1(BaseStrategy):
      def __init__(self, config_dict):
                super().__init__(config_dict)

      def generate_signals(self, market_data):
                # Example logic for generating buy/sell signals
                signals = []
                for symbol in market_data:
                              price_data = market_data[symbol]['close']
                              short_mavg = price_data.rolling(window=10).mean()
                              long_mavg = price_data.rolling(window=50).mean()

                    # Generate buy signal
                              if short_mavg[-1] > long_mavg[-1]:
                                                signals.append({'symbol': symbol, 'action': 'buy'})
                                            # Generate sell signal
elif short_mavg[-1] < long_mavg[-1]:
                signals.append({'symbol': symbol, 'action': 'sell'})

        return signals

    def set_parameters(self, **params):
              # Method to set parameters for optimization
              self.params = params

    def optimize_parameters(self):
              # Dummy method for parameter optimization
              # Should iterate over possible parameter values
              best_params = {'short_window': 10, 'long_window': 50}
              self.set_parameters(**best_params)

# Example usage:
# strategy = Strategy1(config_dict)
# strategy.optimize_parameters()
# signals = strategy.generate_signals(market_data)
