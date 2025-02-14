# strategy_3.py

class Strategy3:
      def __init__(self, short_window=40, long_window=100):
                self.short_window = short_window
                self.long_window = long_window

      def generate_signals(self, market_data_df):
                signals = pd.DataFrame(index=market_data_df.index)
                signals['signal'] = 0.0

          # Create short simple moving average
                signals['short_mavg'] = market_data_df['close'].rolling(window=self.short_window, min_periods=1).mean()

          # Create long simple moving average
                signals['long_mavg'] = market_data_df['close'].rolling(window=self.long_window, min_periods=1).mean()

          # Create signals
                signals['signal'][self.short_window:] = np.where(
                    signals['short_mavg'][self.short_window:] > signals['long_mavg'][self.short_window:], 1.0, 0.0)

          # Generate trading orders
                signals['positions'] = signals['signal'].diff()

          return signals
