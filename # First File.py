# First File
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import GOOG # Built-in historical data for Google

class MyFirstSimulation(Strategy):
    # 1. Define your strategy variables
    short_window = 10
    long_window = 20

    def init(self):
        # Calculate moving averages based on historical closing prices
        price = self.data.Close
        self.ma1 = self.I(lambda x: pd.Series(x).rolling(self.short_window).mean(), price)
        self.ma2 = self.I(lambda x: pd.Series(x).rolling(self.long_window).mean(), price)

    def next(self):
        # 2. Define the "what to do" rules
        if crossover(self.ma1, self.ma2):
            self.buy() # Buy when short MA crosses above long MA
        elif crossover(self.ma2, self.ma1):
            self.position.close() # Sell / close position when it crosses back

# 3. Set up the account variables and run the simulation
bt = Backtest(GOOG, MyFirstSimulation, cash=10000, commission=.002)
stats = bt.run()

# 4. See how it would have done
print(stats)
bt.plot() # Opens an interactive HTML performance chart
