"""
TheoreticallyOptimalStrategy.py
Theoretically Optimal Strategy
Trades optimally by knowing future prices
"""

import datetime as dt
import pandas as pd
from util import get_data

def author():
    return 'dcarbono3'  

def study_group():
    return 'NONE'  

class OptimalTrader:
    """
    A class to determine the best possible sequence of trades by looking ahead at price data.
    """
    def __init__(self, symbol="JPM", start_date=dt.datetime(2008, 1, 1), 
                 end_date=dt.datetime(2009, 12, 31)):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.dates = pd.date_range(self.start_date, self.end_date)
        
    def _get_price_data(self):
        """Fetches and cleans price data for the specified symbol."""
        price_df = get_data([self.symbol], self.dates, addSPY=False, colname='Adj Close')
        price_df = price_df.fillna(method='ffill').fillna(method='bfill')
        return price_df[self.symbol]

    def generate_trades(self):
        """
        Main method to create the trades DataFrame.

        The logic is to first determine the ideal holding for each day (-1000, 0, or 1000)
        and then calculate the trades needed to shift from one day's holding to the next.
        """
        prices = self._get_price_data()
        
        # 1. Determine the ideal holding for each day based on the *next* day's price movement.
        # A positive shift means we should have been long (+1000). A negative shift means short (-1000).
        holdings = pd.Series(0, index=prices.index)
        price_deltas = prices.diff().shift(-1) # Look ahead one day
        
        holdings[price_deltas > 0] = 1000  # If price goes up tomorrow, we want to be long today.
        holdings[price_deltas < 0] = -1000 # If price goes down tomorrow, we want to be short today.
        
        # 2. The trade for any given day is the difference between that day's
        # desired holding and the previous day's holding.
        trades = holdings.diff().fillna(holdings.iloc[0])
        
        # Create a DataFrame in the required format
        trades_df = trades.to_frame(name=self.symbol)
        
        return trades_df

# Standalone function to match the project API
def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), 
               ed=dt.datetime(2009, 12, 31), sv=100000):
    trader = OptimalTrader(symbol=symbol, start_date=sd, end_date=ed)
    return trader.generate_trades()


# Test code (optional - for your own testing)
if __name__ == "__main__":
    trades = testPolicy(symbol="JPM", 
                       sd=dt.datetime(2008, 1, 1), 
                       ed=dt.datetime(2009, 12, 31), 
                       sv=100000)
    print("Trades generated:")
    print(trades[trades['JPM'] != 0])  # Print only days with trades
    print(f"\nTotal trading days: {len(trades)}")
    print(f"Days with trades: {(trades['JPM'] != 0).sum()}")