import pandas as pd
import numpy as np
import datetime as dt
from util import get_data


def author():
    return 'dcarbono3'


def study_group():
    return 'dcarbono3'


def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), 
               ed=dt.datetime(2009, 12, 31), sv=100000):
    """
    Generate optimal trades using perfect future knowledge
    """
    date_range = pd.date_range(sd, ed)
    price_data = get_data([symbol], date_range, addSPY=True, colname='Adj Close')
    stock_prices = price_data[[symbol]].copy()  
    
    stock_prices.ffill(inplace=True)
    stock_prices.bfill(inplace=True)
    daily_trades = pd.DataFrame(0.0, index=stock_prices.index, columns=[symbol])
    
    # Track current holdings
    current_holdings = 0
    
    # Process each day
    total_days = len(stock_prices)
    
    for day_idx in range(total_days - 1):
        today_price = stock_prices.iloc[day_idx, 0]
        tomorrow_price = stock_prices.iloc[day_idx + 1, 0]
        
        if tomorrow_price > today_price:
            target_position = 1000
        elif tomorrow_price < today_price:
            target_position = -1000
        else:
            target_position = current_holdings
        
        # Calculate trade needed
        shares_to_trade = target_position - current_holdings
        
        if shares_to_trade != 0:
            daily_trades.iloc[day_idx, 0] = shares_to_trade
            current_holdings = target_position
    
    return daily_trades


if __name__ == "__main__":
    print("Testing Theoretically Optimal Strategy...")
    
    test_symbol = "JPM"
    start = dt.datetime(2008, 1, 1)
    end = dt.datetime(2009, 12, 31)
    
    optimal_trades = testPolicy(symbol=test_symbol, sd=start, ed=end, sv=100000)
    
    print(f"\nTrades Summary:")
    print(f"Total trading days: {len(optimal_trades)}")
    print(f"Days with trades: {(optimal_trades[test_symbol] != 0).sum()}")