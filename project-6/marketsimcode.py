"""
marketsimcode.py
Market Simulator
Computes portfolio value over time given a trades DataFrame
"""

import pandas as pd
import numpy as np
from util import get_data

def author():
    return 'dcarbono3'

def compute_portvals(trades_df, start_val=100000, commission=9.95, impact=0.005):
    """
    Compute portfolio values over time given trades
    
    Parameters:
    trades_df: DataFrame with dates as index and symbols as columns
               Values represent shares traded (positive=buy, negative=sell)
    start_val: Starting cash value
    commission: Commission per trade
    impact: Market impact (slippage)
    
    Returns:
    portvals: Series of portfolio values indexed by date
    """
    
    # Get start and end dates
    start_date = trades_df.index.min()
    end_date = trades_df.index.max()
    
    # Get all symbols being traded
    symbols = list(trades_df.columns)
    
    # Get price data for all symbols
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates, addSPY=True, colname='Adj Close')
    prices = prices_all[symbols]  # Only keep symbols we're trading
    
    # Forward fill and backward fill to handle missing data
    prices = prices.fillna(method='ffill').fillna(method='bfill')
    
    # Initialize holdings and cash
    # holdings: DataFrame tracking shares of each symbol owned each day
    holdings = pd.DataFrame(0, index=prices.index, columns=symbols)
    cash = pd.Series(start_val, index=prices.index)
    
    # Process trades day by day
    for date in prices.index:
        # Update holdings and cash based on trades
        if date in trades_df.index:
            for symbol in symbols:
                trade = trades_df.loc[date, symbol]
                
                if trade != 0:
                    # Price at which we trade (including impact)
                    if trade > 0:  # Buying
                        price = prices.loc[date, symbol] * (1 + impact)
                    else:  # Selling
                        price = prices.loc[date, symbol] * (1 - impact)
                    
                    # Update cash (subtract cost of shares and commission)
                    cash[date] = cash[date] - (trade * price) - commission
        
        # Update holdings to reflect trades
        if date in trades_df.index:
            for symbol in symbols:
                trade = trades_df.loc[date, symbol]
                if date == prices.index[0]:
                    holdings.loc[date, symbol] = trade
                else:
                    prev_date = prices.index[prices.index.get_loc(date) - 1]
                    holdings.loc[date, symbol] = holdings.loc[prev_date, symbol] + trade
        else:
            if date != prices.index[0]:
                prev_date = prices.index[prices.index.get_loc(date) - 1]
                holdings.loc[date, :] = holdings.loc[prev_date, :]
        
        # Carry forward cash from previous day if no trade today
        if date != prices.index[0] and date not in trades_df.index:
            prev_date = prices.index[prices.index.get_loc(date) - 1]
            cash[date] = cash[prev_date]
    
    # Calculate portfolio value each day
    holdings_value = (holdings * prices).sum(axis=1)
    portvals = cash + holdings_value
    
    return portvals


# Alternative simpler implementation
def compute_portvals_simple(trades_df, start_val=100000, commission=0.0, impact=0.0):
    """
    Simplified version - easier to understand and debug
    """
    # Get date range
    start_date = trades_df.index.min()
    end_date = trades_df.index.max()
    dates = pd.date_range(start_date, end_date)
    
    # Get symbol(s)
    symbols = list(trades_df.columns)
    
    # Get prices
    prices_all = get_data(symbols, dates, addSPY=True, colname='Adj Close')
    prices = prices_all[symbols]
    prices = prices.fillna(method='ffill').fillna(method='bfill')
    
    # Initialize
    cash = start_val
    holdings = {symbol: 0 for symbol in symbols}
    portvals = []
    
    for date in prices.index:
        # Process any trades for this date
        if date in trades_df.index:
            for symbol in symbols:
                trade = trades_df.loc[date, symbol]
                if trade != 0:
                    # Calculate trade cost
                    if trade > 0:  # Buying
                        price = prices.loc[date, symbol] * (1 + impact)
                    else:  # Selling
                        price = prices.loc[date, symbol] * (1 - impact)
                    
                    # Update cash and holdings
                    cash -= (trade * price + commission)
                    holdings[symbol] += trade
        
        # Calculate portfolio value
        holdings_value = sum(holdings[symbol] * prices.loc[date, symbol] 
                           for symbol in symbols)
        portval = cash + holdings_value
        portvals.append(portval)
    
    return pd.Series(portvals, index=prices.index)