# marketsimcode.py

"""Market Simulator - Accepts DataFrame trades"""
import pandas as pd
from util import get_data


def author():
    return 'dcarbono3' 


def compute_portvals(orders_df, start_val=100000, commission=9.95, impact=0.005):
    """
    Computes portfolio values from trades DataFrame.
    
    Parameters:
    orders_df: DataFrame with dates as index, symbol columns with trade amounts
    start_val: Starting cash
    commission: Fixed commission per trade
    impact: Market impact
    
    Returns:
    portvals: Series of portfolio values
    """
    start_date = orders_df.index.min()
    end_date = orders_df.index.max()
    symbols = orders_df.columns.tolist()
    
    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices = prices[symbols]
    prices.ffill(inplace=True)
    prices.bfill(inplace=True)
    
    cash = start_val
    holdings = pd.Series(0, index=symbols)
    portvals = pd.Series(0.0, index=prices.index)
    
    for date in prices.index:
        if date in orders_df.index:
            for symbol in symbols:
                shares = orders_df.loc[date, symbol]
                
                if shares != 0:
                    price = prices.loc[date, symbol]
                    
                    if shares > 0:  # BUY
                        trade_price = price * (1.0 + impact)
                        cash -= shares * trade_price
                        cash -= commission
                        holdings[symbol] += shares
                    else:  # SELL
                        trade_price = price * (1.0 - impact)
                        cash -= shares * trade_price
                        cash -= commission
                        holdings[symbol] += shares
        
        equity_value = sum(holdings[symbol] * prices.loc[date, symbol] for symbol in symbols)
        portvals[date] = cash + equity_value
    
    return portvals