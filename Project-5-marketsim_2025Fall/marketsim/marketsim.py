""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""MC2-P1: Market simulator.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	 	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	 	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 		  		  		    	 		 		   		 		  
or edited.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	 	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Student Name: (replace with your name)  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT User ID: dcarbono3 (replace with your User ID)  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT ID: (replace with your GT ID)  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		 	 	 		  		  		    	 		 		   		 		  
import os  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		 	 	 		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def author():
    """
    Returns the GT username of the student.
    """
    return 'dcarbono3'


def study_group():
    """
    Returns a comma-separated string of GT usernames of members in study group.
    """
    return 'dcarbono3'


def compute_portvals(  		  	   		 	 	 		  		  		    	 		 		   		 		  
    orders_file="./orders/orders.csv",  		  	   		 	 	 		  		  		    	 		 		   		 		  
    start_val=1000000,  		  	   		 	 	 		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		 	 	 		  		  		    	 		 		   		 		  
    impact=0.005,  		  	   		 	 	 		  		  		    	 		 		   		 		  
):  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param orders_file: Path of the order file or the file object  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type orders_file: str or file object  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    
    # Read orders file
    orders_df = pd.read_csv(orders_file, index_col='Date', 
                            parse_dates=True, na_values=['nan'])
    orders_df = orders_df.sort_index()
    
    # Get date range from orders
    start_date = orders_df.index.min()
    end_date = orders_df.index.max()
    
    # Get list of unique symbols
    symbols = orders_df['Symbol'].unique().tolist()
    
    # Get price data for all symbols
    prices_df = get_data(symbols, pd.date_range(start_date, end_date))
    prices_df = prices_df[symbols]  # Remove SPY
    
    # Initialize cash and holdings
    cash = start_val
    holdings = {symbol: 0 for symbol in symbols}
    
    # Create DataFrame to store portfolio values
    portvals = pd.DataFrame(index=prices_df.index, columns=['value'])
    
    # Process each day
    for date in prices_df.index:
        # Execute orders for this day if any
        if date in orders_df.index:
            day_orders = orders_df.loc[date]
            
            # Handle case where there's only one order (Series) vs multiple orders (DataFrame)
            if isinstance(day_orders, pd.Series):
                day_orders = day_orders.to_frame().T
            
            # Process each order for this day
            for idx, order in day_orders.iterrows():
                symbol = order['Symbol']
                shares = order['Shares']
                order_type = order['Order']
                
                # Get the price for this symbol on this date
                price = prices_df.loc[date, symbol]
                
                if order_type == 'BUY':
                    # Market impact: price moves against us (up for buy)
                    adjusted_price = price * (1.0 + impact)
                    # Cost = shares * adjusted_price + commission
                    transaction_cost = shares * adjusted_price + commission
                    cash -= transaction_cost
                    holdings[symbol] += shares
                    
                elif order_type == 'SELL':
                    # Market impact: price moves against us (down for sell)
                    adjusted_price = price * (1.0 - impact)
                    # Proceeds = shares * adjusted_price - commission
                    transaction_proceeds = shares * adjusted_price - commission
                    cash += transaction_proceeds
                    holdings[symbol] -= shares
        
        # Calculate portfolio value for this day
        stocks_value = sum(holdings[symbol] * prices_df.loc[date, symbol] 
                          for symbol in symbols)
        portvals.loc[date, 'value'] = cash + stocks_value
    
    return portvals


def test_code():  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    Helper function to test code  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # this is a helper function you can use to test your code  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # Define input parameters  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    of = "./orders/orders-short.csv"  		  	   		 	 	 		  		  		    	 		 		   		 		  
    sv = 1000000  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # Process orders  		  	   		 	 	 		  		  		    	 		 		   		 		  
    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    if isinstance(portvals, pd.DataFrame):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		 	 	 		  		  		    	 		 		   		 		  
    else:  		  	   		 	 	 		  		  		    	 		 		   		 		  
        print("warning, code did not return a DataFrame")
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # Get portfolio stats  		  	   		 	 	 		  		  		    	 		 		   		 		  
    start_date = portvals.index.min()
    end_date = portvals.index.max()
    
    # Calculate daily returns
    daily_returns = portvals.pct_change()[1:]
    
    # Calculate statistics
    cum_ret = (portvals[-1] / portvals[0]) - 1
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std()
    sharpe_ratio = np.sqrt(252) * (avg_daily_ret / std_daily_ret)
    
    # Get SPY for comparison
    spy_df = get_data(['SPY'], pd.date_range(start_date, end_date))
    spy_df = spy_df[['SPY']]
    spy_returns = spy_df.pct_change()[1:]
    
    cum_ret_SPY = (spy_df['SPY'][-1] / spy_df['SPY'][0]) - 1
    avg_daily_ret_SPY = spy_returns['SPY'].mean()
    std_daily_ret_SPY = spy_returns['SPY'].std()
    sharpe_ratio_SPY = np.sqrt(252) * (avg_daily_ret_SPY / std_daily_ret_SPY)
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # Compare portfolio against $SPX  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Date Range: {start_date} to {end_date}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print()  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of $SPX : {sharpe_ratio_SPY}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print()  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of $SPX : {cum_ret_SPY}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print()  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of $SPX : {std_daily_ret_SPY}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print()  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of $SPX : {avg_daily_ret_SPY}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print()  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Final Portfolio Value: {portvals[-1]}")
    print()
    print("Portfolio values:")
    print(portvals)
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	 	 		  		  		    	 		 		   		 		  
    test_code()