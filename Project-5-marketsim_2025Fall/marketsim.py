# marketsim.py

"""MC2-P1: Market simulator."""

import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data


def author():
    """Returns the GT username of the student."""
    return 'dcarbono3'


def study_group():
    """Returns a comma-separated string of GT usernames of members in study group."""
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
    :param commission: The fixed amount in dollars charged for each transaction
    :type commission: float
    :param impact: The amount the price moves against the trader
    :type impact: float
    :return: Portfolio values as a single-column dataframe
    :rtype: pandas.DataFrame
    """

    # Read the trades file 
    trades_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    trades_df.sort_index(inplace=True)

    # Get the date range 
    start_date = trades_df.index.min()
    end_date = trades_df.index.max()
    syms = trades_df['Symbol'].unique().tolist()

    prices = get_data(syms, pd.date_range(start_date, end_date))
    prices = prices[syms]

    # Set up portfolio tracking
    cash_on_hand = start_val
    positions = {s: 0 for s in syms}
    portfolio_values = pd.DataFrame(index=prices.index, columns=['Total Value'])

    # Step through each day
    for day, price_data in prices.iterrows():
        # Check if any trades happen on this day
        if day in trades_df.index:
            todays_trades = trades_df.loc[day]

            # If only one trade today, pandas returns a Series. Convert it to a DataFrame.
            if isinstance(todays_trades, pd.Series):
                todays_trades = pd.DataFrame(todays_trades).T

            for _, trade in todays_trades.iterrows():
                symbol = trade['Symbol']
                num_shares = trade['Shares']
                trade_type = trade['Order']
                stock_price = price_data[symbol]

                if trade_type == 'BUY':
                    trade_price = stock_price * (1.0 + impact)
                    cost_of_trade = trade_price * num_shares
                    cash_on_hand -= cost_of_trade
                    cash_on_hand -= commission
                    positions[symbol] += num_shares
                else:  # SELL
                    trade_price = stock_price * (1.0 - impact)
                    revenue_from_trade = trade_price * num_shares
                    cash_on_hand += revenue_from_trade
                    cash_on_hand -= commission
                    positions[symbol] -= num_shares

        # After any trades, calculate the new total portfolio value for the day
        equity_value = 0
        for s in syms:
            equity_value += positions[s] * prices.loc[day, s]
        
        portfolio_values.loc[day, 'Total Value'] = cash_on_hand + equity_value

    return portfolio_values


def test_code():
    """Helper function to test code"""
    
    # Test with one of the actual order files you have
    of = "./orders/orders-01.csv"  # Changed to a file that exists
    sv = 1000000
    
    print(f"Testing with {of}")
    print("=" * 60)
    
    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]
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
    print("\n" + "=" * 60)
    
    # Test with other order files if they exist
    test_files = ["./orders/orders-02.csv", "./orders/orders-03.csv"]
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\nTesting with {test_file}")
            print("=" * 60)
            portvals_test = compute_portvals(orders_file=test_file, start_val=1000000)
            if isinstance(portvals_test, pd.DataFrame):
                portvals_test = portvals_test[portvals_test.columns[0]]
            
            daily_ret_test = portvals_test.pct_change()[1:]
            cum_ret_test = (portvals_test[-1] / portvals_test[0]) - 1
            avg_daily_ret_test = daily_ret_test.mean()
            std_daily_ret_test = daily_ret_test.std()
            sharpe_ratio_test = np.sqrt(252) * (avg_daily_ret_test / std_daily_ret_test)
            
            print(f"Sharpe Ratio: {sharpe_ratio_test}")
            print(f"Cumulative Return: {cum_ret_test}")
            print(f"Final Portfolio Value: {portvals_test[-1]}")


if __name__ == "__main__":
    test_code()