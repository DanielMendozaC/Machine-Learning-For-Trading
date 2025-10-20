import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import get_data
import TheoreticallyOptimalStrategy as tos
import indicators


def author():
    return 'dcarbono3'


def simulate_market(trades_dataframe, starting_cash=100000, 
                   commission=0.0, impact=0.0):
    """
    Simulate portfolio performance based on trades
    """
    stock_symbol = trades_dataframe.columns[0]
    first_date = trades_dataframe.index.min()
    last_date = trades_dataframe.index.max()
    
    date_list = pd.date_range(first_date, last_date)
    price_data = get_data([stock_symbol], date_list, addSPY=True, colname='Adj Close')
    stock_price = price_data[[stock_symbol]].copy()  
    stock_price.ffill(inplace=True)
    stock_price.bfill(inplace=True)
    
    available_cash = starting_cash
    owned_shares = 0
    value_history = []
    
    for current_date in stock_price.index:
        price_today = stock_price.loc[current_date, stock_symbol]
        
        if current_date in trades_dataframe.index:
            trade_size = trades_dataframe.loc[current_date, stock_symbol]
            
            if trade_size != 0:
                if trade_size > 0:
                    effective_price = price_today * (1.0 + impact)
                else:
                    effective_price = price_today * (1.0 - impact)
                
                transaction_cost = abs(trade_size) * effective_price
                available_cash -= (trade_size * effective_price + commission)
                owned_shares += trade_size
        
        stock_value = owned_shares * price_today
        total_value = available_cash + stock_value
        value_history.append(total_value)
    
    return pd.Series(value_history, index=stock_price.index)


def calculate_performance_metrics(portfolio_values):
    """
    Compute performance statistics for a portfolio
    """
    daily_rets = portfolio_values.pct_change()
    daily_rets = daily_rets[1:]
    
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1.0
    return_std = daily_rets.std()
    return_mean = daily_rets.mean()
    
    return total_return, return_std, return_mean


def run_tos_analysis():
    """
    Execute Theoretically Optimal Strategy analysis
    """    
    stock_ticker = "JPM"
    period_start = dt.datetime(2008, 1, 1)
    period_end = dt.datetime(2009, 12, 31)
    initial_value = 100000
    
    print(f"\nGenerating optimal trades for {stock_ticker}...")
    optimal_trades = tos.testPolicy(symbol=stock_ticker, sd=period_start, 
                                    ed=period_end, sv=initial_value)
    
    num_trades = (optimal_trades[stock_ticker] != 0).sum()
    print(f"Generated {num_trades} trades")
    
    benchmark_trades = pd.DataFrame(0.0, index=optimal_trades.index, 
                                   columns=[stock_ticker])
    benchmark_trades.iloc[0][stock_ticker] = 1000
    
    print("\nSimulating portfolio performance...")
    tos_portfolio = simulate_market(optimal_trades, starting_cash=initial_value,
                                    commission=0.0, impact=0.0)
    benchmark_portfolio = simulate_market(benchmark_trades, starting_cash=initial_value,
                                         commission=0.0, impact=0.0)
    
    tos_normalized = tos_portfolio / tos_portfolio.iloc[0]
    benchmark_normalized = benchmark_portfolio / benchmark_portfolio.iloc[0]
    
    tos_stats = calculate_performance_metrics(tos_portfolio)
    bench_stats = calculate_performance_metrics(benchmark_portfolio)
    
    print("PERFORMANCE COMPARISON")
    print(f"{'Metric':<35} {'Benchmark':<18} {'TOS':<18}")
    print(f"{'Cumulative Return':<35} {bench_stats[0]:<18.6f} {tos_stats[0]:<18.6f}")
    print(f"{'Std Dev Daily Returns':<35} {bench_stats[1]:<18.6f} {tos_stats[1]:<18.6f}")
    print(f"{'Mean Daily Returns':<35} {bench_stats[2]:<18.6f} {tos_stats[2]:<18.6f}")
    
    # Generate comparison chart (FIXED: removed problematic style)
    print("\nCreating comparison chart...")
    
    fig, ax = plt.subplots(figsize=(13, 7.5))
    
    # CRITICAL: Keep required colors (purple for benchmark, red for TOS)
    ax.plot(benchmark_normalized, label='Benchmark Portfolio', 
            color='purple', linewidth=2.5, linestyle='-', alpha=0.9)
    ax.plot(tos_normalized, label='Theoretically Optimal Strategy', 
            color='red', linewidth=2.5, linestyle='-', alpha=0.9)
    
    ax.set_title('Performance Comparison: Theoretically Optimal Strategy vs Benchmark\nJPM Stock (January 2008 - December 2009)',
                fontsize=13, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Portfolio Value', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    
    # Add reference line at 1.0
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('TOS_vs_Benchmark.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: TOS_vs_Benchmark.png")
    
    with open('p6_results.txt', 'w') as output_file:
        output_file.write("PROJECT 6: THEORETICALLY OPTIMAL STRATEGY RESULTS\n")
        output_file.write(f"Symbol: {stock_ticker}\n")
        output_file.write(f"Period: {period_start.date()} to {period_end.date()}\n")
        output_file.write(f"Starting Value: ${initial_value:,.2f}\n\n")
        output_file.write("PERFORMANCE METRICS\n")
        output_file.write(f"{'Metric':<40} {'Benchmark':<15} {'TOS':<15}\n")
        output_file.write(f"{'Cumulative Return':<40} {bench_stats[0]:<15.6f} {tos_stats[0]:<15.6f}\n")
        output_file.write(f"{'Standard Deviation of Daily Returns':<40} {bench_stats[1]:<15.6f} {tos_stats[1]:<15.6f}\n")
        output_file.write(f"{'Mean of Daily Returns':<40} {bench_stats[2]:<15.6f} {tos_stats[2]:<15.6f}\n")
    
    print(" ANALYSIS COMPLETE")


def run_indicator_generation():
    """
    Generate all technical indicator charts
    """
    print("GENERATING TECHNICAL INDICATOR CHARTS")
    
    indicators.create_all_charts(ticker="JPM",
                                 start_date=dt.datetime(2008, 1, 1),
                                 end_date=dt.datetime(2009, 12, 31))
    
    print("INDICATOR GENERATION COMPLETE")


if __name__ == "__main__":
    print("# PROJECT 6: INDICATOR EVALUATION")
    print("# Starting test execution...")
    
    run_tos_analysis()
    run_indicator_generation()
    
    print("# ALL TESTS COMPLETE")
    print("# Generated files:")
    print("#   - TOS_vs_Benchmark.png")
    print("#   - indicator_bbp.png")
    print("#   - indicator_rsi.png")
    print("#   - indicator_macd.png")
    print("#   - indicator_momentum.png")
    print("#   - indicator_stochastic.png")
    print("#   - p6_results.txt")
