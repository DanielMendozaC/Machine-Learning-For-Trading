"""
testproject.py
Test Project - Entry point for Project 6
Runs TOS and generates all charts and statistics
"""

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import TheoreticallyOptimalStrategy as tos
import indicators
import marketsimcode as msc

def author():
    return 'dcarbono3'


def test_theoretically_optimal_strategy():
    """
    Test the Theoretically Optimal Strategy and generate charts/stats
    """
    print("\n" + "="*60)
    print("TESTING THEORETICALLY OPTIMAL STRATEGY")
    print("="*60)
    
    # Parameters
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    
    # Get TOS trades
    print(f"\nGenerating TOS trades for {symbol}...")
    tos_trades = tos.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    print(f"Trades generated: {(tos_trades[symbol] != 0).sum()} trading days")
    
    # Create benchmark trades (buy 1000 shares on day 1, hold)
    benchmark_trades = pd.DataFrame(0, index=tos_trades.index, columns=[symbol])
    benchmark_trades.iloc[0][symbol] = 1000
    
    # Compute portfolio values (using zero commission and impact for TOS)
    print("\nComputing portfolio values...")
    tos_portvals = msc.compute_portvals_simple(tos_trades, start_val=sv, 
                                               commission=0.0, impact=0.0)
    benchmark_portvals = msc.compute_portvals_simple(benchmark_trades, start_val=sv, 
                                                     commission=0.0, impact=0.0)
    
    # Normalize to 1.0 at start
    tos_portvals_norm = tos_portvals / tos_portvals.iloc[0]
    benchmark_portvals_norm = benchmark_portvals / benchmark_portvals.iloc[0]
    
    # Calculate statistics
    def calculate_stats(portvals):
        daily_returns = (portvals / portvals.shift(1)) - 1
        daily_returns = daily_returns[1:]  # Remove first NaN
        
        cum_return = (portvals.iloc[-1] / portvals.iloc[0]) - 1
        avg_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        
        return cum_return, std_daily_return, avg_daily_return
    
    tos_cum_ret, tos_std, tos_mean = calculate_stats(tos_portvals)
    bench_cum_ret, bench_std, bench_mean = calculate_stats(benchmark_portvals)
    
    # Print statistics
    print("\n" + "-"*60)
    print("PERFORMANCE STATISTICS")
    print("-"*60)
    print(f"{'Metric':<30} {'Benchmark':<20} {'TOS':<20}")
    print("-"*60)
    print(f"{'Cumulative Return':<30} {bench_cum_ret:<20.6f} {tos_cum_ret:<20.6f}")
    print(f"{'Std Dev of Daily Returns':<30} {bench_std:<20.6f} {tos_std:<20.6f}")
    print(f"{'Mean of Daily Returns':<30} {bench_mean:<20.6f} {tos_mean:<20.6f}")
    print("-"*60)
    
    # Generate chart
    print("\nGenerating TOS comparison chart...")
    plt.figure(figsize=(12, 7))
    plt.plot(benchmark_portvals_norm.index, benchmark_portvals_norm, 
             label='Benchmark', color='purple', linewidth=2)
    plt.plot(tos_portvals_norm.index, tos_portvals_norm, 
             label='Theoretically Optimal Strategy', color='red', linewidth=2)
    plt.title('Theoretically Optimal Strategy vs Benchmark (JPM 2008-2009)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Normalized Portfolio Value', fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('TOS_vs_Benchmark.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: TOS_vs_Benchmark.png")
    
    # Save statistics to file
    with open('p6_results.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("PROJECT 6: THEORETICALLY OPTIMAL STRATEGY RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Date Range: {sd.date()} to {ed.date()}\n")
        f.write(f"Starting Value: ${sv:,.2f}\n\n")
        f.write("-"*70 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Metric':<40} {'Benchmark':<15} {'TOS':<15}\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Cumulative Return':<40} {bench_cum_ret:<15.6f} {tos_cum_ret:<15.6f}\n")
        f.write(f"{'Standard Deviation of Daily Returns':<40} {bench_std:<15.6f} {tos_std:<15.6f}\n")
        f.write(f"{'Mean of Daily Returns':<40} {bench_mean:<15.6f} {tos_mean:<15.6f}\n")
        f.write("-"*70 + "\n")
    
    print("\nStatistics saved to: p6_results.txt")
    print("\n" + "="*60)
    print("TOS TESTING COMPLETE")
    print("="*60)


def test_indicators():
    """
    Generate all indicator charts
    """
    print("\n" + "="*60)
    print("GENERATING INDICATOR CHARTS")
    print("="*60 + "\n")
    
    indicators.generate_indicator_charts(symbol="JPM", 
                                        sd=dt.datetime(2008, 1, 1), 
                                        ed=dt.datetime(2009, 12, 31))
    
    print("\n" + "="*60)
    print("INDICATOR CHART GENERATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# PROJECT 6: INDICATOR EVALUATION")
    print("# Starting test execution...")
    print("#"*60)
    
    # Test TOS
    test_theoretically_optimal_strategy()
    
    # Generate indicator charts
    test_indicators()
    
    print("\n" + "#"*60)
    print("# ALL TESTS COMPLETE")
    print("# Check generated PNG files and p6_results.txt")
    print("#"*60 + "\n")