# experiment2.py

"""Experiment 2: Test impact sensitivity of Strategy Learner"""
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from StrategyLearner import StrategyLearner
from marketsimcode import compute_portvals


def author():
    return 'dcarbono3'


def run_experiment2():
    """Test how impact values affect Strategy Learner"""
    ticker = "JPM"
    initial_cash = 100000
    trading_commission = 0.0  # As specified
    
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    
    test_impacts = [0.000, 0.005, 0.010, 0.025, 0.050]
    
    print("EXPERIMENT 2: Impact Sensitivity")
    print(f"Symbol: {ticker}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Impact values: {test_impacts}")
    
    summary_data = {
        'impact': [],
        'cumulative_return': [],
        'num_trades': [],
        'mean_daily_ret': [],
        'std_daily_ret': []
    }
    
    portfolio_history = {}
    
    for market_impact in test_impacts:
        print(f"\n\n Testing Impact = {market_impact}")
        
        strategy_learner = StrategyLearner(
            verbose=False, 
            impact=market_impact, 
            commission=trading_commission
        )
        print("Training...")
        strategy_learner.add_evidence(ticker, start_date, end_date, initial_cash)
        
        print("Testing...")
        trade_signals = strategy_learner.testPolicy(ticker, start_date, end_date, initial_cash)
        portfolio_values = compute_portvals(
            trade_signals, 
            start_val=initial_cash, 
            commission=trading_commission, 
            impact=market_impact
        )
        
        performance_stats = compute_statistics(portfolio_values)
        total_trades = (trade_signals != 0).sum().values[0]
        
        summary_data['impact'].append(market_impact)
        summary_data['cumulative_return'].append(performance_stats['cumulative_return'])
        summary_data['num_trades'].append(total_trades)
        summary_data['mean_daily_ret'].append(performance_stats['mean_daily_ret'])
        summary_data['std_daily_ret'].append(performance_stats['std_daily_ret'])
        
        portfolio_history[market_impact] = portfolio_values
        
        print(f"Trades: {total_trades}")
        print(f"CR: {performance_stats['cumulative_return']:.6f}")
        print(f"Mean: {performance_stats['mean_daily_ret']:.6f}")
        print(f"Std: {performance_stats['std_daily_ret']:.6f}")
    
    summary_df = pd.DataFrame(summary_data)
    
    print("SUMMARY")
    print(summary_df.to_string(index=False))
    
    generate_visualizations(summary_df, portfolio_history, test_impacts)
    
    print("\nExperiment 2 complete!")


def compute_statistics(portfolio_values):
    """Calculate statistics"""
    daily_returns = (portfolio_values / portfolio_values.shift(1)) - 1
    daily_returns = daily_returns[1:]
    cumulative_ret = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    avg_daily_ret = daily_returns.mean()
    std_dev_daily_ret = daily_returns.std()
    
    return {
        'cumulative_return': cumulative_ret,
        'mean_daily_ret': avg_daily_ret,
        'std_daily_ret': std_dev_daily_ret
    }


def generate_visualizations(summary_df, portfolio_history, test_impacts):
    """Create visualization charts"""
    # Chart 1: Impact vs CR and Trades
    figure, (axis1, axis2) = plt.subplots(1, 2, figsize=(14, 5))
    
    axis1.plot(
        summary_df['impact'], 
        summary_df['cumulative_return'],
        marker='o', 
        linewidth=2, 
        markersize=8, 
        color='blue'
    )
    axis1.set_xlabel('Market Impact')
    axis1.set_ylabel('Cumulative Return')
    axis1.set_title('Impact vs Cumulative Return')
    axis1.grid(True, alpha=0.3)
    
    axis2.plot(
        summary_df['impact'], 
        summary_df['num_trades'],
        marker='s', 
        linewidth=2, 
        markersize=8, 
        color='red'
    )
    axis2.set_xlabel('Market Impact')
    axis2.set_ylabel('Number of Trades')
    axis2.set_title('Impact vs Number of Trades')
    axis2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment2_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: experiment2_metrics.png")
    
    # Chart 2: Portfolio values
    figure, axis = plt.subplots(figsize=(12, 6))
    color_palette = ['blue', 'green', 'orange', 'red', 'purple']
    
    for idx, market_impact in enumerate(test_impacts):
        normalized_portfolio = portfolio_history[market_impact] / portfolio_history[market_impact].iloc[0]
        axis.plot(
            normalized_portfolio.index,
            normalized_portfolio.values,
            label=f'Impact = {market_impact}',
            linewidth=2,
            color=color_palette[idx]
        )
    
    axis.set_xlabel('Date')
    axis.set_ylabel('Normalized Portfolio Value')
    axis.set_title('Portfolio Performance with Different Impact Values')
    axis.legend()
    axis.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiment2_portfolios.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: experiment2_portfolios.png")


if __name__ == "__main__":
    run_experiment2()