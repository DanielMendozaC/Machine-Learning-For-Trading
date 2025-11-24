# experiment1.py

"""Experiment 1: Compare Manual Strategy vs Strategy Learner"""
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from marketsimcode import compute_portvals


def author():
    return 'dcarbono3'


def run_experiment1():
    """Compare Manual Strategy with Strategy Learner"""
    # Trading parameters
    stock_symbol = "JPM"
    initial_capital = 100000
    trading_commission = 9.95
    market_impact = 0.005
    
    # Training period dates
    train_start_date = dt.datetime(2008, 1, 1)
    train_end_date = dt.datetime(2009, 12, 31)
    
    # Testing period dates
    test_start_date = dt.datetime(2010, 1, 1)
    test_end_date = dt.datetime(2011, 12, 31)
    
    print("EXPERIMENT 1: Manual Strategy vs Strategy Learner")
    
    # Initialize strategies
    manual_strat = ManualStrategy(verbose=False, impact=market_impact, commission=trading_commission)
    learning_strat = StrategyLearner(verbose=False, impact=market_impact, commission=trading_commission)
    
    print("\nTraining Strategy Learner...")
    learning_strat.add_evidence(stock_symbol, train_start_date, train_end_date, initial_capital)
    
    # IN-SAMPLE ANALYSIS
    print("IN-SAMPLE: 2008-2009")
    
    manual_trades_in_sample = manual_strat.testPolicy(stock_symbol, train_start_date, train_end_date, initial_capital)
    learner_trades_in_sample = learning_strat.testPolicy(stock_symbol, train_start_date, train_end_date, initial_capital)
    benchmark_trades_in_sample = manual_strat.benchmark(stock_symbol, train_start_date, train_end_date, initial_capital)
    
    portfolio_values_manual_in = compute_portvals(manual_trades_in_sample, start_val=initial_capital,
                                                   commission=trading_commission, impact=market_impact)
    portfolio_values_learner_in = compute_portvals(learner_trades_in_sample, start_val=initial_capital,
                                                    commission=trading_commission, impact=market_impact)
    portfolio_values_benchmark_in = compute_portvals(benchmark_trades_in_sample, start_val=initial_capital,
                                                      commission=trading_commission, impact=market_impact)
    
    normalized_manual_in = portfolio_values_manual_in / portfolio_values_manual_in.iloc[0]
    normalized_learner_in = portfolio_values_learner_in / portfolio_values_learner_in.iloc[0]
    normalized_benchmark_in = portfolio_values_benchmark_in / portfolio_values_benchmark_in.iloc[0]
    
    metrics_manual_in = calculate_portfolio_metrics(portfolio_values_manual_in)
    metrics_learner_in = calculate_portfolio_metrics(portfolio_values_learner_in)
    metrics_benchmark_in = calculate_portfolio_metrics(portfolio_values_benchmark_in)
    
    print("\nBenchmark:")
    display_statistics(metrics_benchmark_in)
    print("\nManual Strategy:")
    display_statistics(metrics_manual_in)
    print("\nStrategy Learner:")
    display_statistics(metrics_learner_in)
    
    generate_comparison_plot(normalized_benchmark_in, normalized_manual_in, normalized_learner_in,
                             "Experiment 1: In-Sample (2008-2009)", "experiment1_in_sample.png")
    print("\nSaved: experiment1_in_sample.png")
    
    # OUT-OF-SAMPLE ANALYSIS
    print("OUT-OF-SAMPLE: 2010-2011")
    
    manual_trades_out_sample = manual_strat.testPolicy(stock_symbol, test_start_date, test_end_date, initial_capital)
    learner_trades_out_sample = learning_strat.testPolicy(stock_symbol, test_start_date, test_end_date, initial_capital)
    benchmark_trades_out_sample = manual_strat.benchmark(stock_symbol, test_start_date, test_end_date, initial_capital)
    
    portfolio_values_manual_out = compute_portvals(manual_trades_out_sample, start_val=initial_capital,
                                                    commission=trading_commission, impact=market_impact)
    portfolio_values_learner_out = compute_portvals(learner_trades_out_sample, start_val=initial_capital,
                                                     commission=trading_commission, impact=market_impact)
    portfolio_values_benchmark_out = compute_portvals(benchmark_trades_out_sample, start_val=initial_capital,
                                                       commission=trading_commission, impact=market_impact)
    
    normalized_manual_out = portfolio_values_manual_out / portfolio_values_manual_out.iloc[0]
    normalized_learner_out = portfolio_values_learner_out / portfolio_values_learner_out.iloc[0]
    normalized_benchmark_out = portfolio_values_benchmark_out / portfolio_values_benchmark_out.iloc[0]
    
    metrics_manual_out = calculate_portfolio_metrics(portfolio_values_manual_out)
    metrics_learner_out = calculate_portfolio_metrics(portfolio_values_learner_out)
    metrics_benchmark_out = calculate_portfolio_metrics(portfolio_values_benchmark_out)
    
    print("\nBenchmark:")
    display_statistics(metrics_benchmark_out)
    print("\nManual Strategy:")
    display_statistics(metrics_manual_out)
    print("\nStrategy Learner:")
    display_statistics(metrics_learner_out)
    
    generate_comparison_plot(normalized_benchmark_out, normalized_manual_out, normalized_learner_out,
                             "Experiment 1: Out-of-Sample (2010-2011)", "experiment1_out_sample.png")
    print("\nSaved: experiment1_out_sample.png")


def calculate_portfolio_metrics(portfolio_values):
    """Calculate portfolio statistics"""
    daily_returns = (portfolio_values / portfolio_values.shift(1)) - 1
    daily_returns = daily_returns[1:]
    cumulative_ret = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std()
    return {'cumulative_return': cumulative_ret, 'mean_daily_ret': avg_daily_ret, 'std_daily_ret': std_daily_ret}


def display_statistics(metrics):
    """Print statistics"""
    print(f"  CR:   {metrics['cumulative_return']:.6f}")
    print(f"  Mean: {metrics['mean_daily_ret']:.6f}")
    print(f"  Std:  {metrics['std_daily_ret']:.6f}")


def generate_comparison_plot(benchmark_data, manual_data, learner_data, chart_title, output_filename):
    """Create comparison plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(benchmark_data.index, benchmark_data.values, color='purple', label='Benchmark', linewidth=2)
    ax.plot(manual_data.index, manual_data.values, color='red', label='Manual Strategy', linewidth=2)
    ax.plot(learner_data.index, learner_data.values, color='green', label='Strategy Learner', linewidth=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Portfolio Value')
    ax.set_title(chart_title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    run_experiment1()