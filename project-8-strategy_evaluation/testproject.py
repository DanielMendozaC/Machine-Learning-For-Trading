# testproject.py

"""
Test Project: Strategy Evaluation Report - Execute All Components
"""
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
from ManualStrategy import ManualStrategy
import experiment1
import experiment2


def author():
    return 'dcarbono3'


def execute_manual_analysis():
    """Execute the Manual Strategy performance analysis"""
    print("RUNNING MANUAL STRATEGY")
    
    ticker = "JPM"
    initial_capital = 100000
    trade_fee = 9.95
    market_impact = 0.005
    
    train_start = dt.datetime(2008, 1, 1)
    train_end = dt.datetime(2009, 12, 31)
    test_start = dt.datetime(2010, 1, 1)
    test_end = dt.datetime(2011, 12, 31)
    
    strategy_obj = ManualStrategy(verbose=True, impact=market_impact, commission=trade_fee)
    
    print("\nIN-SAMPLE (2008-2009) ")
    training_metrics = strategy_obj.get_statistics(ticker, train_start, train_end, initial_capital)
    print("\nBenchmark:")
    print("  CR:   {:.6f}".format(training_metrics['benchmark']['cumulative_return']))
    print("  Mean: {:.6f}".format(training_metrics['benchmark']['mean_daily_ret']))
    print("  Std:  {:.6f}".format(training_metrics['benchmark']['std_daily_ret']))
    print("\nManual Strategy:")
    print("  CR:   {:.6f}".format(training_metrics['manual']['cumulative_return']))
    print("  Mean: {:.6f}".format(training_metrics['manual']['mean_daily_ret']))
    print("  Std:  {:.6f}".format(training_metrics['manual']['std_daily_ret']))
    
    strategy_obj.plot_strategy(ticker, train_start, train_end, initial_capital,
                    title="Manual Strategy vs Benchmark (In-Sample: 2008-2009)",
                    filename="manual_in_sample.png")
    print("\n Generated: manual_in_sample.png")
    
    print("\n OUT-OF-SAMPLE (2010-2011) ")
    testing_metrics = strategy_obj.get_statistics(ticker, test_start, test_end, initial_capital)
    print("\nBenchmark:")
    print("  CR:   {:.6f}".format(testing_metrics['benchmark']['cumulative_return']))
    print("  Mean: {:.6f}".format(testing_metrics['benchmark']['mean_daily_ret']))
    print("  Std:  {:.6f}".format(testing_metrics['benchmark']['std_daily_ret']))
    print("\nManual Strategy:")
    print("  CR:   {:.6f}".format(testing_metrics['manual']['cumulative_return']))
    print("  Mean: {:.6f}".format(testing_metrics['manual']['mean_daily_ret']))
    print("  Std:  {:.6f}".format(testing_metrics['manual']['std_daily_ret']))
    
    strategy_obj.plot_strategy(ticker, test_start, test_end, initial_capital,
                    title="Manual Strategy vs Benchmark (Out-of-Sample: 2010-2011)",
                    filename="manual_out_sample.png")
    print("\n Generated: manual_out_sample.png")


def main():
    """Primary execution entry point"""

    import numpy as np
    import random
    
    np.random.seed(904060775)  
    random.seed(904060775)

    print("PROJECT 8: STRATEGY EVALUATION")
    print("Author: {}".format(author()))
    
    try:
        # Execute Part 1: Manual Strategy
        execute_manual_analysis()
        
        # Execute Part 2: Experiment 1
        print("\n Starting Experiment 1...")
        experiment1.run_experiment1()
        print("\n Experiment 1 complete")

        
        
        # Execute Part 3: Experiment 2
        print("\n Starting Experiment 2...")
        experiment2.run_experiment2()
        print("\n Experiment 2 complete")
        
        print("ALL TASKS COMPLETED")
        print("\nGenerated files:")
        print("  1. manual_in_sample.png")
        print("  2. manual_out_sample.png")
        print("  3. experiment1_in_sample.png")
        print("  4. experiment1_out_sample.png")
        print("  5. experiment2_metrics.png")
        print("  6. experiment2_portfolios.png")
        
    except Exception as e:
        print("\nERROR: {}".format(str(e)))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()