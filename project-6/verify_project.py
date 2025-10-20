# Create a file called verify_project.py
"""
Verification script for Project 6
Run this BEFORE submitting to check everything
"""

import os
import datetime as dt
import pandas as pd
import numpy as np
from util import get_data
import TheoreticallyOptimalStrategy as tos
import indicators
import marketsimcode as msc

def verify_files_exist():
    """Check all required files exist"""
    print("\n" + "="*70)
    print("1. CHECKING FILE EXISTENCE")
    print("="*70)
    
    required_files = [
        'TheoreticallyOptimalStrategy.py',
        'indicators.py',
        'testproject.py',
        'marketsimcode.py'  # Optional but recommended
    ]
    
    all_exist = True
    for file in required_files:
        exists = os.path.exists(file)
        status = "✓ EXISTS" if exists else "✗ MISSING"
        print(f"{status}: {file}")
        if not exists:
            all_exist = False
    
    return all_exist


def verify_author_functions():
    """Check all files have author() function"""
    print("\n" + "="*70)
    print("2. CHECKING AUTHOR() FUNCTIONS")
    print("="*70)
    
    modules = [tos, indicators, msc]
    module_names = ['TheoreticallyOptimalStrategy', 'indicators', 'marketsimcode']
    
    all_have_author = True
    for mod, name in zip(modules, module_names):
        if hasattr(mod, 'author'):
            author_id = mod.author()
            print(f"✓ {name}.author() returns: '{author_id}'")
            if len(author_id) < 2 or len(author_id) > 10:
                print(f"  ⚠ Warning: Author ID seems unusual (length: {len(author_id)})")
        else:
            print(f"✗ {name}.author() NOT FOUND")
            all_have_author = False
    
    # Check study_group in TOS
    if hasattr(tos, 'study_group'):
        print(f"✓ TheoreticallyOptimalStrategy.study_group() exists")
    else:
        print(f"✗ TheoreticallyOptimalStrategy.study_group() NOT FOUND")
        all_have_author = False
    
    return all_have_author


def verify_tos_function():
    """Check TOS testPolicy function works correctly"""
    print("\n" + "="*70)
    print("3. CHECKING TOS testPolicy() FUNCTION")
    print("="*70)
    
    try:
        # Test with default parameters
        trades = tos.testPolicy(symbol="JPM", 
                               sd=dt.datetime(2008, 1, 1), 
                               ed=dt.datetime(2009, 12, 31), 
                               sv=100000)
        
        print(f"✓ testPolicy() executed successfully")
        print(f"  - DataFrame shape: {trades.shape}")
        print(f"  - Columns: {list(trades.columns)}")
        print(f"  - Date range: {trades.index[0]} to {trades.index[-1]}")
        print(f"  - Total days: {len(trades)}")
        print(f"  - Days with trades: {(trades != 0).sum().sum()}")
        
        # Check trade values
        unique_trades = trades['JPM'].unique()
        print(f"  - Unique trade values: {sorted(unique_trades)}")
        
        # Verify only legal values
        legal_values = {-2000, -1000, 0, 1000, 2000}
        illegal_values = set(unique_trades) - legal_values
        
        if illegal_values:
            print(f"✗ ILLEGAL trade values found: {illegal_values}")
            return False
        else:
            print(f"✓ All trade values are legal")
        
        # Check holdings never exceed limits
        holdings = trades['JPM'].cumsum()
        max_holding = holdings.max()
        min_holding = holdings.min()
        print(f"  - Max holding: {max_holding}")
        print(f"  - Min holding: {min_holding}")
        
        if max_holding > 1000 or min_holding < -1000:
            print(f"✗ Holdings exceed limits! Max=1000, Min=-1000")
            return False
        else:
            print(f"✓ Holdings within limits")
        
        return True
        
    except Exception as e:
        print(f"✗ ERROR in testPolicy(): {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_indicators():
    """Check all 5 indicators work correctly"""
    print("\n" + "="*70)
    print("4. CHECKING INDICATOR FUNCTIONS")
    print("="*70)
    
    # Get price data
    dates = pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))
    prices_all = get_data(['JPM'], dates, addSPY=True, colname='Adj Close')
    prices = prices_all['JPM'].fillna(method='ffill').fillna(method='bfill')
    
    indicator_functions = [
        ('bollinger_band_percent', indicators.bollinger_band_percent),
        ('rsi', indicators.rsi),
        ('macd_histogram', indicators.macd_histogram),
        ('momentum', indicators.momentum),
        ('stochastic_oscillator', indicators.stochastic_oscillator)
    ]
    
    all_work = True
    for name, func in indicator_functions:
        try:
            result = func(prices)
            
            # Check it returns a Series or array
            if not isinstance(result, (pd.Series, np.ndarray)):
                print(f"✗ {name}: Returns {type(result)}, should return Series/array")
                all_work = False
                continue
            
            # Check it's a 1D vector
            if isinstance(result, pd.Series):
                is_1d = True
            else:
                is_1d = result.ndim == 1
            
            if not is_1d:
                print(f"✗ {name}: Returns multi-dimensional data, should be 1D")
                all_work = False
                continue
            
            # Check length
            print(f"✓ {name}:")
            print(f"    Type: {type(result).__name__}")
            print(f"    Length: {len(result)}")
            print(f"    Non-NaN values: {pd.Series(result).notna().sum()}")
            print(f"    Sample values: {pd.Series(result).dropna().head(3).values}")
            
        except Exception as e:
            print(f"✗ {name}: ERROR - {e}")
            all_work = False
    
    return all_work


def verify_charts_generated():
    """Check all required charts are generated"""
    print("\n" + "="*70)
    print("5. CHECKING CHART GENERATION")
    print("="*70)
    
    required_charts = [
        'indicator_bbp.png',
        'indicator_rsi.png',
        'indicator_macd.png',
        'indicator_momentum.png',
        'indicator_stochastic.png',
        'TOS_vs_Benchmark.png'
    ]
    
    print("Running testproject.py to generate charts...")
    print("(This may take a moment...)\n")
    
    # Run testproject
    try:
        import testproject
        # This should generate all charts
    except Exception as e:
        print(f"✗ Error running testproject: {e}")
        return False
    
    all_exist = True
    for chart in required_charts:
        exists = os.path.exists(chart)
        if exists:
            size = os.path.getsize(chart)
            print(f"✓ {chart} ({size:,} bytes)")
        else:
            print(f"✗ {chart} NOT FOUND")
            all_exist = False
    
    return all_exist


def verify_tos_chart_requirements():
    """Check TOS chart meets specific requirements"""
    print("\n" + "="*70)
    print("6. VERIFYING TOS CHART REQUIREMENTS")
    print("="*70)
    
    # Get the data used in TOS
    dates = pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))
    
    # Get TOS trades and compute values
    trades = tos.testPolicy(symbol="JPM", 
                           sd=dt.datetime(2008, 1, 1), 
                           ed=dt.datetime(2009, 12, 31), 
                           sv=100000)
    
    # Create benchmark
    benchmark_trades = pd.DataFrame(0, index=trades.index, columns=['JPM'])
    benchmark_trades.iloc[0]['JPM'] = 1000
    
    # Compute portfolio values
    tos_portvals = msc.compute_portvals_simple(trades, start_val=100000, 
                                               commission=0.0, impact=0.0)
    benchmark_portvals = msc.compute_portvals_simple(benchmark_trades, start_val=100000,
                                                     commission=0.0, impact=0.0)
    
    # Check normalization
    tos_norm_start = tos_portvals.iloc[0]
    bench_norm_start = benchmark_portvals.iloc[0]
    
    print(f"TOS first value: {tos_portvals.iloc[0]:.2f}")
    print(f"Benchmark first value: {benchmark_portvals.iloc[0]:.2f}")
    
    # They should both start at 100000
    if abs(tos_norm_start - 100000) > 0.01:
        print(f"⚠ TOS doesn't start at 100000")
    else:
        print(f"✓ TOS starts at correct value")
    
    if abs(bench_norm_start - 100000) > 0.01:
        print(f"⚠ Benchmark doesn't start at 100000")
    else:
        print(f"✓ Benchmark starts at correct value")
    
    # Calculate statistics
    def calc_stats(portvals):
        daily_rets = (portvals / portvals.shift(1)) - 1
        daily_rets = daily_rets[1:]
        return {
            'cum_ret': (portvals.iloc[-1] / portvals.iloc[0]) - 1,
            'std': daily_rets.std(),
            'mean': daily_rets.mean()
        }
    
    tos_stats = calc_stats(tos_portvals)
    bench_stats = calc_stats(benchmark_portvals)
    
    print(f"\nBenchmark Cumulative Return: {bench_stats['cum_ret']:.6f}")
    print(f"TOS Cumulative Return: {tos_stats['cum_ret']:.6f}")
    
    if tos_stats['cum_ret'] <= bench_stats['cum_ret']:
        print(f"⚠ WARNING: TOS should outperform benchmark significantly!")
        print(f"  (TOS has perfect knowledge, so it should have much higher returns)")
    else:
        print(f"✓ TOS outperforms benchmark (as expected)")
    
    return True


def verify_no_display():
    """Check that charts aren't displayed (only saved)"""
    print("\n" + "="*70)
    print("7. CHECKING NO CHART DISPLAY")
    print("="*70)
    
    print("⚠ MANUAL CHECK REQUIRED:")
    print("  When you run testproject.py, do any chart windows pop up?")
    print("  - If YES: You're using plt.show() - REMOVE IT")
    print("  - If NO: Good! Charts are only saved")
    print("\n  Make sure you only use plt.savefig() and plt.close()")
    
    return True


def main():
    """Run all verification checks"""
    print("\n" + "#"*70)
    print("# PROJECT 6 VERIFICATION SCRIPT")
    print("#"*70)
    
    results = {
        'Files Exist': verify_files_exist(),
        'Author Functions': verify_author_functions(),
        'TOS Function': verify_tos_function(),
        'Indicators': verify_indicators(),
        'Charts Generated': verify_charts_generated(),
        'TOS Requirements': verify_tos_chart_requirements(),
        'No Display': verify_no_display()
    }
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for check, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {check}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED! Your project looks good!")
        print("Next steps:")
        print("  1. Create your report (see report guide below)")
        print("  2. Test in Gradescope TESTING")
        print("  3. Submit to Gradescope SUBMISSION")
        print("  4. Submit report to Canvas")
    else:
        print("\n⚠ SOME CHECKS FAILED - Fix issues before submitting")
    
    return all_passed


if __name__ == "__main__":
    main()
