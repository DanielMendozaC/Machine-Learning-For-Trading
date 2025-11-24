# ManualStrategy.py

"""Manual Strategy using BBP, MACD, RSI"""
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import get_data
from marketsimcode import compute_portvals
from indicators import bollinger_band_percent, rsi, macd_histogram


def author():
    return 'dcarbono3'


class ManualStrategy:
    """Manual trading strategy using BBP, MACD, and RSI"""
    
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        # Position limits
        self.MAX_POSITION = 1000
        self.MIN_POSITION = -1000
        self.NEUTRAL_POSITION = 0
        
    def _load_price_data(self, ticker, start_date, end_date):
        """Load and clean price data for given symbol"""
        date_range = pd.date_range(start_date, end_date)
        price_data = get_data([ticker], date_range)
        price_data = price_data[[ticker]]
        price_data.ffill(inplace=True)
        price_data.bfill(inplace=True)
        return price_data
    
    def _calculate_technical_indicators(self, price_series):
        """Calculate all technical indicators"""
        bbp_values = bollinger_band_percent(price_series, lookback=20)
        rsi_values = rsi(price_series, period=14)
        macd_values = macd_histogram(price_series)
        return bbp_values, rsi_values, macd_values
    
    def _evaluate_signals(self, bbp_value, rsi_value, macd_value):
        """Evaluate trading signals using voting system"""
        # Count bullish indicators
        bullish_count = 0
        if bbp_value < 0.3:  # Oversold (loosened from 0.2)
            bullish_count += 1
        if rsi_value < 40:  # Oversold (loosened from 35)
            bullish_count += 1
        if macd_value > 0:  # Positive momentum
            bullish_count += 1
        
        # Count bearish indicators
        bearish_count = 0
        if bbp_value > 0.7:  # Overbought (loosened from 0.8)
            bearish_count += 1
        if rsi_value > 60:  # Overbought (loosened from 65)
            bearish_count += 1
        if macd_value < 0:  # Negative momentum
            bearish_count += 1
        
        # Determine signal based on votes (need 2+ votes)
        if bullish_count >= 2:
            return 1  # BUY signal
        elif bearish_count >= 2:
            return -1  # SELL signal
        else:
            return 0  # HOLD signal
    
    def _calculate_order_size(self, trade_signal, current_position):
        """Calculate order size based on signal and current position"""
        if trade_signal == 1 and current_position <= 0:
            if current_position == 0:
                return 1000
            elif current_position == -1000:
                return 2000
        elif trade_signal == -1 and current_position >= 0:
            if current_position == 0:
                return -1000
            elif current_position == 1000:
                return -2000
        return 0
    
    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), 
                   ed=dt.datetime(2009, 12, 31), sv=100000):
        """Generate trades based on manual rules"""
        # Load price data
        price_df = self._load_price_data(symbol, sd, ed)
        
        # Calculate technical indicators
        price_time_series = price_df[symbol]
        bbp_indicator, rsi_indicator, macd_indicator = self._calculate_technical_indicators(price_time_series)
        
        # Initialize trade dataframe
        trades_df = pd.DataFrame(0, index=price_df.index, columns=[symbol])
        current_holdings = 0
        
        # Generate trading signals
        for idx in range(len(price_df)):
            trading_date = price_df.index[idx]
            
            # Skip initial period for MACD calculation
            if idx < 26:
                continue
            
            # Get indicator values
            bbp_val = bbp_indicator.loc[trading_date]
            rsi_val = rsi_indicator.loc[trading_date]
            macd_val = macd_indicator.loc[trading_date]
            
            # Skip if any indicator is NaN
            if pd.isna(bbp_val) or pd.isna(rsi_val) or pd.isna(macd_val):
                continue
            
            # Evaluate trading signal
            decision = self._evaluate_signals(bbp_val, rsi_val, macd_val)
            
            # Calculate order size
            order_shares = self._calculate_order_size(decision, current_holdings)
            
            # Execute trade if order size is non-zero
            if order_shares != 0:
                trades_df.loc[trading_date, symbol] = order_shares
                current_holdings += order_shares
        
        if self.verbose:
            num_trades = (trades_df != 0).sum()[0]
            print(f"Trades: {num_trades}")
        
        return trades_df
    
    def benchmark(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), 
                  ed=dt.datetime(2009, 12, 31), sv=100000):
        """Buy and hold benchmark"""
        date_range = pd.date_range(sd, ed)
        price_data = get_data([symbol], date_range)
        price_data = price_data[[symbol]]
        
        trades_df = pd.DataFrame(0, index=price_data.index, columns=[symbol])
        trades_df.iloc[0] = 1000
        return trades_df
    
    def _compute_normalized_portfolio(self, trades, initial_capital):
        """Compute and normalize portfolio values"""
        portfolio_values = compute_portvals(trades, start_val=initial_capital,
                                           commission=self.commission, impact=self.impact)
        normalized_values = portfolio_values / portfolio_values.iloc[0]
        return normalized_values
    
    def plot_strategy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1),
                     ed=dt.datetime(2009, 12, 31), sv=100000, 
                     title="Manual Strategy vs Benchmark", filename=None):
        """Plot strategy performance"""
        # Generate trades
        manual_trades = self.testPolicy(symbol, sd, ed, sv)
        benchmark_trades = self.benchmark(symbol, sd, ed, sv)
        
        # Calculate normalized portfolio values
        manual_portfolio = self._compute_normalized_portfolio(manual_trades, sv)
        benchmark_portfolio = self._compute_normalized_portfolio(benchmark_trades, sv)
        
        # Create plot
        figure, axes = plt.subplots(figsize=(12, 6))
        axes.plot(benchmark_portfolio.index, benchmark_portfolio.values, 
                 color='purple', label='Benchmark', linewidth=2)
        axes.plot(manual_portfolio.index, manual_portfolio.values, 
                 color='red', label='Manual Strategy', linewidth=2)
        
        # Mark long entries
        long_trade_dates = manual_trades[manual_trades[symbol] > 0].index
        for trade_date in long_trade_dates:
            axes.axvline(x=trade_date, color='blue', alpha=0.5, linewidth=1)
        
        # Mark short entries
        short_trade_dates = manual_trades[manual_trades[symbol] < 0].index
        for trade_date in short_trade_dates:
            axes.axvline(x=trade_date, color='black', alpha=0.5, linewidth=1)
        
        axes.set_xlabel('Date')
        axes.set_ylabel('Normalized Portfolio Value')
        axes.set_title(title)
        axes.legend()
        axes.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _compute_performance_metrics(self, portfolio_values):
        """Calculate performance statistics for portfolio"""
        daily_returns = (portfolio_values / portfolio_values.shift(1)) - 1
        daily_returns = daily_returns[1:]
        cumulative_ret = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        avg_daily_ret = daily_returns.mean()
        std_daily_ret = daily_returns.std()
        return {
            'cumulative_return': cumulative_ret, 
            'mean_daily_ret': avg_daily_ret, 
            'std_daily_ret': std_daily_ret
        }
    
    def get_statistics(self, symbol="JPM", sd=dt.datetime(2008, 1, 1),
                      ed=dt.datetime(2009, 12, 31), sv=100000):
        """Calculate performance statistics"""
        # Generate trades
        manual_trades = self.testPolicy(symbol, sd, ed, sv)
        benchmark_trades = self.benchmark(symbol, sd, ed, sv)
        
        # Calculate portfolio values
        manual_portfolio = compute_portvals(manual_trades, start_val=sv,
                                           commission=self.commission, impact=self.impact)
        benchmark_portfolio = compute_portvals(benchmark_trades, start_val=sv,
                                              commission=self.commission, impact=self.impact)
        
        # Calculate statistics
        manual_stats = self._compute_performance_metrics(manual_portfolio)
        benchmark_stats = self._compute_performance_metrics(benchmark_portfolio)
        
        return {'manual': manual_stats, 'benchmark': benchmark_stats}