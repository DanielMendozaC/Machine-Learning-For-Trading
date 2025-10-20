import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import get_data
import datetime as dt


def author():
    return 'dcarbono3'


def bollinger_band_percent(price_series, lookback=20):
    """Calculate Bollinger Band Percentage"""
    rolling_mean = price_series.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = price_series.rolling(window=lookback, min_periods=lookback).std()
    
    top_band = rolling_mean + (2.0 * rolling_std)
    bottom_band = rolling_mean - (2.0 * rolling_std)
    
    band_width = top_band - bottom_band
    price_position = price_series - bottom_band
    bbp_indicator = price_position / band_width
    
    return bbp_indicator


def rsi(price_series, period=14):
    """Relative Strength Index calculation"""
    price_deltas = price_series.diff()
    
    gains = price_deltas.copy()
    losses = price_deltas.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = losses.abs()
    
    avg_gains = gains.rolling(window=period, min_periods=period).mean()
    avg_losses = losses.rolling(window=period, min_periods=period).mean()
    
    relative_strength = avg_gains / avg_losses
    rsi_values = 100.0 - (100.0 / (1.0 + relative_strength))
    
    return rsi_values


def macd_histogram(price_series, fast_period=12, slow_period=26, signal_period=9):
    """MACD Histogram calculation"""
    fast_ema = price_series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = price_series.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram_values = macd_line - signal_line
    
    return histogram_values


def momentum(price_series, lookback_days=10):
    """Price momentum calculation"""
    past_prices = price_series.shift(lookback_days)
    momentum_values = (price_series / past_prices) - 1.0
    
    return momentum_values


def stochastic_oscillator(price_series, window_size=14):
    """Stochastic Oscillator calculation"""
    rolling_high = price_series.rolling(window=window_size, min_periods=window_size).max()
    rolling_low = price_series.rolling(window=window_size, min_periods=window_size).min()
    
    price_range = rolling_high - rolling_low
    price_above_low = price_series - rolling_low
    stoch_values = (price_above_low / price_range) * 100.0
    
    return stoch_values


def create_all_charts(ticker="JPM", start_date=dt.datetime(2008, 1, 1),
                     end_date=dt.datetime(2009, 12, 31)):
    """Generate all five indicator charts"""
    dates = pd.date_range(start_date, end_date)
    all_prices = get_data([ticker], dates, addSPY=True, colname='Adj Close')
    stock_prices = all_prices[[ticker]].copy()  
    stock_prices.ffill(inplace=True)
    stock_prices.bfill(inplace=True)
    price_data = stock_prices[ticker]
    
    # Normalize prices
    normalized_prices = price_data / price_data.iloc[0]
    
    bbp_values = bollinger_band_percent(price_data, lookback=20)
    rsi_values = rsi(price_data, period=14)
    macd_values = macd_histogram(price_data, fast_period=12, slow_period=26, signal_period=9)
    mom_values = momentum(price_data, lookback_days=10)
    stoch_values = stochastic_oscillator(price_data, window_size=14)
    
    window = 20
    rolling_avg = price_data.rolling(window=window).mean()
    rolling_dev = price_data.rolling(window=window).std()
    upper = rolling_avg + (2.0 * rolling_dev)
    lower = rolling_avg - (2.0 * rolling_dev)
    norm_avg = rolling_avg / price_data.iloc[0]
    norm_upper = upper / price_data.iloc[0]
    norm_lower = lower / price_data.iloc[0]
    
    #  CHART 1: BBP 
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    
    axes[0].plot(normalized_prices, label='JPM Stock Price', color='#2C3E50', linewidth=2)
    axes[0].plot(norm_avg, label='SMA-20', color='#3498DB', linewidth=1.5, linestyle=':')
    axes[0].plot(norm_upper, label='Upper Band', color='#E74C3C', linestyle='-.', linewidth=1.5)
    axes[0].plot(norm_lower, label='Lower Band', color='#27AE60', linestyle='-.', linewidth=1.5)
    axes[0].set_ylabel('Price (Normalized)', fontsize=12, fontweight='bold')
    axes[0].set_title('JPM with Bollinger Bands - 2008 to 2009', fontsize=13, fontweight='bold', pad=15)
    axes[0].legend(loc='upper left', fontsize=10, framealpha=0.9)
    axes[0].grid(True, alpha=0.25, linestyle='--')
    
    axes[1].plot(bbp_values, label='BBP Indicator', color='#8E44AD', linewidth=2)
    axes[1].axhline(y=0, color='#27AE60', linestyle='-', linewidth=1.5, alpha=0.8, label='Lower Threshold')
    axes[1].axhline(y=1, color='#E74C3C', linestyle='-', linewidth=1.5, alpha=0.8, label='Upper Threshold')
    axes[1].fill_between(bbp_values.index, 0, bbp_values, where=(bbp_values <= 0), 
                         alpha=0.25, color='#27AE60', label='Oversold Region')
    axes[1].fill_between(bbp_values.index, 1, bbp_values, where=(bbp_values >= 1), 
                         alpha=0.25, color='#E74C3C', label='Overbought Region')
    axes[1].set_xlabel('Trading Date', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('BBP Value', fontsize=12, fontweight='bold')
    axes[1].set_title('Bollinger Band Percentage Indicator', fontsize=13, fontweight='bold', pad=15)
    axes[1].legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)
    axes[1].grid(True, alpha=0.25, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('indicator_bbp.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: indicator_bbp.png")
    
    #  CHART 2: RSI 
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    
    axes[0].plot(normalized_prices, label='JPM Price', color='#34495E', linewidth=2)
    axes[0].set_ylabel('Price (Normalized)', fontsize=12, fontweight='bold')
    axes[0].set_title('JPM Stock Price with RSI Analysis - 2008 to 2009', fontsize=13, fontweight='bold', pad=15)
    axes[0].legend(loc='upper left', fontsize=10, framealpha=0.9)
    axes[0].grid(True, alpha=0.25, linestyle='--')
    
    axes[1].plot(rsi_values, label='RSI (14-day)', color='#16A085', linewidth=2)
    axes[1].axhline(y=70, color='#C0392B', linestyle='--', linewidth=1.8, alpha=0.8, label='Overbought (70)')
    axes[1].axhline(y=30, color='#229954', linestyle='--', linewidth=1.8, alpha=0.8, label='Oversold (30)')
    axes[1].axhline(y=50, color='#7F8C8D', linestyle=':', linewidth=1, alpha=0.5)
    axes[1].fill_between(rsi_values.index, 70, 100, alpha=0.15, color='#C0392B')
    axes[1].fill_between(rsi_values.index, 0, 30, alpha=0.15, color='#229954')
    axes[1].set_xlabel('Trading Date', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('RSI Value', fontsize=12, fontweight='bold')
    axes[1].set_title('Relative Strength Index', fontsize=13, fontweight='bold', pad=15)
    axes[1].set_ylim([-5, 105])
    axes[1].legend(loc='upper left', fontsize=10, framealpha=0.9)
    axes[1].grid(True, alpha=0.25, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('indicator_rsi.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: indicator_rsi.png")
    
    # CHART 3: MACD 
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    
    axes[0].plot(normalized_prices, label='JPM Price', color='#1A1A1A', linewidth=2)
    axes[0].set_ylabel('Price (Normalized)', fontsize=12, fontweight='bold')
    axes[0].set_title('JPM Price with MACD Momentum Indicator - 2008 to 2009', fontsize=13, fontweight='bold', pad=15)
    axes[0].legend(loc='upper left', fontsize=10, framealpha=0.9)
    axes[0].grid(True, alpha=0.25, linestyle='--')
    
    bar_colors = ['#2ECC71' if x > 0 else '#E67E22' for x in macd_values]
    axes[1].bar(macd_values.index, macd_values, color=bar_colors, alpha=0.75, width=1.5)
    axes[1].axhline(y=0, color='#2C3E50', linewidth=2)
    axes[1].set_xlabel('Trading Date', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Histogram Value', fontsize=12, fontweight='bold')
    axes[1].set_title('MACD Histogram (12-26-9)', fontsize=13, fontweight='bold', pad=15)
    axes[1].grid(True, alpha=0.25, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('indicator_macd.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: indicator_macd.png")
    
    # CHART 4: MOMENTUM 
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    
    axes[0].plot(normalized_prices, label='JPM Price', color='#154360', linewidth=2)
    axes[0].set_ylabel('Price (Normalized)', fontsize=12, fontweight='bold')
    axes[0].set_title('JPM Price Momentum Analysis - 2008 to 2009', fontsize=13, fontweight='bold', pad=15)
    axes[0].legend(loc='upper left', fontsize=10, framealpha=0.9)
    axes[0].grid(True, alpha=0.25, linestyle='--')
    
    axes[1].plot(mom_values, label='Momentum (10-day)', color='#D68910', linewidth=2.5)
    axes[1].axhline(y=0, color='#1C2833', linewidth=2)
    axes[1].fill_between(mom_values.index, 0, mom_values, where=(mom_values > 0),
                         alpha=0.2, color='#28B463', label='Positive Momentum')
    axes[1].fill_between(mom_values.index, 0, mom_values, where=(mom_values < 0),
                         alpha=0.2, color='#CB4335', label='Negative Momentum')
    axes[1].set_xlabel('Trading Date', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Momentum Value', fontsize=12, fontweight='bold')
    axes[1].set_title('Price Momentum Indicator', fontsize=13, fontweight='bold', pad=15)
    axes[1].legend(loc='upper left', fontsize=10, framealpha=0.9)
    axes[1].grid(True, alpha=0.25, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('indicator_momentum.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: indicator_momentum.png")
    
    # CHART 5: STOCHASTIC 
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    
    axes[0].plot(normalized_prices, label='JPM Price', color='#212F3C', linewidth=2)
    axes[0].set_ylabel('Price (Normalized)', fontsize=12, fontweight='bold')
    axes[0].set_title('JPM with Stochastic Oscillator - 2008 to 2009', fontsize=13, fontweight='bold', pad=15)
    axes[0].legend(loc='upper left', fontsize=10, framealpha=0.9)
    axes[0].grid(True, alpha=0.25, linestyle='--')
    
    axes[1].plot(stoch_values, label='Stochastic %K (14-day)', color='#6C3483', linewidth=2)
    axes[1].axhline(y=80, color='#922B21', linestyle='--', linewidth=1.8, alpha=0.8, label='Overbought (80)')
    axes[1].axhline(y=20, color='#0E6655', linestyle='--', linewidth=1.8, alpha=0.8, label='Oversold (20)')
    axes[1].axhline(y=50, color='#566573', linestyle=':', linewidth=1, alpha=0.5)
    axes[1].fill_between(stoch_values.index, 80, 100, alpha=0.15, color='#922B21')
    axes[1].fill_between(stoch_values.index, 0, 20, alpha=0.15, color='#0E6655')
    axes[1].set_xlabel('Trading Date', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Oscillator Value', fontsize=12, fontweight='bold')
    axes[1].set_title('Stochastic Oscillator (%K)', fontsize=13, fontweight='bold', pad=15)
    axes[1].set_ylim([-5, 105])
    axes[1].legend(loc='upper left', fontsize=10, framealpha=0.9)
    axes[1].grid(True, alpha=0.25, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('indicator_stochastic.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: indicator_stochastic.png")
    
    print("\nAll customized indicator charts created successfully!")


if __name__ == "__main__":
    print("Generating customized indicator charts...")
    create_all_charts(ticker="JPM", 
                     start_date=dt.datetime(2008, 1, 1),
                     end_date=dt.datetime(2009, 12, 31))