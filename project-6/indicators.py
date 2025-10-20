"""
indicators.py
Technical Indicators
Implements 5 indicators that return single result vectors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import get_data
import datetime as dt

def author():
    return 'dcarbono3'


def bollinger_band_percent(prices, window=20):
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    bbp = (prices - lower_band) / (upper_band - lower_band)
    return bbp


def rsi(prices, window=14):
    delta = prices.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd_histogram(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    histogram = macd_line - signal_line
    return histogram


def momentum(prices, window=10):
    mom = (prices / prices.shift(window)) - 1 
    return mom


def stochastic_oscillator(prices, window=14):
    lowest_low = prices.rolling(window=window).min()
    highest_high = prices.rolling(window=window).max()
    
    stoch = ((prices - lowest_low) / (highest_high - lowest_low)) * 100
    
    return stoch



def generate_indicator_charts(symbol="JPM", sd=dt.datetime(2008, 1, 1), 
                              ed=dt.datetime(2009, 12, 31)):

    dates = pd.date_range(sd, ed)
    prices_all = get_data([symbol], dates, addSPY=True, colname='Adj Close')
    prices = prices_all[[symbol]].fillna(method='ffill').fillna(method='bfill')
    prices_series = prices[symbol]
    
    prices_norm = prices_series / prices_series.iloc[0]
    
    bbp = bollinger_band_percent(prices_series, window=20)
    rsi_values = rsi(prices_series, window=14)
    macd_hist = macd_histogram(prices_series, fast=12, slow=26, signal=9)
    mom = momentum(prices_series, window=10)
    stoch = stochastic_oscillator(prices_series, window=14)
    
    sma_20 = prices_series.rolling(window=20).mean()
    std_20 = prices_series.rolling(window=20).std()
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    
    sma_20_norm = sma_20 / prices_series.iloc[0]
    upper_band_norm = upper_band / prices_series.iloc[0]
    lower_band_norm = lower_band / prices_series.iloc[0]
    
    # ===== CHART 1: Bollinger Band Percentage =====
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(prices_norm.index, prices_norm, label='JPM Price', color='black', linewidth=1.5)
    ax1.plot(sma_20_norm.index, sma_20_norm, label='SMA (20)', color='blue', linewidth=1)
    ax1.plot(upper_band_norm.index, upper_band_norm, label='Upper Band', 
             color='red', linestyle='--', linewidth=1)
    ax1.plot(lower_band_norm.index, lower_band_norm, label='Lower Band', 
             color='green', linestyle='--', linewidth=1)
    ax1.set_ylabel('Normalized Price', fontsize=11)
    ax1.set_title('Bollinger Bands Indicator (Window=20 days)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(bbp.index, bbp, label='BBP', color='purple', linewidth=1.5)
    ax2.axhline(y=0, color='green', linestyle='--', linewidth=1, label='Oversold (BBP=0)')
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=1, label='Overbought (BBP=1)')
    ax2.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax2.fill_between(bbp.index, 0, bbp, where=(bbp <= 0), alpha=0.3, color='green', label='Below Lower Band')
    ax2.fill_between(bbp.index, 1, bbp, where=(bbp >= 1), alpha=0.3, color='red', label='Above Upper Band')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('BBP Value', fontsize=11)
    ax2.set_title('Bollinger Band Percentage (BBP)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('indicator_bbp.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: indicator_bbp.png")
    
    
    # ===== CHART 2: RSI =====
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(prices_norm.index, prices_norm, label='JPM Price', color='black', linewidth=1.5)
    ax1.set_ylabel('Normalized Price', fontsize=11)
    ax1.set_title('Relative Strength Index (RSI) with Price', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(rsi_values.index, rsi_values, label='RSI (14-day)', color='blue', linewidth=1.5)
    ax2.axhline(y=70, color='red', linestyle='--', linewidth=1, label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', linewidth=1, label='Oversold (30)')
    ax2.axhline(y=50, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax2.fill_between(rsi_values.index, 70, 100, alpha=0.2, color='red')
    ax2.fill_between(rsi_values.index, 0, 30, alpha=0.2, color='green')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('RSI Value', fontsize=11)
    ax2.set_title('RSI Indicator Values', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 100])
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('indicator_rsi.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: indicator_rsi.png")
    
    
    # ===== CHART 3: MACD Histogram =====
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(prices_norm.index, prices_norm, label='JPM Price', color='black', linewidth=1.5)
    ax1.set_ylabel('Normalized Price', fontsize=11)
    ax1.set_title('MACD Histogram with Price', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    colors = ['green' if val > 0 else 'red' for val in macd_hist]
    ax2.bar(macd_hist.index, macd_hist, label='MACD Histogram', color=colors, alpha=0.7, width=1)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Histogram Value', fontsize=11)
    ax2.set_title('MACD Histogram (12, 26, 9)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('indicator_macd.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: indicator_macd.png")
    
    
    # ===== CHART 4: Momentum =====
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(prices_norm.index, prices_norm, label='JPM Price', color='black', linewidth=1.5)
    ax1.set_ylabel('Normalized Price', fontsize=11)
    ax1.set_title('Momentum Indicator (10-day) with Price', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(mom.index, mom, label='Momentum', color='orange', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.fill_between(mom.index, 0, mom, where=(mom > 0), alpha=0.3, color='green', label='Positive Momentum')
    ax2.fill_between(mom.index, 0, mom, where=(mom < 0), alpha=0.3, color='red', label='Negative Momentum')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Momentum Value', fontsize=11)
    ax2.set_title('Momentum Indicator Values', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('indicator_momentum.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: indicator_momentum.png")
    
    
    # ===== CHART 5: Stochastic Oscillator =====
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(prices_norm.index, prices_norm, label='JPM Price', color='black', linewidth=1.5)
    ax1.set_ylabel('Normalized Price', fontsize=11)
    ax1.set_title('Stochastic Oscillator (14-day) with Price', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(stoch.index, stoch, label='Stochastic %K', color='purple', linewidth=1.5)
    ax2.axhline(y=80, color='red', linestyle='--', linewidth=1, label='Overbought (80)')
    ax2.axhline(y=20, color='green', linestyle='--', linewidth=1, label='Oversold (20)')
    ax2.axhline(y=50, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax2.fill_between(stoch.index, 80, 100, alpha=0.2, color='red')
    ax2.fill_between(stoch.index, 0, 20, alpha=0.2, color='green')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Stochastic Value', fontsize=11)
    ax2.set_title('Stochastic Oscillator Values', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 100])
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('indicator_stochastic.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: indicator_stochastic.png")
    
    print("\nAll 5 indicator charts generated successfully!")


if __name__ == "__main__":
    print("Generating indicator charts...")
    generate_indicator_charts(symbol="JPM", 
                             sd=dt.datetime(2008, 1, 1), 
                             ed=dt.datetime(2009, 12, 31))