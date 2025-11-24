# indicators.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import get_data
import datetime as dt


def author():
    return 'dcarbono3'


def bollinger_band_percent(price_series, lookback=20):
    """Calculate Bollinger Band Percentage"""
    sma = price_series.rolling(window=lookback, min_periods=lookback).mean()
    std_dev = price_series.rolling(window=lookback, min_periods=lookback).std()
    
    upper_band = sma + (2.0 * std_dev)
    lower_band = sma - (2.0 * std_dev)
    
    bandwidth = upper_band - lower_band
    position = price_series - lower_band
    bbp = position / bandwidth
    
    return bbp


def rsi(price_series, period=14):
    """Relative Strength Index calculation"""
    delta = price_series.diff()
    
    ups = delta.copy()
    downs = delta.copy()
    ups[ups < 0] = 0
    downs[downs > 0] = 0
    downs = downs.abs()
    
    avg_up = ups.rolling(window=period, min_periods=period).mean()
    avg_down = downs.rolling(window=period, min_periods=period).mean()
    
    rs = avg_up / avg_down
    rsi_result = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi_result


def macd_histogram(price_series, fast_period=12, slow_period=26, signal_period=9):
    """MACD Histogram calculation"""
    ema_fast = price_series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = price_series.ewm(span=slow_period, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    
    return histogram


def momentum(price_series, lookback_days=10):
    """Price momentum calculation"""
    lagged_price = price_series.shift(lookback_days)
    mom = (price_series / lagged_price) - 1.0
    
    return mom


def stochastic_oscillator(price_series, window_size=14):
    """Stochastic Oscillator calculation"""
    highest = price_series.rolling(window=window_size, min_periods=window_size).max()
    lowest = price_series.rolling(window=window_size, min_periods=window_size).min()
    
    range_hl = highest - lowest
    current_position = price_series - lowest
    stoch = (current_position / range_hl) * 100.0
    
    return stoch