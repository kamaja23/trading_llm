"""
Technical indicator calculations for token generation.

This module calculates simple technical indicators from OHLCV data
and converts them into trading language tokens.
"""

import pandas as pd
import numpy as np


def calculate_trend(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate trend state based on price position relative to moving average.
    
    Args:
        df: DataFrame with 'Close' column
        window: Moving average window
        
    Returns:
        Series of trend tokens: ST_UpTrend, ST_DownTrend, ST_Flat
    """
    ma = df['Close'].rolling(window=window).mean()
    
    # Calculate percentage above/below MA
    pct_diff = ((df['Close'] - ma) / ma * 100)
    
    # Define thresholds for trend classification
    conditions = [
        pct_diff > 2,   # More than 2% above MA = uptrend
        pct_diff < -2,  # More than 2% below MA = downtrend
    ]
    choices = ['ST_UpTrend', 'ST_DownTrend']
    
    return pd.Series(
        np.select(conditions, choices, default='ST_Flat'),
        index=df.index
    )


def calculate_volume_state(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate volume state relative to moving average.
    
    Args:
        df: DataFrame with 'Volume' column
        window: Moving average window
        
    Returns:
        Series of volume tokens: Hi_Volume, Lo_Volume, Avg_Volume
    """
    avg_volume = df['Volume'].rolling(window=window).mean()
    ratio = df['Volume'] / avg_volume
    
    conditions = [
        ratio > 1.2,   # 20% above average
        ratio < 0.8,   # 20% below average
    ]
    choices = ['Hi_Volume', 'Lo_Volume']
    
    return pd.Series(
        np.select(conditions, choices, default='Avg_Volume'),
        index=df.index
    )


def calculate_heikin_ashi_signal(df: pd.DataFrame) -> pd.Series:
    """
    Simplified Heikin Ashi signal.
    
    Args:
        df: DataFrame with OHLC columns
        
    Returns:
        Series of HA tokens: HA_UpCross, HA_DownCross, HA_Neutral
    """
    # Calculate Heikin Ashi candles
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = df['Open'].iloc[0]
    
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    
    # Signal: bullish if HA close > HA open
    current_bullish = ha_close > ha_open
    prev_bullish = current_bullish.shift(1).fillna(False)  # Fix: Fill NaN with False
    
    # Detect crosses
    conditions = [
        (current_bullish) & (~prev_bullish),  # Bullish cross
        (~current_bullish) & (prev_bullish),  # Bearish cross
    ]
    choices = ['HA_UpCross', 'HA_DownCross']
    
    return pd.Series(
        np.select(conditions, choices, default='HA_Neutral'),
        index=df.index
    )


def calculate_stochastic_signal(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.Series:
    """
    Calculate Stochastic oscillator signal.
    
    Args:
        df: DataFrame with High, Low, Close columns
        k_window: %K period
        d_window: %D period (smoothing)
        
    Returns:
        Series of stochastic tokens
    """
    # Calculate %K
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_window).mean()
    
    # Detect crosses and overbought/oversold
    k_cross_up = (k > d) & (k.shift(1).fillna(0) <= d.shift(1).fillna(0))  # Fix: Fill NaN
    k_cross_down = (k < d) & (k.shift(1).fillna(100) >= d.shift(1).fillna(100))  # Fix: Fill NaN
    
    conditions = [
        k < 20,           # Oversold
        k > 80,           # Overbought
        k_cross_up,       # Bullish cross
        k_cross_down,     # Bearish cross (not used in current tokens but could be)
    ]
    choices = ['STO_Oversold', 'STO_Overbought', 'STO_Cross', 'STO_NoCross']
    
    return pd.Series(
        np.select(conditions, choices, default='STO_NoCross'),
        index=df.index
    )


def calculate_action_token(df: pd.DataFrame, future_window: int = 5, threshold: float = 0.02) -> pd.Series:
    """
    Calculate action token based on future price movement.
    
    This is the "label" we want the model to learn to predict.
    
    Args:
        df: DataFrame with 'Close' column
        future_window: Days to look ahead
        threshold: Minimum price change (as fraction) to trigger BUY/SELL
        
    Returns:
        Series of action tokens: BUY, SELL, HOLD
    """
    # Look ahead at future price
    future_price = df['Close'].shift(-future_window)
    current_price = df['Close']
    
    # Calculate future return
    future_return = (future_price - current_price) / current_price
    
    conditions = [
        future_return > threshold,   # Price will go up significantly -> BUY
        future_return < -threshold,  # Price will go down significantly -> SELL
    ]
    choices = ['BUY', 'SELL']
    
    return pd.Series(
        np.select(conditions, choices, default='HOLD'),
        index=df.index
    )


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicator columns to a DataFrame.
    
    Args:
        df: DataFrame with OHLCV columns
        
    Returns:
        DataFrame with added indicator columns as tokens
    """
    df = df.copy()
    
    df['trend_token'] = calculate_trend(df)
    df['volume_token'] = calculate_volume_state(df)
    df['ha_token'] = calculate_heikin_ashi_signal(df)
    df['sto_token'] = calculate_stochastic_signal(df)
    df['action_token'] = calculate_action_token(df)
    
    return df


if __name__ == "__main__":
    # Test with sample data
    print("Testing indicator calculations...")
    
    # Create sample OHLCV data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 102,
        'Low': np.random.randn(100).cumsum() + 98,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # Add indicators
    result = add_all_indicators(sample_data)
    
    print("\nSample output:")
    print(result[['Close', 'trend_token', 'volume_token', 'ha_token', 'sto_token', 'action_token']].tail(10))
