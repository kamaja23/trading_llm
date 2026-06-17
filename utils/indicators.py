"""
Technical indicator calculations for token generation.

This module calculates a comprehensive set of technical indicators from OHLCV data
and converts them into trading language tokens.
"""

import pandas as pd
import numpy as np


def calculate_trend(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate trend state with finer granularity based on price position
    relative to moving average.

    Args:
        df: DataFrame with 'Close' column
        window: Moving average window

    Returns:
        Series of trend tokens: TR_StrongUp, TR_Up, TR_WeakUp, TR_Flat, etc.
    """
    ma = df['Close'].rolling(window=window).mean()
    pct_diff = ((df['Close'] - ma) / ma * 100)

    conditions = [
        pct_diff > 5,
        pct_diff > 2,
        pct_diff > 0.5,
        pct_diff < -5,
        pct_diff < -2,
        pct_diff < -0.5,
    ]
    choices = [
        'TR_StrongUp',
        'TR_Up',
        'TR_WeakUp',
        'TR_StrongDown',
        'TR_Down',
        'TR_WeakDown',
    ]

    return pd.Series(
        np.select(conditions, choices, default='TR_Flat'),
        index=df.index
    )


def calculate_volume_state(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate volume state relative to moving average.

    Args:
        df: DataFrame with 'Volume' column
        window: Moving average window

    Returns:
        Series of volume tokens
    """
    avg_volume = df['Volume'].rolling(window=window).mean()
    ratio = df['Volume'] / avg_volume

    conditions = [
        ratio > 2.0,
        ratio > 1.2,
        ratio < 0.5,
        ratio < 0.8,
    ]
    choices = ['VOL_Surge', 'VOL_High', 'VOL_Dead', 'VOL_Low']

    return pd.Series(
        np.select(conditions, choices, default='VOL_Normal'),
        index=df.index
    )


def calculate_heikin_ashi_signal(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced Heikin Ashi signal detection.

    Args:
        df: DataFrame with OHLC columns

    Returns:
        Series of HA tokens: HA_UpCross, HA_DownCross, HA_UpTrend, HA_DownTrend, HA_Neutral
    """
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = df['Open'].iloc[0]

    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2

    current_bullish = ha_close > ha_open
    prev_bullish = current_bullish.shift(1).fillna(False)

    # Count consecutive bullish/bearish candles
    bullish_streak = 0
    bearish_streak = 0
    streaks = []
    for i in range(len(df)):
        if current_bullish.iloc[i]:
            bullish_streak += 1
            bearish_streak = 0
        else:
            bearish_streak += 1
            bullish_streak = 0
        streaks.append((bullish_streak, bearish_streak))

    streak_df = pd.DataFrame(streaks, columns=['bull_streak', 'bear_streak'], index=df.index)

    conditions = [
        (current_bullish) & (~prev_bullish),
        (~current_bullish) & (prev_bullish),
        (current_bullish) & (streak_df['bull_streak'] >= 3),
        (~current_bullish) & (streak_df['bear_streak'] >= 3),
    ]
    choices = ['HA_UpCross', 'HA_DownCross', 'HA_UpTrend', 'HA_DownTrend']

    return pd.Series(
        np.select(conditions, choices, default='HA_Neutral'),
        index=df.index
    )


def calculate_stochastic_signal(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.Series:
    """
    Calculate Stochastic oscillator signal with both cross and level detection.

    Args:
        df: DataFrame with High, Low, Close columns
        k_window: %K period
        d_window: %D period (smoothing)

    Returns:
        Series of stochastic tokens
    """
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()

    k = 100 * ((df['Close'] - low_min) / (high_max - low_min + 1e-10))
    d = k.rolling(window=d_window).mean()

    k_cross_up = (k > d) & (k.shift(1).fillna(0) <= d.shift(1).fillna(0))
    k_cross_down = (k < d) & (k.shift(1).fillna(100) >= d.shift(1).fillna(100))

    conditions = [
        k_cross_up,
        k_cross_down,
        k < 20,
        k > 80,
    ]
    choices = ['STO_Cross', 'STO_DownCross', 'STO_Oversold', 'STO_Overbought']

    return pd.Series(
        np.select(conditions, choices, default='STO_NoCross'),
        index=df.index
    )


def calculate_rsi_signal(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index) signal.

    Args:
        df: DataFrame with 'Close' column
        window: RSI period

    Returns:
        Series of RSI tokens
    """
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    conditions = [
        rsi < 30,
        rsi > 70,
        rsi >= 50,
        rsi < 50,
    ]
    choices = ['RSI_Oversold', 'RSI_Overbought', 'RSI_Bullish', 'RSI_Bearish']

    return pd.Series(
        np.select(conditions, choices, default='RSI_Neutral'),
        index=df.index
    )


def calculate_macd_signal(df: pd.DataFrame) -> pd.Series:
    """
    Calculate MACD (Moving Average Convergence Divergence) signal.

    Args:
        df: DataFrame with 'Close' column

    Returns:
        Series of MACD tokens
    """
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    # Detect crossovers
    cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

    # Divergence detection (simplified)
    price_high = df['Close'].rolling(20).max()
    price_low = df['Close'].rolling(20).min()
    macd_high = macd_line.rolling(20).max()
    macd_low = macd_line.rolling(20).min()

    # Bullish divergence: price making lower low, MACD making higher low
    is_bull_div = (df['Close'] == price_low) & (macd_line > macd_low.shift(1))
    # Bearish divergence: price making higher high, MACD making lower high
    is_bear_div = (df['Close'] == price_high) & (macd_line < macd_high.shift(1))

    conditions = [
        cross_up,
        cross_down,
        macd_line > 0,
        macd_line < 0,
        is_bull_div,
        is_bear_div,
    ]
    choices = [
        'MACD_BullishCross',
        'MACD_BearishCross',
        'MACD_Positive',
        'MACD_Negative',
        'MACD_DivBullish',
        'MACD_DivBearish',
    ]

    return pd.Series(
        np.select(conditions, choices, default='MACD_Neutral'),
        index=df.index
    )


def calculate_bollinger_bands_signal(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Calculate Bollinger Bands signal.

    Args:
        df: DataFrame with 'Close' column
        window: Moving average window
        num_std: Number of standard deviations

    Returns:
        Series of BB tokens
    """
    ma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()

    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    bandwidth = (upper - lower) / ma * 100

    conditions = [
        df['Close'] >= upper,
        df['Close'] <= lower,
        df['Close'] > ma,
        df['Close'] < ma,
        bandwidth < 5,
        bandwidth > 15,
    ]
    choices = [
        'BB_UpperTouch',
        'BB_LowerTouch',
        'BB_AboveMid',
        'BB_BelowMid',
        'BB_Squeeze',
        'BB_Expand',
    ]

    return pd.Series(
        np.select(conditions, choices, default='BB_Neutral'),
        index=df.index
    )


def calculate_ma_crossover_signal(df: pd.DataFrame, short_window: int = 50, long_window: int = 200) -> pd.Series:
    """
    Calculate Moving Average crossover signals.

    Args:
        df: DataFrame with 'Close' column
        short_window: Short-term MA period (e.g., 50)
        long_window: Long-term MA period (e.g., 200)

    Returns:
        Series of MA crossover tokens
    """
    if len(df) < long_window:
        return pd.Series(['MA_Neutral'] * len(df), index=df.index)

    short_ma = df['Close'].rolling(window=short_window).mean()
    long_ma = df['Close'].rolling(window=long_window).mean()

    golden_cross = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
    death_cross = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
    bullish = short_ma > long_ma
    bearish = short_ma < long_ma

    conditions = [
        golden_cross,
        death_cross,
        bullish,
        bearish,
    ]
    choices = ['MA_GoldenCross', 'MA_DeathCross', 'MA_Bullish', 'MA_Bearish']

    return pd.Series(
        np.select(conditions, choices, default='MA_Neutral'),
        index=df.index
    )


def calculate_volatility_signal(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate volatility signal based on ATR relative to price.

    Args:
        df: DataFrame with OHLC data
        window: ATR window

    Returns:
        Series of volatility tokens
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()

    atr_pct = atr / df['Close'] * 100

    conditions = [
        atr_pct > atr_pct.rolling(window=window*3).mean() * 1.5,
        atr_pct < atr_pct.rolling(window=window*3).mean() * 0.5,
    ]
    choices = ['VIX_High', 'VIX_Low']

    return pd.Series(
        np.select(conditions, choices, default='VIX_Moderate'),
        index=df.index
    )


def detect_candle_pattern(df: pd.DataFrame) -> pd.Series:
    """
    Detect common single and multi-candle patterns.

    Args:
        df: DataFrame with OHLC columns

    Returns:
        Series of candle pattern tokens
    """
    body = abs(df['Close'] - df['Open'])
    upper_wick = df['High'] - df[['Close', 'Open']].max(axis=1)
    lower_wick = df[['Close', 'Open']].min(axis=1) - df['Low']
    total_range = df['High'] - df['Low'] + 1e-10

    doji = body <= total_range * 0.1

    hammer = (~doji) & (lower_wick >= total_range * 0.6) & (upper_wick <= total_range * 0.1) & (body <= total_range * 0.3)

    shooting_star = (~doji) & (upper_wick >= total_range * 0.6) & (lower_wick <= total_range * 0.1) & (body <= total_range * 0.3)

    bullish_marubozu = (body >= total_range * 0.8) & (upper_wick <= total_range * 0.05) & (lower_wick <= total_range * 0.05) & (df['Close'] > df['Open'])

    bearish_marubozu = (body >= total_range * 0.8) & (upper_wick <= total_range * 0.05) & (lower_wick <= total_range * 0.05) & (df['Close'] < df['Open'])

    spinning_top = (~doji) & (upper_wick >= total_range * 0.3) & (lower_wick >= total_range * 0.3) & (body <= total_range * 0.3)

    # Multi-candle patterns (require previous day)
    engulfing_bullish = pd.Series(False, index=df.index)
    engulfing_bearish = pd.Series(False, index=df.index)
    morning_star = pd.Series(False, index=df.index)
    evening_star = pd.Series(False, index=df.index)

    for i in range(2, len(df)):
        prev_body1 = abs(df['Close'].iloc[i-2] - df['Open'].iloc[i-2])
        prev_body2 = abs(df['Close'].iloc[i-1] - df['Open'].iloc[i-1])

        # Engulfing patterns
        if df['Close'].iloc[i-1] < df['Open'].iloc[i-1]:  # Previous bearish
            if df['Close'].iloc[i] > df['Open'].iloc[i]:  # Current bullish
                if df['Open'].iloc[i] <= df['Close'].iloc[i-1] and df['Close'].iloc[i] >= df['Open'].iloc[i-1]:
                    engulfing_bullish.iloc[i] = True

        if df['Close'].iloc[i-1] > df['Open'].iloc[i-1]:  # Previous bullish
            if df['Close'].iloc[i] < df['Open'].iloc[i]:  # Current bearish
                if df['Open'].iloc[i] >= df['Close'].iloc[i-1] and df['Close'].iloc[i] <= df['Open'].iloc[i-1]:
                    engulfing_bearish.iloc[i] = True

        # Morning/Evening star (3-candle pattern)
        prev_body1_size = prev_body1 / (total_range.iloc[i-2] + 1e-10)
        prev_body2_size = prev_body2 / (total_range.iloc[i-1] + 1e-10)
        curr_body = body.iloc[i] / total_range.iloc[i]

        if prev_body1_size > 0.5 and prev_body2_size < 0.2 and curr_body > 0.5:
            if df['Close'].iloc[i-2] < df['Open'].iloc[i-2] and df['Close'].iloc[i] > df['Open'].iloc[i]:
                if df['Close'].iloc[i] > (df['High'].iloc[i-2] + df['Low'].iloc[i-2]) / 2 + (df['High'].iloc[i-2] - df['Low'].iloc[i-2]) * 0.3:
                    morning_star.iloc[i] = True
            if df['Close'].iloc[i-2] > df['Open'].iloc[i-2] and df['Close'].iloc[i] < df['Open'].iloc[i]:
                if df['Close'].iloc[i] < (df['High'].iloc[i-2] + df['Low'].iloc[i-2]) / 2 - (df['High'].iloc[i-2] - df['Low'].iloc[i-2]) * 0.3:
                    evening_star.iloc[i] = True

    conditions = [
        doji,
        hammer,
        shooting_star,
        engulfing_bullish,
        engulfing_bearish,
        morning_star,
        evening_star,
        bullish_marubozu,
        bearish_marubozu,
        spinning_top,
    ]
    choices = [
        'CP_Doji',
        'CP_Hammer',
        'CP_ShootingStar',
        'CP_EngulfBull',
        'CP_EngulfBear',
        'CP_MorningStar',
        'CP_EveningStar',
        'CP_MarubozuBull',
        'CP_MarubozuBear',
        'CP_SpinningTop',
    ]

    return pd.Series(
        np.select(conditions, choices, default='CP_NoPattern'),
        index=df.index
    )


def calculate_obv_signal(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate On-Balance Volume signal.

    Args:
        df: DataFrame with 'Close' and 'Volume' columns
        window: Smoothing window

    Returns:
        Series of OBV tokens
    """
    obv = (df['Volume'] * (~df['Close'].diff().le(0) * 2 - 1)).cumsum()
    obv_ma = obv.rolling(window=window).mean()

    # OBV trend
    obv_trend = obv - obv_ma

    # Divergence detection: price direction vs OBV direction
    price_trend = df['Close'].diff(window)
    obv_change = obv.diff(window)

    bullish_div = (price_trend < 0) & (obv_change > 0) & (obv_trend > 0)
    bearish_div = (price_trend > 0) & (obv_change < 0) & (obv_trend < 0)

    conditions = [
        bullish_div,
        bearish_div,
        obv_trend > obv_ma * 0.02,
        obv_trend < -obv_ma * 0.02,
    ]
    choices = ['OBV_DivBull', 'OBV_DivBear', 'OBV_Rising', 'OBV_Falling']

    return pd.Series(
        np.select(conditions, choices, default='OBV_Flat'),
        index=df.index
    )


def calculate_atr_signal(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate Average True Range signal.

    Args:
        df: DataFrame with OHLC data
        window: ATR window

    Returns:
        Series of ATR tokens
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()

    atr_slope = atr.diff(5)
    atr_pct = atr / df['Close'] * 100
    atr_pct_ma = atr_pct.rolling(window=window*2).mean()

    conditions = [
        (atr_pct > atr_pct_ma * 1.5) & (atr_slope > 0),
        atr_slope > 0,
        atr_slope < 0,
        atr_pct < atr_pct_ma * 0.5,
    ]
    choices = ['ATR_High', 'ATR_Rising', 'ATR_Falling', 'ATR_Low']

    return pd.Series(
        np.select(conditions, choices, default='ATR_Rising'),
        index=df.index
    )


def calculate_price_action_signal(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Detect price action patterns like breakouts and support/resistance tests.

    Args:
        df: DataFrame with OHLC data
        window: Lookback window

    Returns:
        Series of price action tokens
    """
    rolling_high = df['High'].rolling(window=window).max()
    rolling_low = df['Low'].rolling(window=window).min()

    recent_high = df['High'].rolling(window=5).max()
    recent_low = df['Low'].rolling(window=5).min()

    # Gap detection
    gap_up = df['Low'] > df['High'].shift(1)
    gap_down = df['High'] < df['Low'].shift(1)

    # Breakout detection
    breakout_up = (df['Close'] > rolling_high.shift(1)) & (df['Volume'] > df['Volume'].rolling(window=window).mean())
    breakout_down = (df['Close'] < rolling_low.shift(1)) & (df['Volume'] > df['Volume'].rolling(window=window).mean())

    # Support/resistance tests
    near_high = (df['High'] >= rolling_high * 0.995) & (df['High'] <= rolling_high * 1.005)
    near_low = (df['Low'] <= rolling_low * 1.005) & (df['Low'] >= rolling_low * 0.995)

    # New highs/lows
    new_high = df['High'] == df['High'].rolling(window=window*2).max()
    new_low = df['Low'] == df['Low'].rolling(window=window*2).min()

    # Range bound
    range_width = (rolling_high - rolling_low) / rolling_low
    in_range = range_width < 0.05

    conditions = [
        breakout_up,
        breakout_down,
        near_high,
        near_low,
        new_high,
        new_low,
        gap_up,
        gap_down,
        in_range,
    ]
    choices = [
        'PA_BreakoutUp',
        'PA_BreakoutDown',
        'PA_ResistanceTest',
        'PA_SupportTest',
        'PA_NewHigh',
        'PA_NewLow',
        'PA_GapUp',
        'PA_GapDown',
        'PA_RangeBound',
    ]

    return pd.Series(
        np.select(conditions, choices, default='PA_RangeBound'),
        index=df.index
    )


def calculate_market_context(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    Calculate broader market context based on recent price action.

    Args:
        df: DataFrame with 'Close' column
        window: Lookback window

    Returns:
        Series of market context tokens
    """
    returns = df['Close'].pct_change(periods=window)
    volatility = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    # Market phase
    high_vol = volatility > volatility.rolling(window=window*2).mean()
    low_vol = volatility < volatility.rolling(window=window*2).mean() * 0.5

    # Overbought/oversold based on distance from 200-day MA
    ma_200 = df['Close'].rolling(window=min(200, len(df))).mean()
    pct_from_ma = ((df['Close'] - ma_200) / ma_200 * 100)

    conditions = [
        returns > 0.1,
        returns < -0.1,
        high_vol & (volatility > volatility.median()),
        low_vol,
        pct_from_ma > 15,
        pct_from_ma < -15,
    ]
    choices = [
        'MKT_Bullish',
        'MKT_Bearish',
        'MKT_Volatile',
        'MKT_Calm',
        'MKT_Overbought',
        'MKT_Oversold',
    ]

    return pd.Series(
        np.select(conditions, choices, default='MKT_Neutral'),
        index=df.index
    )


def calculate_relative_strength(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate relative strength (proxy using price momentum vs rolling average).

    Args:
        df: DataFrame with 'Close' column
        window: Lookback window

    Returns:
        Series of relative strength tokens
    """
    momentum = df['Close'].pct_change(periods=window)
    avg_momentum = momentum.rolling(window=window).mean()

    conditions = [
        momentum > avg_momentum + avg_momentum.std() if len(avg_momentum.dropna()) > 0 else momentum > 0,
        momentum < avg_momentum - avg_momentum.std() if len(avg_momentum.dropna()) > 0 else momentum < 0,
    ]
    choices = ['REL_Strong', 'REL_Weak']

    return pd.Series(
        np.select(conditions, choices, default='REL_Inline'),
        index=df.index
    )


def calculate_action_token(df: pd.DataFrame, future_window: int = 5, threshold: float = 0.01) -> pd.Series:
    """
    Calculate action token based on future price movement.

    Args:
        df: DataFrame with 'Close' column
        future_window: Days to look ahead
        threshold: Minimum price change (as fraction) to trigger BUY/SELL

    Returns:
        Series of action tokens: BUY, SELL, HOLD
    """
    future_price = df['Close'].shift(-future_window)
    current_price = df['Close']

    future_return = (future_price - current_price) / current_price

    conditions = [
        future_return > threshold,
        future_return < -threshold,
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
    df['rsi_token'] = calculate_rsi_signal(df)
    df['macd_token'] = calculate_macd_signal(df)
    df['bb_token'] = calculate_bollinger_bands_signal(df)
    df['ma_cross_token'] = calculate_ma_crossover_signal(df)
    df['volatility_token'] = calculate_volatility_signal(df)
    df['candle_token'] = detect_candle_pattern(df)
    df['obv_token'] = calculate_obv_signal(df)
    df['atr_token'] = calculate_atr_signal(df)
    df['price_action_token'] = calculate_price_action_signal(df)
    df['market_context_token'] = calculate_market_context(df)
    df['relative_token'] = calculate_relative_strength(df)
    df['action_token'] = calculate_action_token(df)

    return df


if __name__ == "__main__":
    print("Testing indicator calculations...")

    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    close = np.random.randn(200).cumsum() + 100
    sample_data = pd.DataFrame({
        'Open': close - np.random.rand(200) * 2,
        'High': close + np.abs(np.random.randn(200) * 2),
        'Low': close - np.abs(np.random.randn(200) * 2),
        'Close': close,
        'Volume': np.random.randint(1000000, 5000000, 200)
    }, index=dates)

    result = add_all_indicators(sample_data)

    token_cols = [c for c in result.columns if c.endswith('_token')]
    print(f"\nGenerated {len(token_cols)} token columns:")
    for col in token_cols:
        unique = result[col].unique()
        print(f"  {col}: {len(unique)} unique values → {unique}")
