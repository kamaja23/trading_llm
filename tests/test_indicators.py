"""Tests for indicators module."""

import numpy as np
import pandas as pd
import pytest
from utils.indicators import (
    calculate_trend,
    calculate_volume_state,
    calculate_heikin_ashi_signal,
    calculate_stochastic_signal,
    calculate_rsi_signal,
    calculate_macd_signal,
    calculate_bollinger_bands_signal,
    calculate_ma_crossover_signal,
    calculate_volatility_signal,
    detect_candle_pattern,
    calculate_obv_signal,
    calculate_atr_signal,
    calculate_price_action_signal,
    calculate_market_context,
    calculate_relative_strength,
    calculate_action_token,
    add_all_indicators,
)


@pytest.fixture
def sample_ohlcv():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    close = np.random.randn(200).cumsum() + 100
    df = pd.DataFrame(
        {
            "Open": close - np.random.rand(200) * 2,
            "High": close + np.abs(np.random.randn(200) * 2),
            "Low": close - np.abs(np.random.randn(200) * 2),
            "Close": close,
            "Volume": np.random.randint(1_000_000, 5_000_000, 200),
        },
        index=dates,
    )
    return df


class TestCalculateTrend:
    def test_returns_series(self, sample_ohlcv):
        result = calculate_trend(sample_ohlcv)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)

    def test_returns_valid_tokens(self, sample_ohlcv):
        result = calculate_trend(sample_ohlcv)
        valid_tokens = {"TR_StrongUp", "TR_Up", "TR_WeakUp", "TR_Flat", "TR_WeakDown", "TR_Down", "TR_StrongDown"}
        for token in result.dropna():
            assert token in valid_tokens

    def test_all_rows_have_values(self, sample_ohlcv):
        result = calculate_trend(sample_ohlcv)
        assert result.notna().all()


class TestCalculateVolumeState:
    def test_returns_series(self, sample_ohlcv):
        result = calculate_volume_state(sample_ohlcv)
        assert isinstance(result, pd.Series)

    def test_returns_valid_tokens(self, sample_ohlcv):
        result = calculate_volume_state(sample_ohlcv)
        valid_tokens = {"VOL_Surge", "VOL_High", "VOL_Normal", "VOL_Low", "VOL_Dead"}
        for token in result.dropna():
            assert token in valid_tokens


class TestHeikinAshiSignal:
    def test_returns_series(self, sample_ohlcv):
        result = calculate_heikin_ashi_signal(sample_ohlcv)
        assert isinstance(result, pd.Series)

    def test_returns_valid_tokens(self, sample_ohlcv):
        result = calculate_heikin_ashi_signal(sample_ohlcv)
        valid_tokens = {"HA_UpCross", "HA_DownCross", "HA_UpTrend", "HA_DownTrend", "HA_Neutral"}
        for token in result.dropna():
            assert token in valid_tokens


class TestStochasticSignal:
    def test_returns_series(self, sample_ohlcv):
        result = calculate_stochastic_signal(sample_ohlcv)
        assert isinstance(result, pd.Series)

    def test_returns_valid_tokens(self, sample_ohlcv):
        result = calculate_stochastic_signal(sample_ohlcv)
        valid_tokens = {"STO_Oversold", "STO_Overbought", "STO_Cross", "STO_DownCross", "STO_NoCross"}
        for token in result.dropna():
            assert token in valid_tokens


class TestRSISignal:
    def test_returns_series(self, sample_ohlcv):
        result = calculate_rsi_signal(sample_ohlcv)
        assert isinstance(result, pd.Series)

    def test_returns_valid_tokens(self, sample_ohlcv):
        result = calculate_rsi_signal(sample_ohlcv)
        valid_tokens = {"RSI_Oversold", "RSI_Overbought", "RSI_Bullish", "RSI_Bearish", "RSI_Neutral"}
        for token in result.dropna():
            assert token in valid_tokens


class TestMACDSignal:
    def test_returns_series(self, sample_ohlcv):
        result = calculate_macd_signal(sample_ohlcv)
        assert isinstance(result, pd.Series)

    def test_returns_valid_tokens(self, sample_ohlcv):
        result = calculate_macd_signal(sample_ohlcv)
        valid_tokens = {"MACD_BullishCross", "MACD_BearishCross", "MACD_Positive", "MACD_Negative",
                        "MACD_Neutral", "MACD_DivBullish", "MACD_DivBearish"}
        for token in result.dropna():
            assert token in valid_tokens


class TestBollingerBandsSignal:
    def test_returns_series(self, sample_ohlcv):
        result = calculate_bollinger_bands_signal(sample_ohlcv)
        assert isinstance(result, pd.Series)

    def test_returns_valid_tokens(self, sample_ohlcv):
        result = calculate_bollinger_bands_signal(sample_ohlcv)
        valid_tokens = {"BB_UpperTouch", "BB_LowerTouch", "BB_AboveMid", "BB_BelowMid",
                        "BB_Squeeze", "BB_Expand", "BB_Neutral"}
        for token in result.dropna():
            assert token in valid_tokens


class TestMACrossoverSignal:
    def test_returns_series(self, sample_ohlcv):
        result = calculate_ma_crossover_signal(sample_ohlcv)
        assert isinstance(result, pd.Series)

    def test_returns_valid_tokens(self, sample_ohlcv):
        result = calculate_ma_crossover_signal(sample_ohlcv)
        valid_tokens = {"MA_GoldenCross", "MA_DeathCross", "MA_Bullish", "MA_Bearish", "MA_Neutral"}
        for token in result.dropna():
            assert token in valid_tokens

    def test_short_dataset_returns_neutral(self):
        df = pd.DataFrame({"Close": [100] * 50})
        result = calculate_ma_crossover_signal(df)
        assert (result == "MA_Neutral").all()


class TestVolatilitySignal:
    def test_returns_series(self, sample_ohlcv):
        result = calculate_volatility_signal(sample_ohlcv)
        assert isinstance(result, pd.Series)

    def test_returns_valid_tokens(self, sample_ohlcv):
        result = calculate_volatility_signal(sample_ohlcv)
        valid_tokens = {"VIX_High", "VIX_Moderate", "VIX_Low"}
        for token in result.dropna():
            assert token in valid_tokens


class TestCandlePattern:
    def test_returns_series(self, sample_ohlcv):
        result = detect_candle_pattern(sample_ohlcv)
        assert isinstance(result, pd.Series)

    def test_returns_valid_tokens(self, sample_ohlcv):
        result = detect_candle_pattern(sample_ohlcv)
        valid_tokens = {"CP_Doji", "CP_Hammer", "CP_ShootingStar", "CP_EngulfBull", "CP_EngulfBear",
                        "CP_MorningStar", "CP_EveningStar", "CP_MarubozuBull", "CP_MarubozuBear",
                        "CP_SpinningTop", "CP_NoPattern"}
        for token in result.dropna():
            assert token in valid_tokens


class TestOBVSignal:
    def test_returns_series(self, sample_ohlcv):
        result = calculate_obv_signal(sample_ohlcv)
        assert isinstance(result, pd.Series)

    def test_returns_valid_tokens(self, sample_ohlcv):
        result = calculate_obv_signal(sample_ohlcv)
        valid_tokens = {"OBV_Rising", "OBV_Falling", "OBV_Flat", "OBV_DivBull", "OBV_DivBear"}
        for token in result.dropna():
            assert token in valid_tokens


class TestCalculateActionToken:
    def test_returns_series(self, sample_ohlcv):
        result = calculate_action_token(sample_ohlcv)
        assert isinstance(result, pd.Series)

    def test_returns_valid_tokens(self, sample_ohlcv):
        result = calculate_action_token(sample_ohlcv)
        valid_tokens = {"BUY", "SELL", "HOLD"}
        for token in result.dropna():
            assert token in valid_tokens

    def test_known_buy_signal(self):
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        close = np.array([100 + i for i in range(20)])
        df = pd.DataFrame({"Close": close}, index=dates)
        result = calculate_action_token(df, future_window=5, threshold=0.01)
        assert result.iloc[0] == "BUY"
        assert result.iloc[10] == "BUY"

    def test_known_sell_signal(self):
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        close = np.array([100 - i for i in range(20)])
        df = pd.DataFrame({"Close": close}, index=dates)
        result = calculate_action_token(df, future_window=5, threshold=0.01)
        assert result.iloc[0] == "SELL"

    def test_all_rows_have_values(self, sample_ohlcv):
        result = calculate_action_token(sample_ohlcv, future_window=10)
        assert result.notna().all()


class TestAddAllIndicators:
    def test_adds_all_indicator_columns(self, sample_ohlcv):
        result = add_all_indicators(sample_ohlcv)
        expected_columns = {
            "trend_token", "volume_token", "ha_token", "sto_token",
            "rsi_token", "macd_token", "bb_token", "ma_cross_token",
            "volatility_token", "candle_token", "obv_token", "atr_token",
            "price_action_token", "market_context_token", "relative_token",
            "action_token",
        }
        assert expected_columns.issubset(set(result.columns))

    def test_returns_dataframe(self, sample_ohlcv):
        result = add_all_indicators(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_original(self, sample_ohlcv):
        original_cols = set(sample_ohlcv.columns)
        add_all_indicators(sample_ohlcv)
        assert set(sample_ohlcv.columns) == original_cols

    def test_all_token_columns_have_valid_values(self, sample_ohlcv):
        result = add_all_indicators(sample_ohlcv)
        token_cols = [c for c in result.columns if c.endswith('_token') and c != 'action_token']
        for col in token_cols:
            assert result[col].notna().all(), f"{col} has NaN values"
