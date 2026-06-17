"""Tests for market_data module."""

import pytest
from utils.market_data import (
    _normalize_ohlcv,
    _resolve_date_range,
    _parse_date,
    _period_to_days,
    _to_stooq_symbol,
    _to_stockanalysis_symbol,
    _display_stockanalysis_symbol,
    _stockanalysis_history_url,
    MarketDataError,
)
import pandas as pd
from datetime import date


class TestNormalizeOHLCV:
    def test_valid_dataframe(self):
        df = pd.DataFrame(
            {
                "Date": ["2023-01-03", "2023-01-04"],
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.5],
                "Volume": [1000000, 1500000],
            }
        )
        result = _normalize_ohlcv(df, "SPY")
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert result.index.name == "Date"
        assert len(result) == 2

    def test_missing_columns_raises_error(self):
        df = pd.DataFrame({"Close": [100.0]})
        with pytest.raises(MarketDataError, match="missing columns"):
            _normalize_ohlcv(df, "SPY")

    def test_empty_after_dropna_raises_error(self):
        df = pd.DataFrame(
            {
                "Date": ["2023-01-03"],
                "Open": [None],
                "High": [None],
                "Low": [None],
                "Close": [None],
                "Volume": [None],
            }
        )
        with pytest.raises(MarketDataError, match="unusable rows"):
            _normalize_ohlcv(df, "SPY")

    def test_numeric_coercion(self):
        df = pd.DataFrame(
            {
                "Date": ["2023-01-03"],
                "Open": ["100.5"],
                "High": ["102.0"],
                "Low": ["99.0"],
                "Close": ["101.5"],
                "Volume": ["1000000"],
            }
        )
        result = _normalize_ohlcv(df, "SPY")
        assert result["Open"].iloc[0] == 100.5
        assert result["Volume"].iloc[0] == 1000000


class TestResolveDateRange:
    def test_with_start_and_end(self):
        start, end = _resolve_date_range("2023-01-01", "2023-12-31", None)
        assert start == date(2023, 1, 1)
        assert end == date(2023, 12, 31)

    def test_without_start_uses_period(self):
        start, end = _resolve_date_range(None, None, "1y")
        assert (end - start).days <= 366

    def test_default_period_is_1y(self):
        start, end = _resolve_date_range(None, None, None)
        assert (end - start).days <= 366


class TestPeriodToDays:
    def test_known_periods(self):
        assert _period_to_days("1d") == 1
        assert _period_to_days("1y") == 366
        assert _period_to_days("5y") == 366 * 5

    def test_unknown_period_defaults_to_1y(self):
        assert _period_to_days("invalid") == 366


class TestStooqSymbol:
    def test_plain_symbol(self):
        assert _to_stooq_symbol("SPY") == "spy.us"

    def test_already_has_suffix(self):
        assert _to_stooq_symbol("spy.us") == "spy.us"

    def test_lowercase(self):
        assert _to_stooq_symbol("AAPL") == "aapl.us"

    def test_dot_to_dash(self):
        assert _to_stooq_symbol("BRK.B") == "brk-b.us"

    def test_empty_raises_error(self):
        with pytest.raises(MarketDataError, match="cannot be empty"):
            _to_stooq_symbol("")


class TestStockAnalysisSymbol:
    def test_valid_symbol(self):
        assert _to_stockanalysis_symbol("AAPL") == "aapl"

    def test_dot_to_dash(self):
        assert _to_stockanalysis_symbol("BRK.B") == "brk-b"

    def test_empty_raises_error(self):
        with pytest.raises(MarketDataError, match="cannot be empty"):
            _to_stockanalysis_symbol("")

    def test_invalid_chars_raises_error(self):
        with pytest.raises(MarketDataError, match="not a valid ticker"):
            _to_stockanalysis_symbol("AAPL!!!")


class TestDisplayStockAnalysisSymbol:
    def test_dollar_prefix(self):
        assert _display_stockanalysis_symbol("$AAPL") == "AAPL"

    def test_exchange_format(self):
        assert _display_stockanalysis_symbol("@NYSE/BRK.A") == "NYSE:BRK.A"

    def test_no_prefix(self):
        assert _display_stockanalysis_symbol("AAPL") == "AAPL"


class TestStockAnalysisHistoryUrl:
    def test_plain_symbol(self):
        url = _stockanalysis_history_url("AAPL")
        assert "aapl" in url.lower()
        assert "stockanalysis.com" in url

    def test_symbol_with_dot(self):
        url = _stockanalysis_history_url("BRK.B")
        assert "brk-b" in url
