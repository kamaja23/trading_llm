"""Tests for data_generator module."""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from utils.data_generator import (
    generate_token_sequences,
    split_sequences,
    save_sequences,
    load_sequences,
    analyze_sequences,
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


class TestGenerateTokenSequences:
    def test_returns_list_of_strings(self, sample_ohlcv):
        sequences = generate_token_sequences(sample_ohlcv, "SPY", "DAILY")
        assert isinstance(sequences, list)
        assert len(sequences) > 0
        assert all(isinstance(s, str) for s in sequences)

    def test_sequence_format(self, sample_ohlcv):
        sequences = generate_token_sequences(sample_ohlcv, "SPY", "DAILY")
        for seq in sequences:
            tokens = seq.split()
            # 17 context tokens + 1 action = 18 tokens (without sentiment)
            assert len(tokens) == 18, f"Expected 18 tokens, got {len(tokens)}: {seq}"

    def test_first_token_is_symbol(self, sample_ohlcv):
        sequences = generate_token_sequences(sample_ohlcv, "SPY", "DAILY")
        for seq in sequences:
            tokens = seq.split()
            assert tokens[0] == "<SYM_SPY>"

    def test_second_token_is_timeframe(self, sample_ohlcv):
        sequences = generate_token_sequences(sample_ohlcv, "SPY", "DAILY")
        for seq in sequences:
            tokens = seq.split()
            assert tokens[1] == "<TF_DAILY>"

    def test_last_token_is_action(self, sample_ohlcv):
        sequences = generate_token_sequences(sample_ohlcv, "SPY", "DAILY")
        for seq in sequences:
            tokens = seq.split()
            assert tokens[-1] in {"BUY", "SELL", "HOLD"}

    def test_different_symbols(self, sample_ohlcv):
        spy_seqs = generate_token_sequences(sample_ohlcv, "SPY", "DAILY")
        qqq_seqs = generate_token_sequences(sample_ohlcv, "QQQ", "DAILY")
        assert spy_seqs[0].startswith("<SYM_SPY>")
        assert qqq_seqs[0].startswith("<SYM_QQQ>")

    def test_rsi_token_present(self, sample_ohlcv):
        sequences = generate_token_sequences(sample_ohlcv, "SPY", "DAILY")
        rsi_tokens = {"RSI_Oversold", "RSI_Overbought", "RSI_Bullish", "RSI_Bearish", "RSI_Neutral"}
        for seq in sequences[:5]:
            tokens = seq.split()
            assert tokens[6] in rsi_tokens, f"Expected RSI token at position 6, got {tokens[6]}"

    def test_macd_token_present(self, sample_ohlcv):
        sequences = generate_token_sequences(sample_ohlcv, "SPY", "DAILY")
        macd_tokens = {"MACD_BullishCross", "MACD_BearishCross", "MACD_Positive", "MACD_Negative",
                       "MACD_Neutral", "MACD_DivBullish", "MACD_DivBearish"}
        for seq in sequences[:5]:
            tokens = seq.split()
            assert tokens[7] in macd_tokens, f"Expected MACD token at position 7, got {tokens[7]}"


class TestSplitSequences:
    @pytest.fixture
    def sequences(self):
        return [f"tok{i}" for i in range(100)]

    def test_default_split_ratios(self, sequences):
        train, val, test = split_sequences(sequences)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_custom_split_ratios(self, sequences):
        train, val, test = split_sequences(sequences, train_ratio=0.7, val_ratio=0.15)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_chronological_order(self, sequences):
        train, val, test = split_sequences(sequences)
        assert train == sequences[:80]
        assert val == sequences[80:90]
        assert test == sequences[90:]

    def test_empty_sequences(self):
        train, val, test = split_sequences([])
        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0

    def test_single_sequence(self):
        train, val, test = split_sequences(["seq1"])
        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 1


class TestSaveAndLoadSequences:
    def test_roundtrip(self):
        sequences = ["seq one", "seq two", "seq three"]
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            tmp_path = f.name

        try:
            save_sequences(sequences, tmp_path)
            loaded = load_sequences(tmp_path)
            assert loaded == sequences
        finally:
            os.unlink(tmp_path)

    def test_save_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "file.txt")
            save_sequences(["test"], path)
            assert os.path.exists(path)

    def test_load_empty_file_returns_empty_list(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n\n")
            tmp_path = f.name

        try:
            loaded = load_sequences(tmp_path)
            assert loaded == []
        finally:
            os.unlink(tmp_path)


class TestAnalyzeSequences:
    def test_empty_sequences_does_not_raise(self):
        analyze_sequences([])

    def test_valid_sequences(self):
        sequences = [
            "<SYM_SPY> <TF_DAILY> TR_Up VOL_Normal HA_Neutral STO_NoCross "
            "RSI_Bullish MACD_Positive BB_Neutral MA_Bullish "
            "VIX_Moderate CP_NoPattern OBV_Rising ATR_Rising "
            "PA_RangeBound MKT_Neutral REL_Inline BUY",
            "<SYM_QQQ> <TF_DAILY> TR_Down VOL_High HA_DownCross STO_Oversold "
            "RSI_Oversold MACD_Negative BB_LowerTouch MA_Bearish "
            "VIX_High CP_Hammer OBV_Falling ATR_High "
            "PA_SupportTest MKT_Bearish REL_Weak SELL",
        ]
        analyze_sequences(sequences)
