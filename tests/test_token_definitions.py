"""Tests for token_definitions module."""

import pytest
from utils.token_definitions import (
    SYMBOL_TOKENS,
    TIMEFRAME_TOKENS,
    STATE_TOKENS,
    VOLUME_TOKENS,
    HA_TOKENS,
    STO_TOKENS,
    RSI_TOKENS,
    MACD_TOKENS,
    BB_TOKENS,
    MA_CROSS_TOKENS,
    VOLATILITY_TOKENS,
    CANDLE_TOKENS,
    OBV_TOKENS,
    ATR_TOKENS,
    PRICE_ACTION_TOKENS,
    MARKET_CONTEXT_TOKENS,
    SENTIMENT_TOKENS,
    NEWS_TOKENS,
    SOCIAL_TOKENS,
    RELATIVE_TOKENS,
    INDICATOR_TOKENS,
    ACTION_TOKENS,
    ALL_CUSTOM_TOKENS,
    TOKEN_TO_ID,
    ID_TO_TOKEN,
    get_symbol_token,
    get_timeframe_token,
    validate_sequence,
    get_token_category,
)


class TestTokenCounts:
    def test_symbol_tokens_count(self):
        assert len(SYMBOL_TOKENS) == 50

    def test_timeframe_tokens_count(self):
        assert len(TIMEFRAME_TOKENS) == 10

    def test_state_tokens_count(self):
        assert len(STATE_TOKENS) == 7

    def test_volume_tokens_count(self):
        assert len(VOLUME_TOKENS) == 5

    def test_ha_tokens_count(self):
        assert len(HA_TOKENS) == 5

    def test_sto_tokens_count(self):
        assert len(STO_TOKENS) == 5

    def test_rsi_tokens_count(self):
        assert len(RSI_TOKENS) == 5

    def test_macd_tokens_count(self):
        assert len(MACD_TOKENS) == 7

    def test_bb_tokens_count(self):
        assert len(BB_TOKENS) == 7

    def test_ma_cross_tokens_count(self):
        assert len(MA_CROSS_TOKENS) == 5

    def test_volatility_tokens_count(self):
        assert len(VOLATILITY_TOKENS) == 3

    def test_candle_tokens_count(self):
        assert len(CANDLE_TOKENS) == 11

    def test_obv_tokens_count(self):
        assert len(OBV_TOKENS) == 5

    def test_atr_tokens_count(self):
        assert len(ATR_TOKENS) == 4

    def test_price_action_tokens_count(self):
        assert len(PRICE_ACTION_TOKENS) == 9

    def test_market_context_tokens_count(self):
        assert len(MARKET_CONTEXT_TOKENS) == 7

    def test_sentiment_tokens_count(self):
        assert len(SENTIMENT_TOKENS) == 6

    def test_news_tokens_count(self):
        assert len(NEWS_TOKENS) == 6

    def test_social_tokens_count(self):
        assert len(SOCIAL_TOKENS) == 4

    def test_relative_tokens_count(self):
        assert len(RELATIVE_TOKENS) == 4

    def test_action_tokens_count(self):
        assert len(ACTION_TOKENS) == 3

    def test_total_custom_tokens(self):
        expected = (
            len(SYMBOL_TOKENS)
            + len(TIMEFRAME_TOKENS)
            + len(INDICATOR_TOKENS)
            + len(ACTION_TOKENS)
        )
        assert len(ALL_CUSTOM_TOKENS) == expected


class TestTokenMappings:
    def test_get_symbol_token_valid(self):
        assert get_symbol_token("SPY") == "<SYM_SPY>"
        assert get_symbol_token("QQQ") == "<SYM_QQQ>"

    def test_get_symbol_token_case_insensitive(self):
        assert get_symbol_token("spy") == "<SYM_SPY>"
        assert get_symbol_token("Spy") == "<SYM_SPY>"

    def test_get_symbol_token_unknown_defaults_to_spy(self):
        assert get_symbol_token("UNKNOWN") == "<SYM_SPY>"

    def test_get_timeframe_token_valid(self):
        assert get_timeframe_token("DAILY") == "<TF_DAILY>"
        assert get_timeframe_token("WEEKLY") == "<TF_WEEKLY>"

    def test_get_timeframe_token_unknown_defaults_to_daily(self):
        assert get_timeframe_token("UNKNOWN") == "<TF_DAILY>"

    def test_to_token_id_and_back(self):
        for token in ALL_CUSTOM_TOKENS:
            idx = TOKEN_TO_ID[token]
            assert ID_TO_TOKEN[idx] == token
            assert isinstance(idx, int)

    def test_all_tokens_have_unique_ids(self):
        assert len(set(TOKEN_TO_ID.values())) == len(TOKEN_TO_ID)


class TestValidateSequence:
    def test_valid_sequence(self):
        seq = (
            "<SYM_SPY> <TF_DAILY> "
            "TR_Up VOL_Normal HA_Neutral STO_NoCross "
            "RSI_Bullish MACD_Positive BB_Neutral MA_Bullish "
            "VIX_Moderate CP_NoPattern OBV_Rising ATR_Rising "
            "PA_RangeBound MKT_Neutral SENT_Neutral NEWS_None "
            "SOC_Low REL_Inline HOLD"
        )
        assert validate_sequence(seq) is True

    def test_invalid_token_in_sequence(self):
        seq = "<SYM_SPY> INVALID_TOKEN BUY"
        assert validate_sequence(seq) is False

    def test_empty_string(self):
        assert validate_sequence("") is False

    def test_partially_valid(self):
        seq = "<SYM_SPY> BUY INVALID"
        assert validate_sequence(seq) is False


class TestGetTokenCategory:
    def test_symbol_category(self):
        assert get_token_category("<SYM_SPY>") == "symbol"

    def test_trend_category(self):
        assert get_token_category("TR_Up") == "trend"

    def test_action_category(self):
        assert get_token_category("BUY") == "action"

    def test_rsi_category(self):
        assert get_token_category("RSI_Oversold") == "rsi"

    def test_unknown_category(self):
        assert get_token_category("UNKNOWN") == "unknown"


class TestActionTokens:
    def test_action_tokens(self):
        assert "BUY" in ACTION_TOKENS
        assert "SELL" in ACTION_TOKENS
        assert "HOLD" in ACTION_TOKENS
        assert len(ACTION_TOKENS) == 3
