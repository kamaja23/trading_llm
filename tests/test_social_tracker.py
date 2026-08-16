"""Tests for the social_tracker module."""

import pytest

from utils.social_tracker import (
    _no_data_token,
    _parse_datetime,
    _score_to_social_token,
    _simple_sentiment_score,
    add_user_figure,
    all_figures,
    compute_social_indicator,
    figure_by_name,
    figure_by_prefix,
    figure_for_ticker,
    load_figures,
    load_user_figures,
    remove_user_figure,
    score_posts,
    social_token_for_date,
    suggested_figures,
    suggest_figures,
)


class TestFigures:
    def test_load_figures(self):
        figures = load_figures()
        assert len(figures) >= 2
        prefixes = {f["prefix"] for f in figures}
        assert "MUSK" in prefixes
        assert "TRUMP" in prefixes

    def test_figure_by_name(self):
        fig = figure_by_name("Elon Musk")
        assert fig is not None
        assert fig["prefix"] == "MUSK"

    def test_figure_by_prefix(self):
        fig = figure_by_prefix("TRUMP")
        assert fig is not None
        assert fig["name"] == "Donald Trump"

    def test_figure_for_ticker(self):
        musk_figs = figure_for_ticker("TSLA")
        assert any(f["prefix"] == "MUSK" for f in musk_figs)
        trump_figs = figure_for_ticker("SPY")
        assert any(f["prefix"] == "TRUMP" for f in trump_figs)


class TestTokens:
    def test_score_to_social_token(self):
        assert _score_to_social_token(0.7, "MUSK") == "SOC_MUSK_StrongPos"
        assert _score_to_social_token(0.3, "TRUMP") == "SOC_TRUMP_Positive"
        assert _score_to_social_token(0.0, "MUSK") == "SOC_MUSK_Neutral"
        assert _score_to_social_token(-0.3, "TRUMP") == "SOC_TRUMP_Negative"
        assert _score_to_social_token(-0.7, "MUSK") == "SOC_MUSK_StrongNeg"

    def test_no_data_token(self):
        assert _no_data_token("MUSK") == "SOC_MUSK_NoData"
        assert _no_data_token("trump") == "SOC_TRUMP_NoData"


class TestScoring:
    def test_simple_sentiment_score(self):
        assert _simple_sentiment_score("up gain profit bullish") > 0
        assert _simple_sentiment_score("down loss crash bearish") < 0
        assert _simple_sentiment_score("a b c") == 0.0

    def test_score_posts_no_relevant(self):
        figure = figure_by_name("Elon Musk")
        posts = [{"text": "hello world this is unrelated", "created_at": None}]
        result = score_posts(posts, "TSLA", figure)
        assert result["count"] == 0
        assert result["token"] == "SOC_MUSK_NoData"

    def test_score_posts_relevant_positive(self):
        figure = figure_by_name("Elon Musk")
        posts = [
            {"text": "Tesla cybertruck launch is a huge win, profits up", "created_at": None},
            {"text": "Tesla breakout rally, strong demand", "created_at": None},
        ]
        result = score_posts(posts, "TSLA", figure)
        assert result["count"] == 2
        assert result["score"] > 0
        assert result["token"] in {"SOC_MUSK_Positive", "SOC_MUSK_StrongPos"}

    def test_score_posts_relevant_negative(self):
        figure = figure_by_name("Elon Musk")
        posts = [
            {"text": "Tesla crash, huge losses and layoffs", "created_at": None},
            {"text": "Tesla fraud investigation, stock down", "created_at": None},
        ]
        result = score_posts(posts, "TSLA", figure)
        assert result["count"] == 2
        assert result["score"] < 0


class TestIndicator:
    def test_compute_social_indicator_returns_token(self):
        result = compute_social_indicator("TSLA")
        assert "token" in result
        assert result["token"].startswith("SOC_")
        assert result["token"] in {
            "SOC_MUSK_StrongPos", "SOC_MUSK_Positive", "SOC_MUSK_Neutral",
            "SOC_MUSK_Negative", "SOC_MUSK_StrongNeg", "SOC_MUSK_NoData",
        }

    def test_compute_social_indicator_untracked(self):
        result = compute_social_indicator("UNKNOWN")
        assert "token" in result
        assert result["token"].startswith("SOC_")

    def test_social_token_for_date(self):
        token = social_token_for_date("TSLA", "2024-01-15")
        assert token == "SOC_MUSK_NoData"
        token2 = social_token_for_date("SPY", "2024-01-15")
        assert token2 == "SOC_TRUMP_NoData"


class TestDatetime:
    def test_parse_datetime_iso(self):
        dt = _parse_datetime("2024-01-15T12:30:00.000Z")
        assert dt is not None
        assert dt.year == 2024

    def test_parse_datetime_rfc822(self):
        dt = _parse_datetime("Mon, 15 Jan 2024 12:30:00 +0000")
        assert dt is not None
        assert dt.year == 2024

    def test_parse_datetime_invalid(self):
        assert _parse_datetime("not a date") is None
        assert _parse_datetime(None) is None


class TestUserFigures:
    @pytest.fixture(autouse=True)
    def _clean_user_figures(self):
        for fig in load_user_figures():
            remove_user_figure(fig["name"])
        yield
        for fig in load_user_figures():
            remove_user_figure(fig["name"])

    def test_add_and_remove_user_figure(self):
        result = add_user_figure({
            "name": "Bill Ackman",
            "prefix": "ACKMAN",
            "x": "BillAckman",
            "tickers": ["SPY", "AAPL"],
        })
        assert result["ok"]
        user = load_user_figures()
        assert len(user) == 1
        assert user[0]["prefix"] == "ACKMAN"
        assert user[0]["added_by_user"] is True
        merged = all_figures()
        assert any(f["prefix"] == "ACKMAN" for f in merged)

        assert remove_user_figure("Bill Ackman") is True
        assert not load_user_figures()

    def test_add_user_figure_requires_name(self):
        result = add_user_figure({"prefix": "X", "x": "h", "tickers": ["SPY"]})
        assert not result["ok"]

    def test_add_user_figure_requires_prefix(self):
        result = add_user_figure({"name": "Foo", "x": "h", "tickers": ["SPY"]})
        assert not result["ok"]

    def test_add_user_figure_requires_handle(self):
        result = add_user_figure({"name": "Foo", "prefix": "FOO", "tickers": ["SPY"]})
        assert not result["ok"]

    def test_add_user_figure_requires_ticker(self):
        result = add_user_figure({"name": "Foo", "prefix": "FOO", "x": "foo"})
        assert not result["ok"]

    def test_add_user_figure_rejects_invalid_prefix(self):
        result = add_user_figure({"name": "Foo", "prefix": "FOO BAR", "x": "h", "tickers": ["SPY"]})
        assert not result["ok"]

    def test_add_user_figure_rejects_duplicate_prefix(self):
        add_user_figure({"name": "Foo", "prefix": "ACKMAN", "x": "h", "tickers": ["SPY"]})
        result = add_user_figure({"name": "Bar", "prefix": "ACKMAN", "x": "h2", "tickers": ["AAPL"]})
        assert not result["ok"]

    def test_figure_for_ticker_includes_user_figures(self):
        add_user_figure({"name": "Cathie Wood", "prefix": "WOOD", "x": "CathieDWood", "tickers": ["NVDA"]})
        figs = figure_for_ticker("NVDA")
        assert any(f["prefix"] == "WOOD" for f in figs)

    def test_dynamic_token_available_for_user_figure(self):
        from utils.token_definitions import dynamic_figure_social_tokens
        add_user_figure({"name": "Cathie Wood", "prefix": "WOOD", "x": "CathieDWood", "tickers": ["NVDA"]})
        dynamic = dynamic_figure_social_tokens()
        assert "SOC_WOOD_NoData" in dynamic


class TestSuggestions:
    @pytest.fixture(autouse=True)
    def _clean_user_figures(self):
        for fig in load_user_figures():
            remove_user_figure(fig["name"])
        yield
        for fig in load_user_figures():
            remove_user_figure(fig["name"])

    def test_suggested_figures(self):
        suggestions = suggested_figures()
        assert len(suggestions) > 0
        for suggestion in suggestions:
            assert suggestion.get("tickers")
            assert suggestion.get("reason")
            assert suggestion.get("is_suggestion") is True

    def test_suggestions_rank_by_watchlist(self):
        ranked = suggest_figures(["NVDA", "MSFT", "AAPL"])
        top = ranked[0]
        assert any(t in top["tickers"] for t in ["NVDA", "MSFT", "AAPL"])

    def test_suggestions_exclude_tracked(self):
        add_user_figure({
            "name": "Jensen Huang",
            "prefix": "HUANG",
            "x": "nvidia",
            "tickers": ["NVDA"],
        })
        ranked = suggest_figures(["NVDA"])
        assert all(s["name"] != "Jensen Huang" for s in ranked)
