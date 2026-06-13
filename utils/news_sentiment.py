"""
News and social media sentiment data sources.

Provides functions to fetch news headlines, social media activity levels,
and compute sentiment tokens for the trading LLM vocabulary.

All providers degrade gracefully when API keys are missing.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from utils.token_definitions import (
    SENTIMENT_TOKENS,
    NEWS_TOKENS,
    SOCIAL_TOKENS,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

NEWSAPI_URL = "https://newsapi.org/v2/everything"
REDDIT_CLIENT_ID_ENV = "REDDIT_CLIENT_ID"
REDDIT_CLIENT_SECRET_ENV = "REDDIT_CLIENT_SECRET"
NEWSAPI_KEY_ENV = "NEWSAPI_API_KEY"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

FINBERT_URL = "https://api.monoxor.com/finbert/analyze"

STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"


class SentimentError(Exception):
    """Raised when a sentiment data source fails."""


def _get_config_value(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value:
        return value
    env_values = _read_dotenv(ENV_PATH)
    return env_values.get(name)


def _read_dotenv(path: str) -> dict[str, str]:
    if not os.path.exists(path):
        return {}
    values = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                values[key] = value
    return values


def _download(url: str, timeout: int = 10) -> bytes:
    request = Request(url, headers={"User-Agent": "tradebot/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except Exception as exc:
        raise SentimentError(f"Sentiment request failed: {exc}") from exc


def _simple_sentiment_score(text: str) -> float:
    """
    Simple lexicon-based sentiment scoring.
    Returns a score from -1 (very negative) to +1 (very positive).
    """
    positive_words = {
        "up", "gain", "profit", "bullish", "growth", "surge", "rally",
        "beat", "upgrade", "positive", "strong", "record", "boom",
        "opportunity", "outperform", "breakthrough", "innovation",
        "expansion", "momentum", "confidence", "optimistic",
        "soar", "climb", "jump", "rise", "recover", "improve",
        "dividend", "buyback", "partnership", "launch", "approval",
    }
    negative_words = {
        "down", "loss", "decline", "bearish", "fall", "drop", "crash",
        "miss", "downgrade", "negative", "weak", "debt", "risk",
        "lawsuit", "investigation", "fraud", "scandal", "volatile",
        "uncertainty", "recession", "inflation", "slowdown",
        "plunge", "slump", "tumble", "sink", "cut", "layoff",
        "firing", "ban", "fine", "penalty", "sanction", "default",
    }

    words = set(re.findall(r"[a-z]+", text.lower()))
    pos_count = len(words & positive_words)
    neg_count = len(words & negative_words)
    total = pos_count + neg_count

    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total


def _score_to_sentiment_token(score: float) -> str:
    """Convert a numeric sentiment score to a sentiment token."""
    if score > 0.5:
        return "SENT_StrongPos"
    elif score > 0.1:
        return "SENT_Positive"
    elif score < -0.5:
        return "SENT_StrongNeg"
    elif score < -0.1:
        return "SENT_Negative"
    return "SENT_Neutral"


def _score_to_news_token(score: float) -> str:
    """Convert a sentiment score to a news impact token."""
    if score > 0.5:
        return "NEWS_MajorPos"
    elif score > 0.1:
        return "NEWS_MinorPos"
    elif score < -0.5:
        return "NEWS_MajorNeg"
    elif score < -0.1:
        return "NEWS_MinorNeg"
    return "NEWS_Neutral"


def fetch_newsapi_sentiment(
    symbol: str,
    days_back: int = 7,
    timeout: int = 10,
) -> dict:
    """
    Fetch news headlines via NewsAPI and compute aggregate sentiment.

    Args:
        symbol: Ticker symbol (e.g., 'AAPL')
        days_back: How many days of news to fetch
        timeout: Request timeout

    Returns:
        Dict with 'sentiment_token', 'news_token', 'headline_count', 'avg_score'
        or None if unavailable.
    """
    api_key = _get_config_value(NEWSAPI_KEY_ENV)
    if not api_key:
        return None

    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "q": symbol,
        "from": from_date,
        "sortBy": "publishedAt",
        "pageSize": 20,
        "language": "en",
        "apiKey": api_key,
    }

    try:
        payload = _download(f"{NEWSAPI_URL}?{urlencode(params)}", timeout)
        data = json.loads(payload.decode("utf-8"))

        if data.get("status") != "ok":
            return None

        articles = data.get("articles", [])
        if not articles:
            return {"sentiment_token": "SENT_NoData", "news_token": "NEWS_None", "headline_count": 0, "avg_score": 0.0}

        scores = []
        for article in articles:
            title = article.get("title", "") or ""
            desc = article.get("description", "") or ""
            text = f"{title} {desc}"
            scores.append(_simple_sentiment_score(text))

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "sentiment_token": _score_to_sentiment_token(avg_score),
            "news_token": _score_to_news_token(avg_score),
            "headline_count": len(articles),
            "avg_score": avg_score,
        }

    except (SentimentError, json.JSONDecodeError, Exception):
        return None


def fetch_gdelt_sentiment(
    symbol: str,
    days_back: int = 3,
    timeout: int = 15,
) -> Optional[dict]:
    """
    Fetch news sentiment from GDELT Project (free, no API key needed).

    Args:
        symbol: Ticker symbol
        days_back: How many days back to search
        timeout: Request timeout

    Returns:
        Dict with sentiment tokens or None
    """
    try:
        company_name = symbol
        params = {
            "query": f"{symbol} stock",
            "mode": "artlist",
            "maxrecords": 10,
            "format": "json",
            "timespan": f"{days_back}d",
        }
        payload = _download(f"{GDELT_URL}?{urlencode(params)}", timeout)
        data = json.loads(payload.decode("utf-8"))

        articles = data.get("articles", []) or data.get("results", [])
        if not articles:
            return None

        scores = []
        for article in articles:
            title = article.get("title", "") or ""
            text = f"{title}"
            scores.append(_simple_sentiment_score(text))

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "sentiment_token": _score_to_sentiment_token(avg_score),
            "news_token": _score_to_news_token(avg_score),
            "headline_count": len(articles),
            "avg_score": avg_score,
        }

    except Exception:
        return None


def fetch_stocktwits_sentiment(
    symbol: str,
    timeout: int = 10,
) -> Optional[dict]:
    """
    Fetch social media sentiment from StockTwits (free, no API key for basic access).

    Args:
        symbol: Ticker symbol
        timeout: Request timeout

    Returns:
        Dict with social sentiment tokens or None
    """
    try:
        payload = _download(STOCKTWITS_URL.format(symbol=symbol.upper()), timeout)
        data = json.loads(payload.decode("utf-8"))

        if data.get("response", {}).get("status") != 200:
            return None

        messages = data.get("messages", [])
        if not messages:
            return {"social_token": "SOC_Silent", "sentiment_token": "SENT_NoData", "message_count": 0}

        bullish_count = 0
        bearish_count = 0
        total = 0

        for msg in messages:
            entities = msg.get("entities", {})
            sentiment_data = entities.get("sentiment", {})
            if sentiment_data.get("basic"):
                total += 1
                if sentiment_data["basic"] == "Bullish":
                    bullish_count += 1
                elif sentiment_data["basic"] == "Bearish":
                    bearish_count += 1

        activity_level = len(messages)

        if activity_level > 30:
            social_token = "SOC_HighBuzz"
        elif activity_level > 10:
            social_token = "SOC_Moderate"
        elif activity_level > 0:
            social_token = "SOC_Low"
        else:
            social_token = "SOC_Silent"

        if total > 0:
            sentiment_ratio = (bullish_count - bearish_count) / total
            sentiment_token = _score_to_sentiment_token(sentiment_ratio)
        else:
            sentiment_token = "SENT_Neutral"

        return {
            "social_token": social_token,
            "sentiment_token": sentiment_token,
            "message_count": activity_level,
        }

    except Exception:
        return None


def fetch_aggregated_sentiment(
    symbol: str,
    days_back: int = 3,
) -> dict:
    """
    Fetch sentiment from all available sources and aggregate.

    Tries each provider in order and returns the best available data.
    All tokens default to 'NoData' / 'None' if no provider works.

    Returns:
        Dict with 'sentiment_token', 'news_token', 'social_token' keys.
    """
    result = {
        "sentiment_token": "SENT_NoData",
        "news_token": "NEWS_None",
        "social_token": "SOC_Silent",
    }

    newsapi = fetch_newsapi_sentiment(symbol, days_back=days_back)
    if newsapi:
        result["sentiment_token"] = newsapi.get("sentiment_token", "SENT_Neutral")
        result["news_token"] = newsapi.get("news_token", "NEWS_Neutral")
        return result

    gdelt = fetch_gdelt_sentiment(symbol, days_back=days_back)
    if gdelt:
        result["sentiment_token"] = gdelt.get("sentiment_token", "SENT_Neutral")
        result["news_token"] = gdelt.get("news_token", "NEWS_Neutral")
        return result

    stocktwits = fetch_stocktwits_sentiment(symbol)
    if stocktwits:
        result["sentiment_token"] = stocktwits.get("sentiment_token", result["sentiment_token"])
        result["social_token"] = stocktwits.get("social_token", "SOC_Silent")

    return result


if __name__ == "__main__":
    symbol = "AAPL"
    print(f"Fetching sentiment for {symbol}...")

    result = fetch_aggregated_sentiment(symbol)
    print(f"  Sentiment: {result['sentiment_token']}")
    print(f"  News: {result['news_token']}")
    print(f"  Social: {result['social_token']}")

    newsapi = fetch_newsapi_sentiment(symbol)
    if newsapi:
        print(f"\nNewsAPI: {newsapi['headline_count']} headlines, avg score: {newsapi['avg_score']:.2f}")
    else:
        print("\nNewsAPI: Not available (no API key or error)")

    stocktwits = fetch_stocktwits_sentiment(symbol)
    if stocktwits:
        print(f"StockTwits: {stocktwits['message_count']} messages, social: {stocktwits['social_token']}")
    else:
        print("StockTwits: Not available")

    gdelt = fetch_gdelt_sentiment(symbol)
    if gdelt:
        print(f"GDELT: {gdelt['headline_count']} articles, score: {gdelt['avg_score']:.2f}")
    else:
        print("GDELT: Not available")
