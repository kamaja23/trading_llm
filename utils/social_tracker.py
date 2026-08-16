"""
Social media tracker for influential figures (X / Truth Social).

Fetches recent posts from public, free data sources, scores them with a
lexicon-based sentiment model plus per-ticker keyword relevance, and
produces figure-specific social sentiment tokens for the trading model.

All providers degrade gracefully: when a source is unreachable the tracker
returns a ``NoData`` token so the rest of the app keeps working.  Fetches
are cached in memory (TTL) and persisted to a JSONL archive under
``data/social/`` so the archive grows over time and improves future retrains.
"""

from __future__ import annotations

import email.utils
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from utils.news_sentiment import SentimentError, _simple_sentiment_score

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "social_figures.json"
USER_FIGURES_PATH = PROJECT_ROOT / "data" / "user_figures.json"
ARCHIVE_DIR = PROJECT_ROOT / "data" / "social"
ENV_PATH = PROJECT_ROOT / ".env"

TRUTHSOCIAL_API = "https://truthsocial.com/api/v1"
TRUTHSOCIAL_TOKEN_ENV = "TRUTHSOCIAL_TOKEN"
NITTER_INSTANCES_ENV = "NITTER_INSTANCES"
X_SYNDICATION_URL = "https://syndication.twitter.com/srv/timeline-profile/screen-name/{user}"
DEFAULT_NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.poast.org",
    "https://nitter.privacydev.net",
]

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# How many days of recent posts count as "current" for indicator purposes
POST_LOOKBACK_DAYS = 14
# In-memory cache TTL in seconds
TTL_SECONDS = 600

_DEFAULT_FIGURES = [
    {
        "name": "Elon Musk",
        "prefix": "MUSK",
        "x": "elonmusk",
        "truth": "elonmusk",
        "tickers": ["TSLA", "DOGE", "BTC"],
        "keywords": {
            "TSLA": ["tesla", "tsla", "cybertruck", "model s", "model x", "model 3", "model y", "roadster", "full self driving", "fsd", "robotaxi", "gigafactory", "autopilot"],
            "DOGE": ["dogecoin", "doge"],
            "BTC": ["bitcoin", "btc", "crypto"],
        },
    },
    {
        "name": "Donald Trump",
        "prefix": "TRUMP",
        "x": "realDonaldTrump",
        "truth": "realDonaldTrump",
        "tickers": ["DJT", "SPY", "META", "T", "MSFT"],
        "keywords": {
            "DJT": ["djt", "truth social", "trump media"],
            "SPY": ["stock market", "stocks", "wall street", "tariff", "tariffs", "fed", "economy", "recession", "jobs"],
            "META": ["meta", "facebook", "instagram"],
            "T": ["at&t", "att", "merger"],
            "MSFT": ["microsoft", "msft"],
        },
    },
]

_cache: Dict[str, Tuple[float, List[dict]]] = {}


# ── Suggestions catalog ──────────────────────────────────────────────

# Curated catalog of notable figures users may want to track.  Each entry
# maps a figure to the tickers their statements most often move.
_SUGGESTED_FIGURES = [
    {
        "name": "Warren Buffett",
        "prefix": "BUFFETT",
        "x": "WarrenBuffett",
        "truth": None,
        "tickers": ["BRK_B", "SPY", "AAPL", "BAC"],
        "reason": "Berkshire chairman; comments sway banks, Apple and broad markets.",
    },
    {
        "name": "Jamie Dimon",
        "prefix": "DIMON",
        "x": "jpmorgan",
        "truth": None,
        "tickers": ["JPM", "SPY", "BAC", "WFC"],
        "reason": "JPMorgan CEO; bank and economy commentary.",
    },
    {
        "name": "Jensen Huang",
        "prefix": "HUANG",
        "x": "nvidia",
        "truth": None,
        "tickers": ["NVDA", "AMD", "INTC"],
        "reason": "NVIDIA CEO; AI-chip commentary moves semiconductor names.",
    },
    {
        "name": "Michael Saylor",
        "prefix": "SAYLOR",
        "x": "saylor",
        "truth": None,
        "tickers": ["BTC", "MSTR", "COIN"],
        "reason": "MicroStrategy founder; prominent bitcoin advocate.",
    },
    {
        "name": "Peter Schiff",
        "prefix": "SCHIFF",
        "x": "PeterSchiff",
        "truth": None,
        "tickers": ["BTC", "GLD", "SLV"],
        "reason": "Gold bug; bitcoin/gold commentary.",
    },
    {
        "name": "Jerome Powell",
        "prefix": "POWELL",
        "x": "federalreserve",
        "truth": None,
        "tickers": ["SPY", "QQQ", "DIA", "TLT"],
        "reason": "Fed chair; rate decisions move the whole market.",
    },
    {
        "name": "Cathie Wood",
        "prefix": "WOOD",
        "x": "CathieDWood",
        "truth": None,
        "tickers": ["TSLA", "NVDA", "COIN", "SQ"],
        "reason": "ARK Invest CEO; disruptive-tech stock commentary.",
    },
    {
        "name": "Jim Cramer",
        "prefix": "CRAMER",
        "x": "jimcramer",
        "truth": None,
        "tickers": ["SPY", "AAPL", "NVDA", "TSLA"],
        "reason": "CNBC host; frequent stock calls on the broad market.",
    },
    {
        "name": "Vitalik Buterin",
        "prefix": "VITALIK",
        "x": "VitalikButerin",
        "truth": None,
        "tickers": ["ETH", "BTC", "COIN"],
        "reason": "Ethereum co-founder; crypto ecosystem commentary.",
    },
    {
        "name": "Mark Cuban",
        "prefix": "CUBAN",
        "x": "mcuban",
        "truth": None,
        "tickers": ["SPY", "AMZN", "GME"],
        "reason": "Investor; markets and meme-stock commentary.",
    },
    {
        "name": "Satya Nadella",
        "prefix": "NADELLA",
        "x": "satyanadella",
        "truth": None,
        "tickers": ["MSFT", "AI"],
        "reason": "Microsoft CEO; AI and enterprise-tech commentary.",
    },
    {
        "name": "Tim Cook",
        "prefix": "COOK",
        "x": "tim_cook",
        "truth": None,
        "tickers": ["AAPL", "SPY"],
        "reason": "Apple CEO; product and supply-chain commentary.",
    },
]

# Auto keywords per ticker used for suggestion defaults
_SUGGESTED_KEYWORDS = {
    "SPY": ["spy", "s&p 500", "s&p", "stock market", "stocks", "fed", "economy", "rates"],
    "QQQ": ["qqq", "nasdaq", "tech stocks"],
    "DIA": ["dow", "dow jones"],
    "TLT": ["bonds", "tlt", "yields", "treasuries"],
    "AAPL": ["apple", "aapl", "iphone"],
    "BAC": ["bank of america", "bac"],
    "WFC": ["wells fargo", "wfc"],
    "JPM": ["jpmorgan", "jpm", "jamie dimon"],
    "NVDA": ["nvidia", "nvda", "chips", "semiconductor", "ai"],
    "AMD": ["amd", "advanced micro"],
    "INTC": ["intel", "intc"],
    "BTC": ["bitcoin", "btc", "crypto", "cryptocurrency"],
    "ETH": ["ethereum", "eth", "crypto", "cryptocurrency"],
    "MSTR": ["microstrategy", "mstr"],
    "COIN": ["coinbase", "coin"],
    "GLD": ["gold", "gld"],
    "SLV": ["silver", "slv"],
    "TSLA": ["tesla", "tsla"],
    "SQ": ["block", "square", "sq"],
    "MSFT": ["microsoft", "msft"],
    "AI": ["artificial intelligence", "ai"],
    "BRK_B": ["berkshire", "brk"],
    "AMZN": ["amazon", "amzn"],
    "GME": ["gamestop", "gme"],
}


def suggested_figures() -> List[dict]:
    """Return the curated suggestion catalog (read-only copies)."""
    suggestions = []
    for fig in _SUGGESTED_FIGURES:
        item = dict(fig)
        keywords = {}
        for ticker in fig.get("tickers", []):
            keywords[ticker] = _SUGGESTED_KEYWORDS.get(ticker, [ticker.lower()])
        item["keywords"] = keywords
        item["is_suggestion"] = True
        suggestions.append(item)
    return suggestions


def suggest_figures(watchlist: Optional[List[str]] = None) -> List[dict]:
    """
    Recommend figures to track.

    Figures whose tickers overlap the user's watchlist are ranked first;
    the remaining catalog follows.  Figures already being tracked are
    filtered out.
    """
    watchlist = {(w or "").upper().strip() for w in (watchlist or [])}
    tracked = {fig.get("name", "").lower() for fig in all_figures()}

    ranked = []
    for fig in suggested_figures():
        if fig.get("name", "").lower() in tracked:
            continue
        overlap = [t for t in fig.get("tickers", []) if t in watchlist]
        ranked.append((len(overlap), fig))
    ranked.sort(key=lambda item: (-item[0], item[1].get("name", "")))
    return [fig for _, fig in ranked]


class SocialTrackerError(Exception):
    """Raised when social tracking data cannot be retrieved."""


# ── Config helpers ───────────────────────────────────────────────────


def _read_dotenv(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    values: Dict[str, str] = {}
    try:
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
    except OSError:
        pass
    return values


def _get_config_value(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value:
        return value
    return _read_dotenv(ENV_PATH).get(name)


def load_figures() -> List[dict]:
    """Load the tracked figures from config (falls back to defaults)."""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                return data
    except (OSError, json.JSONDecodeError):
        pass
    return _DEFAULT_FIGURES


def load_user_figures() -> List[dict]:
    """Load user-added figures persisted under data/user_figures.json."""
    if not USER_FIGURES_PATH.exists():
        return []
    try:
        with open(USER_FIGURES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return []


def save_user_figures(figures: List[dict]) -> None:
    """Persist user-added figures so tracking survives restarts."""
    USER_FIGURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = USER_FIGURES_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(figures, f, indent=2)
    os.replace(tmp, USER_FIGURES_PATH)


def all_figures() -> List[dict]:
    """Default + user-added figures (deduplicated by name)."""
    merged = {fig.get("name", "").lower(): fig for fig in load_figures()}
    for fig in load_user_figures():
        merged[fig.get("name", "").lower()] = fig
    return [fig for _, fig in merged.items()]


def add_user_figure(figure: dict) -> dict:
    """
    Add a figure to the user's tracked list and persist it.

    Validates required fields and prefix uniqueness.  Returns a result dict
    with ``ok``, ``error`` (if any) and the resulting figure.
    """
    name = (figure.get("name") or "").strip()
    prefix = (figure.get("prefix") or "").strip().upper()
    tickers = [t.upper().strip() for t in (figure.get("tickers") or []) if t.strip()]

    if not name:
        return {"ok": False, "error": "Name is required."}
    if not prefix or not prefix.isalnum():
        return {"ok": False, "error": "Prefix must be letters/numbers only (e.g. MUSK)."}
    if not (figure.get("x") or figure.get("truth")):
        return {"ok": False, "error": "Provide at least one X or Truth Social handle."}

    existing = all_figures()
    for fig in existing:
        if fig.get("prefix", "").upper() == prefix:
            return {"ok": False, "error": f"Prefix '{prefix}' is already used by {fig.get('name')}."}
        if fig.get("name", "").lower() == name.lower():
            return {"ok": False, "error": f"A figure named '{name}' is already tracked."}

    if not tickers:
        return {"ok": False, "error": "Add at least one ticker."}

    new_figure = {
        "name": name,
        "prefix": prefix,
        "x": (figure.get("x") or "").strip() or None,
        "truth": (figure.get("truth") or "").strip() or None,
        "tickers": tickers,
        "keywords": figure.get("keywords") or {t: [t.lower()] for t in tickers},
        "added_by_user": True,
    }

    figures = load_user_figures()
    figures.append(new_figure)
    save_user_figures(figures)
    return {"ok": True, "figure": new_figure}


def remove_user_figure(name: str) -> bool:
    """Remove a user-added figure by name; returns True if removed."""
    name = name.strip().lower()
    figures = load_user_figures()
    kept = [f for f in figures if f.get("name", "").strip().lower() != name]
    if len(kept) == len(figures):
        return False
    save_user_figures(kept)
    return True


def figure_by_name(name: str) -> Optional[dict]:
    for fig in all_figures():
        if fig.get("name", "").lower() == name.lower():
            return fig
    return None


def figure_by_prefix(prefix: str) -> Optional[dict]:
    for fig in all_figures():
        if fig.get("prefix", "").upper() == prefix.upper():
            return fig
    return None


def figure_for_ticker(ticker: str) -> List[dict]:
    """Return all configured figures that track the given ticker."""
    ticker = ticker.upper()
    return [fig for fig in all_figures() if ticker in fig.get("tickers", [])]


# ── HTTP helpers ─────────────────────────────────────────────────────


def _download(url: str, timeout: int = 12, headers: Optional[dict] = None) -> bytes:
    req_headers = {"User-Agent": USER_AGENT}
    if headers:
        req_headers.update(headers)
    request = Request(url, headers=req_headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except Exception as exc:
        raise SocialTrackerError(f"Social request failed: {exc}") from exc


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        pass
    try:
        return email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None


# ── Archive helpers ──────────────────────────────────────────────────


def _archive_path(figure: dict) -> Path:
    prefix = figure.get("prefix", "FIG").upper()
    path = ARCHIVE_DIR / f"{prefix}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_archive(figure: dict) -> List[dict]:
    path = _archive_path(figure)
    posts: List[dict] = []
    if not path.exists():
        return posts
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    posts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return posts


def _append_to_archive(figure: dict, posts: List[dict]) -> None:
    if not posts:
        return
    path = _archive_path(figure)
    existing = set()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        existing.add(json.loads(line).get("uri", ""))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
    with open(path, "a", encoding="utf-8") as f:
        for post in posts:
            uri = post.get("uri", "")
            if uri and uri in existing:
                continue
            if uri:
                existing.add(uri)
            f.write(json.dumps(post) + "\n")


# ── Truth Social (Mastodon-compatible public API) ────────────────────


def _truth_account_id(handle: str) -> Optional[str]:
    token = _get_config_value(TRUTHSOCIAL_TOKEN_ENV)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        url = f"{TRUTHSOCIAL_API}/accounts/lookup?acct={quote(handle)}"
        payload = _download(url, headers=headers)
        data = json.loads(payload.decode("utf-8"))
        return data.get("id")
    except (SocialTrackerError, json.JSONDecodeError, AttributeError):
        return None


def fetch_truth_social_posts(
    handle: str,
    limit: int = 20,
    source: str = "truth",
    figure: Optional[dict] = None,
) -> List[dict]:
    """Fetch recent posts from a Truth Social account via the public API."""
    account_id = _truth_account_id(handle)
    if not account_id:
        return []

    token = _get_config_value(TRUTHSOCIAL_TOKEN_ENV)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    params = urlencode({
        "limit": limit,
        "exclude_replies": "true",
        "exclude_reblogs": "true",
    })
    try:
        url = f"{TRUTHSOCIAL_API}/accounts/{quote(str(account_id))}/statuses?{params}"
        payload = _download(url, headers=headers)
        data = json.loads(payload.decode("utf-8"))
    except (SocialTrackerError, json.JSONDecodeError):
        return []

    posts = []
    for item in data or []:
        content = re.sub(r"<[^>]+>", " ", item.get("content", "") or "")
        text = re.sub(r"\s+", " ", content).strip()
        created_at = _parse_datetime(item.get("created_at"))
        posts.append({
            "source": source,
            "text": text,
            "created_at": created_at.isoformat() if created_at else None,
            "uri": item.get("uri") or item.get("id"),
            "url": item.get("url"),
            "figure": figure.get("name") if figure else None,
        })
    return posts


# ── X (unofficial backends, best-effort) ─────────────────────────────


def _x_via_nitter(user: str, instance: str, limit: int) -> List[dict]:
    url = f"{instance.rstrip('/')}/{quote(user)}/rss"
    payload = _download(url, timeout=10)
    text = payload.decode("utf-8", errors="replace")
    posts = []
    for item in re.findall(r"<item>(.*?)</item>", text, re.DOTALL):
        title_m = re.search(r"<title>(.*?)</title>", item, re.DOTALL)
        link_m = re.search(r"<link>(.*?)</link>", item, re.DOTALL)
        date_m = re.search(r"<pubDate>(.*?)</pubDate>", item, re.DOTALL)
        if not title_m:
            continue
        title = re.sub(r"\s+", " ", title_m.group(1)).strip()
        if not title or title.startswith("@"):
            continue
        created_at = _parse_datetime(date_m.group(1) if date_m else None)
        posts.append({
            "source": "x_nitter",
            "text": title,
            "created_at": created_at.isoformat() if created_at else None,
            "uri": link_m.group(1) if link_m else None,
            "url": link_m.group(1) if link_m else None,
        })
        if len(posts) >= limit:
            break
    return posts


def _x_via_syndication(user: str, limit: int) -> List[dict]:
    url = X_SYNDICATION_URL.format(user=quote(user))
    payload = _download(url, timeout=10)
    text = payload.decode("utf-8", errors="replace")
    posts = []
    for tweet in re.findall(r'<li class="timeline-Item">(.*?)</li>', text, re.DOTALL):
        text_m = re.search(r'class="tweet-text"[^>]*>(.*?)</p>', tweet, re.DOTALL)
        link_m = re.search(r'href="(https?://twitter\.com/[^"]+)"', tweet)
        time_m = re.search(r'datetime="([^"]+)"', tweet)
        if not text_m:
            continue
        body = re.sub(r"<[^>]+>", " ", text_m.group(1))
        body = re.sub(r"\s+", " ", body).strip()
        created_at = _parse_datetime(time_m.group(1) if time_m else None)
        posts.append({
            "source": "x_syndication",
            "text": body,
            "created_at": created_at.isoformat() if created_at else None,
            "uri": link_m.group(1) if link_m else None,
            "url": link_m.group(1) if link_m else None,
        })
        if len(posts) >= limit:
            break
    return posts


def fetch_x_posts(
    handle: str,
    limit: int = 20,
    source: str = "x",
    figure: Optional[dict] = None,
) -> List[dict]:
    """Fetch recent posts from an X account (best-effort, free backends)."""
    instances = [
        i.strip().rstrip("/")
        for i in (_get_config_value(NITTER_INSTANCES_ENV) or "").split(",")
        if i.strip()
    ] or DEFAULT_NITTER_INSTANCES

    errors = []
    for instance in instances:
        try:
            posts = _x_via_nitter(handle, instance, limit)
            if posts:
                return posts
        except SocialTrackerError as exc:
            errors.append(str(exc))
            continue

    try:
        posts = _x_via_syndication(handle, limit)
        if posts:
            return posts
    except SocialTrackerError as exc:
        errors.append(str(exc))

    if errors and _get_config_value("X_GUEST_TOKEN"):
        try:
            posts = _x_via_graphql(handle, limit)
            if posts:
                return posts
        except SocialTrackerError:
            pass

    return []


def _x_via_graphql(user: str, limit: int) -> List[dict]:
    """Experimental X GraphQL backend (requires a guest token)."""
    guest_token = _get_config_value("X_GUEST_TOKEN")
    if not guest_token:
        return []

    bearer = ("Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs"
              "=1Zv7ttbk8LVmHivzgFVVYzOK1WjoNL0HmMf3Wph3b4")
    headers = {
        "Authorization": bearer,
        "x-guest-token": guest_token,
        "User-Agent": USER_AGENT,
    }
    variables = {
        "screen_name": user,
        "count": limit,
        "include_user_interests": False,
        "include_profile_interests_type": False,
        "with_highlighted_label": False,
    }
    features = {
        "responsive_web_graphql_timeline_navigation_enabled": True,
        "responsive_web_graphql_exclude_directive_enabled": True,
        "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": False,
        "longform_notetweets_richtext_consumption_enabled": True,
        "longform_notetweets_inline_media_enabled": False,
    }
    url = ("https://twitter.com/i/api/graphql/BDhMJW1Zx_9sHVdEBBCbcg/UserTweets?"
           + urlencode({"variables": json.dumps(variables), "features": json.dumps(features)}))
    payload = _download(url, timeout=12, headers=headers)
    data = json.loads(payload.decode("utf-8"))
    entries = (data.get("data", {})
               .get("user", {})
               .get("result", {})
               .get("timeline", {})
               .get("timeline", {})
               .get("instructions", []))
    posts = []
    for instruction in entries:
        for entry in instruction.get("entries", []):
            result = entry.get("content", {}).get("itemContent", {}).get("tweet_results", {}).get("result", {})
            legacy = result.get("legacy", {})
            text = legacy.get("full_text", "")
            if not text:
                continue
            created_at = _parse_datetime(legacy.get("created_at"))
            post_id = legacy.get("id_str")
            posts.append({
                "source": "x_graphql",
                "text": text,
                "created_at": created_at.isoformat() if created_at else None,
                "uri": f"https://twitter.com/{user}/status/{post_id}" if post_id else None,
                "url": f"https://twitter.com/{user}/status/{post_id}" if post_id else None,
            })
            if len(posts) >= limit:
                return posts
    return posts


# ── Sentiment scoring ────────────────────────────────────────────────


def _score_to_social_token(score: float, prefix: str) -> str:
    """Convert a sentiment score to a figure-specific social token."""
    prefix = prefix.upper()
    if score > 0.5:
        return f"SOC_{prefix}_StrongPos"
    elif score > 0.1:
        return f"SOC_{prefix}_Positive"
    elif score < -0.5:
        return f"SOC_{prefix}_StrongNeg"
    elif score < -0.1:
        return f"SOC_{prefix}_Negative"
    return f"SOC_{prefix}_Neutral"


def _no_data_token(prefix: str) -> str:
    return f"SOC_{prefix.upper()}_NoData"


def _post_relevance(post: dict, ticker: str, figure: dict) -> bool:
    """Check whether a post mentions the target ticker for this figure."""
    keywords = figure.get("keywords", {}).get(ticker.upper(), [])
    if not keywords:
        return False
    text = (post.get("text", "") or "").lower()
    return any(keyword in text for keyword in keywords)


def score_posts(posts: List[dict], ticker: str, figure: dict) -> Dict:
    """
    Aggregate sentiment for a figure's posts that mention a ticker.

    Returns dict with 'score', 'relevant_posts', 'count', 'token',
    'last_post' and 'last_post_time'.
    """
    relevant = [p for p in posts if _post_relevance(p, ticker, figure)]
    if not relevant:
        return {
            "score": 0.0,
            "count": 0,
            "token": _no_data_token(figure.get("prefix", "")),
            "posts": [],
            "last_post": None,
            "last_post_time": None,
        }

    scores = [_simple_sentiment_score(p.get("text", "")) for p in relevant]
    avg_score = sum(scores) / len(scores)

    posts_sorted = sorted(
        relevant,
        key=lambda p: p.get("created_at") or "",
        reverse=True,
    )
    last_post = posts_sorted[0] if posts_sorted else None

    return {
        "score": avg_score,
        "count": len(relevant),
        "token": _score_to_social_token(avg_score, figure.get("prefix", "")),
        "posts": posts_sorted,
        "last_post": last_post.get("text", "") if last_post else None,
        "last_post_time": last_post.get("created_at") if last_post else None,
    }


# ── Top-level tracker API ────────────────────────────────────────────


def fetch_figure_posts(figure: dict, force: bool = False) -> List[dict]:
    """
    Fetch recent posts for a figure (cached with TTL + disk archive).

    Returns a merged list of posts from archive and fresh fetches.
    """
    prefix = figure.get("prefix", "FIG").upper()
    cache_key = f"posts_{prefix}"

    now = time.time()
    cached = _cache.get(cache_key)
    if cached and not force and (now - cached[0]) < TTL_SECONDS:
        return cached[1]

    fresh: List[dict] = []
    x_handle = figure.get("x")
    truth_handle = figure.get("truth")

    if truth_handle:
        try:
            fresh += fetch_truth_social_posts(truth_handle, figure=figure, source="truth")
        except SocialTrackerError:
            pass

    if x_handle:
        try:
            fresh += fetch_x_posts(x_handle, figure=figure, source="x")
        except SocialTrackerError:
            pass

    _append_to_archive(figure, fresh)
    merged = _load_archive(figure)
    _cache[cache_key] = (now, merged)
    return merged


def compute_social_indicator(
    ticker: str,
    force: bool = False,
    as_of: Optional[datetime] = None,
) -> Dict:
    """
    Compute the figure-level social sentiment indicator for a ticker.

    Returns a dict with:
      - 'token': dominant figure social sentiment token (or NoData)
      - 'signals': per-figure details for UI display
      - 'figure': name of the dominant figure (or None)
      - 'total_posts': number of relevant posts found

    Never raises; always returns a valid token.
    """
    ticker = ticker.upper()
    figures = figure_for_ticker(ticker)

    if not figures:
        first = all_figures()[0] if all_figures() else _DEFAULT_FIGURES[0]
        return {
            "token": _no_data_token(first.get("prefix", "")),
            "signals": [],
            "figure": None,
            "total_posts": 0,
        }

    signals = []
    for figure in figures:
        try:
            posts = fetch_figure_posts(figure, force=force)
        except Exception:
            posts = []

        lookback = as_of or datetime.now(timezone.utc)
        cutoff = lookback - timedelta(days=POST_LOOKBACK_DAYS)
        posts = [
            p for p in posts
            if (p.get("created_at") and _parse_datetime(p.get("created_at"))
                and _parse_datetime(p.get("created_at")) >= cutoff)
        ]

        signal = score_posts(posts, ticker, figure)
        signals.append({
            "figure": figure.get("name", ""),
            "prefix": figure.get("prefix", ""),
            **signal,
        })

    dominant = max(signals, key=lambda s: abs(s.get("score", 0.0)) if s.get("count", 0) > 0 else 0.0)
    if dominant.get("count", 0) == 0:
        token = _no_data_token(signals[0].get("prefix", ""))
    else:
        token = dominant["token"]

    return {
        "token": token,
        "signals": signals,
        "figure": dominant.get("figure") if dominant.get("count", 0) > 0 else None,
        "total_posts": sum(s.get("count", 0) for s in signals),
    }


def social_token_for_date(ticker: str, as_of_date) -> str:
    """
    Figure social sentiment token as of a historical date (for training data).

    Reads from the disk archive only (no network).  Historical rows before the
    archive has accumulated posts naturally resolve to the NoData token.
    """
    ticker = ticker.upper()
    figures = figure_for_ticker(ticker)
    if not figures:
        first = all_figures()[0] if all_figures() else _DEFAULT_FIGURES[0]
        return _no_data_token(first.get("prefix", ""))

    if isinstance(as_of_date, str):
        try:
            as_of_date = datetime.fromisoformat(as_of_date)
        except ValueError:
            as_of_date = None

    if as_of_date is not None and as_of_date.tzinfo is None:
        as_of_date = as_of_date.replace(tzinfo=timezone.utc)

    signals = []
    for figure in figures:
        posts = []
        try:
            posts = _load_archive(figure)
        except Exception:
            posts = []

        if as_of_date is not None:
            posts = [
                p for p in posts
                if (p.get("created_at")
                    and (d := _parse_datetime(p.get("created_at")))
                    and d <= as_of_date)
            ]
        signal = score_posts(posts, ticker, figure)
        signals.append(signal)

    if not signals:
        return _no_data_token(figures[0].get("prefix", ""))

    dominant = max(signals, key=lambda s: abs(s.get("score", 0.0)) if s.get("count", 0) > 0 else 0.0)
    if dominant.get("count", 0) == 0:
        return _no_data_token(figures[0].get("prefix", ""))
    return dominant["token"]


def compute_statement_reaction(
    figure_name: str,
    ticker: str,
    closes: Dict[str, float],
    horizon_days: int = 1,
    min_posts: int = 5,
) -> Dict:
    """
    Correlate a figure's scored posts with subsequent market moves.

    Args:
        figure_name: Name of the tracked figure.
        ticker: Target ticker symbol.
        closes: {date_str: close} mapping (date_str ISO yyyy-mm-dd).
        horizon_days: Number of trading days after a post to measure return.
        min_posts: Minimum number of posts before reporting a correlation.

    Returns:
        Dict with 'correlation', 'posts', 'r2', 'is_sufficient',
        'avg_post_score', 'avg_following_return'.
    """
    figure = figure_by_name(figure_name)
    if not figure:
        return {"correlation": 0.0, "posts": 0, "r2": 0.0, "is_sufficient": False,
                "avg_post_score": 0.0, "avg_following_return": 0.0}

    ticker = ticker.upper()
    posts = []
    try:
        posts = fetch_figure_posts(figure)
    except Exception:
        posts = []

    dated = sorted(
        [p for p in posts if _post_relevance(p, ticker, figure)
         and p.get("created_at") and _parse_datetime(p.get("created_at"))],
        key=lambda p: p["created_at"],
    )

    close_dates = sorted(closes.keys())
    if len(close_dates) < horizon_days + 2:
        return {"correlation": 0.0, "posts": 0, "r2": 0.0, "is_sufficient": False,
                "avg_post_score": 0.0, "avg_following_return": 0.0}

    date_index = {d: i for i, d in enumerate(close_dates)}
    xs: List[float] = []
    ys: List[float] = []
    for post in dated:
        post_dt = _parse_datetime(post["created_at"])
        post_day = post_dt.date().isoformat()
        # Use the closest trading date on or after the post
        if post_day in date_index:
            i = date_index[post_day]
        else:
            candidate = [d for d in close_dates if d >= post_day]
            if not candidate:
                continue
            i = date_index[candidate[0]]
        if i + horizon_days >= len(close_dates):
            continue
        base = closes[close_dates[i]]
        future = closes[close_dates[i + horizon_days]]
        if base <= 0:
            continue
        score = _simple_sentiment_score(post.get("text", ""))
        following_return = (future - base) / base
        xs.append(score)
        ys.append(following_return)

    n = len(xs)
    if n < min_posts:
        return {"correlation": 0.0, "posts": n, "r2": 0.0, "is_sufficient": False,
                "avg_post_score": 0.0, "avg_following_return": 0.0}

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    if var_x == 0 or var_y == 0:
        correlation = 0.0
        r2 = 0.0
    else:
        correlation = cov / ((var_x * var_y) ** 0.5)
        r2 = correlation ** 2

    return {
        "correlation": correlation,
        "posts": n,
        "r2": r2,
        "is_sufficient": True,
        "avg_post_score": mean_x,
        "avg_following_return": mean_y,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Social tracker CLI test")
    parser.add_argument("ticker", nargs="?", default="TSLA")
    args = parser.parse_args()

    print(f"Figures configured: {len(all_figures())}")
    for fig in all_figures():
        extra = " (user)" if fig.get("added_by_user") else ""
        print(f"  - {fig['name']} ({fig.get('prefix')}) x={fig.get('x')} truth={fig.get('truth')}{extra}")

    print(f"\nFetching social indicator for {args.ticker}...")
    result = compute_social_indicator(args.ticker, force=True)
    print(f"  Token: {result['token']}")
    print(f"  Figure: {result['figure']}")
    print(f"  Total relevant posts: {result['total_posts']}")
    for s in result["signals"]:
        print(f"  {s['figure']}: score={s.get('score', 0.0):+.2f} "
              f"posts={s.get('count', 0)} token={s.get('token')}")
