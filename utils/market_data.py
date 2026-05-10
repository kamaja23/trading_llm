"""Market data helpers for non-Yahoo OHLCV providers."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from io import BytesIO
import os
import re
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


STOOQ_DAILY_URL = "https://stooq.com/q/d/l/"
ALPHA_VANTAGE_DAILY_URL = "https://www.alphavantage.co/query"
STOCKANALYSIS_LOOKUP_URL = "https://stockanalysis.com/symbol-lookup/"
STOCKANALYSIS_HISTORY_URL = "https://stockanalysis.com/stocks/{symbol}/history/"
STOCKANALYSIS_QUOTE_HISTORY_URL = "https://stockanalysis.com/quote/{exchange}/{symbol}/history/"
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
STOCKANALYSIS_ROW_RE = re.compile(
    r"\{a:(?P<adj>-?\d+(?:\.\d+)?),"
    r"c:(?P<close>-?\d+(?:\.\d+)?),"
    r"h:(?P<high>-?\d+(?:\.\d+)?),"
    r"l:(?P<low>-?\d+(?:\.\d+)?),"
    r"o:(?P<open>-?\d+(?:\.\d+)?),"
    r't:"(?P<date>\d{4}-\d{2}-\d{2})",'
    r"v:(?P<volume>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)
LOOKUP_RESULT_RE = re.compile(
    r'\{s:"(?P<symbol>[$!@/\w.-]+)",'
    r'n:"(?P<name>[^"]+)",'
    r't:"(?P<type>[^"]+)"',
)
LOOKUP_STOPWORDS = {
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "limited",
    "ltd",
    "plc",
    "group",
    "holdings",
    "holding",
    "the",
}


class MarketDataError(Exception):
    """Raised when a market data provider cannot return usable OHLCV data."""


def fetch_market_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    timeout: int = 15,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data from configured non-Yahoo providers.

    Set ALPHA_VANTAGE_API_KEY for Alpha Vantage, or STOOQ_API_KEY for Stooq.
    """
    errors = []

    try:
        return fetch_alpha_vantage_data(
            symbol,
            start_date=start_date,
            end_date=end_date,
            period=period,
            timeout=timeout,
        )
    except MarketDataError as exc:
        errors.append(str(exc))

    try:
        return fetch_stooq_data(
            symbol,
            start_date=start_date,
            end_date=end_date,
            period=period,
            timeout=timeout,
        )
    except MarketDataError as exc:
        errors.append(str(exc))

    try:
        return fetch_stockanalysis_data(
            symbol,
            start_date=start_date,
            end_date=end_date,
            period=period,
            timeout=timeout,
        )
    except MarketDataError as exc:
        errors.append(str(exc))

    raise MarketDataError(" | ".join(errors))


def resolve_stockanalysis_symbol(query: str, timeout: int = 15) -> Optional[dict]:
    """Resolve a company name or ticker to a primary public StockAnalysis symbol."""
    value = query.strip()
    if not value:
        return None

    params = {"q": value}
    payload = _download(f"{STOCKANALYSIS_LOOKUP_URL}?{urlencode(params)}", timeout)
    text = payload.decode("utf-8", errors="replace")

    matches = []
    for match in LOOKUP_RESULT_RE.finditer(text):
        raw_symbol = match.group("symbol")
        result_type = match.group("type")
        if result_type != "Stock" or not raw_symbol.startswith(("$", "@")):
            continue
        ticker = _display_stockanalysis_symbol(raw_symbol)
        name = _clean_html_text(match.group("name"))
        if not _is_relevant_lookup_match(value, ticker, name):
            continue
        matches.append(
            {
                "ticker": ticker,
                "raw_symbol": raw_symbol,
                "name": name,
                "source": "StockAnalysis",
            }
        )

    if not matches:
        return None

    for preferred in (
        lambda item: item["raw_symbol"].startswith("$"),
        lambda item: item["raw_symbol"].lower().startswith("@otc/"),
    ):
        for item in matches:
            if preferred(item):
                return {
                    "ticker": item["ticker"],
                    "name": item["name"],
                    "source": item["source"],
                }

    first = matches[0]
    return {
        "ticker": first["ticker"],
        "name": first["name"],
        "source": first["source"],
    }


def fetch_alpha_vantage_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    timeout: int = 15,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV data from Alpha Vantage."""
    key = api_key or _get_config_value("ALPHA_VANTAGE_API_KEY")
    if not key:
        raise MarketDataError("ALPHA_VANTAGE_API_KEY is not set.")

    start, end = _resolve_date_range(start_date, end_date, period)
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol.strip().upper(),
        "outputsize": "full",
        "datatype": "csv",
        "apikey": key,
    }
    payload = _download(f"{ALPHA_VANTAGE_DAILY_URL}?{urlencode(params)}", timeout)

    if payload.lstrip().startswith((b"{", b"Thank you", b"Our standard")):
        message = payload.decode("utf-8", errors="replace").strip()
        raise MarketDataError(f"Alpha Vantage returned an error: {message}")

    df = pd.read_csv(BytesIO(payload))
    if df.empty or "timestamp" not in df.columns:
        raise MarketDataError(f"Alpha Vantage returned no daily data for {symbol}.")

    rename = {
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename)
    df = _normalize_ohlcv(df, symbol)
    return df[(df.index.date >= start) & (df.index.date <= end)]


def fetch_stooq_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    timeout: int = 15,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data from Stooq.

    Args:
        symbol: Ticker symbol, such as SPY or AAPL. US symbols are mapped to
            Stooq's ".us" suffix automatically.
        start_date: Optional inclusive start date in YYYY-MM-DD format.
        end_date: Optional inclusive end date in YYYY-MM-DD format.
        period: Optional period such as "1y" or "6mo".
        timeout: Request timeout in seconds.

    Returns:
        DataFrame indexed by Date with Open, High, Low, Close, Volume columns.
    """
    start, end = _resolve_date_range(start_date, end_date, period)
    key = api_key or _get_config_value("STOOQ_API_KEY")
    if not key:
        raise MarketDataError("STOOQ_API_KEY is not set.")

    params = {
        "s": _to_stooq_symbol(symbol),
        "i": "d",
        "d1": start.strftime("%Y%m%d"),
        "d2": end.strftime("%Y%m%d"),
        "apikey": key,
    }
    payload = _download(f"{STOOQ_DAILY_URL}?{urlencode(params)}", timeout)

    if b"Get your apikey" in payload:
        raise MarketDataError("Stooq requires a valid STOOQ_API_KEY.")

    df = pd.read_csv(BytesIO(payload))
    if df.empty or "Date" not in df.columns:
        raise MarketDataError(f"Stooq returned no daily data for {symbol}.")

    return _normalize_ohlcv(df, symbol)


def fetch_stockanalysis_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    timeout: int = 15,
) -> pd.DataFrame:
    """Fetch daily OHLCV data from a StockAnalysis history page."""
    start, end = _resolve_date_range(start_date, end_date, period)
    payload = _download(_stockanalysis_history_url(symbol), timeout)
    text = payload.decode("utf-8", errors="replace")

    rows = []
    for match in STOCKANALYSIS_ROW_RE.finditer(text):
        row_date = _parse_date(match.group("date"))
        if row_date < start or row_date > end:
            continue
        rows.append(
            {
                "Date": row_date.isoformat(),
                "Open": match.group("open"),
                "High": match.group("high"),
                "Low": match.group("low"),
                "Close": match.group("close"),
                "Volume": match.group("volume"),
            }
        )

    if not rows:
        raise MarketDataError(f"StockAnalysis returned no daily data for {symbol}.")

    return _normalize_ohlcv(pd.DataFrame(rows), symbol)


def _download(url: str, timeout: int) -> bytes:
    request = Request(url, headers={"User-Agent": "trading-llm/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except Exception as exc:
        raise MarketDataError(f"Market data request failed: {exc}") from exc


def _clean_html_text(value: str) -> str:
    return (
        value.replace("&amp;", "&")
        .replace("&#39;", "'")
        .replace("&quot;", '"')
        .strip()
    )


def _lookup_tokens(value: str) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 1 and token not in LOOKUP_STOPWORDS
    }
    return tokens


def _is_relevant_lookup_match(query: str, ticker: str, name: str) -> bool:
    clean_query = re.sub(r"[^a-z0-9]", "", query.lower())
    clean_ticker = re.sub(r"[^a-z0-9]", "", ticker.lower())
    if clean_query and clean_query == clean_ticker:
        return True

    query_tokens = _lookup_tokens(query)
    if not query_tokens:
        return False

    searchable = _lookup_tokens(f"{ticker} {name}")
    return bool(query_tokens & searchable)


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


def _normalize_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise MarketDataError(f"{symbol} data is missing columns: {', '.join(missing)}")

    df = df[required_cols].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)

    if df.empty:
        raise MarketDataError(f"Provider returned only unusable rows for {symbol}.")

    return df


def _resolve_date_range(
    start_date: Optional[str],
    end_date: Optional[str],
    period: Optional[str],
) -> tuple[date, date]:
    end = _parse_date(end_date) if end_date else date.today()
    if start_date:
        return _parse_date(start_date), end

    days = _period_to_days(period or "1y")
    return end - timedelta(days=days), end


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _period_to_days(period: str) -> int:
    periods = {
        "1d": 1,
        "5d": 7,
        "1mo": 31,
        "3mo": 93,
        "6mo": 186,
        "1y": 366,
        "2y": 366 * 2,
        "5y": 366 * 5,
        "10y": 366 * 10,
        "ytd": max((date.today() - date(date.today().year, 1, 1)).days, 1),
        "max": 366 * 30,
    }
    return periods.get(period, periods["1y"])


def _to_stooq_symbol(symbol: str) -> str:
    clean = symbol.strip().lower()
    if not clean:
        raise MarketDataError("Ticker symbol cannot be empty.")
    if clean.endswith(".us"):
        return clean
    return f"{clean.replace('.', '-')}.us"


def _to_stockanalysis_symbol(symbol: str) -> str:
    clean = symbol.strip().lower()
    if not clean:
        raise MarketDataError("Ticker symbol cannot be empty.")
    if not re.fullmatch(r"[a-z0-9.-]+", clean):
        raise MarketDataError(f"{symbol} is not a valid ticker for StockAnalysis.")
    return clean.replace(".", "-")


def _display_stockanalysis_symbol(raw_symbol: str) -> str:
    if raw_symbol.startswith("$"):
        return raw_symbol[1:].upper()
    if raw_symbol.startswith("@") and "/" in raw_symbol:
        exchange, symbol = raw_symbol[1:].split("/", 1)
        return f"{exchange.upper()}:{symbol.upper()}"
    return raw_symbol.upper()


def _stockanalysis_history_url(symbol: str) -> str:
    clean = symbol.strip()
    if not clean:
        raise MarketDataError("Ticker symbol cannot be empty.")

    if clean.startswith("$"):
        return STOCKANALYSIS_HISTORY_URL.format(
            symbol=_to_stockanalysis_symbol(clean[1:])
        )
    if clean.startswith("@") and "/" in clean:
        exchange, quote_symbol = clean[1:].split("/", 1)
        return STOCKANALYSIS_QUOTE_HISTORY_URL.format(
            exchange=_to_stockanalysis_symbol(exchange),
            symbol=_to_stockanalysis_symbol(quote_symbol),
        )
    if ":" in clean:
        exchange, quote_symbol = clean.split(":", 1)
        return STOCKANALYSIS_QUOTE_HISTORY_URL.format(
            exchange=_to_stockanalysis_symbol(exchange),
            symbol=_to_stockanalysis_symbol(quote_symbol),
        )

    return STOCKANALYSIS_HISTORY_URL.format(symbol=_to_stockanalysis_symbol(clean))
