"""
TradeBot - Stock Analysis Frontend

Streamlit app that uses the trained model to analyze stocks
and provide BUY/SELL/HOLD recommendations with visual charts.

Usage:
  pip install streamlit plotly
  streamlit run app.py
"""

import time
from datetime import date, timedelta

import streamlit as st
import streamlit.components.v1 as components
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
from streamlit_autorefresh import st_autorefresh

if get_script_run_ctx(suppress_warning=True) is None:
    print(
        "This is a Streamlit app. Run it with:\n\n"
        "  .venv/bin/streamlit run app.py\n\n"
        "or:\n\n"
        "  streamlit run app.py\n"
    )
    raise SystemExit(1)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from agent.stock_agent import (
    BEARISH_INDICATORS,
    BULLISH_INDICATORS,
    INDICATOR_LABELS,
    INDICATOR_VALUES,
    StockAnalysisAgent,
    StockAnalysisError,
)
from auth.auth import (
    authenticate_user,
    create_session,
    delete_session,
    get_saved_stocks,
    init_db,
    register_user,
    remove_stock,
    save_stock,
    verify_session,
)
from utils.market_data import MarketDataError, resolve_stockanalysis_symbol
from utils.paper_trader import PaperTrader

st.set_page_config(
    page_title="TradeBot — Stock Analysis Agent",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Auto-refresh via streamlit-autorefresh (no full page reload)
if st.session_state.get("live_mode"):
    interval_ms = st.session_state.get("live_interval", 30) * 1000
    refresh_count = st_autorefresh(interval=interval_ms, key="live_autorefresh")
    prev_count = st.session_state.get("_autorefresh_count", -1)
    if refresh_count != prev_count:
        st.session_state._autorefresh_count = refresh_count
        st.session_state.analyses = {}
        st.session_state.last_live_refresh = time.time()


@st.cache_resource
def get_agent():
    try:
        return StockAnalysisAgent()
    except StockAnalysisError as e:
        st.error(str(e))
        st.stop()


agent = get_agent()
init_db()

# ── Auth session state ──────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "_auth_restored" not in st.session_state:
    st.session_state._auth_restored = False

# Restore session from query param token (survives page refresh)
if not st.session_state._auth_restored:
    token_from_url = st.query_params.get("token")
    if token_from_url:
        valid, payload = verify_session(token_from_url)
        if valid:
            st.session_state.user = payload
            st.session_state.auth_token = token_from_url
    st.session_state._auth_restored = True

# Load saved stocks when user logs in (detect transition)
if st.session_state.user and not st.session_state.watchlist:
    saved = get_saved_stocks(st.session_state.user["user_id"])
    for s in saved:
        ticker = s["ticker"]
        if ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(ticker)
            if s["company_name"]:
                st.session_state.company_names[ticker] = s["company_name"]

DEMO_VERSION = "clarify_unresolved_stock_lookup"
DEFAULT_WATCHLIST: list[str] = []
DEFAULT_PRIVATE_COMPANIES: list[tuple[str, str]] = []
DEMO_END_DATE = (date.today() - timedelta(days=1)).isoformat()
DEMO_START_DATE = (date.fromisoformat(DEMO_END_DATE) - timedelta(days=365)).isoformat()
COMPANY_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "GOOGL": "Alphabet",
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq-100 ETF",
}
TICKER_ALIASES = {
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "NVIDIA": "NVDA",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
}
_DEFAULT_PRIVATE_ALIASES = {
    "OPENAI": ("OpenAI", "Private company · no public stock ticker"),
    "OPEN AI": ("OpenAI", "Private company · no public stock ticker"),
    "ANTHROPIC": ("Anthropic", "Private company · no public stock ticker"),
    "CLAUDE": ("Anthropic", "Private company · no public stock ticker"),
    "SPACEX": ("SpaceX", "Private company · no public stock ticker"),
    "SPACE X": ("SpaceX", "Private company · no public stock ticker"),
    "XAI": ("xAI", "Private company · no public stock ticker"),
    "X.AI": ("xAI", "Private company · no public stock ticker"),
    "STRIPE": ("Stripe", "Private company · no public stock ticker"),
    "DATABRICKS": ("Databricks", "Private company · no public stock ticker"),
    "PERPLEXITY": ("Perplexity", "Private company · no public stock ticker"),
    "ANDURIL": ("Anduril", "Private company · no public stock ticker"),
    "VALVE": ("Valve Corporation", "Private company · no public stock ticker"),
    "VALVE CORPORATION": (
        "Valve Corporation",
        "Private company · no public stock ticker",
    ),
}


def normalize_ticker(raw: str) -> str:
    value = raw.upper().strip()
    return TICKER_ALIASES.get(value, value)


def private_company_for(raw: str):
    return st.session_state.private_aliases.get(raw.upper().strip())


def analyze_demo_stock(ticker: str):
    return agent.analyze(
        ticker,
        start_date=DEMO_START_DATE,
        end_date=DEMO_END_DATE,
    )


def analyze_live_stock(ticker: str):
    return agent.live_analyze(
        ticker,
        start_date=DEMO_START_DATE,
        end_date=DEMO_END_DATE,
    )


if st.session_state.get("demo_version") != DEMO_VERSION:
    st.session_state.demo_version = DEMO_VERSION
    st.session_state.watchlist = DEFAULT_WATCHLIST.copy()
    st.session_state.analyses = {}
    st.session_state.private_companies = DEFAULT_PRIVATE_COMPANIES.copy()
    st.session_state.company_names = COMPANY_NAMES.copy()
    st.session_state.private_aliases = _DEFAULT_PRIVATE_ALIASES.copy()
elif "watchlist" not in st.session_state:
    st.session_state.watchlist = DEFAULT_WATCHLIST.copy()
if "analyses" not in st.session_state:
    st.session_state.analyses = {}
if "private_companies" not in st.session_state:
    st.session_state.private_companies = DEFAULT_PRIVATE_COMPANIES.copy()
if "company_names" not in st.session_state:
    st.session_state.company_names = COMPANY_NAMES.copy()
if "pending_unresolved_stock" not in st.session_state:
    st.session_state.pending_unresolved_stock = None
if "stock_lookup_dialog" not in st.session_state:
    st.session_state.stock_lookup_dialog = None
if "private_aliases" not in st.session_state:
    st.session_state.private_aliases = _DEFAULT_PRIVATE_ALIASES.copy()
if "pending_private_add" not in st.session_state:
    st.session_state.pending_private_add = None
if "live_mode" not in st.session_state:
    st.session_state.live_mode = False
if "live_interval" not in st.session_state:
    st.session_state.live_interval = 30
if "last_live_refresh" not in st.session_state:
    st.session_state.last_live_refresh = 0.0
if "paper_trader" not in st.session_state:
    st.session_state.paper_trader = None
if "paper_trading" not in st.session_state:
    st.session_state.paper_trading = False

# Restore watchlist from query params on page reload (for live mode auto-refresh)
if "w" in st.query_params and not st.session_state.watchlist:
    raw = st.query_params["w"]
    if raw:
        st.session_state.watchlist = [t for t in raw.split(",") if t]


def company_name(ticker: str) -> str:
    return st.session_state.company_names.get(ticker.upper(), ticker.upper())


def company_label_html(ticker: str, analyzed: bool = False) -> str:
    status = "Ready" if analyzed else "Pending"
    ticker = ticker.upper()
    return (
        f"<strong>{company_name(ticker)}</strong><br>"
        f"<span style='color:#777;font-size:0.8rem;'>{ticker} · {status}</span>"
    )


def prediction_hover_colors(ticker: str) -> tuple[str, str]:
    analysis = st.session_state.analyses.get(ticker)
    if not analysis:
        return "#6b7280", "#f3f4f6"

    return {
        "BUY": ("#1b8f5a", "#e8f5e9"),
        "SELL": ("#c0392b", "#ffebee"),
    }.get(analysis.prediction, ("#6b7280", "#f3f4f6"))


def prediction_card_colors(prediction: str) -> tuple[str, str, str]:
    return {
        "BUY": ("#e8f5e9", "#166534", "#bbf7d0"),
        "SELL": ("#ffebee", "#991b1b", "#fecdd3"),
    }.get(prediction, ("#f5f5f5", "#374151", "#d1d5db"))


def stock_tab_hover_css() -> str:
    rules = []
    for idx, ticker in enumerate(st.session_state.watchlist, start=1):
        hover_color, hover_bg = prediction_hover_colors(ticker)
        rules.append(
            f"""
            div[data-testid="stTabs"] button[role="tab"]:nth-of-type({idx}) {{
                transition: background 140ms ease, border-color 140ms ease;
            }}

            div[data-testid="stTabs"] button[role="tab"]:nth-of-type({idx}):hover {{
                background: {hover_bg};
                border-color: {hover_color};
            }}

            div[data-testid="stTabs"] button[role="tab"]:nth-of-type({idx}):hover p {{
                color: {hover_color};
            }}
            """
        )

    return "\n".join(rules)


def indicator_bias(key: str, value: str) -> str:
    if key in BULLISH_INDICATORS and value in BULLISH_INDICATORS[key]:
        return "bullish"
    if key in BEARISH_INDICATORS and value in BEARISH_INDICATORS[key]:
        return "bearish"
    return "neutral"


def prediction_explanation(result) -> tuple[str, list[str]]:
    prediction = result.prediction
    probabilities = sorted(
        result.action_probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    _, top_prob = probabilities[0]
    runner_up_action, runner_up_prob = probabilities[1]
    probability_gap = top_prob - runner_up_prob

    aligned_signals = []
    opposing_signals = []
    neutral_signals = []

    target_bias = {
        "BUY": "bullish",
        "SELL": "bearish",
    }.get(prediction)

    indicator_keys = ["trend", "volume", "heikin_ashi", "stochastic", "rsi", "macd", "bb", "ma_cross", "volatility", "candle", "obv", "price_action", "market_context", "sentiment"]
    for key in indicator_keys:
        value = result.indicators.get(key, "")
        if not value:
            continue
        bias = indicator_bias(key, value)
        description = INDICATOR_VALUES.get(value, value)
        label = INDICATOR_LABELS.get(key, key)
        signal_text = f"{label}: {description}"

        if target_bias and bias == target_bias:
            aligned_signals.append(signal_text)
        elif target_bias and bias in ("bullish", "bearish"):
            opposing_signals.append(signal_text)
        else:
            neutral_signals.append(signal_text)

    if probability_gap >= 0.25:
        confidence_reason = (
            f"The confidence is higher because {prediction} leads "
            f"{runner_up_action} by {probability_gap:.1%}."
        )
    elif probability_gap >= 0.10:
        confidence_reason = (
            f"The confidence is moderate because {prediction} is ahead of "
            f"{runner_up_action}, but only by {probability_gap:.1%}."
        )
    else:
        confidence_reason = (
            f"The confidence is cautious because {prediction} and "
            f"{runner_up_action} are close, separated by {probability_gap:.1%}."
        )

    if prediction == "BUY":
        direction_reason = "The model is leaning BUY because current signals skew bullish."
    elif prediction == "SELL":
        direction_reason = "The model is leaning SELL because current signals skew bearish."
    else:
        direction_reason = "The model is leaning HOLD because the current signals are mixed or muted."

    details = [confidence_reason]
    if aligned_signals:
        details.append("Supporting signals: " + "; ".join(aligned_signals) + ".")
    if opposing_signals:
        details.append("Signals working against it: " + "; ".join(opposing_signals) + ".")
    if neutral_signals:
        details.append("Neutral signals: " + "; ".join(neutral_signals) + ".")

    details.append(
        f"Overall indicator score is {result.indicator_score:+.2f} on a -1 to +1 scale."
    )

    return direction_reason, details


def show_lookup_dialog(message: str):
    if hasattr(st, "dialog"):
        @st.dialog("Stock not found")
        def _dialog():
            st.write(message)
            if st.button("OK", width="stretch"):
                st.session_state.stock_lookup_dialog = None
                st.rerun()

        _dialog()
    else:
        st.warning(message)


def add_private_company(private_company, alias: str = ""):
    name, desc = private_company
    entry = (name, desc)
    if entry not in st.session_state.private_companies:
        st.session_state.private_companies.append(entry)
        if alias:
            st.session_state.private_aliases[alias.upper().strip()] = entry
        st.toast(f"{name} added to private list")
        st.rerun()
    else:
        st.toast(f"{name} is already in private list")


def unresolved_stock_message(raw_ticker: str) -> str:
    return (
        f"I could not find a public listing for '{raw_ticker}'. "
        "Try a different public company or a related parent company."
    )


def add_public_stock(ticker: str):
    if ticker in st.session_state.watchlist:
        st.toast(f"{ticker} already in watchlist")
        return

    with st.spinner(f"Training on {ticker} data..."):
        try:
            analysis = analyze_demo_stock(ticker)
        except Exception:
            st.session_state.stock_lookup_dialog = (
                "Could not find any relevant information."
            )
            st.rerun()
            return

    st.session_state.watchlist.append(ticker)
    st.session_state.analyses[ticker] = analysis
    if st.session_state.user:
        save_stock(st.session_state.user["user_id"], ticker,
                   st.session_state.company_names.get(ticker, ""))
    st.rerun()


# ── Auth helpers ────────────────────────────────────────────────────

def _set_cookie(name: str, value: str, days: int = 30):
    components.html(
        f"""
        <script>
        (function() {{
            var d = new Date();
            d.setTime(d.getTime() + ({days} * 24 * 60 * 60 * 1000));
            document.cookie = "{name}=" + encodeURIComponent("{value}")
                + ";expires=" + d.toUTCString() + ";path=/;SameSite=Lax";
        }})();
        </script>
        """,
        height=0,
    )


def _clear_cookie(name: str):
    components.html(
        f"""
        <script>
        (function() {{
            document.cookie = "{name}=;expires=Thu, 01 Jan 1970 00:00:00 UTC;path=/;SameSite=Lax";
        }})();
        </script>
        """,
        height=0,
    )


def _try_restore_from_cookie():
    """Inject JS to read auth cookie and redirect with token in URL."""
    if st.session_state.user or st.query_params.get("token"):
        return
    components.html(
        """
        <script>
        (function() {
            function getCookie(name) {
                var value = "; " + document.cookie;
                var parts = value.split("; " + name + "=");
                if (parts.length === 2) return decodeURIComponent(parts.pop().split(";").shift());
                return null;
            }
            var token = getCookie("auth_token");
            if (token) {
                var url = new URL(window.location.href);
                if (!url.searchParams.has("token")) {
                    url.searchParams.set("token", token);
                    window.parent.location.replace(url.toString());
                }
            }
        })();
        </script>
        """,
        height=0,
    )


def _show_auth_section():
    if st.session_state.user:
        _show_logged_in_ui()
    else:
        _show_login_signup_ui()


def _show_logged_in_ui():
    user = st.session_state.user
    st.sidebar.markdown(
        f"**Logged in as:** {user['username']}",
    )
    if st.sidebar.button("Logout", use_container_width=True, type="secondary"):
        token = st.session_state.auth_token
        if token:
            delete_session(token)
        st.session_state.user = None
        st.session_state.auth_token = None
        st.session_state.watchlist = []
        st.session_state.analyses = {}
        st.query_params.clear()
        _clear_cookie("auth_token")
        st.rerun()


def _show_login_signup_ui():
    with st.sidebar.expander("Login / Sign Up", expanded=True):
        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                l_user = st.text_input("Username", placeholder="Enter username")
                l_pass = st.text_input("Password", type="password", placeholder="Enter password")
                l_submit = st.form_submit_button("Login", use_container_width=True)
            if l_submit:
                ok, msg, user = authenticate_user(l_user, l_pass)
                if ok and user:
                    token = create_session(user["id"], user["username"])
                    st.session_state.user = {"user_id": user["id"], "username": user["username"]}
                    st.session_state.auth_token = token
                    st.query_params["token"] = token
                    _set_cookie("auth_token", token)
                    saved = get_saved_stocks(user["id"])
                    for s in saved:
                        t = s["ticker"]
                        if t not in st.session_state.watchlist:
                            st.session_state.watchlist.append(t)
                            if s["company_name"]:
                                st.session_state.company_names[t] = s["company_name"]
                    st.toast(msg)
                    st.rerun()
                else:
                    st.error(msg)

        with tab_signup:
            with st.form("signup_form", clear_on_submit=False):
                s_user = st.text_input("Choose username", placeholder="Username (min 2 chars)")
                s_pass = st.text_input("Choose password", type="password", placeholder="Password (min 4 chars)")
                s_confirm = st.text_input("Confirm password", type="password", placeholder="Repeat password")
                s_submit = st.form_submit_button("Create Account", use_container_width=True)
            if s_submit:
                if s_pass != s_confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register_user(s_user, s_pass)
                    if ok:
                        st.success(msg)
                        st.toast("You can now log in.")
                    else:
                        st.error(msg)


if st.session_state.stock_lookup_dialog:
    show_lookup_dialog(st.session_state.stock_lookup_dialog)

_try_restore_from_cookie()
_show_auth_section()

st.sidebar.title("Watchlist")

with st.sidebar:
    with st.form("add_stock_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        raw_ticker = col1.text_input(
            "Ticker", placeholder="e.g. Apple or AAPL", label_visibility="collapsed"
        ).strip()
        submitted = col2.form_submit_button(
            "",
            width="stretch",
            icon=":material/add:",
            help="Add stock",
        )

    new_ticker = normalize_ticker(raw_ticker)
    if submitted:
        st.session_state.pending_private_add = None
        if new_ticker:
            private_company = private_company_for(raw_ticker)
            if private_company:
                st.session_state.pending_unresolved_stock = None
                add_private_company(private_company)
            else:
                if new_ticker in COMPANY_NAMES:
                    st.session_state.pending_unresolved_stock = None
                    add_public_stock(new_ticker)
                else:
                    with st.spinner(f"Looking up {raw_ticker}..."):
                        try:
                            resolved = resolve_stockanalysis_symbol(raw_ticker)
                        except MarketDataError:
                            resolved = None
                    if resolved:
                        new_ticker = resolved["ticker"]
                        st.session_state.company_names[new_ticker] = resolved["name"]
                        st.session_state.pending_unresolved_stock = None
                        add_public_stock(new_ticker)
                    elif st.session_state.pending_unresolved_stock == raw_ticker.upper():
                        st.session_state.pending_unresolved_stock = None
                        st.session_state.pending_private_add = raw_ticker.upper()
                        st.rerun()
                    else:
                        st.session_state.pending_unresolved_stock = raw_ticker.upper()
                        st.warning(unresolved_stock_message(raw_ticker))

    pending = st.session_state.get("pending_private_add")
    if pending:
        st.warning(f"'{pending}' is not a public stock.")
        col_a, col_b = st.columns(2)
        if col_a.button("Add as private company", use_container_width=True):
            entry = (pending.title(), "Private company · no public stock ticker")
            st.session_state.private_aliases[pending] = entry
            st.session_state.pending_private_add = None
            add_private_company(entry)
        if col_b.button("Dismiss", use_container_width=True):
            st.session_state.pending_private_add = None
            st.rerun()

    st.divider()

    for ticker in list(st.session_state.watchlist):
        cols = st.columns([3, 1, 1])
        analyzed = ticker in st.session_state.analyses
        cols[0].markdown(company_label_html(ticker, analyzed), unsafe_allow_html=True)
        if cols[1].button(
            "",
            key=f"ref_{ticker}",
            help="Refresh",
            icon=":material/refresh:",
        ):
            st.session_state.analyses.pop(ticker, None)
            st.rerun()
        if cols[2].button("✕", key=f"rm_{ticker}", help="Remove"):
            st.session_state.watchlist.remove(ticker)
            st.session_state.analyses.pop(ticker, None)
            if st.session_state.user:
                remove_stock(st.session_state.user["user_id"], ticker)
            st.rerun()

    if st.session_state.private_companies:
        st.divider()
        st.markdown("**Private companies**")
        for idx, (name, note) in enumerate(list(st.session_state.private_companies)):
            cols = st.columns([4, 1])
            cols[0].markdown(
                f"<strong>{name}</strong><br>"
                f"<span style='color:#777;font-size:0.8rem;'>{note}</span>",
                unsafe_allow_html=True,
            )
            if cols[1].button("✕", key=f"private_rm_{idx}", help="Remove"):
                st.session_state.private_companies.pop(idx)
                st.rerun()

    with st.expander("Add private company", expanded=False):
        with st.form("add_private_form", clear_on_submit=True):
            priv_name = st.text_input("Company name", placeholder="e.g. SpaceX")
            priv_alias = st.text_input("Search alias (optional)", placeholder="e.g. SPACEX")
            priv_desc = st.text_input("Description (optional)", placeholder="Private company")
            priv_submitted = st.form_submit_button("Add", use_container_width=True)
        if priv_submitted and priv_name:
            alias = (priv_alias or priv_name).upper().strip()
            desc = priv_desc or "Private company · no public stock ticker"
            entry = (priv_name.strip(), desc)
            add_private_company(entry, alias=alias)

    st.divider()
    cols = st.columns(2)
    if cols[0].button("Refresh All", width="stretch", type="primary"):
        st.session_state.analyses = {}
        st.rerun()
    if cols[1].button("Clear All", width="stretch"):
        if st.session_state.user:
            for t in list(st.session_state.watchlist):
                remove_stock(st.session_state.user["user_id"], t)
        st.session_state.watchlist = []
        st.session_state.analyses = {}
        st.rerun()

    st.divider()
    st.caption(
        "Model analyzes 4 technical indicators (trend, volume, "
        "Heikin Ashi, Stochastic). Demo loads local big-tech data from "
        f"{DEMO_START_DATE} to {DEMO_END_DATE} (auto, refreshed daily). "
        "Not financial advice."
    )

    # === Live Mode ===
    st.divider()
    live_mode = st.toggle("Live Mode", value=st.session_state.live_mode, key="live_toggle")
    st.session_state.live_mode = live_mode

    if live_mode:
        interval = st.select_slider(
            "Refresh interval",
            options=[10, 30, 60, 120],
            value=st.session_state.live_interval,
            key="live_interval_slider",
        )
        st.session_state.live_interval = interval

        elapsed = time.time() - st.session_state.get("last_live_refresh", 0)
        if st.session_state.last_live_refresh > 0:
            st.caption(f"{int(elapsed)}s since last refresh · auto-refreshes every {interval}s")
        else:
            st.caption(f"Auto-refreshes every {interval}s (waiting for first refresh)")

    # === Paper Trading ===
    st.divider()
    st.markdown("**Paper Trading**")

    if not st.session_state.paper_trading:
        col_cash, col_btn = st.columns([2, 1])
        initial_cash = col_cash.number_input(
            "Initial capital",
            min_value=100.0,
            max_value=1_000_000.0,
            value=10_000.0,
            step=1_000.0,
            format="%.0f",
            label_visibility="collapsed",
        )
        if col_btn.button("Start", type="primary", use_container_width=True):
            st.session_state.paper_trader = PaperTrader(initial_cash)
            st.session_state.paper_trading = True
            st.rerun()
    else:
        trader = st.session_state.paper_trader
        prices = {
            t: a.price
            for t, a in st.session_state.analyses.items()
            if hasattr(a, "price")
        }

        port_value = trader.portfolio_value(prices)
        st.metric(
            "Portfolio",
            f"${port_value:,.2f}",
            delta=f"${port_value - trader.initial_cash:+,.2f}",
        )

        col1, col2 = st.columns(2)
        col1.metric("Cash", f"${trader.cash:,.2f}")
        col2.metric("Positions", str(trader.position_count))

        if trader.holdings:
            st.markdown("**Holdings**")
            for ticker, shares in sorted(trader.holdings.items()):
                current_price = prices.get(ticker, 0)
                cost = trader.cost_basis(ticker)
                value = shares * current_price
                delta = value - (shares * cost)
                st.metric(
                    f"{ticker} ({shares} sh)",
                    f"${value:,.2f}",
                    delta=f"${delta:+,.2f}",
                )

        if st.button("Stop Trading", use_container_width=True, type="secondary"):
            st.session_state.paper_trading = False
            st.rerun()

st.title("TradeBot — Stock Analysis Agent")

if not st.session_state.watchlist:
    st.info("Add stocks to your watchlist using the sidebar.")
    st.stop()

st.markdown(f"<style>{stock_tab_hover_css()}</style>", unsafe_allow_html=True)

tabs = st.tabs([
    f"{company_name(t)} ({t})"
    for t in st.session_state.watchlist
])

analyze_fn = analyze_live_stock if st.session_state.live_mode else analyze_demo_stock

for idx, ticker in enumerate(st.session_state.watchlist):
    with tabs[idx]:
        if ticker not in st.session_state.analyses:
            with st.spinner(f"{'Live' if st.session_state.live_mode else 'Fetching'} data, training, and analyzing {ticker}..."):
                try:
                    result = analyze_fn(ticker)
                    st.session_state.analyses[ticker] = result
                    st.rerun()
                except StockAnalysisError as e:
                    st.error(f"**{ticker}** — {e}")
                    st.button(
                        "Retry",
                        key=f"retry_{ticker}",
                        on_click=lambda t=ticker: st.session_state.analyses.pop(t, None),
                    )
                    continue

        result = st.session_state.analyses[ticker]

        chart_col, info_col = st.columns([2, 1])

        with chart_col:
            if result.price_history:
                ph = result.price_history
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3],
                )

                fig.add_trace(
                    go.Candlestick(
                        x=ph["dates"],
                        open=ph["Open"],
                        high=ph["High"],
                        low=ph["Low"],
                        close=ph["Close"],
                        name=result.ticker,
                        increasing_line_color="#26a69a",
                        decreasing_line_color="#ef5350",
                    ),
                    row=1,
                    col=1,
                )

                fig.add_trace(
                    go.Bar(
                        x=ph["dates"],
                        y=ph["Volume"],
                        name="Volume",
                        marker_color="rgba(100,100,200,0.3)",
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

                fig.update_layout(
                    height=450,
                    margin=dict(l=0, r=0, t=20, b=0),
                    xaxis_rangeslider_visible=False,
                    template="plotly_white",
                    hovermode="x unified",
                    font=dict(size=12),
                )
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)

                st.plotly_chart(fig, width="stretch")

        with info_col:
            pred = result.prediction
            conf = result.confidence
            rec_text, _rec_color = agent.recommendation_text(pred, conf)

            st.metric(
                label=f"{company_name(result.ticker)} · {result.ticker} · {result.date}",
                value=f"${result.price:.2f}",
            )

            if result.live_data:
                ld = result.live_data
                change = ld.get("change", 0)
                change_pct = ld.get("change_pct", 0)
                arrow = "▲" if change >= 0 else "▼"
                color = "#26a69a" if change >= 0 else "#ef5350"
                st.markdown(
                    f"<span style='color:{color};font-size:1.1rem;'>"
                    f"{arrow} ${abs(change):.2f} ({change_pct:+.2f}%)</span>"
                    f"<br><span style='color:#777;font-size:0.8rem;'>"
                    f"H:{ld.get('today_high', 0):.2f} "
                    f"L:{ld.get('today_low', 0):.2f} "
                    f"O:{ld.get('today_open', 0):.2f} "
                    f"Vol:{ld.get('today_volume', 0):,}"
                    f" · {ld.get('market_state', '')}"
                    f" · {ld.get('last_updated', '')[:19]}</span>",
                    unsafe_allow_html=True,
                )

            bg_color, card_text_color, border_color = prediction_card_colors(pred)
            st.markdown(
                f"""
                <div style="
                    background: {bg_color};
                    border: 1px solid {border_color};
                    border-radius: 12px;
                    padding: 16px;
                    text-align: center;
                    margin: 12px 0;
                    color: {card_text_color};
                ">
                    <div style="font-size: 14px; color: #4b5563;">Model Prediction</div>
                    <div style="font-size: 38px; font-weight: 700; color: {card_text_color};">
                        {pred}
                    </div>
                    <div style="font-size: 16px; margin-top: 4px; color: #374151;">
                        {conf:.1%} confidence
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.session_state.paper_trading:
                trader = st.session_state.paper_trader
                already_holding = result.ticker in trader.holdings
                col_exec, col_status = st.columns([1, 1])
                if pred == "BUY" and not already_holding:
                    if col_exec.button(
                        "Execute BUY", type="primary", key=f"buy_{result.ticker}",
                        use_container_width=True,
                    ):
                        trade = trader.buy(result.ticker, result.price, result.date, conf)
                        if trade:
                            st.toast(f"Bought {trade.shares} shares of {result.ticker} for ${trade.total:,.2f}")
                            st.rerun()
                        else:
                            st.toast("Insufficient cash to buy", icon="⚠️")
                elif pred == "SELL" and already_holding:
                    if col_exec.button(
                        "Execute SELL", type="secondary", key=f"sell_{result.ticker}",
                        use_container_width=True,
                    ):
                        trade = trader.sell(result.ticker, result.price, result.date, conf)
                        if trade:
                            st.toast(f"Sold {trade.shares} shares of {result.ticker} for ${trade.total:,.2f}")
                            st.rerun()
                elif already_holding:
                    col_exec.markdown("**HOLD** — already in portfolio")
                else:
                    col_exec.markdown("**HOLD** — no action recommended")

                if already_holding:
                    cost = trader.cost_basis(result.ticker)
                    shares = trader.holdings[result.ticker]
                    value = shares * result.price
                    pl = value - (shares * cost)
                    col_status.metric(
                        f"{shares} sh",
                        f"${value:,.2f}",
                        delta=f"${pl:+,.2f}",
                    )
                else:
                    col_status.empty()
                st.divider()

            st.markdown(f"**{rec_text}**")
            st.caption(
                f"{result.prediction_source}"
                + (
                    f" · trained on {result.training_samples} historical rows"
                    f" · train fit {result.training_accuracy:.0%}"
                    if result.training_accuracy is not None
                    else ""
                )
            )

            explanation_intro, explanation_details = prediction_explanation(result)
            st.markdown("**Why This Prediction**")
            st.write(explanation_intro)
            for detail in explanation_details:
                st.caption(detail)

            neutral_color = "#90a4ae"
            buy_color = "#26a69a"
            sell_color = "#ef5350"

            for action, prob in sorted(
                result.action_probabilities.items(), key=lambda x: -x[1]
            ):
                bar_color = (
                    buy_color if action == "BUY"
                    else sell_color if action == "SELL"
                    else neutral_color
                )
                pct = int(prob * 100)
                cols = st.columns([1, 3])
                cols[0].write(f"{action}")
                cols[1].markdown(
                    f"""
                    <div style="background:#eee;border-radius:4px;height:20px;width:100%;position:relative;">
                        <div style="background:{bar_color};width:{pct}%;height:20px;border-radius:4px;"></div>
                        <div style="position:absolute;top:0;right:4px;font-size:11px;line-height:20px;">{prob:.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.divider()
            st.markdown("**Market State — Indicators**")
            for key in ["trend", "volume", "heikin_ashi", "stochastic", "rsi", "macd", "bb", "ma_cross", "volatility", "candle", "obv", "atr", "price_action", "market_context", "relative"]:
                val = result.indicators.get(key, "")
                if not val:
                    continue
                description = INDICATOR_VALUES.get(val, val)
                with st.expander(
                    f"**{INDICATOR_LABELS.get(key, key)}** — `{val}`",
                    expanded=True,
                ):
                    st.caption(description)

            st.divider()
            st.markdown("**Model Next-Token Predictions**")
            st.caption("What the model expects to see next:")
            for token, prob in result.top_k_tokens[:5]:
                is_action = token in ("BUY", "SELL", "HOLD")
                st.markdown(
                    f"`{token}` — {prob:.1%}"
                )

            st.divider()
            score = result.indicator_score
            score_color = "green" if score > 0 else "red" if score < 0 else "gray"
            st.markdown(
                f"**Indicator Score:** "
                f"<span style='color:{score_color};font-size:1.2em;'>"
                f"{score:+.2f}</span> (-1 to +1)",
                unsafe_allow_html=True,
            )

            st.caption(
                "Score based on current indicator signals. "
                "Positive = bullish, Negative = bearish."
            )
