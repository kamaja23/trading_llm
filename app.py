"""
Trading LLM - Stock Analysis Frontend

Streamlit app that uses the trained model to analyze stocks
and provide BUY/SELL/HOLD recommendations with visual charts.

Usage:
  pip install streamlit plotly
  streamlit run app.py
"""

import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

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
    StockAnalysisAgent,
    StockAnalysisError,
    INDICATOR_VALUES,
    INDICATOR_LABELS,
    BULLISH_INDICATORS,
    BEARISH_INDICATORS,)
from utils.market_data import MarketDataError, resolve_stockanalysis_symbol

st.set_page_config(
    page_title="Trading LLM — Stock Analysis Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_agent():
    try:
        return StockAnalysisAgent()
    except StockAnalysisError as e:
        st.error(str(e))
        st.stop()


agent = get_agent()

DEMO_VERSION = "clarify_unresolved_stock_lookup"
DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "GOOGL"]
DEFAULT_PRIVATE_COMPANIES = [
    ("OpenAI", "Private company · no public stock ticker"),
    ("Anthropic", "Private company · no public stock ticker"),
]
DEMO_START_DATE = "2026-02-27"
DEMO_END_DATE = "2026-04-30"
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
PRIVATE_ALIASES = {
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
    return PRIVATE_ALIASES.get(raw.upper().strip())


def analyze_demo_stock(ticker: str):
    return agent.analyze(
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


def add_private_company(private_company):
    if private_company not in st.session_state.private_companies:
        st.session_state.private_companies.append(private_company)
        st.toast(f"{private_company[0]} added to private list")
        st.rerun()
    else:
        st.toast(f"{private_company[0]} is already in private list")


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
    st.rerun()


if st.session_state.stock_lookup_dialog:
    show_lookup_dialog(st.session_state.stock_lookup_dialog)

st.sidebar.title("Watchlist")

with st.sidebar:
    with st.form("add_stock_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        raw_ticker = col1.text_input(
            "Ticker", placeholder="e.g. Apple or AAPL", label_visibility="collapsed"
        ).strip()
        submitted = col2.form_submit_button("+ Add", width="stretch")

    new_ticker = normalize_ticker(raw_ticker)
    if submitted:
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
                        st.session_state.stock_lookup_dialog = (
                            "Could not find any relevant information."
                        )
                        st.rerun()
                    else:
                        st.session_state.pending_unresolved_stock = raw_ticker.upper()
                        st.warning(unresolved_stock_message(raw_ticker))

    st.divider()

    for ticker in list(st.session_state.watchlist):
        cols = st.columns([3, 1, 1])
        analyzed = ticker in st.session_state.analyses
        cols[0].markdown(company_label_html(ticker, analyzed), unsafe_allow_html=True)
        if cols[1].button("🔄", key=f"ref_{ticker}", help="Refresh"):
            st.session_state.analyses.pop(ticker, None)
            st.rerun()
        if cols[2].button("✕", key=f"rm_{ticker}", help="Remove"):
            st.session_state.watchlist.remove(ticker)
            st.session_state.analyses.pop(ticker, None)
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

    st.divider()
    cols = st.columns(2)
    if cols[0].button("Refresh All", width="stretch", type="primary"):
        st.session_state.analyses = {}
        st.rerun()
    if cols[1].button("Clear All", width="stretch"):
        st.session_state.watchlist = []
        st.session_state.analyses = {}
        st.rerun()

    st.divider()
    st.caption(
        "Model analyzes 4 technical indicators (trend, volume, "
        "Heikin Ashi, Stochastic). Demo loads local big-tech data from "
        f"{DEMO_START_DATE} to {DEMO_END_DATE}. "
        "Not financial advice."
    )

st.title("Trading LLM — Stock Analysis Agent")

if not st.session_state.watchlist:
    st.info("Add stocks to your watchlist using the sidebar.")
    st.stop()

st.markdown(f"<style>{stock_tab_hover_css()}</style>", unsafe_allow_html=True)

tabs = st.tabs([
    f"{company_name(t)} ({t})"
    for t in st.session_state.watchlist
])

for idx, ticker in enumerate(st.session_state.watchlist):
    with tabs[idx]:
        if ticker not in st.session_state.analyses:
            with st.spinner(f"Fetching data, training, and analyzing {ticker}..."):
                try:
                    result = analyze_demo_stock(ticker)
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
            rec_text, rec_color = agent.recommendation_text(pred, conf)

            st.metric(
                label=f"{company_name(result.ticker)} · {result.ticker} · {result.date}",
                value=f"${result.price:.2f}",
            )

            bg_color = (
                "#e8f5e9" if pred == "BUY"
                else "#ffebee" if pred == "SELL"
                else "#f5f5f5"
            )
            st.markdown(
                f"""
                <div style="
                    background: {bg_color};
                    border-radius: 12px;
                    padding: 16px;
                    text-align: center;
                    margin: 12px 0;
                ">
                    <div style="font-size: 14px; color: #666;">Model Prediction</div>
                    <div style="font-size: 38px; font-weight: 700; color: {rec_color};">
                        {pred}
                    </div>
                    <div style="font-size: 16px; margin-top: 4px;">
                        {conf:.1%} confidence
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
            for key in ["trend", "volume", "heikin_ashi", "stochastic"]:
                val = result.indicators[key]
                description = INDICATOR_VALUES.get(val, val)
                is_bullish = "Up" in val or (
                    key == "stochastic" and val == "STO_Oversold"
                )
                is_bearish = "Down" in val or (
                    key == "stochastic" and val == "STO_Overbought"
                )
                with st.expander(
                    f"**{INDICATOR_LABELS[key]}** — `{val}`",
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
