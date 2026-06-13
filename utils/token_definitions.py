"""
Token definitions for the trading language.

This module defines all the discrete tokens that can appear in trading sequences.
Each token represents a specific market fact or state.
"""

# Symbol tokens - represent the asset being traded
SYMBOL_TOKENS = [
    "<SYM_SPY>",   # S&P 500 ETF
    "<SYM_QQQ>",   # NASDAQ ETF
    "<SYM_DIA>",   # Dow Jones ETF
    "<SYM_IWM>",   # Russell 2000 ETF
    "<SYM_AAPL>",  # Apple
    "<SYM_MSFT>",  # Microsoft
    "<SYM_NVDA>",  # NVIDIA
    "<SYM_GOOGL>", # Alphabet
    "<SYM_AMZN>",  # Amazon
    "<SYM_META>",  # Meta
    "<SYM_TSLA>",  # Tesla
    "<SYM_BRK_B>", # Berkshire Hathaway
    "<SYM_JPM>",   # JPMorgan Chase
    "<SYM_V>",     # Visa
    "<SYM_MA>",    # Mastercard
    "<SYM_UNH>",   # UnitedHealth
    "<SYM_HD>",    # Home Depot
    "<SYM_JNJ>",   # Johnson & Johnson
    "<SYM_PG>",    # Procter & Gamble
    "<SYM_XOM>",   # Exxon Mobil
    "<SYM_CVX>",   # Chevron
    "<SYM_WMT>",   # Walmart
    "<SYM_BA>",    # Boeing
    "<SYM_CAT>",   # Caterpillar
    "<SYM_DIS>",   # Disney
    "<SYM_KO>",    # Coca-Cola
    "<SYM_PEP>",   # PepsiCo
    "<SYM_NKE>",   # Nike
    "<SYM_INTC>",  # Intel
    "<SYM_AMD>",   # AMD
    "<SYM_CRM>",   # Salesforce
    "<SYM_ADBE>",  # Adobe
    "<SYM_NFLX>",  # Netflix
    "<SYM_AVGO>",  # Broadcom
    "<SYM_TMO>",   # Thermo Fisher
    "<SYM_ABBV>",  # AbbVie
    "<SYM_COST>",  # Costco
    "<SYM_MCD>",   # McDonald's
    "<SYM_VZ>",    # Verizon
    "<SYM_T>",     # AT&T
    "<SYM_WFC>",   # Wells Fargo
    "<SYM_GS>",    # Goldman Sachs
    "<SYM_MS>",    # Morgan Stanley
    "<SYM_BAC>",   # Bank of America
    "<SYM_C>",     # Citigroup
    "<SYM_ORCL>",  # Oracle
    "<SYM_IBM>",   # IBM
    "<SYM_CSCO>",  # Cisco
    "<SYM_QCOM>",  # Qualcomm
    "<SYM_TXN>",   # Texas Instruments
]

# Timeframe tokens - represent the granularity of data
TIMEFRAME_TOKENS = [
    "<TF_DAILY>",
    "<TF_WEEKLY>",
    "<TF_MONTHLY>",
    "<TF_60MIN>",
    "<TF_30MIN>",
    "<TF_15MIN>",
    "<TF_5MIN>",
    "<TF_1MIN>",
    "<TF_4HOUR>",
    "<TF_1HOUR>",
]

# State/Trend tokens - represent overall market direction
STATE_TOKENS = [
    "TR_StrongUp",    # Strong uptrend
    "TR_Up",          # Moderate uptrend
    "TR_WeakUp",      # Weak uptrend
    "TR_Flat",        # Sideways/ranging
    "TR_WeakDown",    # Weak downtrend
    "TR_Down",        # Moderate downtrend
    "TR_StrongDown",  # Strong downtrend
]

# Volume tokens - represent trading volume relative to average
VOLUME_TOKENS = [
    "VOL_Surge",    # Very high volume (extreme)
    "VOL_High",     # Above average
    "VOL_Normal",   # Near average
    "VOL_Low",      # Below average
    "VOL_Dead",     # Very low volume (extreme)
]

# Heikin Ashi tokens
HA_TOKENS = [
    "HA_UpCross",     # Bullish crossover
    "HA_DownCross",   # Bearish crossover
    "HA_UpTrend",     # Sustained bullish
    "HA_DownTrend",   # Sustained bearish
    "HA_Neutral",     # No clear signal
]

# Stochastic oscillator tokens
STO_TOKENS = [
    "STO_Cross",      # Bullish crossover
    "STO_DownCross",  # Bearish crossover
    "STO_Oversold",   # Below 20
    "STO_Overbought", # Above 80
    "STO_NoCross",    # No crossover
]

# RSI tokens
RSI_TOKENS = [
    "RSI_Oversold",     # Below 30
    "RSI_Overbought",   # Above 70
    "RSI_Bullish",      # 50-70
    "RSI_Bearish",      # 30-50
    "RSI_Neutral",      # Exactly around 50
]

# MACD tokens
MACD_TOKENS = [
    "MACD_BullishCross",   # Signal line cross above
    "MACD_BearishCross",   # Signal line cross below
    "MACD_Positive",       # MACD above zero
    "MACD_Negative",       # MACD below zero
    "MACD_Neutral",        # No signal
    "MACD_DivBullish",     # Bullish divergence
    "MACD_DivBearish",     # Bearish divergence
]

# Bollinger Bands tokens
BB_TOKENS = [
    "BB_UpperTouch",    # Price touches upper band
    "BB_LowerTouch",    # Price touches lower band
    "BB_AboveMid",      # Price above middle band
    "BB_BelowMid",      # Price below middle band
    "BB_Squeeze",       # Band squeeze (low volatility)
    "BB_Expand",        # Band expansion (high volatility)
    "BB_Neutral",       # Normal range
]

# Moving Average Crossover tokens
MA_CROSS_TOKENS = [
    "MA_GoldenCross",   # 50-day crosses above 200-day
    "MA_DeathCross",    # 50-day crosses below 200-day
    "MA_Bullish",       # Short-term above long-term
    "MA_Bearish",       # Short-term below long-term
    "MA_Neutral",       # Mixed signals
]

# Volatility tokens
VOLATILITY_TOKENS = [
    "VIX_High",     # High volatility
    "VIX_Moderate", # Moderate volatility
    "VIX_Low",      # Low volatility
]

# Candle pattern tokens
CANDLE_TOKENS = [
    "CP_Doji",             # Indecision
    "CP_Hammer",           # Bullish reversal
    "CP_ShootingStar",     # Bearish reversal
    "CP_EngulfBull",       # Bullish engulfing
    "CP_EngulfBear",       # Bearish engulfing
    "CP_MorningStar",      # Bullish reversal 3-candle
    "CP_EveningStar",      # Bearish reversal 3-candle
    "CP_MarubozuBull",     # Strong bullish
    "CP_MarubozuBear",     # Strong bearish
    "CP_SpinningTop",      # Indecision
    "CP_NoPattern",        # No recognizable pattern
]

# On-Balance Volume tokens
OBV_TOKENS = [
    "OBV_Rising",     # OBV trending up
    "OBV_Falling",    # OBV trending down
    "OBV_Flat",       # OBV sideways
    "OBV_DivBull",    # Bullish divergence (price down, OBV up)
    "OBV_DivBear",    # Bearish divergence (price up, OBV down)
]

# Average True Range tokens
ATR_TOKENS = [
    "ATR_High",     # High volatility
    "ATR_Rising",   # Volatility increasing
    "ATR_Falling",  # Volatility decreasing
    "ATR_Low",      # Low volatility
]

# Price action tokens
PRICE_ACTION_TOKENS = [
    "PA_BreakoutUp",      # Price breaks above resistance
    "PA_BreakoutDown",    # Price breaks below support
    "PA_ResistanceTest",  # Testing resistance level
    "PA_SupportTest",     # Testing support level
    "PA_NewHigh",         # New high
    "PA_NewLow",          # New low
    "PA_RangeBound",      # Trading in range
    "PA_GapUp",           # Gap up open
    "PA_GapDown",         # Gap down open
]

# Market context tokens
MARKET_CONTEXT_TOKENS = [
    "MKT_Bullish",     # Market in bullish phase
    "MKT_Bearish",     # Market in bearish phase
    "MKT_Neutral",     # Mixed market
    "MKT_Volatile",    # High volatility environment
    "MKT_Calm",        # Low volatility environment
    "MKT_Overbought",  # Broad market overbought
    "MKT_Oversold",    # Broad market oversold
]

# Sentiment tokens (news + social media)
SENTIMENT_TOKENS = [
    "SENT_StrongPos",    # Very positive sentiment
    "SENT_Positive",     # Positive sentiment
    "SENT_Neutral",      # Neutral sentiment
    "SENT_Negative",     # Negative sentiment
    "SENT_StrongNeg",    # Very negative sentiment
    "SENT_NoData",       # No sentiment data available
]

# News impact tokens
NEWS_TOKENS = [
    "NEWS_MajorPos",    # Major positive news
    "NEWS_MinorPos",    # Minor positive news
    "NEWS_Neutral",     # Neutral news
    "NEWS_MinorNeg",    # Minor negative news
    "NEWS_MajorNeg",    # Major negative news
    "NEWS_None",        # No significant news
]

# Social media activity tokens
SOCIAL_TOKENS = [
    "SOC_HighBuzz",     # High social media activity
    "SOC_Moderate",     # Moderate activity
    "SOC_Low",          # Low activity
    "SOC_Silent",       # Very low / no activity
]

# Sector-relative performance tokens
RELATIVE_TOKENS = [
    "REL_Strong",    # Strong vs sector
    "REL_Inline",    # In line with sector
    "REL_Weak",      # Weak vs sector
    "REL_Unknown",   # Cannot determine
]

# Action tokens - represent trading decisions
ACTION_TOKENS = [
    "BUY",
    "SELL",
    "HOLD",
]

# Indicator tokens (legacy grouping, all non-action state tokens)
INDICATOR_TOKENS = (
    STATE_TOKENS +
    VOLUME_TOKENS +
    HA_TOKENS +
    STO_TOKENS +
    RSI_TOKENS +
    MACD_TOKENS +
    BB_TOKENS +
    MA_CROSS_TOKENS +
    VOLATILITY_TOKENS +
    CANDLE_TOKENS +
    OBV_TOKENS +
    ATR_TOKENS +
    PRICE_ACTION_TOKENS +
    MARKET_CONTEXT_TOKENS +
    SENTIMENT_TOKENS +
    NEWS_TOKENS +
    SOCIAL_TOKENS +
    RELATIVE_TOKENS
)

# All custom tokens combined
ALL_CUSTOM_TOKENS = (
    SYMBOL_TOKENS +
    TIMEFRAME_TOKENS +
    INDICATOR_TOKENS +
    ACTION_TOKENS
)

# Token to numeric mapping (for analysis)
TOKEN_TO_ID = {token: idx for idx, token in enumerate(ALL_CUSTOM_TOKENS)}
ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}

# Token category sets for validation/generation
TREND_TOKENS = set(STATE_TOKENS)
VOLUME_STATE_TOKENS = set(VOLUME_TOKENS)
SENTIMENT_STATE_TOKENS = set(SENTIMENT_TOKENS)
NEWS_STATE_TOKENS = set(NEWS_TOKENS)
SOCIAL_STATE_TOKENS = set(SOCIAL_TOKENS)

# Colormap for visualization
TOKEN_CATEGORY_COLORS = {
    "symbol": "#4CAF50",
    "timeframe": "#2196F3",
    "trend": "#FF9800",
    "volume": "#9C27B0",
    "ha": "#E91E63",
    "stochastic": "#00BCD4",
    "rsi": "#FF5722",
    "macd": "#795548",
    "bb": "#607D8B",
    "ma_cross": "#3F51B5",
    "volatility": "#009688",
    "candle": "#CDDC39",
    "obv": "#FFC107",
    "atr": "#E040FB",
    "price_action": "#536DFE",
    "market_context": "#FF6F00",
    "sentiment": "#C51162",
    "news": "#1B5E20",
    "social": "#311B92",
    "relative": "#00838F",
    "action": "#D32F2F",
}


def get_symbol_token(symbol: str) -> str:
    """Convert a symbol string to its token representation."""
    token = f"<SYM_{symbol.upper()}>"
    if token not in SYMBOL_TOKENS:
        return "<SYM_SPY>"
    return token


def get_timeframe_token(timeframe: str) -> str:
    """Convert a timeframe string to its token representation."""
    token = f"<TF_{timeframe.upper()}>"
    if token not in TIMEFRAME_TOKENS:
        return "<TF_DAILY>"
    return token


def validate_sequence(sequence: str) -> bool:
    """
    Validate that a sequence contains only known tokens.
    
    Args:
        sequence: Space-separated token sequence
        
    Returns:
        True if all tokens are valid, False otherwise
    """
    tokens = sequence.split()
    if not tokens:
        return False
    return all(token in ALL_CUSTOM_TOKENS for token in tokens)


def get_token_category(token: str) -> str:
    """Determine which category a token belongs to."""
    if token in SYMBOL_TOKENS:
        return "symbol"
    if token in TIMEFRAME_TOKENS:
        return "timeframe"
    if token in STATE_TOKENS:
        return "trend"
    if token in VOLUME_TOKENS:
        return "volume"
    if token in HA_TOKENS:
        return "ha"
    if token in STO_TOKENS:
        return "stochastic"
    if token in RSI_TOKENS:
        return "rsi"
    if token in MACD_TOKENS:
        return "macd"
    if token in BB_TOKENS:
        return "bb"
    if token in MA_CROSS_TOKENS:
        return "ma_cross"
    if token in VOLATILITY_TOKENS:
        return "volatility"
    if token in CANDLE_TOKENS:
        return "candle"
    if token in OBV_TOKENS:
        return "obv"
    if token in ATR_TOKENS:
        return "atr"
    if token in PRICE_ACTION_TOKENS:
        return "price_action"
    if token in MARKET_CONTEXT_TOKENS:
        return "market_context"
    if token in SENTIMENT_TOKENS:
        return "sentiment"
    if token in NEWS_TOKENS:
        return "news"
    if token in SOCIAL_TOKENS:
        return "social"
    if token in RELATIVE_TOKENS:
        return "relative"
    if token in ACTION_TOKENS:
        return "action"
    return "unknown"


if __name__ == "__main__":
    print("=== Trading Language Token Definitions ===\n")
    
    print(f"Total custom tokens: {len(ALL_CUSTOM_TOKENS)}")
    print(f"  Symbols: {len(SYMBOL_TOKENS)}")
    print(f"  Timeframes: {len(TIMEFRAME_TOKENS)}")
    print(f"  Trend: {len(STATE_TOKENS)}")
    print(f"  Volume: {len(VOLUME_TOKENS)}")
    print(f"  Heikin Ashi: {len(HA_TOKENS)}")
    print(f"  Stochastic: {len(STO_TOKENS)}")
    print(f"  RSI: {len(RSI_TOKENS)}")
    print(f"  MACD: {len(MACD_TOKENS)}")
    print(f"  Bollinger Bands: {len(BB_TOKENS)}")
    print(f"  MA Cross: {len(MA_CROSS_TOKENS)}")
    print(f"  Volatility: {len(VOLATILITY_TOKENS)}")
    print(f"  Candle Patterns: {len(CANDLE_TOKENS)}")
    print(f"  OBV: {len(OBV_TOKENS)}")
    print(f"  ATR: {len(ATR_TOKENS)}")
    print(f"  Price Action: {len(PRICE_ACTION_TOKENS)}")
    print(f"  Market Context: {len(MARKET_CONTEXT_TOKENS)}")
    print(f"  Sentiment: {len(SENTIMENT_TOKENS)}")
    print(f"  News: {len(NEWS_TOKENS)}")
    print(f"  Social: {len(SOCIAL_TOKENS)}")
    print(f"  Relative: {len(RELATIVE_TOKENS)}")
    print(f"  Actions: {len(ACTION_TOKENS)}")
    
    # Example sequence
    example = (
        "<SYM_SPY> <TF_DAILY> "
        "TR_Up VOL_Normal HA_Neutral STO_NoCross "
        "RSI_Bullish MACD_Positive BB_Neutral MA_Bullish "
        "VIX_Moderate CP_NoPattern OBV_Rising ATR_Rising "
        "PA_RangeBound MKT_Neutral SENT_Neutral NEWS_None "
        "SOC_Low REL_Inline "
        "HOLD"
    )
    print(f"\nExample sequence ({len(example.split())} tokens):\n{example}")
    print(f"Valid: {validate_sequence(example)}")
