"""
Token definitions for the trading language.

This module defines all the discrete tokens that can appear in trading sequences.
Each token represents a specific market fact or state.
"""

# Symbol tokens - represent the asset being traded
SYMBOL_TOKENS = [
    "<SYM_SPY>",  # S&P 500 ETF
    "<SYM_QQQ>",  # NASDAQ ETF
    "<SYM_DIA>",  # Dow Jones ETF
    "<SYM_IWM>",  # Russell 2000 ETF
]

# Timeframe tokens - represent the granularity of data
TIMEFRAME_TOKENS = [
    "<TF_DAILY>",
    "<TF_WEEKLY>",
    "<TF_60MIN>",
    "<TF_15MIN>",
]

# State/Trend tokens - represent overall market direction
STATE_TOKENS = [
    "ST_UpTrend",      # Price trending upward
    "ST_DownTrend",    # Price trending downward
    "ST_Flat",         # Sideways/ranging market
]

# Volume tokens - represent trading volume relative to average
VOLUME_TOKENS = [
    "Hi_Volume",   # Above average volume
    "Lo_Volume",   # Below average volume
    "Avg_Volume",  # Near average volume
]

# Indicator tokens - represent technical indicator signals
# HA = Heikin Ashi, STO = Stochastic
INDICATOR_TOKENS = [
    "HA_UpCross",     # Heikin Ashi bullish signal
    "HA_DownCross",   # Heikin Ashi bearish signal
    "HA_Neutral",     # No clear HA signal
    "STO_Cross",      # Stochastic crossover
    "STO_NoCross",    # No stochastic crossover
    "STO_Oversold",   # Stochastic oversold
    "STO_Overbought", # Stochastic overbought
]

# Action tokens - represent trading decisions
ACTION_TOKENS = [
    "BUY",
    "SELL",
    "HOLD",
]

# All custom tokens combined
ALL_CUSTOM_TOKENS = (
    SYMBOL_TOKENS +
    TIMEFRAME_TOKENS +
    STATE_TOKENS +
    VOLUME_TOKENS +
    INDICATOR_TOKENS +
    ACTION_TOKENS
)

# Token to numeric mapping (for analysis)
TOKEN_TO_ID = {token: idx for idx, token in enumerate(ALL_CUSTOM_TOKENS)}
ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}


def get_symbol_token(symbol: str) -> str:
    """Convert a symbol string to its token representation."""
    token = f"<SYM_{symbol.upper()}>"
    if token not in SYMBOL_TOKENS:
        # Default to SPY if unknown symbol
        return "<SYM_SPY>"
    return token


def get_timeframe_token(timeframe: str) -> str:
    """Convert a timeframe string to its token representation."""
    token = f"<TF_{timeframe.upper()}>"
    if token not in TIMEFRAME_TOKENS:
        # Default to daily if unknown
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
    return all(token in ALL_CUSTOM_TOKENS for token in tokens)


if __name__ == "__main__":
    # Print all token categories for reference
    print("=== Trading Language Token Definitions ===\n")
    
    print("Symbol Tokens:", SYMBOL_TOKENS)
    print("Timeframe Tokens:", TIMEFRAME_TOKENS)
    print("State Tokens:", STATE_TOKENS)
    print("Volume Tokens:", VOLUME_TOKENS)
    print("Indicator Tokens:", INDICATOR_TOKENS)
    print("Action Tokens:", ACTION_TOKENS)
    
    print(f"\nTotal custom tokens: {len(ALL_CUSTOM_TOKENS)}")
    
    # Example sequence
    example = "<SYM_SPY> <TF_DAILY> ST_UpTrend Hi_Volume HA_UpCross STO_Cross BUY"
    print(f"\nExample sequence:\n{example}")
    print(f"Valid: {validate_sequence(example)}")
