"""
Data generator module for converting OHLCV price data into token sequences.

This module downloads historical data and converts it into the trading language
format suitable for training an LLM. Now with expanded vocabulary including
RSI, MACD, Bollinger Bands, candle patterns, and sentiment.
"""

import pandas as pd
from typing import List, Optional, Tuple
import os
from pathlib import Path

from utils.market_data import MarketDataError, fetch_market_data
from utils.token_definitions import get_symbol_token, get_timeframe_token
from utils.indicators import add_all_indicators
from utils.news_sentiment import fetch_aggregated_sentiment


def download_price_data(
    symbol: str,
    start_date: str = "2020-01-01",
    end_date: str = "2024-01-01",
    save_path: str = None,
    use_sample_data: bool = False,
    sample_data_path: str = None
) -> pd.DataFrame:
    """
    Download historical price data, or load sample data if offline.
    
    Args:
        symbol: Ticker symbol (e.g., 'SPY')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        save_path: Optional path to save raw data
        use_sample_data: If True, load from sample data instead of downloading
        sample_data_path: Path to sample CSV file
        
    Returns:
        DataFrame with OHLCV data
    """
    if use_sample_data or sample_data_path:
        if sample_data_path and os.path.exists(sample_data_path):
            print(f"Loading sample data from {sample_data_path}...")
            df = pd.read_csv(sample_data_path, index_col=0, parse_dates=True)
        else:
            default_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'SPY_sample_data.csv')
            if os.path.exists(default_path):
                print(f"Loading sample data from {default_path}...")
                df = pd.read_csv(default_path, index_col=0, parse_dates=True)
            else:
                raise FileNotFoundError(
                    f"Sample data not found. Please provide sample_data_path or ensure "
                    f"sample data exists at {default_path}"
                )
        
        print(f"Loaded {len(df)} days of sample data")
        
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index < end_date]
        
        print(f"Filtered to {len(df)} days in date range")
    else:
        print(f"Downloading {symbol} data ({start_date} to {end_date})...")
        
        try:
            df = fetch_market_data(symbol, start_date=start_date, end_date=end_date)
        except MarketDataError as e:
            print(f"\n  Failed to download data: {e}")
            print("  Attempting to use sample data instead...")
            return download_price_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                save_path=save_path,
                use_sample_data=True
            )
        
        if df.empty:
            print("\n  No data downloaded, attempting to use sample data instead...")
            return download_price_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                save_path=save_path,
                use_sample_data=True
            )
        
        print(f"Downloaded {len(df)} days of data")
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns in data. Found: {df.columns.tolist()}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print(f"Saved data to {save_path}")
    
    return df


def generate_token_sequences(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "DAILY",
    include_sentiment: bool = False,
) -> List[str]:
    """
    Convert OHLCV DataFrame into token sequences with expanded vocabulary.

    Args:
        df: DataFrame with OHLCV data and indicators
        symbol: Trading symbol
        timeframe: Timeframe string
        include_sentiment: If True, attempt to fetch news/sentiment data

    Returns:
        List of token sequences (strings)
    """
    sequences = []
    
    sym_token = get_symbol_token(symbol)
    tf_token = get_timeframe_token(timeframe)
    
    df = add_all_indicators(df)
    df = df.dropna()
    
    sentiment_data = None
    if include_sentiment:
        try:
            sentiment_data = fetch_aggregated_sentiment(symbol)
        except Exception:
            sentiment_data = None
    
    for idx, row in df.iterrows():
        sequence_parts = [
            sym_token,
            tf_token,
            row['trend_token'],
            row['volume_token'],
            row['ha_token'],
            row['sto_token'],
            row['rsi_token'],
            row['macd_token'],
            row['bb_token'],
            row['ma_cross_token'],
            row['volatility_token'],
            row['candle_token'],
            row['obv_token'],
            row['atr_token'],
            row['price_action_token'],
            row['market_context_token'],
            row['relative_token'],
        ]
        
        if sentiment_data:
            sequence_parts.append(sentiment_data['sentiment_token'])
            sequence_parts.append(sentiment_data['news_token'])
            sequence_parts.append(sentiment_data['social_token'])
        
        sequence = " ".join(sequence_parts) + " " + row['action_token']
        sequences.append(sequence)
    
    return sequences


def split_sequences(
    sequences: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split sequences into train, validation, and test sets.
    
    Args:
        sequences: List of token sequences
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        
    Returns:
        Tuple of (train_sequences, val_sequences, test_sequences)
    """
    n = len(sequences)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:train_size + val_size]
    test_sequences = sequences[train_size + val_size:]
    
    return train_sequences, val_sequences, test_sequences


def save_sequences(
    sequences: List[str],
    filepath: str
) -> None:
    """
    Save sequences to a text file (one per line).
    
    Args:
        sequences: List of token sequences
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        for seq in sequences:
            f.write(seq + '\n')
    
    print(f"Saved {len(sequences)} sequences to {filepath}")


def load_sequences(filepath: str) -> List[str]:
    """
    Load sequences from a text file.
    
    Args:
        filepath: Input file path
        
    Returns:
        List of token sequences
    """
    with open(filepath, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]
    
    return sequences


def analyze_sequences(sequences: List[str]) -> None:
    """
    Print statistics about generated sequences.
    
    Args:
        sequences: List of token sequences
    """
    print(f"\n=== Sequence Analysis ===")
    print(f"Total sequences: {len(sequences)}")
    
    if not sequences:
        return
    
    lengths = [len(seq.split()) for seq in sequences]
    print(f"Tokens per sequence: {lengths[0]} (all sequences should be same length)")
    
    action_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    for seq in sequences:
        tokens = seq.split()
        action = tokens[-1]
        if action in action_counts:
            action_counts[action] += 1
    
    print(f"\nAction distribution:")
    for action, count in action_counts.items():
        pct = (count / len(sequences)) * 100
        print(f"  {action}: {count} ({pct:.1f}%)")
    
    print(f"\nToken category distribution (first tokens only):")
    from utils.token_definitions import get_token_category
    for seq in sequences[:1]:
        tokens = seq.split()
        for i, token in enumerate(tokens):
            cat = get_token_category(token)
            print(f"  [{i:2d}] {token:25s} → {cat}")
    
    print(f"\nFirst 3 sequences:")
    for seq in sequences[:3]:
        print(f"  {seq}")
    
    print(f"\nLast 3 sequences:")
    for seq in sequences[-3:]:
        print(f"  {seq}")


if __name__ == "__main__":
    print("Testing expanded data generation pipeline...\n")
    
    from utils.token_definitions import ALL_CUSTOM_TOKENS
    print(f"Vocabulary size: {len(ALL_CUSTOM_TOKENS)} custom tokens\n")
    
    df = download_price_data('SPY', start_date='2023-01-01', end_date='2024-01-01')
    
    sequences = generate_token_sequences(df, symbol='SPY', timeframe='DAILY')
    
    analyze_sequences(sequences)
    
    train, val, test = split_sequences(sequences)
    print(f"\nSplit sizes: Train={len(train)}, Val={len(val)}, Test={len(test)}")
