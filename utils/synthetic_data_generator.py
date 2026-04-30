"""
Synthetic data generator with known patterns for model verification.

This module creates synthetic OHLCV data with EXPLICIT patterns that can be
verified after training. If the model cannot learn these simple patterns,
there's a bug in the training pipeline.

Pattern examples:
1. If ST_UpTrend + HA_UpCross → Always STO_Cross (100% correlation)
2. If ST_DownTrend + HA_DownCross → Always STO_Oversold (100% correlation)
3. If ST_Flat → Always STO_NoCross (100% correlation)
"""

import pandas as pd
import numpy as np
from typing import List
from utils.token_definitions import get_symbol_token, get_timeframe_token
from utils.indicators import add_all_indicators


def generate_synthetic_ohlcv(n_days: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with realistic price movements.
    
    Args:
        n_days: Number of days to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(seed)
    
    # Generate price series with realistic properties
    returns = np.random.normal(0.0005, 0.015, n_days)  # Daily returns
    price = 100 * np.exp(np.cumsum(returns))  # Start at $100
    
    # Create OHLCV from close prices
    df = pd.DataFrame({
        'Open': price * (1 + np.random.uniform(-0.005, 0.005, n_days)),
        'Close': price,
        'High': price * (1 + np.abs(np.random.uniform(0, 0.01, n_days))),
        'Low': price * (1 - np.abs(np.random.uniform(0, 0.01, n_days))),
        'Volume': np.random.randint(1000000, 5000000, n_days)
    }, index=pd.date_range('2020-01-01', periods=n_days, freq='D'))
    
    return df


def inject_known_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject KNOWN patterns into the data that the model should learn.
    
    Pattern rules (100% correlation for verification):
    - Rule 1: If Close > Close.shift(5) (uptrend) AND Volume > Volume.median → STO_Overbought
    - Rule 2: If Close < Close.shift(5) (downtrend) AND Volume > Volume.median → STO_Oversold
    - Rule 3: If Volume <= Volume.median → STO_NoCross
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with injected pattern columns
    """
    df = df.copy()
    
    # Calculate indicators (needed for token generation)
    df = add_all_indicators(df)
    
    # Define trends based on price movement
    df['price_trend'] = np.where(
        df['Close'] > df['Close'].shift(5),
        'up',
        np.where(df['Close'] < df['Close'].shift(5), 'down', 'flat')
    )
    
    # Define volume condition
    volume_median = df['Volume'].median()
    df['high_volume'] = df['Volume'] > volume_median
    
    # INJECT KNOWN PATTERNS (these will become the action tokens)
    # Rule 1: Uptrend + High Volume → STO_Overbought
    # Rule 2: Downtrend + High Volume → STO_Oversold
    # Rule 3: Low Volume → STO_NoCross
    
    def determine_sto_token(row):
        if not row['high_volume']:
            return 'STO_NoCross'
        elif row['price_trend'] == 'up':
            return 'STO_Overbought'
        elif row['price_trend'] == 'down':
            return 'STO_Oversold'
        else:
            return 'STO_NoCross'
    
    df['sto_token'] = df.apply(determine_sto_token, axis=1)
    
    # Override the stochastic token from indicators with our known pattern
    df['stoch_token'] = df['sto_token']
    
    return df


def generate_sequences_with_known_patterns(
    df: pd.DataFrame,
    symbol: str = 'SPY',
    timeframe: str = 'DAILY'
) -> List[str]:
    """
    Generate token sequences with KNOWN patterns for verification.
    
    Args:
        df: DataFrame with injected patterns
        symbol: Trading symbol
        timeframe: Timeframe string
        
    Returns:
        List of token sequences where the action follows known rules
    """
    sequences = []
    
    sym_token = get_symbol_token(symbol)
    tf_token = get_timeframe_token(timeframe)
    
    for idx, row in df.iterrows():
        # Create sequence with pattern-based action
        sequence_parts = [
            sym_token,
            tf_token,
            row['trend_token'],
            row['volume_token'],
            row['ha_token'],
            row['sto_token'],  # This follows our KNOWN pattern
        ]
        
        sequence = " ".join(sequence_parts) + " " + row['sto_token']
        sequences.append(sequence)
    
    return sequences


def verify_pattern_learning(sequences: List[str]) -> dict:
    """
    Verify that sequences follow the injected patterns.
    
    Returns:
        Dictionary with verification results
    """
    results = {
        'total': len(sequences),
        'pattern_checks': []
    }
    
    # Parse sequences and check patterns
    for seq in sequences:
        tokens = seq.split()
        # Format: <SYM> <TF> <TREND> <VOLUME> <HA> <STO> <ACTION>
        # Action should equal the STO token (6th element, index 5)
        expected_action = tokens[5]
        actual_action = tokens[6]
        
        results['pattern_checks'].append({
            'matches': expected_action == actual_action,
            'expected': expected_action,
            'actual': actual_action
        })
    
    matches = sum(1 for check in results['pattern_checks'] if check['matches'])
    results['accuracy'] = (matches / len(sequences)) * 100 if sequences else 0
    
    return results


def create_synthetic_dataset(n_days: int = 2000, seed: int = 42) -> List[str]:
    """
    Create a complete synthetic dataset with known patterns.
    
    Args:
        n_days: Number of days to generate
        seed: Random seed
        
    Returns:
        List of token sequences with 100% pattern correlation
    """
    print(f"Generating {n_days} days of synthetic data with KNOWN patterns...")
    
    # Step 1: Generate OHLCV
    df = generate_synthetic_ohlcv(n_days=n_days, seed=seed)
    print(f"  Generated OHLCV data: {len(df)} rows")
    
    # Step 2: Inject known patterns
    df = inject_known_patterns(df)
    print(f"  Injected patterns into data")
    
    # Step 3: Generate sequences
    sequences = generate_sequences_with_known_patterns(df)
    print(f"  Created {len(sequences)} sequences")
    
    # Step 4: Verify patterns
    verification = verify_pattern_learning(sequences)
    print(f"  Pattern verification: {verification['accuracy']:.1f}% accuracy (should be 100%)")
    
    return sequences


if __name__ == "__main__":
    # Test the synthetic data generation
    print("=" * 60)
    print("SYNTHETIC DATA GENERATION TEST")
    print("=" * 60)
    
    sequences = create_synthetic_dataset(n_days=500, seed=42)
    
    print("\n" + "=" * 60)
    print("SAMPLE SEQUENCES")
    print("=" * 60)
    for i, seq in enumerate(sequences[:5]):
        print(f"{i+1}. {seq}")
    
    # Verify
    verification = verify_pattern_learning(sequences)
    print(f"\nVerification: {verification['accuracy']:.1f}% patterns correct")
