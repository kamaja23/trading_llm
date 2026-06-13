#!/usr/bin/env python3
"""
Step 1: Generate Training Data

This script downloads historical price data and converts it into
token sequences suitable for training an LLM.
"""

import sys
from pathlib import Path
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.data_generator import (
    download_price_data,
    generate_token_sequences,
    split_sequences,
    save_sequences,
    analyze_sequences
)


def main():
    """Generate and save training data."""
    
    print("=" * 60)
    print("STEP 1: GENERATE TRAINING DATA")
    print("=" * 60)
    
    # Configuration - expanded symbols and timeframes for richer data
    SYMBOLS = [
        'SPY', 'QQQ', 'DIA', 'IWM',
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META',
        'TSLA', 'JPM', 'V', 'MA', 'UNH',
    ]
    TIMEFRAMES = ['DAILY']
    START_DATE = '2015-01-01'
    END_DATE = '2025-01-01'
    
    # Paths
    TRAIN_PATH = project_root / 'data' / 'train_test_split' / 'train.txt'
    VAL_PATH = project_root / 'data' / 'train_test_split' / 'val.txt'
    TEST_PATH = project_root / 'data' / 'train_test_split' / 'test.txt'
    ALL_SEQUENCES_PATH = project_root / 'data' / 'processed' / 'all_sequences.txt'
    
    all_sequences = []
    
    # Step 1: Download data for all symbols and generate sequences
    print(f"\n1. Downloading data for {len(SYMBOLS)} symbols...")
    
    for symbol in SYMBOLS:
        print(f"\n   Processing {symbol}...")
        
        for timeframe in TIMEFRAMES:
            raw_path = project_root / 'data' / 'raw' / f'{symbol}_{timeframe.lower()}.csv'
            
            df = download_price_data(
                symbol=symbol,
                start_date=START_DATE,
                end_date=END_DATE,
                save_path=str(raw_path)
            )
            
            sequences = generate_token_sequences(
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                include_sentiment=False,
            )
            
            print(f"      Generated {len(sequences)} sequences")
            all_sequences.extend(sequences)
    
    print(f"\n   Total sequences: {len(all_sequences)}")
    
    # Step 2: Analyze sequences
    print(f"\n2. Analyzing sequences...")
    analyze_sequences(all_sequences)
    
    # Step 3: Save all sequences
    print(f"\n3. Saving all sequences...")
    save_sequences(all_sequences, str(ALL_SEQUENCES_PATH))
    
    # Step 4: Split into train/val/test (shuffled for better distribution)
    print(f"\n4. Splitting into train/val/test sets...")
    
    import random
    random.seed(42)
    shuffled = all_sequences.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)
    
    train_sequences = shuffled[:train_size]
    val_sequences = shuffled[train_size:train_size + val_size]
    test_sequences = shuffled[train_size + val_size:]
    
    print(f"   Train: {len(train_sequences)} sequences")
    print(f"   Val:   {len(val_sequences)} sequences")
    print(f"   Test:  {len(test_sequences)} sequences")
    
    # Check distribution
    train_actions = Counter(s.split()[-1] for s in train_sequences)
    print(f"\n   Training action distribution: {dict(train_actions)}")
    
    # Step 5: Save splits
    print(f"\n5. Saving train/val/test files...")
    save_sequences(train_sequences, str(TRAIN_PATH))
    save_sequences(val_sequences, str(VAL_PATH))
    save_sequences(test_sequences, str(TEST_PATH))
    
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  - All sequences: {ALL_SEQUENCES_PATH}")
    print(f"  - Train set: {TRAIN_PATH}")
    print(f"  - Val set: {VAL_PATH}")
    print(f"  - Test set: {TEST_PATH}")
    print(f"\nNext step: Run 02_train_model.py")


if __name__ == "__main__":
    main()
