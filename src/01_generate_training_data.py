#!/usr/bin/env python3
"""
Step 1: Generate Training Data

This script downloads historical price data and converts it into
token sequences suitable for training an LLM.
"""

import sys
from pathlib import Path

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
    
    # Configuration
    SYMBOL = 'SPY'
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'
    TIMEFRAME = 'DAILY'
    
    # Paths
    RAW_DATA_PATH = project_root / 'data' / 'raw' / f'{SYMBOL}_daily.csv'
    TRAIN_PATH = project_root / 'data' / 'train_test_split' / 'train.txt'
    VAL_PATH = project_root / 'data' / 'train_test_split' / 'val.txt'
    TEST_PATH = project_root / 'data' / 'train_test_split' / 'test.txt'
    ALL_SEQUENCES_PATH = project_root / 'data' / 'processed' / 'all_sequences.txt'
    
    # Step 1: Download price data
    print(f"\n1. Downloading {SYMBOL} data...")
    df = download_price_data(
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        save_path=str(RAW_DATA_PATH)
    )
    
    # Step 2: Generate token sequences
    print(f"\n2. Generating token sequences...")
    sequences = generate_token_sequences(
        df=df,
        symbol=SYMBOL,
        timeframe=TIMEFRAME
    )
    
    # Step 3: Analyze sequences
    print(f"\n3. Analyzing sequences...")
    analyze_sequences(sequences)
    
    # Step 4: Save all sequences
    print(f"\n4. Saving all sequences...")
    save_sequences(sequences, str(ALL_SEQUENCES_PATH))
    
    # Step 5: Split into train/val/test
    print(f"\n5. Splitting into train/val/test sets...")
    train_sequences, val_sequences, test_sequences = split_sequences(
        sequences,
        train_ratio=0.7,  # 70% for training
        val_ratio=0.15    # 15% for validation, 15% for test
    )
    
    print(f"   Train: {len(train_sequences)} sequences")
    print(f"   Val:   {len(val_sequences)} sequences")
    print(f"   Test:  {len(test_sequences)} sequences")
    
    # Step 6: Save splits
    print(f"\n6. Saving train/val/test files...")
    save_sequences(train_sequences, str(TRAIN_PATH))
    save_sequences(val_sequences, str(VAL_PATH))
    save_sequences(test_sequences, str(TEST_PATH))
    
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  - Raw data: {RAW_DATA_PATH}")
    print(f"  - All sequences: {ALL_SEQUENCES_PATH}")
    print(f"  - Train set: {TRAIN_PATH}")
    print(f"  - Val set: {VAL_PATH}")
    print(f"  - Test set: {TEST_PATH}")
    print(f"\nNext step: Run 02_train_model.py")


if __name__ == "__main__":
    main()
