#!/usr/bin/env python3
"""
Verify Model Training with Synthetic Data - Simplified Version

This script generates synthetic data with KNOWN patterns, trains a simple classifier,
and verifies that the model learned those patterns.

Expected patterns (100% correlation):
- Uptrend + High Volume → STO_Overbought
- Downtrend + High Volume → STO_Oversold
- Low Volume → STO_NoCross
"""

import sys
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.synthetic_data_generator import create_synthetic_dataset
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def extract_features_and_labels(sequences, tokenizer, gpt2_model, device):
    """
    Extract GPT-2 features for each sequence and get the true action labels.
    
    Returns:
        features: numpy array of shape (n_samples, hidden_size)
        labels: numpy array of shape (n_samples,) with action indices
    """
    gpt2_model.eval()
    
    action_to_idx = {'STO_Overbought': 0, 'STO_Oversold': 1, 'STO_NoCross': 2, 'STO_Cross': 3}
    idx_to_action = {v: k for k, v in action_to_idx.items()}
    
    features = []
    labels = []
    
    for seq in sequences:
        tokens = seq.split()
        input_text = ' '.join(tokens[:-1])  # Everything except action
        action = tokens[-1]
        
        # Tokenize
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=128).to(device)
        
        # Get GPT-2 features (use [CLS] token or average of last hidden states)
        with torch.no_grad():
            outputs = gpt2_model(**inputs)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
            # Use the last token's hidden state as the feature
            feature = hidden_states[0, -1, :].cpu().numpy()
        
        features.append(feature)
        labels.append(action_to_idx.get(action, 0))
    
    return np.array(features), np.array(labels), idx_to_action


def train_and_test_classifier(train_features, train_labels, test_features, test_labels):
    """
    Train a simple classifier on top of GPT-2 features.
    """
    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_features, train_labels)
    
    # Test
    predictions = clf.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    
    return clf, predictions, accuracy


def main():
    print("=" * 60)
    print("MODEL VERIFICATION WITH SYNTHETIC DATA (SIMPLIFIED)")
    print("=" * 60)
    
    # Step 1: Generate synthetic data with KNOWN patterns
    print("\n1. Generating synthetic data with known patterns...")
    all_sequences = create_synthetic_dataset(n_days=3000, seed=42)
    
    # Step 2: Split into train/val/test
    print("\n2. Splitting data...")
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
    
    # Step 3: Load GPT-2 and tokenizer
    print("\n3. Loading GPT-2 model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpt2_model = GPT2Model.from_pretrained('gpt2').to(device)
    print(f"   Using device: {device}")
    
    # Step 4: Extract features
    print("\n4. Extracting features from GPT-2...")
    
    print("   Processing training data...")
    train_features, train_labels, idx_to_action = extract_features_and_labels(
        train_sequences, tokenizer, gpt2_model, device
    )
    print(f"   Training features shape: {train_features.shape}")
    
    print("   Processing test data...")
    test_features, test_labels, _ = extract_features_and_labels(
        test_sequences, tokenizer, gpt2_model, device
    )
    print(f"   Test features shape: {test_features.shape}")
    
    # Step 5: Train classifier
    print("\n5. Training classifier on GPT-2 features...")
    clf, predictions, accuracy = train_and_test_classifier(
        train_features, train_labels, test_features, test_labels
    )
    
    # Step 6: Show results
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {accuracy * 100:.1f}%")
    
    if accuracy >= 0.95:
        print("\n✓ SUCCESS: Model learned the patterns almost perfectly!")
    elif accuracy >= 0.80:
        print("\n≈ PARTIAL SUCCESS: Model learned the patterns reasonably well")
    else:
        print("\n✗ FAILURE: Model did NOT learn the patterns")
        print("  This indicates a problem with the feature extraction or data generation")
    
    # Show some predictions
    print("\nSample predictions:")
    for i in range(min(10, len(test_sequences))):
        tokens = test_sequences[i].split()
        input_text = ' '.join(tokens[:-1])
        true_action = tokens[-1]
        pred_action = idx_to_action[predictions[i]]
        status = "✓" if pred_action == true_action else "✗"
        print(f"  {status} Input: {input_text}")
        print(f"      Expected: {true_action}, Predicted: {pred_action}")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
