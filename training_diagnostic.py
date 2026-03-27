#!/usr/bin/env python3
"""
Comprehensive training diagnostic to identify issues.
Run this BEFORE training to verify everything is set up correctly.
"""

import sys
from pathlib import Path
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("TRAINING DIAGNOSTIC")
print("=" * 60)

# 1. GPU Check
print("\n1. GPU AVAILABILITY:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU name: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("   ❌ GPU NOT DETECTED - training will be slow!")

# 2. PyTorch Check
print(f"\n2. PYTORCH:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA compiled: {torch.version.cuda}")

# 3. Data Check
print(f"\n3. TRAINING DATA:")
train_path = project_root / 'data' / 'train_test_split' / 'train.txt'
val_path = project_root / 'data' / 'train_test_split' / 'val.txt'

if not train_path.exists():
    print(f"   ❌ Training data not found at {train_path}")
    print("   Run: python src/01_generate_training_data.py")
    sys.exit(1)

with open(train_path) as f:
    train_lines = f.readlines()
with open(val_path) as f:
    val_lines = f.readlines()

print(f"   Train sequences: {len(train_lines)}")
print(f"   Val sequences: {len(val_lines)}")

# Check action distribution
from collections import Counter
train_actions = [line.strip().split()[-1] for line in train_lines if line.strip()]
action_counts = Counter(train_actions)
print(f"\n   Action distribution:")
for action, count in action_counts.items():
    pct = count / len(train_actions) * 100
    print(f"     {action}: {count} ({pct:.1f}%)")

# 4. Sample sequences
print(f"\n4. SAMPLE TRAINING SEQUENCES:")
for i, line in enumerate(train_lines[:3], 1):
    print(f"   {i}. {line.strip()}")

# 5. Tokenizer Check
print(f"\n5. TOKENIZER CHECK:")
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
print(f"   Base vocab size: {len(tokenizer)}")

# Add custom tokens
from utils.token_definitions import ALL_CUSTOM_TOKENS
tokenizer.add_tokens(ALL_CUSTOM_TOKENS)
print(f"   After adding custom: {len(tokenizer)}")
print(f"   Custom tokens added: {len(ALL_CUSTOM_TOKENS)}")

# Test tokenization of a sequence
test_seq = train_lines[0].strip()
tokens = tokenizer.encode(test_seq, add_special_tokens=False)
print(f"\n   Test sequence: {test_seq}")
print(f"   Tokenized to {len(tokens)} tokens: {tokens[:10]}...")

# 6. Model Size Check
print(f"\n6. MODEL SIZE:")
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
model.resize_token_embeddings(len(tokenizer))
param_count = sum(p.numel() for p in model.parameters())
print(f"   Total parameters: {param_count:,}")
print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 7. Training Config
print(f"\n7. TRAINING CONFIGURATION:")
BATCH_SIZE = 8
NUM_EPOCHS = 3
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Train batches per epoch: {len(train_lines) // BATCH_SIZE}")
print(f"   Total training steps: {(len(train_lines) // BATCH_SIZE) * NUM_EPOCHS}")

# 8. Time Estimate
steps = (len(train_lines) // BATCH_SIZE) * NUM_EPOCHS
if torch.cuda.is_available():
    time_per_step = 0.5  # seconds on GPU
    total_time = steps * time_per_step / 60
    print(f"\n8. ESTIMATED TRAINING TIME:")
    print(f"   ~{total_time:.1f} minutes on GPU")
    print(f"   ({steps} steps × ~0.5 sec/step)")
else:
    time_per_step = 3.0  # seconds on CPU
    total_time = steps * time_per_step / 60
    print(f"\n8. ESTIMATED TRAINING TIME:")
    print(f"   ~{total_time:.1f} minutes on CPU")
    print(f"   ({steps} steps × ~3 sec/step)")

# 9. Check for existing model
print(f"\n9. EXISTING MODEL CHECK:")
model_path = project_root / 'models' / 'trading_llm' / 'final_model'
if model_path.exists():
    print(f"   ⚠️  Model already exists at {model_path}")
    print(f"   This is the OLD model trained on bad data!")
    print(f"   DELETE IT: rm -rf {model_path}")
else:
    print(f"   ✓ No existing model (good - will train fresh)")

checkpoint_dirs = list((project_root / 'models' / 'trading_llm').glob('checkpoint-*'))
if checkpoint_dirs:
    print(f"   ⚠️  {len(checkpoint_dirs)} checkpoint directories exist")
    print(f"   DELETE THEM: rm -rf models/trading_llm/checkpoint-*")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

# Summary
issues = []
if not torch.cuda.is_available():
    issues.append("GPU not available")
if len(train_lines) < 500:
    issues.append(f"Too few training sequences ({len(train_lines)})")
if model_path.exists():
    issues.append("Old model exists - needs deletion")
if action_counts.get('HOLD', 0) / len(train_actions) > 0.5:
    issues.append("Data still imbalanced (too much HOLD)")

if issues:
    print("\n⚠️  ISSUES FOUND:")
    for issue in issues:
        print(f"   - {issue}")
    print("\nFix these before training!")
else:
    print("\n✓ All checks passed!")
    print("\nReady to train. Expected time: 15-25 minutes")
    print("Run: python src/02_train_model.py")
