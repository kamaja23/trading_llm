#!/usr/bin/env python3
"""
Verify Model Training with Synthetic Data

This script generates synthetic data with KNOWN patterns, trains the model,
and verifies that the model learned those patterns.

Expected patterns (100% correlation):
- Uptrend + High Volume → STO_Overbought
- Downtrend + High Volume → STO_Oversold
- Low Volume → STO_NoCross
"""

import sys
from pathlib import Path
import random
from collections import Counter

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.synthetic_data_generator import create_synthetic_dataset, verify_pattern_learning
from utils.data_generator import save_sequences, load_sequences
from utils.token_definitions import ALL_CUSTOM_TOKENS
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
import torch


class ClassificationTrainer(Trainer):
    """Custom trainer that computes loss only on action tokens."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels")
        
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        pad_token_id = self.data_collator.tokenizer.pad_token_id
        
        # Flatten logits and labels
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        
        # Only compute loss on non-masked labels
        valid_mask = (flat_labels != -100) & (flat_labels != pad_token_id)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        flat_logits = flat_logits[valid_mask]
        flat_labels = flat_labels[valid_mask]
        
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(flat_logits, flat_labels)
        
        return (loss, outputs) if return_outputs else loss


def create_dataset_for_action_prediction(sequences, tokenizer, block_size=128):
    """
    Create dataset where the GOAL is to predict the action token.
    Input: <SYM> <TF> <TREND> <VOL> <HA> <STO>
    Target: <ACTION>
    
    We'll use the [last hidden state] to predict the action via classification.
    """
    # Separate inputs and actions
    input_texts = []
    action_labels = []
    
    for text in sequences:
        tokens = text.split()
        input_text = ' '.join(tokens[:-1])  # Everything except action
        action = tokens[-1]
        input_texts.append(input_text)
        action_labels.append(action)
    
    # Tokenize inputs
    inputs = tokenizer(
        input_texts,
        truncation=True,
        max_length=block_size,
        padding='max_length',
        return_tensors=None
    )
    
    # Convert action tokens to IDs for classification
    action_to_id = {'STO_Overbought': 0, 'STO_Oversold': 1, 'STO_NoCross': 2, 'STO_Cross': 3}
    labels = [action_to_id.get(action, 0) for action in action_labels]
    
    inputs['labels'] = labels
    return inputs


def test_model_predictions(model, tokenizer, test_cases):
    """
    Test model on specific cases with known expected outputs.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_cases: List of (input_text, expected_action) tuples
        
    Returns:
        Dictionary with test results
    """
    model.eval()
    device = next(model.parameters()).device
    
    results = {
        'total': len(test_cases),
        'correct': 0,
        'details': []
    }
    
    for input_text, expected_action in test_cases:
        # Tokenize input (everything except action)
        inputs = tokenizer(input_text, return_tensors='pt').to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Get the last token's prediction
        last_token_logits = logits[0, -1, :]
        predicted_token_id = last_token_logits.argmax().item()
        predicted_action = tokenizer.decode([predicted_token_id]).strip()
        
        # Check if correct
        is_correct = predicted_action == expected_action
        if is_correct:
            results['correct'] += 1
        
        results['details'].append({
            'input': input_text,
            'expected': expected_action,
            'predicted': predicted_action,
            'correct': is_correct
        })
    
    results['accuracy'] = (results['correct'] / results['total']) * 100 if results['total'] > 0 else 0
    
    return results


def main():
    print("=" * 60)
    print("MODEL VERIFICATION WITH SYNTHETIC DATA")
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
    
    # Save sequences
    data_dir = project_root / 'data' / 'synthetic'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    save_sequences(train_sequences, str(data_dir / 'train.txt'))
    save_sequences(val_sequences, str(data_dir / 'val.txt'))
    save_sequences(test_sequences, str(data_dir / 'test.txt'))
    print(f"   Saved to {data_dir}")
    
    # Step 3: Prepare tokenizer and model
    print("\n3. Preparing model and tokenizer...")
    from utils.token_definitions import ALL_CUSTOM_TOKENS
    
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(ALL_CUSTOM_TOKENS)
    
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model.resize_token_embeddings(len(tokenizer))
    
    # Fix lm_head if needed
    if model.lm_head.weight.shape[0] != len(tokenizer):
        model.lm_head = torch.nn.Linear(
            model.lm_head.in_features,
            len(tokenizer),
            bias=model.lm_head.bias is not None
        )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"   Using device: {device}")
    
    # Step 4: Create datasets for ACTION PREDICTION
    print("\n4. Creating datasets for action prediction...")
    train_dataset = create_dataset_for_action_prediction(train_sequences, tokenizer)
    val_dataset = create_dataset_for_action_prediction(val_sequences, tokenizer)
    
    # Step 5: Train model
    print("\n5. Training model on synthetic data...")
    print("   (This should learn patterns quickly since they're 100% correlated)")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    training_args = TrainingArguments(
        output_dir=str(project_root / 'models' / 'synthetic_verification'),
        num_train_epochs=10,  # Fewer epochs since patterns are simple
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        fp16=(device == "cuda"),
    )
    
    trainer = ClassificationTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    print("   Training complete!")
    
    # Step 6: Create test cases with KNOWN expected outputs
    print("\n6. Creating test cases...")
    test_cases = []
    
    for seq in test_sequences[:20]:  # Test 20 cases
        tokens = seq.split()
        input_text = ' '.join(tokens[:-1])
        expected_action = tokens[-1]
        test_cases.append((input_text, expected_action))
    
    # Step 7: Load best model for testing
    print("\n7. Loading best model for testing...")
    best_model_path = project_root / 'models' / 'synthetic_verification' / 'checkpoint-1500'
    model = GPT2LMHeadModel.from_pretrained(str(best_model_path))
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    results = test_model_predictions(model, tokenizer, test_cases)
    
    print(f"\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    print(f"Test cases: {results['total']}")
    print(f"Correct predictions: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.1f}%")
    
    if results['accuracy'] == 100.0:
        print("\n✓ SUCCESS: Model learned the patterns perfectly!")
    elif results['accuracy'] >= 90.0:
        print("\n≈ PARTIAL SUCCESS: Model mostly learned the patterns")
    else:
        print("\n✗ FAILURE: Model did NOT learn the patterns")
        print("  This indicates a problem with the training pipeline")
    
    print("\nSample predictions:")
    for i, detail in enumerate(results['details'][:5]):
        status = "✓" if detail['correct'] else "✗"
        print(f"  {status} Input: {detail['input']}")
        print(f"    Expected: {detail['expected']}, Predicted: {detail['predicted']}")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
