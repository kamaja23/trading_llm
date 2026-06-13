#!/usr/bin/env python3
"""
Step 3: Test the Model

This script evaluates the trained model on held-out test data
and measures prediction accuracy.
"""

import sys
from pathlib import Path
import torch
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils.data_generator import load_sequences
from utils.token_definitions import ACTION_TOKENS


def load_trained_model(model_path):
    """
    Load the trained model and tokenizer.
    
    Args:
        model_path: Path to saved model directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")
    
    tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
    model = GPT2LMHeadModel.from_pretrained(str(model_path))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    
    return model, tokenizer, device


def predict_next_token(model, tokenizer, input_sequence, device):
    """
    Predict the next token given an input sequence.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        input_sequence: String of space-separated tokens
        device: Device (cpu/cuda)
        
    Returns:
        Predicted token string
    """
    # Tokenize input
    inputs = tokenizer(input_sequence, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate next token
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    
    # Get the predicted token for the last position
    next_token_logits = predictions[0, -1, :]
    next_token_id = torch.argmax(next_token_logits).item()
    predicted_token = tokenizer.decode([next_token_id])
    
    return predicted_token.strip()


def evaluate_model(model, tokenizer, test_sequences, device):
    """
    Evaluate model accuracy on test sequences.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_sequences: List of test sequences
        device: Device (cpu/cuda)
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating on {len(test_sequences)} test sequences...")
    
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    
    for i, sequence in enumerate(test_sequences):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(test_sequences)}...")
        
        # Split sequence into input and target
        tokens = sequence.split()
        
        # Input: all tokens except the last (action token)
        input_tokens = tokens[:-1]
        input_sequence = " ".join(input_tokens)
        
        # Target: the last token (action)
        true_action = tokens[-1]
        
        # Only evaluate if true action is valid
        if true_action not in ACTION_TOKENS:
            continue
        
        # Predict
        predicted_token = predict_next_token(model, tokenizer, input_sequence, device)
        
        # Check if prediction is an action token
        # (Model might predict other tokens, we only care about actions)
        if predicted_token in ACTION_TOKENS:
            predictions.append(predicted_token)
            true_labels.append(true_action)
            
            if predicted_token == true_action:
                correct += 1
            total += 1
    
    # Calculate metrics
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Action distribution
    pred_distribution = Counter(predictions)
    true_distribution = Counter(true_labels)
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'predictions': predictions,
        'true_labels': true_labels,
        'pred_distribution': pred_distribution,
        'true_distribution': true_distribution
    }


def print_evaluation_results(results):
    """
    Print evaluation results in a readable format.
    
    Args:
        results: Dictionary from evaluate_model
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nOverall Accuracy: {results['accuracy']:.2f}%")
    print(f"Correct predictions: {results['correct']}/{results['total']}")
    
    print("\n--- True Label Distribution ---")
    for action, count in results['true_distribution'].most_common():
        pct = (count / results['total'] * 100)
        print(f"  {action}: {count} ({pct:.1f}%)")
    
    print("\n--- Model Prediction Distribution ---")
    for action, count in results['pred_distribution'].most_common():
        pct = (count / results['total'] * 100)
        print(f"  {action}: {count} ({pct:.1f}%)")
    
    # Baseline (random guessing)
    num_classes = len(ACTION_TOKENS)
    random_accuracy = 100.0 / num_classes
    
    print(f"\n--- Comparison to Baseline ---")
    print(f"Random guessing: {random_accuracy:.2f}%")
    print(f"Model accuracy:  {results['accuracy']:.2f}%")
    
    if results['accuracy'] > random_accuracy:
        improvement = results['accuracy'] - random_accuracy
        print(f"✓ Model is {improvement:.2f}% better than random!")
    else:
        print(f"✗ Model is not better than random guessing")
    
    print("\n" + "=" * 60)


def show_sample_predictions(model, tokenizer, test_sequences, device, n=5):
    """
    Show sample predictions for inspection.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_sequences: List of test sequences
        device: Device
        n: Number of samples to show
    """
    print("\n--- Sample Predictions ---")
    
    for i, sequence in enumerate(test_sequences[:n]):
        tokens = sequence.split()
        input_tokens = tokens[:-1]
        true_action = tokens[-1]
        
        input_sequence = " ".join(input_tokens)
        predicted_action = predict_next_token(model, tokenizer, input_sequence, device)
        
        correct_mark = "✓" if predicted_action == true_action else "✗"
        
        print(f"\n{i+1}. {correct_mark}")
        print(f"   Input:     {input_sequence}")
        print(f"   True:      {true_action}")
        print(f"   Predicted: {predicted_action}")


def main():
    """Test the trained model."""
    
    print("=" * 60)
    print("STEP 3: TEST THE MODEL")
    print("=" * 60)
    
    # Paths
    MODEL_PATH = project_root / 'models' / 'tradebot' / 'final_model'
    TEST_PATH = project_root / 'data' / 'train_test_split' / 'test.txt'
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        print("Please run 02_train_model.py first")
        sys.exit(1)
    
    # Check if test data exists
    if not TEST_PATH.exists():
        print(f"\nERROR: Test data not found at {TEST_PATH}")
        print("Please run 01_generate_training_data.py first")
        sys.exit(1)
    
    # Step 1: Load model
    print("\n1. Loading trained model...")
    model, tokenizer, device = load_trained_model(MODEL_PATH)
    
    # Step 2: Load test data
    print("\n2. Loading test data...")
    test_sequences = load_sequences(str(TEST_PATH))
    print(f"   Loaded {len(test_sequences)} test sequences")
    
    # Step 3: Show sample predictions
    print("\n3. Sample predictions...")
    show_sample_predictions(model, tokenizer, test_sequences, device, n=10)
    
    # Step 4: Full evaluation
    print("\n4. Full evaluation...")
    results = evaluate_model(model, tokenizer, test_sequences, device)
    
    # Step 5: Print results
    print_evaluation_results(results)
    
    print("\nNext step: Run 04_interactive_inference.py to test with custom inputs")


if __name__ == "__main__":
    main()
