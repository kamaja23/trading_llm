#!/usr/bin/env python3
"""
Step 4: Interactive Inference

This script allows you to test the model interactively with custom
market state inputs.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils.token_definitions import (
    SYMBOL_TOKENS,
    TIMEFRAME_TOKENS,
    STATE_TOKENS,
    VOLUME_TOKENS,
    INDICATOR_TOKENS,
    ACTION_TOKENS
)


def load_trained_model(model_path):
    """Load the trained model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
    tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
    model = GPT2LMHeadModel.from_pretrained(str(model_path))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}\n")
    
    return model, tokenizer, device


def predict_action(model, tokenizer, input_sequence, device, top_k=3):
    """
    Predict action and show probabilities for top predictions.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        input_sequence: String of space-separated tokens
        device: Device (cpu/cuda)
        top_k: Number of top predictions to show
        
    Returns:
        List of (token, probability) tuples
    """
    # Tokenize input
    inputs = tokenizer(input_sequence, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    
    # Get logits for last position and apply softmax
    next_token_logits = predictions[0, -1, :]
    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
    
    # Get top k predictions
    top_probs, top_indices = torch.topk(probs, top_k)
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        token = tokenizer.decode([idx]).strip()
        results.append((token, prob.item()))
    
    return results


def print_token_options():
    """Print available token options for user reference."""
    print("\n" + "=" * 60)
    print("AVAILABLE TOKENS")
    print("=" * 60)
    
    print("\nSymbols:", ", ".join(SYMBOL_TOKENS))
    print("Timeframes:", ", ".join(TIMEFRAME_TOKENS))
    print("Trends:", ", ".join(STATE_TOKENS))
    print("Volume:", ", ".join(VOLUME_TOKENS))
    print("Indicators (HA):", ", ".join([t for t in INDICATOR_TOKENS if 'HA_' in t]))
    print("Indicators (STO):", ", ".join([t for t in INDICATOR_TOKENS if 'STO_' in t]))
    print("\n" + "=" * 60)


def interactive_mode(model, tokenizer, device):
    """
    Run interactive prediction mode.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        device: Device
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE PREDICTION MODE")
    print("=" * 60)
    print("\nEnter market state tokens separated by spaces.")
    print("Format: <SYM_X> <TF_X> <Trend> <Volume> <HA_Signal> <STO_Signal>")
    print("\nExamples:")
    print("  <SYM_SPY> <TF_DAILY> ST_UpTrend Hi_Volume HA_UpCross STO_Cross")
    print("  <SYM_SPY> <TF_DAILY> ST_DownTrend Lo_Volume HA_DownCross STO_Oversold")
    print("\nType 'help' to see available tokens")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("Market state > ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Exiting...")
                break
            
            if user_input.lower() == 'help':
                print_token_options()
                continue
            
            # Predict
            predictions = predict_action(model, tokenizer, user_input, device, top_k=3)
            
            print("\n--- Predictions ---")
            for i, (token, prob) in enumerate(predictions, 1):
                # Highlight if it's an action token
                if token in ACTION_TOKENS:
                    print(f"{i}. {token:15s} {prob*100:6.2f}% ← ACTION")
                else:
                    print(f"{i}. {token:15s} {prob*100:6.2f}%")
            
            # Show most likely action
            action_predictions = [(t, p) for t, p in predictions if t in ACTION_TOKENS]
            if action_predictions:
                best_action, best_prob = action_predictions[0]
                print(f"\nMost likely action: {best_action} ({best_prob*100:.1f}% confidence)")
            else:
                print("\nNote: Top predictions are not action tokens.")
                print("      Model may need more training or input may be unusual.")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")


def demo_mode(model, tokenizer, device):
    """
    Run a quick demo with predefined examples.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        device: Device
    """
    print("\n" + "=" * 60)
    print("DEMO MODE - Predefined Examples")
    print("=" * 60)
    
    examples = [
        "<SYM_SPY> <TF_DAILY> ST_UpTrend Hi_Volume HA_UpCross STO_Cross",
        "<SYM_SPY> <TF_DAILY> ST_DownTrend Hi_Volume HA_DownCross STO_Oversold",
        "<SYM_SPY> <TF_DAILY> ST_Flat Lo_Volume HA_Neutral STO_NoCross",
        "<SYM_SPY> <TF_DAILY> ST_DownTrend Hi_Volume HA_UpCross STO_Cross",
        "<SYM_SPY> <TF_DAILY> ST_UpTrend Lo_Volume HA_DownCross STO_Overbought",
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Input: {example}")
        
        predictions = predict_action(model, tokenizer, example, device, top_k=3)
        
        print("Predictions:")
        for token, prob in predictions:
            if token in ACTION_TOKENS:
                print(f"  {token:10s} {prob*100:6.2f}% ← ACTION")
            else:
                print(f"  {token:10s} {prob*100:6.2f}%")
    
    print("\n" + "=" * 60)


def main():
    """Run interactive inference."""
    
    print("=" * 60)
    print("STEP 4: INTERACTIVE INFERENCE")
    print("=" * 60)
    
    # Paths
    MODEL_PATH = project_root / 'models' / 'trading_llm' / 'final_model'
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        print("Please run 02_train_model.py first")
        sys.exit(1)
    
    # Load model
    model, tokenizer, device = load_trained_model(MODEL_PATH)
    
    # Ask user for mode
    print("Choose mode:")
    print("  1. Demo mode (predefined examples)")
    print("  2. Interactive mode (custom inputs)")
    print("  3. Both")
    
    choice = input("\nYour choice (1/2/3): ").strip()
    
    if choice == '1':
        demo_mode(model, tokenizer, device)
    elif choice == '2':
        interactive_mode(model, tokenizer, device)
    elif choice == '3':
        demo_mode(model, tokenizer, device)
        interactive_mode(model, tokenizer, device)
    else:
        print("Invalid choice. Running demo mode.")
        demo_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
