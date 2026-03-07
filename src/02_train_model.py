#!/usr/bin/env python3
"""
Step 2: Train the Model

This script fine-tunes a distilgpt2 model on the trading token sequences.
Updated to use modern transformers API (TextDataset is deprecated).
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from utils.token_definitions import ALL_CUSTOM_TOKENS
from utils.data_generator import load_sequences


def prepare_tokenizer():
    """
    Load GPT-2 tokenizer and add custom trading tokens.
    
    Returns:
        Configured tokenizer
    """
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    
    # Add padding token (GPT-2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add custom trading tokens
    print(f"Adding {len(ALL_CUSTOM_TOKENS)} custom tokens...")
    num_added = tokenizer.add_tokens(ALL_CUSTOM_TOKENS)
    print(f"Added {num_added} new tokens to vocabulary")
    
    return tokenizer


def prepare_model(tokenizer):
    """
    Load distilgpt2 model and resize embeddings for new tokens.
    
    Args:
        tokenizer: Configured tokenizer with custom tokens
        
    Returns:
        Model ready for fine-tuning
    """
    print("Loading distilgpt2 model...")
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    
    # Resize model embeddings to account for new tokens
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized embeddings to {len(tokenizer)} tokens")
    
    return model


def create_dataset_from_file(file_path, tokenizer, block_size=128):
    """
    Create a Dataset from token sequences file (modern approach).
    
    Args:
        file_path: Path to text file with sequences
        tokenizer: Configured tokenizer
        block_size: Maximum sequence length
        
    Returns:
        Dataset object
    """
    # Load sequences from file
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Tokenize all sequences
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=block_size,
            padding='max_length',
            return_tensors=None
        )
    
    # Create dataset
    dataset = Dataset.from_dict({'text': lines})
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing sequences"
    )
    
    return tokenized_dataset


def main():
    """Train the trading LLM."""
    
    print("=" * 60)
    print("STEP 2: TRAIN THE MODEL")
    print("=" * 60)
    
    # Paths
    TRAIN_PATH = project_root / 'data' / 'train_test_split' / 'train.txt'
    VAL_PATH = project_root / 'data' / 'train_test_split' / 'val.txt'
    MODEL_OUTPUT_DIR = project_root / 'models' / 'trading_llm'
    
    # Check if data exists
    if not TRAIN_PATH.exists():
        print(f"\nERROR: Training data not found at {TRAIN_PATH}")
        print("Please run 01_generate_training_data.py first")
        sys.exit(1)
    
    # Try to create model directory, provide helpful error if permissions fail
    try:
        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"\nERROR: Permission denied creating directory: {MODEL_OUTPUT_DIR}")
        print("\nTry one of these solutions:")
        print(f"  1. Fix permissions: sudo chown -R $USER:$USER {project_root / 'models'}")
        print(f"  2. Create manually: mkdir -p {MODEL_OUTPUT_DIR}")
        print(f"  3. Run script with: sudo python src/02_train_model.py (not recommended)")
        sys.exit(1)
    
    # Configuration
    BLOCK_SIZE = 128  # Max sequence length (our sequences are short, ~7 tokens)
    BATCH_SIZE = 8
    NUM_EPOCHS = 3
    LEARNING_RATE = 5e-5
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Step 1: Prepare tokenizer
    print("\n1. Preparing tokenizer...")
    tokenizer = prepare_tokenizer()
    
    # Step 2: Prepare model
    print("\n2. Preparing model...")
    model = prepare_model(tokenizer)
    model.to(device)
    
    # Step 3: Create datasets
    print("\n3. Creating datasets...")
    train_dataset = create_dataset_from_file(TRAIN_PATH, tokenizer, BLOCK_SIZE)
    val_dataset = create_dataset_from_file(VAL_PATH, tokenizer, BLOCK_SIZE)
    
    print(f"   Train dataset size: {len(train_dataset)} examples")
    print(f"   Val dataset size: {len(val_dataset)} examples")
    
    # Step 4: Prepare data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling, not masked LM
    )
    
    # Step 5: Configure training
    print("\n4. Configuring training...")
    
    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        save_total_limit=2,  # Only keep 2 best checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Step 6: Create trainer
    print("\n5. Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Step 7: Train!
    print("\n6. Starting training...")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print("\nThis may take 1-2 hours on CPU, 15-30 minutes on GPU...\n")
    
    trainer.train()
    
    # Step 8: Save final model
    print("\n7. Saving final model...")
    final_model_path = MODEL_OUTPUT_DIR / 'final_model'
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to: {final_model_path}")
    print(f"\nNext step: Run 03_test_model.py")


if __name__ == "__main__":
    main()
