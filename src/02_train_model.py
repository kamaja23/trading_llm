#!/usr/bin/env python3
"""
Step 2: Train the Model

This script fine-tunes a distilgpt2 model on the trading token sequences.
Uses torch.compile with ROCm Triton for numerical stability on AMD GPUs.
"""

import sys
from pathlib import Path
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from utils.token_definitions import ALL_CUSTOM_TOKENS
from utils.data_generator import load_sequences


def prepare_tokenizer():
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

    tokenizer.pad_token = tokenizer.eos_token

    print(f"Adding {len(ALL_CUSTOM_TOKENS)} custom tokens...")
    num_added = tokenizer.add_tokens(ALL_CUSTOM_TOKENS)
    print(f"Added {num_added} new tokens to vocabulary")
    print(f"Total vocabulary size: {len(tokenizer)}")

    return tokenizer


def prepare_model(tokenizer, freeze_base=True):
    print("Loading distilgpt2 model...")
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')

    model.resize_token_embeddings(len(tokenizer))

    if model.lm_head.weight.shape[0] != len(tokenizer):
        print("Fixing lm_head dimension mismatch...")
        old_weight = model.lm_head.weight.data.clone()
        model.lm_head = torch.nn.Linear(
            model.lm_head.in_features,
            len(tokenizer),
            bias=model.lm_head.bias is not None
        )
        old_vocab_size = old_weight.shape[0]
        model.lm_head.weight.data[:old_vocab_size] = old_weight.to(model.lm_head.weight.device)

    if freeze_base:
        print("Freezing transformer layers (training embeddings + lm_head)...")
        for name, param in model.transformer.named_parameters():
            if 'h.' in name:
                param.requires_grad = False

    print(f"Resized embeddings and lm_head to {len(tokenizer)} tokens")
    if freeze_base:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,} (out of {sum(p.numel() for p in model.parameters()):,})")

    return model


def create_dataset_from_file(file_path, tokenizer, block_size=128):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    def tokenize_function(examples):
        texts = examples['text']
        all_input_ids = []
        all_labels = []

        for text in texts:
            tokens = text.split()
            full_text = ' '.join(tokens)

            encoded = tokenizer(
                full_text,
                truncation=True,
                max_length=block_size,
                add_special_tokens=False,
            )
            ids = encoded['input_ids']

            input_ids = ids
            labels = [-100] * (len(ids) - 1) + [ids[-1]]

            padding_len = block_size - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_len
            labels = labels + [-100] * padding_len

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        return {
            'input_ids': all_input_ids,
            'labels': all_labels,
            'attention_mask': [
                [1] * min(len(t.split()), block_size) + [0] * max(0, block_size - len(t.split()))
                for t in texts
            ],
        }

    dataset = Dataset.from_dict({'text': lines})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing sequences",
    )

    return tokenized_dataset


class ActionOnlyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            return_tensors="pt",
        )
        return batch


class ClassificationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels")

        if labels is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.get("labels")
        if labels is None:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return (loss.detach(), None, None)


def main():
    print("=" * 60)
    print("STEP 2: TRAIN THE MODEL")
    print("=" * 60)

    TRAIN_PATH = project_root / 'data' / 'train_test_split' / 'train.txt'
    VAL_PATH = project_root / 'data' / 'train_test_split' / 'val.txt'
    MODEL_OUTPUT_DIR = project_root / 'models' / 'tradebot'

    if not TRAIN_PATH.exists():
        print(f"\nERROR: Training data not found at {TRAIN_PATH}")
        print("Please run 01_generate_training_data.py first")
        sys.exit(1)

    try:
        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"\nERROR: Permission denied creating directory: {MODEL_OUTPUT_DIR}")
        print("\nTry one of these solutions:")
        print(f"  1. Fix permissions: sudo chown -R $USER:$USER {project_root / 'models'}")
        print(f"  2. Create manually: mkdir -p {MODEL_OUTPUT_DIR}")
        print(f"  3. Run script with: sudo python src/02_train_model.py (not recommended)")
        sys.exit(1)

    BLOCK_SIZE = 256
    BATCH_SIZE = 8
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n1. Preparing tokenizer...")
    tokenizer = prepare_tokenizer()

    print("\n2. Preparing model...")
    model = prepare_model(tokenizer, freeze_base=True)
    model.to(device)

    if device == "cuda":
        model = torch.compile(model, mode="default")

    print("\n3. Creating datasets...")
    train_dataset = create_dataset_from_file(TRAIN_PATH, tokenizer, BLOCK_SIZE)
    val_dataset = create_dataset_from_file(VAL_PATH, tokenizer, BLOCK_SIZE)

    print(f"   Train dataset size: {len(train_dataset)} examples")
    print(f"   Val dataset size: {len(val_dataset)} examples")

    data_collator = ActionOnlyDataCollator(tokenizer=tokenizer)

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
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    print("\n5. Creating trainer...")
    trainer = ClassificationTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("\n6. Checking for existing checkpoints...")
    checkpoints = sorted(MODEL_OUTPUT_DIR.glob("checkpoint-*"))
    resume_from = None
    if checkpoints:
        resume_from = str(checkpoints[-1])
        print(f"   Resuming from checkpoint: {resume_from}")
    else:
        print("   No checkpoints found, starting fresh training...")

    print(f"\n7. Starting training...")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"\n   {NUM_EPOCHS} epochs × {len(train_dataset) // BATCH_SIZE} steps/epoch = {NUM_EPOCHS * len(train_dataset) // BATCH_SIZE:,} total steps")
    print(f"   At ~7.9 it/s (AMD RX 7800 XT), expect ~{NUM_EPOCHS * (len(train_dataset) // BATCH_SIZE) / 7.9 / 3600:.1f}h total\n")

    trainer.train(resume_from_checkpoint=resume_from)

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
