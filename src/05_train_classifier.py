"""
Step 5: Train a Classification Head on top of the frozen trading LLM.

Trains a lightweight classifier (3-layer MLP) that takes the last hidden
state from the distilgpt2 model and predicts BUY/SELL/HOLD.

This avoids the issues with causal LM training where action tokens
are indistinguishable from random initialization.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils.token_definitions import ACTION_TOKENS
from utils.data_generator import load_sequences


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def load_model_and_tokenizer(model_path):
    print(f"Loading model from {model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
    model = GPT2LMHeadModel.from_pretrained(str(model_path))
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("Model loaded and frozen.")
    return model, tokenizer


class TradingClassifier(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=3, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.drop1(F.relu(self.ln1(self.fc1(x))))
        x = self.fc2(x)
        return x


def encode_for_classifier(sequences, tokenizer, model, max_len=128):
    input_ids_list = []
    all_logits = []

    with torch.no_grad():
        for seq in sequences:
            tokens = seq.split()
            input_text = " ".join(tokens[:-1])
            true_action = tokens[-1]

            encoding = tokenizer(
                input_text + " ",
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attn_mask = encoding["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]
            mask_expanded = attn_mask.unsqueeze(-1).expand_as(hidden)
            sum_hidden = (hidden * mask_expanded).sum(dim=1)
            valid_counts = mask_expanded.sum(dim=1).clamp(min=1)
            pooled = (sum_hidden / valid_counts).squeeze(0).cpu()

            input_ids_list.append((pooled, ACTION_TOKENS.index(true_action)))

    return input_ids_list


def create_dataloader(data, batch_size=32, shuffle=True):
    xs = torch.cat([x.unsqueeze(0) for x, _ in data], dim=0)
    ys = torch.tensor([y for _, y in data])
    dataset = torch.utils.data.TensorDataset(xs, ys)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )


def train_classifier():
    MODEL_PATH = project_root / "models" / "tradebot" / "final_model"
    TRAIN_PATH = project_root / "data" / "train_test_split" / "train.txt"
    VAL_PATH = project_root / "data" / "train_test_split" / "val.txt"
    TEST_PATH = project_root / "data" / "train_test_split" / "test.txt"
    SAVE_PATH = project_root / "models" / "tradebot" / "classifier.pt"

    print("=" * 60)
    print("STEP 5: TRAIN CLASSIFICATION HEAD")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

    print("\nLoading sequences and extracting features...")
    train_seqs = load_sequences(str(TRAIN_PATH))
    val_seqs = load_sequences(str(VAL_PATH))
    test_seqs = load_sequences(str(TEST_PATH))

    train_data = encode_for_classifier(train_seqs, tokenizer, model)
    val_data = encode_for_classifier(val_seqs, tokenizer, model)
    test_data = encode_for_classifier(test_seqs, tokenizer, model)

    train_loader = create_dataloader(train_data, shuffle=True)
    val_loader = create_dataloader(val_data, shuffle=False)
    test_loader = create_dataloader(test_data, shuffle=False)

    # Compute class weights (inverse frequency)
    action_counts = [0, 0, 0]
    for _, y in train_data:
        action_counts[y] += 1
    total = sum(action_counts)
    class_weights = torch.tensor([total / c for c in action_counts], dtype=torch.float).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    classifier = TradingClassifier().to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=3e-4, weight_decay=1e-3)
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0

    print(f"\nTrain: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    print(f"\nTraining for up to 100 epochs (early stopping)...\n")

    for epoch in range(100):
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xs, ys in train_loader:
            xs, ys = xs.to(device), ys.to(device)
            optimizer.zero_grad()
            logits = classifier(xs)
            loss = F.cross_entropy(logits, ys, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == ys).sum().item()
            train_total += ys.size(0)

        classifier.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xs, ys in val_loader:
                xs, ys = xs.to(device), ys.to(device)
                logits = classifier(xs)
                preds = logits.argmax(dim=1)
                val_correct += (preds == ys).sum().item()
                val_total += ys.size(0)

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d} | Train loss: {train_loss/len(train_loader):.4f} | "
                f"Train acc: {train_acc:.1f}% | Val acc: {val_acc:.1f}%"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(classifier.state_dict(), str(SAVE_PATH))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
    print(f"Classifier saved to: {SAVE_PATH}")

    classifier.load_state_dict(torch.load(str(SAVE_PATH)))
    classifier.eval()

    test_correct = 0
    test_total = 0
    confusion = torch.zeros(3, 3)
    with torch.no_grad():
        for xs, ys in test_loader:
            xs, ys = xs.to(device), ys.to(device)
            logits = classifier(xs)
            preds = logits.argmax(dim=1)
            test_correct += (preds == ys).sum().item()
            test_total += ys.size(0)
            for i in range(len(ys)):
                confusion[ys[i], preds[i]] += 1

    test_acc = test_correct / test_total * 100
    random_baseline = 100.0 / len(ACTION_TOKENS)
    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test accuracy: {test_acc:.1f}% (random: {random_baseline:.1f}%)")
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    print(f"           BUY  SELL HOLD")
    labels = ACTION_TOKENS
    for i, label in enumerate(labels):
        row = "  ".join(f"{confusion[i, j]:5.0f}" for j in range(3))
        print(f"  {label:6s}: {row}")
    print()


if __name__ == "__main__":
    train_classifier()
