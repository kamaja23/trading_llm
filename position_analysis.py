#!/usr/bin/env python3
"""
Check what tokens the model predicts for each position in the sequence.
"""

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from pathlib import Path
import torch

model_path = Path('models/tradebot/final_model')
tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
model = GPT2LMHeadModel.from_pretrained(str(model_path))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

print("=" * 60)
print("POSITION-BY-POSITION PREDICTION ANALYSIS")
print("=" * 60)

# Test with a sequence WITHOUT the action
test_input = '<SYM_SPY> <TF_DAILY> ST_UpTrend Hi_Volume HA_UpCross STO_Cross'

print(f"\nInput: {test_input}")
print(f"\nTokenization:")

# Tokenize
inputs = tokenizer(test_input, return_tensors='pt').to(device)
input_ids = inputs['input_ids'][0]

# Show what each position tokenized to
tokens_list = []
for i, token_id in enumerate(input_ids):
    token_str = tokenizer.decode([token_id])
    tokens_list.append(token_str)
    print(f"  Position {i}: ID {token_id:5d} = '{token_str}'")

print(f"\nTotal input tokens: {len(input_ids)}")

# Get predictions for EACH position
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
    
    print(f"\n{'Position':<10} {'Input Token':<20} {'Predicted Next':<20} {'Probability':<15}")
    print("-" * 70)
    
    for pos in range(len(input_ids)):
        # Get prediction for this position
        probs = torch.nn.functional.softmax(logits[pos], dim=-1)
        top_pred_idx = probs.argmax().item()
        top_pred_prob = probs[top_pred_idx].item()
        top_pred_token = tokenizer.decode([top_pred_idx])
        
        input_token = tokens_list[pos]
        
        print(f"{pos:<10} {input_token:<20} '{top_pred_token}'  {top_pred_prob*100:>6.2f}%")

# Now check specifically the LAST position (what we care about)
print("\n" + "=" * 60)
print("FINAL POSITION (What action should be predicted)")
print("=" * 60)

last_position_logits = logits[-1]
probs = torch.nn.functional.softmax(last_position_logits, dim=-1)

# Get top 10
top_10 = torch.topk(probs, 10)

print(f"\nTop 10 predictions for next token after sequence:")
for i, (prob, idx) in enumerate(zip(top_10.values, top_10.indices), 1):
    token_str = tokenizer.decode([idx.item()])
    print(f"  {i:2d}. '{token_str:15s}' - {prob*100:6.2f}%  (ID: {idx.item()})")

# Check where BUY, SELL, HOLD rank
print(f"\nAction token rankings:")
for action in ['BUY', 'SELL', 'HOLD']:
    action_ids = tokenizer.encode(action, add_special_tokens=False)
    if len(action_ids) == 1:
        action_id = action_ids[0]
        action_prob = probs[action_id].item()
        rank = (probs > probs[action_id]).sum().item() + 1
        print(f"  {action}: Rank #{rank:4d}, Prob {action_prob*100:8.4f}%")
    else:
        print(f"  {action}: Multi-token! IDs={action_ids}")

print("\n" + "=" * 60)
