from transformers import GPT2Tokenizer, GPT2LMHeadModel
from pathlib import Path
import torch

model_path = Path('models/trading_llm/final_model')
tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
model = GPT2LMHeadModel.from_pretrained(str(model_path))

print("=== MODEL DIAGNOSIS ===\n")

# Check vocabulary
print(f"Total vocabulary size: {len(tokenizer)}")
print(f"Base GPT-2 vocab: 50257")
print(f"Custom tokens added: {len(tokenizer) - 50257}\n")

# Check if custom tokens exist
custom_tokens = ['BUY', 'SELL', 'HOLD', '<SYM_SPY>', 'ST_UpTrend']
print("Custom token IDs:")
for token in custom_tokens:
    token_id = tokenizer.encode(token, add_special_tokens=False)
    print(f"  '{token}': {token_id}")

print("\n=== PREDICTIONS ===\n")

# Get top predictions with actual token IDs
test_input = '<SYM_SPY> <TF_DAILY> ST_UpTrend Hi_Volume HA_UpCross STO_Cross'
inputs = tokenizer(test_input, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits[0, -1, :]
    probs = torch.nn.functional.softmax(predictions, dim=-1)
    top_10 = torch.topk(probs, 10)
    
    print('Top 10 predicted tokens (with IDs):')
    for prob, idx in zip(top_10.values, top_10.indices):
        idx_val = idx.item()
        token_str = tokenizer.decode([idx_val])
        print(f'  ID {idx_val:5d}: "{token_str}" - {prob*100:.4f}%')

print("\n=== ACTION TOKEN CHECK ===\n")

# Check where BUY, SELL, HOLD rank
action_tokens = {'BUY': None, 'SELL': None, 'HOLD': None}
for action in action_tokens.keys():
    token_ids = tokenizer.encode(action, add_special_tokens=False)
    if len(token_ids) == 1:
        token_id = token_ids[0]
        action_prob = probs[token_id].item()
        action_rank = (probs > probs[token_id]).sum().item() + 1
        print(f"{action}: ID {token_id}, Prob {action_prob*100:.6f}%, Rank #{action_rank}")
    else:
        print(f"{action}: Multi-token encoding {token_ids} (PROBLEM!)")

