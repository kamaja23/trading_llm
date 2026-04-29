# Trading LLM Hello World

**Version 1.1** - Bug fixes and NVIDIA GPU optimization

A proof-of-concept implementation demonstrating that a small LLM can learn to predict trading actions (BUY/SELL/HOLD) from tokenized market data sequences.

## 🆕 What's New in v1.1

- ✅ **Fixed**: NaN handling errors in indicator calculations
- ✅ **Fixed**: Deprecated transformers API compatibility
- ✅ **Added**: Accelerate library support for modern training
- ✅ **Added**: Comprehensive .gitignore for model files
- ✅ **Optimized**: NVIDIA RTX 2070 support with FP16 mixed precision
- ✅ **Improved**: Better error messages and debugging

See [CHANGELOG.md](CHANGELOG.md) for complete details.

## Project Goal

Demonstrate technical feasibility of using an LLM to learn trading patterns by:
1. Converting historical price data into a token-based trading language
2. Fine-tuning a small language model (distilgpt2) on these sequences
3. Testing if the model can predict trading actions better than random chance

**This is a research/educational project, not investment advice.**

## Architecture Overview

```
Historical Price Data (SPY via yfinance)
    ↓
Token Generator (converts OHLCV → trading language tokens)
    ↓
Training Sequences (e.g., "<SYM_SPY> <TF_DAILY> ST_UpTrend Hi_Volume BUY")
    ↓
Fine-tuned distilgpt2 model
    ↓
Predictions (next token = BUY/SELL/HOLD)
```

## Trading Language Definition

Each sequence is composed of discrete tokens representing market state:

- **Symbol**: `<SYM_SPY>`, `<SYM_QQQ>`, etc.
- **Timeframe**: `<TF_DAILY>`, `<TF_WEEKLY>`, `<TF_60MIN>`
- **State Trend**: `ST_UpTrend`, `ST_DownTrend`, `ST_Flat`
- **Volume**: `Hi_Volume`, `Lo_Volume`, `Avg_Volume`
- **Indicators**: `HA_UpCross`, `HA_DownCross`, `STO_Cross`, `STO_NoCross`
- **Actions**: `BUY`, `SELL`, `HOLD`

Example sequence:
```
<SYM_SPY> <TF_DAILY> ST_DownTrend Hi_Volume HA_UpCross STO_Cross BUY
```

## Installation

### Quick Start (NVIDIA RTX 2070)

```bash
# 1. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

See [RTX2070_SETUP.md](RTX2070_SETUP.md) for detailed setup guide.

## Usage

### Step 1: Generate Training Data
```bash
python src/01_generate_training_data.py
```
This downloads SPY historical data and generates token sequences.

### Step 2: Train the Model
```bash
python src/02_train_model.py
```
Fine-tunes distilgpt2 on the trading sequences.

### Step 3: Test the Model
```bash
python src/03_test_model.py
```
Evaluates model accuracy on held-out test sequences.

### Step 4: Interactive Inference
```bash
python src/04_interactive_inference.py
```
Test the model with custom market state inputs.

## Project Structure

```
trading_llm_hello_world/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # Downloaded price data
│   ├── processed/        # Token sequences
│   └── train_test_split/ # Training/test datasets
├── models/
│   └── trading_llm/      # Saved model checkpoints
├── src/
│   ├── 01_generate_training_data.py
│   ├── 02_train_model.py
│   ├── 03_test_model.py
│   └── 04_interactive_inference.py
└── utils/
    ├── data_generator.py
    ├── token_definitions.py
    └── indicators.py
```

## Success Metrics (Phase 1)

- **Primary**: Model predicts correct action token >50% of the time (better than random 33%)
- **Secondary**: Model learns symbol/timeframe context (different predictions for same pattern under different symbols)

## Limitations & Risks

1. **Correlation ≠ Causation**: Model learns statistical patterns, not market dynamics
2. **Overfitting**: May memorize historical sequences without generalizing
3. **No Risk Management**: Predictions are binary (BUY/SELL), no position sizing or stop losses
4. **Lookback Bias**: All training data is "from the future" relative to prediction point
5. **No Transaction Costs**: Ignores fees, slippage, and execution delays

## Next Steps (Beyond Hello World)

- [ ] Add more sophisticated indicators (RSI, MACD, Bollinger Bands)
- [ ] Implement proper backtesting with transaction costs
- [ ] Test on multiple symbols and timeframes
- [ ] Paper trading with live data
- [ ] Explore reinforcement learning approach (reward = profit)
- [ ] Adapt to prediction markets (Polymarket)

## Timeline

- **Data Generation**: 30 minutes
- **Model Training**: 1-2 hours (CPU) / 15-30 minutes (GPU)
- **Testing & Evaluation**: 30 minutes
- **Total**: ~3-4 hours for complete Hello World

## License

Educational/Research use only. Not financial advice.
