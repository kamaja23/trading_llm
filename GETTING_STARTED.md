# Getting Started - Quick Reference

This is a 5-minute quick start guide. For detailed information, see:
- **README.md** - Project overview
- **IMPLEMENTATION_GUIDE.md** - Detailed technical guide
- **PROJECT_SUMMARY.md** - Complete project documentation

## 30-Second Overview

This project trains a small AI model to predict when to BUY/SELL/HOLD stocks based on market data patterns.

## Installation (5 minutes)

```bash
# 1. Clone or download this project
# 2. Navigate to the project directory
cd trading_llm_hello_world

# 3. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify setup
python verify_setup.py
```

## Quick Start (Automatic)

Run everything with one command:

```bash
./quick_start.sh
```

This will:
1. Set up the environment
2. Download stock data
3. Train the model (takes 1-2 hours on CPU)
4. Test the results

## Manual Execution (Step-by-Step)

If you prefer to run each step individually:

### Step 1: Generate Training Data (~2 minutes)
```bash
python src/01_generate_training_data.py
```
Downloads SPY stock data and converts it to token sequences.

### Step 2: Train the Model (1-2 hours on CPU, 20 min on GPU)
```bash
python src/02_train_model.py
```
Fine-tunes a small AI model on the trading data.

**Note:** Training takes time. Consider:
- Running overnight on CPU
- Using a GPU (Google Colab free tier works)
- Taking a coffee break ☕

### Step 3: Test the Model (~1 minute)
```bash
python src/03_test_model.py
```
Evaluates how well the model learned to predict.

### Step 4: Interactive Mode
```bash
python src/04_interactive_inference.py
```
Test the model with custom market conditions.

## What to Expect

### Good Results
- Accuracy > 33% (better than random guessing)
- Model predicts BUY for uptrends, SELL for downtrends
- Different predictions for different market conditions

### Example Output
```
Overall Accuracy: 47.3%
✓ Model is 13.97% better than random!

Sample Prediction:
Input:  <SYM_SPY> <TF_DAILY> ST_UpTrend Hi_Volume HA_UpCross STO_Cross
Output: BUY (78.5% confidence)
```

## File Structure

```
trading_llm_hello_world/
├── README.md              ← Start here
├── GETTING_STARTED.md     ← You are here
├── IMPLEMENTATION_GUIDE.md ← Detailed guide
├── PROJECT_SUMMARY.md     ← Full documentation
│
├── src/                   ← Python scripts (run these)
├── utils/                 ← Helper functions
├── data/                  ← Downloaded data goes here
└── models/                ← Trained models saved here
```

## Common Issues

### "Package not found" error
```bash
pip install -r requirements.txt
```

### "CUDA out of memory" (if using GPU)
Edit `src/02_train_model.py`, reduce `BATCH_SIZE` from 8 to 4.

### Training takes forever
- Normal on CPU (1-2 hours)
- Use Google Colab for free GPU: https://colab.research.google.com/
- Or run overnight and check results in the morning

### "No module named 'utils'"
Make sure you're in the project root directory:
```bash
cd trading_llm_hello_world
```

## Next Steps

After completing the Hello World:

1. **Read the results** - Did it beat random guessing?
2. **Try interactive mode** - Test different market conditions
3. **Read IMPLEMENTATION_GUIDE.md** - Learn how to extend it
4. **Experiment** - Add more symbols, indicators, timeframes

## Important Warnings

⚠️ **This is a research project, not a real trading system**
- Do NOT use with real money without extensive testing
- Historical performance ≠ future results  
- Consult a financial advisor before trading

⚠️ **What this proves:**
- ✅ The approach works technically
- ✅ The model can learn patterns
- ❌ Does NOT prove profitability

⚠️ **Before using real money:**
- Paper trade for 6+ months
- Understand maximum loss scenarios
- Start with tiny amounts ($10-100)
- Never risk more than you can afford to lose

## Getting Help

1. **Check the docs:**
   - IMPLEMENTATION_GUIDE.md (technical details)
   - PROJECT_SUMMARY.md (complete overview)

2. **Common questions:**
   - "How long does training take?" → 1-2 hours (CPU), 20 min (GPU)
   - "Do I need a GPU?" → No, but it's faster
   - "Can I use this for real trading?" → Not yet, extensive testing needed
   - "What accuracy should I expect?" → 40-60% (vs 33% random)

3. **Debugging:**
   - Run `python verify_setup.py` to check installation
   - Check file paths are correct
   - Make sure you're in the project root directory

## Quick Commands Reference

```bash
# Verify setup
python verify_setup.py

# Run entire pipeline
./quick_start.sh

# Run individual steps
python src/01_generate_training_data.py
python src/02_train_model.py
python src/03_test_model.py
python src/04_interactive_inference.py

# Activate virtual environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Deactivate virtual environment
deactivate
```

## What's Next?

Once you complete the Hello World:

**Short term (1-2 days):**
- Add more stock symbols (QQQ, IWM, DIA)
- Add more indicators (RSI, MACD)
- Test different time periods

**Medium term (1-2 weeks):**
- Implement backtesting with transaction costs
- Try larger models (Phi-2, Llama 3.2)
- Multiple timeframes (daily + weekly)

**Long term (1-3 months):**
- Paper trading with live data
- Risk management system
- Polymarket adaptation

**Full roadmap in PROJECT_SUMMARY.md**

---

**Ready to start? Run this:**

```bash
python verify_setup.py  # Check if ready
./quick_start.sh        # Run everything
```

Good luck! 🚀
