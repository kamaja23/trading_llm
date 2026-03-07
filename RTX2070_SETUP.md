# Quick Setup Guide - NVIDIA RTX 2070

## For Your RTX 2070 Setup

This is a streamlined guide specifically for your NVIDIA RTX 2070 GPU.

## Prerequisites

- NVIDIA RTX 2070 GPU
- NVIDIA drivers installed (check with `nvidia-smi`)
- Python 3.8+ installed
- 8GB+ RAM recommended

## Installation (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/trading_llm.git
cd trading_llm

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows

# 3. Install PyTorch with CUDA 11.8 (for RTX 2070)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install other dependencies
pip install -r requirements.txt

# 5. Verify GPU is detected
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Should show: CUDA: True, GPU: NVIDIA GeForce RTX 2070
```

## Run Training Pipeline (20-25 minutes)

```bash
# Step 1: Generate training data (~2 minutes)
python src/01_generate_training_data.py

# Step 2: Train model with GPU (~20-25 minutes)
python src/02_train_model.py

# Step 3: Test model (~1 minute)
python src/03_test_model.py

# Step 4: Interactive predictions
python src/04_interactive_inference.py
```

## Expected Performance on RTX 2070

| Metric | Value |
|--------|-------|
| Training Time | 20-25 minutes |
| VRAM Usage | 4-6 GB (of 8GB) |
| GPU Utilization | 80-95% |
| Speedup vs CPU | ~5x faster |
| Accuracy | 40-60% |

## Optimization Tips for RTX 2070

### Enable Mixed Precision (Faster Training)

Edit `src/02_train_model.py` around line 170:

```python
training_args = TrainingArguments(
    # ... existing args ...
    fp16=True,  # Enable FP16 mixed precision
)
```

**Benefit**: ~30% faster training, same accuracy

### Increase Batch Size (Optional)

If you have headroom (check VRAM with `nvidia-smi`):

```python
# In src/02_train_model.py, line ~95
BATCH_SIZE = 16  # Increase from 8 if you have VRAM
```

**Benefit**: Faster training, but uses more VRAM

### Monitor GPU Usage

```bash
# In a separate terminal, watch GPU usage
watch -n 1 nvidia-smi

# You should see:
# - GPU utilization: 80-95%
# - Memory usage: 4-6 GB
# - Temperature: 60-75°C
```

## Troubleshooting

### Out of Memory Error

```python
# Reduce batch size in src/02_train_model.py
BATCH_SIZE = 4  # Reduce from 8
```

### CUDA Not Available

```bash
# Verify NVIDIA drivers
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Slow Training

```bash
# Check GPU is being used
nvidia-smi
# If GPU-Util is 0%, PyTorch isn't using GPU

# Verify in Python
python -c "import torch; print(torch.cuda.is_available())"
# Should be True
```

## Docker Alternative (Optional)

```bash
# Build Docker image
docker-compose build trading-llm-gpu

# Run training in Docker
docker-compose run --rm trading-llm-gpu python src/02_train_model.py
```

## File Structure

```
trading_llm/
├── src/                    # Training scripts
├── utils/                  # Helper modules
├── data/                   # Training data (created when you run)
├── models/                 # Trained models (created when you run)
├── requirements.txt        # Python dependencies
└── *.md                    # Documentation
```

## What Gets Created

After running the pipeline:

```
data/
├── raw/SPY_daily.csv              # Downloaded stock data
├── processed/all_sequences.txt    # Token sequences
└── train_test_split/
    ├── train.txt                  # Training data
    ├── val.txt                    # Validation data
    └── test.txt                   # Test data

models/trading_llm/
├── checkpoint-*/                  # Training checkpoints (DON'T commit)
└── final_model/                   # Final trained model (DON'T commit)
```

## Common Commands

```bash
# Verify setup
python verify_setup.py

# Generate data only
python src/01_generate_training_data.py

# Train only
python src/02_train_model.py

# Test only
python src/03_test_model.py

# Interactive mode
python src/04_interactive_inference.py

# Check GPU
nvidia-smi
```

## Next Steps

1. ✅ Run the pipeline and get a working model
2. Read `IMPLEMENTATION_GUIDE.md` for details
3. Try adding more indicators
4. Experiment with different symbols
5. Implement backtesting

## Performance Comparison

| Hardware | Training Time | Notes |
|----------|--------------|-------|
| RTX 2070 (8GB) | ~20-25 min | Your setup ✓ |
| RX 7800 XT (16GB) | ~15-20 min | AMD alternative |
| RTX 3060 (12GB) | ~18-22 min | Similar performance |
| CPU only | ~90-120 min | 5x slower |

## Resources

- **Full Documentation**: `IMPLEMENTATION_GUIDE.md`
- **Docker Guide**: `DOCKER_GUIDE.md`
- **Troubleshooting**: `NETWORK_TROUBLESHOOTING.md`
- **Project Details**: `PROJECT_SUMMARY.md`

---

**Ready to start? Run the installation commands above!** 🚀
