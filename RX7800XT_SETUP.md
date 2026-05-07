# Quick Setup Guide - AMD RX 7800 XT with ROCm

## For Your RX 7800 XT Setup

This is a streamlined guide specifically for your AMD RX 7800 XT GPU using ROCm.

## Prerequisites

- AMD RX 7800 XT GPU (16GB VRAM)
- ROCm 5.7+ installed (check with `rocm-smi`)
- Python 3.8+ installed
- 8GB+ RAM recommended

## Installing ROCm on Linux

### Ubuntu/Debian:
```bash
# Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo dpkg -i amdgpu-install_6.0.60000-1_all.deb
sudo apt update
sudo amdgpu-install -y --usecase=rocm

# Add user to render group
sudo usermod -a -G render $USER
sudo usermod -a -G video $USER
reboot
```

### Verify ROCm Installation:
```bash
rocm-smi
# Should show your RX 7800 XT
```

## Installation (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/trading_llm.git
cd trading_llm

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows (ROCm not supported on Windows)

# 3. Install PyTorch with ROCm 5.7 (for RX 7800 XT)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# 4. Install other dependencies
pip install -r requirements.txt

# 5. Verify GPU is detected
python -c "import torch; print(f'CUDA API (ROCm): {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Should show: CUDA API (ROCm): True, GPU: AMD Radeon RX 7800 XT
```

## Run Training Pipeline (15-20 minutes)

```bash
# Step 1: Generate training data (~2 minutes)
python src/01_generate_training_data.py

# Step 2: Train model with GPU (~15-20 minutes)
python src/02_train_model.py

# Step 3: Test model (~1 minute)
python src/03_test_model.py

# Step 4: Interactive predictions
python src/04_interactive_inference.py
```

## Expected Performance on RX 7800 XT

| Metric | Value |
|--------|-------|
| Training Time | 15-20 minutes |
| VRAM Usage | 4-8 GB (of 16GB) |
| GPU Utilization | 80-95% |
| Speedup vs CPU | ~5x faster |
| Accuracy | 40-60% |

## Optimization Tips for RX 7800 XT

### Enable Mixed Precision (Faster Training)

Edit `src/02_train_model.py` around line 170:

```python
training_args = TrainingArguments(
    # ... existing args ...
    fp16=True,  # Enable FP16 mixed precision (works on ROCm)
)
```

**Benefit**: ~30% faster training, same accuracy

### Increase Batch Size (Optional)

With 16GB VRAM, you can use larger batch sizes:

```python
# In src/02_train_model.py, line ~95
BATCH_SIZE = 32  # Increase from 8 with your 16GB VRAM
```

**Benefit**: Faster training, but uses more VRAM

### Monitor GPU Usage

```bash
# In a separate terminal, watch GPU usage
watch -n 1 rocm-smi

# You should see:
# - GPU utilization: 80-95%
# - Memory usage: 4-8 GB
# - Temperature: 60-75°C
```

## Troubleshooting

### Out of Memory Error

```python
# Reduce batch size in src/02_train_model.py
BATCH_SIZE = 4  # Reduce from 8
```

### ROCm Not Available

```bash
# Verify ROCm drivers
rocm-smi

# Reinstall PyTorch with ROCm
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# Verify PyTorch can see the GPU
python -c "import torch; print(torch.cuda.is_available())"
# Should be True
```

### Slow Training

```bash
# Check GPU is being used
rocm-smi
# If GPU-Util is 0%, PyTorch isn't using GPU

# Verify in Python
python -c "import torch; print(torch.cuda.is_available())"
# Should be True
```

### PyTorch ROCm Compatibility Note

PyTorch uses the CUDA API naming even when running on AMD ROCm. This is normal:
- `torch.cuda.is_available()` returns `True` when ROCm GPU is available
- `torch.cuda.get_device_name(0)` returns the AMD GPU name
- All CUDA functions work transparently on ROCm

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

# Check GPU (ROCm)
rocm-smi
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
| RX 7800 XT (16GB) | ~15-20 min | Your setup ✓ |
| RTX 2070 (8GB) | ~20-25 min | Older NVIDIA alternative |
| RTX 3060 (12GB) | ~18-22 min | Similar performance |
| CPU only | ~90-120 min | 5x slower |

## Resources

- **Full Documentation**: `IMPLEMENTATION_GUIDE.md`
- **Troubleshooting**: `NETWORK_TROUBLESHOOTING.md`
- **Project Details**: `PROJECT_SUMMARY.md`
- **ROCm Docs**: https://rocm.docs.amd.com/

---

**Ready to start? Run the installation commands above!** 🚀
