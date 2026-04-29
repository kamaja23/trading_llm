# Makefile for Trading LLM operations
.PHONY: help install verify data train test clean

# Default target
help:
	@echo "Trading LLM Commands:"
	@echo ""
	@echo "  make install     - Install dependencies (RTX 2070 CUDA)"
	@echo "  make verify      - Verify setup and GPU detection"
	@echo "  make data        - Generate training data"
	@echo "  make train       - Run complete training pipeline"
	@echo "  make test        - Run model testing"
	@echo "  make interactive - Start interactive inference"
	@echo "  make clean       - Remove generated data and models"
	@echo ""

# Install dependencies with CUDA support for RTX 2070
install:
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	pip install -r requirements.txt

# Verify setup
verify:
	python verify_setup.py

# Generate training data
data:
	python src/01_generate_training_data.py

# Train model
train: data
	python src/02_train_model.py

# Run tests
test:
	python src/03_test_model.py

# Interactive inference
interactive:
	python src/04_interactive_inference.py

# Clean up generated files
clean:
	rm -rf data/raw/* data/processed/* data/train_test_split/*
	rm -rf models/trading_llm/*
	@echo "Cleaned data and model files"

# Full pipeline
pipeline: data train test
	@echo "Full pipeline complete!"
