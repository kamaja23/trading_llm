# Makefile for TradeBot operations
.PHONY: help install verify data train test interactive clean pipeline \
        lint typecheck test-coverage precommit docker-build docker-run

# Default target
help:
	@echo "TradeBot Commands:"
	@echo ""
	@echo "  Pipeline:"
	@echo "    make install     - Install dependencies (RX 7800 XT ROCm)"
	@echo "    make verify      - Verify setup and GPU detection"
	@echo "    make data        - Generate training data"
	@echo "    make train       - Run complete training pipeline"
	@echo "    make test        - Run model testing"
	@echo "    make interactive - Start interactive inference"
	@echo "    make pipeline    - Run data -> train -> test"
	@echo "    make clean       - Remove generated data and models"
	@echo ""
	@echo "  Code Quality:"
	@echo "    make lint        - Run ruff linter"
	@echo "    make typecheck   - Run mypy type checker"
	@echo "    make format      - Format code with ruff"
	@echo "    make precommit   - Run all pre-commit hooks"
	@echo ""
	@echo "  Testing:"
	@echo "    make test-unit   - Run unit tests with pytest"
	@echo "    make test-coverage - Run tests with coverage report"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build - Build Docker image"
	@echo "    make docker-run   - Run Streamlit app via docker compose"
	@echo ""

# ---------------------------------------------------------------------------
# Pipeline commands
# ---------------------------------------------------------------------------

# Install dependencies with ROCm support for RX 7800 XT
install:
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
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
	rm -rf models/tradebot/*
	@echo "Cleaned data and model files"

# Full pipeline
pipeline: data train test
	@echo "Full pipeline complete!"

# ---------------------------------------------------------------------------
# Code quality commands
# ---------------------------------------------------------------------------

# Lint with ruff
lint:
	@echo "Running ruff linter..."
	ruff check .
	@echo "OK"

# Type-check with mypy
typecheck:
	@echo "Running mypy type checker..."
	mypy . --ignore-missing-imports
	@echo "OK"

# Format code with ruff
format:
	@echo "Formatting with ruff..."
	ruff format .
	@echo "OK"

# Run all pre-commit hooks
precommit:
	@echo "Running pre-commit hooks..."
	pre-commit run --all-files
	@echo "OK"

# ---------------------------------------------------------------------------
# Testing commands
# ---------------------------------------------------------------------------

# Run unit tests
test-unit:
	@echo "Running unit tests..."
	python -m pytest tests/ -v
	@echo "OK"

# Run tests with coverage
test-coverage:
	@echo "Running unit tests with coverage..."
	python -m pytest tests/ -v --cov=utils --cov=src --cov-report=term-missing
	@echo "OK"

# ---------------------------------------------------------------------------
# Docker commands
# ---------------------------------------------------------------------------

# Build Docker image
docker-build:
	docker build -t tradebot .

# Run via docker compose
docker-run:
	docker compose up

# Stop docker compose
docker-stop:
	docker compose down
