# Makefile for Trading LLM Docker operations
.PHONY: help build up down shell train test clean logs

# Default target
help:
	@echo "Trading LLM Docker Commands:"
	@echo ""
	@echo "  make build       - Build Docker image"
	@echo "  make up          - Start containers in background"
	@echo "  make down        - Stop and remove containers"
	@echo "  make shell       - Access container shell"
	@echo "  make train       - Run complete training pipeline"
	@echo "  make test        - Run tests"
	@echo "  make data        - Generate training data"
	@echo "  make gpu-train   - Train with GPU (requires nvidia-docker)"
	@echo "  make jupyter     - Start Jupyter notebook server"
	@echo "  make logs        - View container logs"
	@echo "  make clean       - Remove containers and volumes"
	@echo "  make rebuild     - Rebuild without cache"
	@echo ""

# Build the Docker image
build:
	docker-compose build

# Start containers
up:
	docker-compose up -d trading-llm

# Stop containers
down:
	docker-compose down

# Access container shell
shell:
	docker-compose exec trading-llm bash

# Generate training data
data:
	docker-compose run --rm trading-llm python src/01_generate_training_data.py

# Train model
train:
	docker-compose run --rm trading-llm bash -c " \
		python src/01_generate_training_data.py && \
		python src/02_train_model.py && \
		python src/03_test_model.py"

# GPU training
gpu-train:
	docker-compose --profile gpu run --rm trading-llm-gpu bash -c " \
		python src/01_generate_training_data.py && \
		python src/02_train_model.py && \
		python src/03_test_model.py"

# Run tests
test:
	docker-compose run --rm trading-llm python verify_setup.py

# Start Jupyter
jupyter:
	@echo "Starting Jupyter at http://localhost:8888"
	docker-compose --profile jupyter up jupyter

# View logs
logs:
	docker-compose logs -f trading-llm

# Clean up everything
clean:
	docker-compose down -v
	docker system prune -f

# Rebuild without cache
rebuild:
	docker-compose build --no-cache

# Quick start (build + run pipeline)
quickstart: build train
	@echo "Complete pipeline finished!"

# Development mode (build + shell)
dev: build
	docker-compose up -d trading-llm
	docker-compose exec trading-llm bash
