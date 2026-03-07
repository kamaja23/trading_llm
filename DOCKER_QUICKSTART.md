# 🐳 Docker Quick Start

## Fastest Way to Run

```bash
# 1. Build image
make build

# 2. Run complete pipeline
make train
```

That's it! Everything runs in Docker.

---

## What You Get

✅ **Isolated Environment** - No conflicts with your system  
✅ **Reproducible** - Same environment everywhere  
✅ **GPU Support** - Automatic GPU detection  
✅ **Portable** - Run anywhere Docker runs  
✅ **Production Ready** - Deploy to cloud instantly  

---

## Common Commands

```bash
# Interactive development
make shell                  # Access container bash

# Individual steps
make data                   # Generate training data
docker-compose run --rm trading-llm python src/02_train_model.py

# With GPU (5-10x faster)
make gpu-train

# Jupyter notebook
make jupyter                # Access at http://localhost:8888

# Monitoring
make logs                   # View container logs

# Cleanup
make clean                  # Remove everything
```

---

## Three Ways to Use Docker

### 1️⃣ Makefile (Easiest)

```bash
make build
make train
```

### 2️⃣ Docker Compose (Flexible)

```bash
docker-compose up -d trading-llm
docker-compose exec trading-llm bash
python src/01_generate_training_data.py
```

### 3️⃣ Plain Docker (Manual)

```bash
docker build -t trading-llm .
docker run -it -v $(pwd)/data:/app/data trading-llm bash
```

---

## Requirements

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed (included with Docker Desktop)
- *Optional:* nvidia-docker for GPU support

---

## Need Help?

📖 **Full Guide:** See `DOCKER_GUIDE.md` for complete documentation  
🐛 **Troubleshooting:** Common issues and solutions in the guide  
🚀 **Production:** Deployment examples for cloud platforms  

---

## File Structure

```
trading_llm_hello_world/
├── Dockerfile              ← Docker image definition
├── docker-compose.yml      ← Orchestration config
├── Makefile               ← Convenient commands
├── DOCKER_GUIDE.md        ← Complete documentation
└── .dockerignore          ← Build optimization
```

---

**Ready?** Run `make build` then `make train` and you're done! 🎉
