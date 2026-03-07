# Docker Deployment Guide

Complete guide to running the Trading LLM project in Docker containers.

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up -d trading-llm

# Access the container
docker-compose exec trading-llm bash

# Inside container, run the pipeline
python src/01_generate_training_data.py
python src/02_train_model.py
python src/03_test_model.py
python src/04_interactive_inference.py
```

### Option 2: Docker (Manual)

```bash
# Build the image
docker build -t trading-llm:latest .

# Run the container
docker run -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  trading-llm:latest bash

# Inside container, run the pipeline
python src/01_generate_training_data.py
```

---

## 📦 What's Included

### Base Container
- Python 3.11 (slim)
- PyTorch, Transformers, HuggingFace libraries
- All project dependencies
- Non-root user (security)
- Sample SPY data (offline mode)

### Docker Compose Services
1. **trading-llm** - Main CPU-based container
2. **trading-llm-gpu** - GPU-enabled container (requires nvidia-docker)
3. **jupyter** - Jupyter notebook server (optional)

---

## 🔧 Configuration Options

### Environment Variables

Set in `docker-compose.yml` or pass with `-e`:

```yaml
environment:
  - PYTHONUNBUFFERED=1        # Python output buffering
  - USE_SAMPLE_DATA=true      # Use offline sample data
  - CUDA_VISIBLE_DEVICES=0    # GPU selection (GPU mode)
```

### Volume Mounts

**Persistent Data:**
```yaml
volumes:
  - ./data:/app/data          # Training data persists
  - ./models:/app/models      # Trained models persist
```

**Development Mode:**
```yaml
volumes:
  - ./src:/app/src            # Live code changes
  - ./utils:/app/utils        # Live code changes
```

### Resource Limits

**CPU/Memory (adjust for your system):**
```yaml
deploy:
  resources:
    limits:
      cpus: '4'       # Max 4 CPU cores
      memory: 8G      # Max 8GB RAM
    reservations:
      cpus: '2'       # Reserve 2 cores
      memory: 4G      # Reserve 4GB
```

---

## 💻 Usage Examples

### 1. Run Complete Pipeline (One Command)

```bash
docker-compose run --rm trading-llm bash -c "
  python src/01_generate_training_data.py &&
  python src/02_train_model.py &&
  python src/03_test_model.py
"
```

### 2. Interactive Development

```bash
# Start container in background
docker-compose up -d trading-llm

# Attach to running container
docker-compose exec trading-llm bash

# Run commands interactively
python verify_setup.py
python src/01_generate_training_data.py
```

### 3. Run Single Script

```bash
# Generate data only
docker-compose run --rm trading-llm \
  python src/01_generate_training_data.py

# Train model only
docker-compose run --rm trading-llm \
  python src/02_train_model.py
```

### 4. GPU Training (Faster)

**Prerequisites:**
- NVIDIA GPU
- nvidia-docker installed

**Usage:**
```bash
# Start GPU-enabled container
docker-compose --profile gpu up -d trading-llm-gpu

# Access container
docker-compose exec trading-llm-gpu bash

# Training will automatically use GPU
python src/02_train_model.py
```

### 5. Jupyter Notebook

```bash
# Start Jupyter server
docker-compose --profile jupyter up -d jupyter

# Access at http://localhost:8888
# No password required (dev mode only!)
```

---

## 🛠️ Advanced Usage

### Custom Build with Build Args

```dockerfile
# Add to Dockerfile
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim
```

```bash
# Build with custom Python version
docker build --build-arg PYTHON_VERSION=3.10 -t trading-llm:py310 .
```

### Multi-Stage Build (Production)

Current Dockerfile uses multi-stage build:
- **Stage 1:** Build dependencies
- **Stage 2:** Minimal runtime image

This reduces final image size significantly.

### Running on Different Architectures

```bash
# Build for ARM64 (Apple Silicon, AWS Graviton)
docker buildx build --platform linux/arm64 -t trading-llm:arm64 .

# Build for AMD64 (Intel/AMD x86_64)
docker buildx build --platform linux/amd64 -t trading-llm:amd64 .

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 \
  -t trading-llm:latest .
```

---

## 📊 Production Deployment

### Option 1: Docker Compose Production

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  trading-llm:
    image: trading-llm:latest
    container_name: trading-llm-prod
    
    # Don't mount source code in production
    volumes:
      - trading-data:/app/data
      - trading-models:/app/models
    
    # Restart policy
    restart: unless-stopped
    
    # Run as daemon
    command: tail -f /dev/null  # Keep running
    
    # Logging
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

volumes:
  trading-data:
  trading-models:
```

**Deploy:**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Option 2: Kubernetes (k8s)

Create `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-llm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: trading-llm
  template:
    metadata:
      labels:
        app: trading-llm
    spec:
      containers:
      - name: trading-llm
        image: trading-llm:latest
        resources:
          limits:
            memory: "8Gi"
            cpu: "4"
          requests:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: data
          mountPath: /app/data
        - name: models
          mountPath: /app/models
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: trading-data-pvc
      - name: models
        persistentVolumeClaim:
          claimName: trading-models-pvc
```

### Option 3: Cloud Run / AWS Fargate

**Build for cloud:**
```bash
# Tag for registry
docker tag trading-llm:latest gcr.io/YOUR-PROJECT/trading-llm:latest

# Push to Google Container Registry
docker push gcr.io/YOUR-PROJECT/trading-llm:latest

# Deploy to Cloud Run
gcloud run deploy trading-llm \
  --image gcr.io/YOUR-PROJECT/trading-llm:latest \
  --memory 8Gi \
  --cpu 4
```

---

## 🔍 Monitoring & Debugging

### View Logs

```bash
# View logs (follow mode)
docker-compose logs -f trading-llm

# View last 100 lines
docker-compose logs --tail=100 trading-llm
```

### Check Resource Usage

```bash
# Container stats (live)
docker stats trading-llm-dev

# Detailed container info
docker inspect trading-llm-dev
```

### Debug Container

```bash
# Access running container
docker-compose exec trading-llm bash

# Check Python environment
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"

# Verify setup
python verify_setup.py
```

### Shell into Exited Container

```bash
# If container exited with error
docker-compose run --rm trading-llm bash
```

---

## 🐛 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs trading-llm

# Rebuild without cache
docker-compose build --no-cache trading-llm

# Remove and recreate
docker-compose down -v
docker-compose up -d trading-llm
```

### Out of Memory

**Reduce batch size** in `src/02_train_model.py`:
```python
BATCH_SIZE = 4  # Reduce from 8
```

**Increase Docker memory** (Docker Desktop):
- Settings → Resources → Memory → 8GB+

### GPU Not Detected

```bash
# Check nvidia-docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# If fails, install nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Permission Denied

```bash
# Fix ownership (if running as root)
docker-compose exec trading-llm chown -R trader:trader /app/data /app/models

# Or run as root (not recommended for production)
docker-compose exec -u root trading-llm bash
```

### Slow Build Times

```bash
# Use BuildKit (faster)
DOCKER_BUILDKIT=1 docker-compose build

# Multi-core build
docker-compose build --parallel
```

---

## 📝 Best Practices

### Security

1. ✅ **Run as non-root user** (already configured)
2. ✅ **Use multi-stage builds** (reduces attack surface)
3. ✅ **Minimal base image** (python:3.11-slim)
4. ⚠️ **Don't expose ports** unless necessary
5. ⚠️ **Use secrets** for API keys (not environment variables)

### Performance

1. ✅ **Layer caching** - Dependencies before code
2. ✅ **Multi-stage builds** - Smaller images
3. ✅ **Volume mounts** - Persistent data
4. ✅ **Resource limits** - Prevent OOM
5. ⚠️ **GPU for training** - 5-10x faster

### Development

1. ✅ **Mount source code** - Live changes
2. ✅ **Use docker-compose** - Easier orchestration
3. ✅ **Version control** - Tag images
4. ⚠️ **Separate dev/prod** - Different compose files
5. ⚠️ **Health checks** - Monitor container status

---

## 🌐 CI/CD Integration

### GitHub Actions

Create `.github/workflows/docker.yml`:

```yaml
name: Docker Build and Push

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: docker build -t trading-llm:${{ github.sha }} .
    
    - name: Run tests
      run: |
        docker run --rm trading-llm:${{ github.sha }} \
          python verify_setup.py
    
    - name: Push to registry
      if: success()
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker tag trading-llm:${{ github.sha }} myrepo/trading-llm:latest
        docker push myrepo/trading-llm:latest
```

---

## 📦 Image Management

### Build and Tag

```bash
# Build with tag
docker build -t trading-llm:v1.0 .

# Multiple tags
docker build -t trading-llm:v1.0 -t trading-llm:latest .
```

### Push to Registry

```bash
# Docker Hub
docker tag trading-llm:latest username/trading-llm:latest
docker push username/trading-llm:latest

# AWS ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker tag trading-llm:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/trading-llm:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/trading-llm:latest
```

### Clean Up

```bash
# Remove containers
docker-compose down

# Remove volumes too
docker-compose down -v

# Remove unused images
docker image prune -a

# Remove everything
docker system prune -a --volumes
```

---

## 🎯 Common Workflows

### Development Workflow

```bash
# 1. Start development container
docker-compose up -d trading-llm

# 2. Attach and develop
docker-compose exec trading-llm bash

# 3. Test changes (code is mounted live)
python src/01_generate_training_data.py

# 4. When done, stop container
docker-compose down
```

### Training Workflow

```bash
# 1. Build image
docker-compose build trading-llm

# 2. Generate data
docker-compose run --rm trading-llm \
  python src/01_generate_training_data.py

# 3. Train model (GPU if available)
docker-compose --profile gpu run --rm trading-llm-gpu \
  python src/02_train_model.py

# 4. Test results
docker-compose run --rm trading-llm \
  python src/03_test_model.py
```

### Production Workflow

```bash
# 1. Build production image
docker build -f Dockerfile.prod -t trading-llm:prod .

# 2. Run with production compose file
docker-compose -f docker-compose.prod.yml up -d

# 3. Monitor logs
docker-compose -f docker-compose.prod.yml logs -f

# 4. Deploy new version
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

## ✅ Quick Reference

```bash
# Build
docker-compose build

# Start
docker-compose up -d

# Access shell
docker-compose exec trading-llm bash

# Run script
docker-compose run --rm trading-llm python src/SCRIPT.py

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

**You're all set! 🎉 Your Trading LLM is now containerized and portable.**
