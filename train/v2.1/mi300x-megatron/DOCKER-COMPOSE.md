# Docker Compose Setup for MegaBlocks Training

This guide covers using Docker Compose for easier container management in the MegaBlocks training environment.

## Quick Start

### Start Training Container

```bash
# Start the main training container
docker-compose up -d megablocks-training

# Access the container shell
docker-compose exec megablocks-training /bin/bash
```

### Start with Jupyter Notebook (Optional)

```bash
# Start both training container and Jupyter Lab
docker-compose --profile notebook up -d

# Access Jupyter at http://localhost:8888
```

## Available Services

### megablocks-training (Main Service)

- **Container Name**: `megablocks_training`
- **Purpose**: Main training environment
- **Status**: Runs continuously in background
- **Access**: Via `docker-compose exec`

### notebook (Optional Service)

- **Container Name**: `megablocks_notebook`
- **Purpose**: Jupyter Lab for development/analysis
- **Status**: Only starts with `--profile notebook`
- **Access**: http://localhost:8888 (no token required)

## Common Commands

### Container Management

```bash
# Start services
docker-compose up -d                          # Start main training container
docker-compose --profile notebook up -d       # Start with Jupyter Lab

# Stop services
docker-compose down                           # Stop and remove containers
docker-compose stop                           # Stop without removing

# View running containers
docker-compose ps

# View logs
docker-compose logs megablocks-training       # Training container logs
docker-compose logs notebook                  # Jupyter container logs
docker-compose logs -f megablocks-training    # Follow logs in real-time
```

### Executing Commands

```bash
# Interactive shell
docker-compose exec megablocks-training /bin/bash

# Run single commands
docker-compose exec megablocks-training python3 --version
docker-compose exec megablocks-training rocm-smi

# Run training scripts
docker-compose exec megablocks-training /bin/bash -c "cd /workspace/shisa-v2.1 && ./gpt2-125m/03-train-dense.sh"
docker-compose exec megablocks-training /bin/bash -c "cd /workspace/shisa-v2.1 && ./gpt2-125m/04-train-moe.sh my_experiment"
```

## Training Workflow with Docker Compose

### 1. Start the Environment

```bash
# Start training container
docker-compose up -d megablocks-training

# Verify container is running
docker-compose ps
```

### 2. Generate Training Data

```bash
# Access container and generate data
docker-compose exec megablocks-training /bin/bash -c "
    cd /workspace/shisa-v2.1 &&
    python3 02-generate.sft.shisa-v2.1-megablocks.py
"
```

### 3. Run Training

```bash
# Standard GPT-2 125M training
docker-compose exec megablocks-training /bin/bash -c "
    cd /workspace/shisa-v2.1 &&
    ./gpt2-125m/03-train-dense.sh
"

# MoE training with custom parameters
docker-compose exec megablocks-training /bin/bash -c "
    cd /workspace/shisa-v2.1 &&
    ./gpt2-125m/04-train-moe.sh my_moe_experiment 128 2 2 0.05 16
"
```

### 4. Monitor Training

```bash
# Follow training logs in real-time
docker-compose exec megablocks-training tail -f /workspace/shisa-v2.1/gpt2-125m/checkpoints/<run_name>/train.log

# Monitor GPU usage
docker-compose exec megablocks-training watch -n 1 rocm-smi

# Check training progress
docker-compose exec megablocks-training ls -la /workspace/shisa-v2.1/gpt2-125m/checkpoints/
```

### 5. Convert to Hugging Face

```bash
docker-compose exec megablocks-training /bin/bash -c "\
    cd /workspace/shisa-v2.1 && \
    ./export-hf.sh gpt2-125m/checkpoints/dense_YYYYMMDD_HHMMSS --iteration iter_0002203"
```

- Add `--output /some/path` to control the Hugging Face export location (defaults to `<run_dir>_hf`).
- Use `--model-dir <path>` if tokenizers live outside the checkpoint directory.
- Use `--hf-dtype bf16` (default) or another dtype to control the saved Hugging Face weights.

## Environment Variables

The docker-compose setup includes several pre-configured environment variables:

```yaml
environment:
  - PYTHONPATH=/workspace/Megatron-LM:$PYTHONPATH
  - CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  - WANDB_PROJECT=shisa-v2-megablocks
  - WANDB_LOG_MODEL=false
  - WANDB_WATCH=false
```

### Customizing Environment Variables

Create a `.env` file in the same directory as `docker-compose.yml`:

```bash
# .env file
WANDB_PROJECT=my-custom-project
WANDB_ENTITY=my-wandb-team
CUDA_VISIBLE_DEVICES=0,1,2,3
```

## Volume Mounts

The following directories are automatically mounted:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `.` (current dir) | `/workspace/shisa-v2.1` | Scripts, data, checkpoints |
| `/root/.cache` | `/root/.cache` | pip/HuggingFace cache |
| `/root/.netrc` | `/root/.netrc` | Wandb token & credentials |

## New Training Features

### Automatic Training Statistics

Both training scripts now provide detailed statistics at completion:

```
=== Training completed! ===
Training end time: Thu Sep 26 10:30:00 UTC 2025
Total training duration: 2h 15m 30s
Training steps completed: 1500
Epochs completed: 3
Total samples processed: 768000
Samples per second: 94
Checkpoints saved to: /workspace/shisa-v2.1/gpt2-125m/checkpoints/<run_name>
Training log: /workspace/shisa-v2.1/gpt2-125m/checkpoints/<run_name>/train.log

Each launch creates a timestamped subdirectory (e.g. `dense_YYYYMMDD_HHMMSS`). Use `RUN_NAME`/`RUN_TIMESTAMP`/`CHECKPOINT_ROOT` to override, or export `OVERWRITE_CHECKPOINTS=1` to reuse an existing folder.
```

### Enhanced Wandb Integration

- **Automatic project naming**: `shisa-v2-megablocks`
- **Unique experiment names**: Includes timestamp
- **Parameter logging**: Logs model parameters and gradients
- **Validation perplexity**: Logged to tensorboard

### Checkpoint Management

- **Automatic cleanup**: Removes existing checkpoint directories
- **Epoch-based saving**: Saves at the end of each epoch
- **No more "directory exists" errors**

## Troubleshooting

### Container Issues

```bash
# Container won't start
docker-compose logs megablocks-training

# Rebuild container (if needed)
docker-compose build megablocks-training

# Reset everything
docker-compose down -v  # Remove volumes too
docker-compose up -d
```

### Training Issues

```bash
# Check data files exist
docker-compose exec megablocks-training ls -la /workspace/shisa-v2.1/data/

# Verify Megatron installation
docker-compose exec megablocks-training ls -la /workspace/Megatron-LM/

# Test GPU access
docker-compose exec megablocks-training rocm-smi
docker-compose exec megablocks-training python3 -c "import torch; print(torch.cuda.is_available())"
```

### Wandb Issues

```bash
# Check wandb login
docker-compose exec megablocks-training wandb status

# Re-login to wandb
docker-compose exec megablocks-training wandb login

# Check .netrc file
docker-compose exec megablocks-training cat /root/.netrc
```

## Comparison: Docker Run vs Docker Compose

### Docker Run (01-run-docker.sh)
- ✅ Quick interactive access
- ✅ Automatic cleanup on exit
- ❌ Manual container name management
- ❌ No service orchestration

### Docker Compose
- ✅ Consistent container names
- ✅ Service orchestration
- ✅ Environment variable management
- ✅ Optional Jupyter service
- ✅ Persistent containers
- ❌ Requires explicit cleanup

## Best Practices

### Development Workflow

1. **Use Docker Compose for long training runs**
2. **Use docker run script for quick testing**
3. **Keep containers running during active development**
4. **Use Jupyter service for data analysis**

### Production Training

1. **Always use docker-compose for production training**
2. **Monitor logs with `docker-compose logs -f`**
3. **Set up proper wandb credentials in `.netrc`**
4. **Use unique experiment names for tracking**

### Cleanup

```bash
# Regular cleanup (keeps volumes)
docker-compose down

# Full cleanup (removes everything)
docker-compose down -v --rmi local

# Clean up old containers/images
docker system prune -a
```

This setup provides a robust, scalable environment for MegaBlocks training with proper monitoring, logging, and easy management.
