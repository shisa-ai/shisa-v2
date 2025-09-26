# Agent Docker Workflow Documentation

This document explains how to work with the MegaBlocks training environment using Docker containers.

## Container Setup

The training environment uses a ROCm 7.0 PyTorch container that provides all necessary dependencies for MegaBlocks training on MI300X GPUs.

### Starting the Container

Use the provided script to start an interactive container:

```bash
./01-run-docker.sh
```

This script:
- Mounts the current directory (`/root/shisa-v2/train/mi300x`) to `/workspace/project` inside the container
- Provides GPU access via `/dev/kfd` and `/dev/dri` devices
- Sets up proper permissions and shared memory
- Uses the ROCm 7.0 PyTorch training image optimized for MI300X

## Working with Running Containers

### Finding Active Containers

To see all running containers:

```bash
docker ps
```

Example output:
```
CONTAINER ID   IMAGE                                               COMMAND     CREATED        STATUS        NAMES
abc123456789   rocm/7.0:rocm7.0_pytorch_training_instinct_...     /bin/bash   5 minutes ago  Up 5 minutes  rocm7_container_1727340123
```

### Executing Commands in Running Containers

To execute commands in a running container, use `docker exec`:

```bash
# Interactive shell
docker exec -it <container_name_or_id> /bin/bash

# Single command
docker exec <container_name_or_id> <command>
```

Examples:
```bash
# Get an interactive shell
docker exec -it rocm7_container_1727340123 /bin/bash

# Check GPU status
docker exec rocm7_container_1727340123 rocm-smi

# Run training script
docker exec rocm7_container_1727340123 /bin/bash -c "cd /workspace/project && ./03-megablocks-gpt2-125m.sh"
```

### File System Access

- **Host Path**: `/root/shisa-v2/train/mi300x`
- **Container Path**: `/workspace/project`
- All files are synchronized between host and container
- Scripts, data, and checkpoints are accessible from both environments

## Training Workflow

### 1. Data Preparation

First, generate the training data (run from host or container):

```bash
# Generate SFT dataset for MegaBlocks
python3 02-generate.sft.shisa-v2.1-megablocks.py
```

This creates:
- `./data/sft.shisa-v2.1_text_document.bin` (binary data)
- `./data/sft.shisa-v2.1_text_document.idx` (index file)
- `./data/gpt2-vocab.json` (vocabulary)
- `./data/gpt2-merges.txt` (merge rules)

### 2. Standard Training

Run dense GPT-2 125M training:

```bash
# From container
cd /workspace/project
./03-megablocks-gpt2-125m.sh
```

### 3. MoE Training

Run Mixture of Experts training:

```bash
# From container
cd /workspace/project

# Default configuration
./04-megablocks-moe-gpt2-125m.sh

# Custom configuration
./04-megablocks-moe-gpt2-125m.sh my_experiment 128 2 2 0.05 16
#                                  ^experiment  ^experts ^capacity ^top_k ^loss_weight ^batch_size
```

## Training Configuration

Both scripts now use **epoch-based training** with automatic step calculation:

- **EPOCHS=3** (configurable at top of each script)
- **Training steps** calculated automatically from dataset size
- **Save intervals** set to save at the end of each epoch
- **Global batch size**: 512 (optimized for MI300X)

### Calculation Details

Training steps are calculated as:
```bash
TRAINING_STEPS = (NUM_SAMPLES * EPOCHS) / GLOBAL_BATCH_SIZE
```

The number of samples is automatically extracted from the `.idx` file header.

## Monitoring Training

### Logs

Training logs are saved to:
- Standard: `/workspace/project/checkpoints/train.log`
- MoE: `/workspace/project/checkpoints/<experiment_name>/train.log`

### Checkpoints

Checkpoints are saved to:
- Standard: `/workspace/project/checkpoints/`
- MoE: `/workspace/project/checkpoints/<experiment_name>/`

### Real-time Monitoring

```bash
# Monitor training progress
docker exec <container_name> tail -f /workspace/project/checkpoints/train.log

# Check GPU utilization
docker exec <container_name> watch -n 1 rocm-smi
```

## Best Practices

### Container Management

1. **Use descriptive container names** by modifying the `--name` parameter in `01-run-docker.sh`
2. **Keep containers running** during training to avoid restart overhead
3. **Use `docker exec`** for running commands rather than starting new containers

### File Management

1. **Work from `/workspace/project`** inside containers
2. **Edit scripts from the host** for easier version control
3. **Monitor disk space** as checkpoints can be large

### Training Management

1. **Verify data generation** completed successfully before starting training
2. **Check available GPU memory** before starting MoE training with many experts
3. **Use screen/tmux** for long-running training sessions

## Troubleshooting

### Container Issues

```bash
# Container not starting
docker logs <container_name>

# Permission issues
docker exec <container_name> ls -la /workspace/project

# GPU not accessible
docker exec <container_name> rocm-smi
```

### Training Issues

```bash
# Check data files exist
ls -la data/sft.shisa-v2.1_text_document.*

# Verify Megatron installation
docker exec <container_name> ls -la /workspace/Megatron-LM

# Check CUDA/ROCm setup
docker exec <container_name> python3 -c "import torch; print(torch.cuda.is_available())"
```

## Testing and Debugging

### Docker Container Commands

To run commands in the ROCm container, first find the container name:

```bash
# List running containers
docker ps

# Example output shows container name like: rocm7_container_1758826316
```

Then execute commands using the full container name:

```bash
# Run training scripts
docker exec rocm7_container_1758826316 bash /workspace/project/04-megablocks-moe-gpt2-125m.sh

# Run test scripts
docker exec rocm7_container_1758826316 python /workspace/project/test-epoch-calculation.py

# Interactive debugging
docker exec -it rocm7_container_1758826316 bash
```

**Note:** Do not use `-it` flags when running from automated scripts or CI, as they require a TTY.

### Dataset Debugging

The training configuration depends heavily on dataset size. Use the test script to verify:

```bash
# Check dataset statistics and epoch calculations
docker exec rocm7_container_1758826316 python /workspace/project/test-epoch-calculation.py
```

This script shows:
- Number of samples in your dataset
- Training steps calculated for different scenarios
- How batch size affects training iterations

### Common Issues

#### Issue: Training Steps = 0

**Symptoms:**
- Script shows "Training steps: 0"
- Training exits immediately after validation
- No actual training iterations occur

**Cause:** Dataset too small for the global batch size (512)

**Solution:**
- Increase dataset size, or
- Reduce global batch size in training scripts

**Example:**
```bash
# If dataset has only 1 sample with batch size 512:
# Training steps = (1 * 3 epochs) / 512 = 0 (rounded down)
# No training occurs because no complete batches can be formed
```

#### Issue: Container Path Errors

**Symptoms:**
- "No such file or directory" errors
- Scripts not found

**Solution:** Files are mounted at `/workspace/project/`, not `/workspace/train/mi300x/`

```bash
# Correct paths in container:
docker exec <container> ls /workspace/project/
docker exec <container> python /workspace/project/test-epoch-calculation.py
docker exec <container> bash /workspace/project/04-megablocks-moe-gpt2-125m.sh
```

### Testing Workflow

Before running full training:

```bash
# 1. Check container is running
docker ps

# 2. Test dataset and epoch calculation
docker exec <container_name> python /workspace/project/test-epoch-calculation.py

# 3. Verify training configuration shows reasonable number of steps
# Look for output like: "Training steps: XXX" where XXX > 0

# 4. Run training with monitoring
docker exec <container_name> bash /workspace/project/04-megablocks-moe-gpt2-125m.sh &

# 5. Monitor progress (from another terminal)
docker exec <container_name> tail -f /workspace/project/checkpoints/moe_experiment/train.log
```

## Example Session

```bash
# 1. Start container
./01-run-docker.sh

# 2. Find container name
docker ps

# 3. Test dataset configuration
docker exec rocm7_container_1758826316 python /workspace/project/test-epoch-calculation.py

# 4. If tests show adequate samples, generate data (if not already done)
docker exec rocm7_container_1758826316 python /workspace/project/02-generate.sft.shisa-v2.1-megablocks.py

# 5. Start training
docker exec rocm7_container_1758826316 bash /workspace/project/03-megablocks-gpt2-125m.sh

# 6. Monitor from host (new terminal)
docker exec rocm7_container_1758826316 tail -f /workspace/project/checkpoints/train.log
```

This workflow ensures reproducible training with proper resource management and monitoring capabilities.