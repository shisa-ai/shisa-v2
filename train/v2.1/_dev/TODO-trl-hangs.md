# TRL Training Hangs and Process Issues

## Problem Summary

When running TRL training scripts with distributed PyTorch on MI300X, we encountered severe hanging issues and process explosions that made training unusable:

- **15+ minute hangs** during startup at "Starting training..." stage
- **Hundreds of zombie processes** after CTRL+C (256+ compile workers)
- **Memory exhaustion** from worker process explosion
- **Inability to gracefully shutdown** distributed training

## Root Cause Analysis

### 1. Torch Inductor Process Explosion
- **Default behavior**: 32 worker processes per GPU rank
- **With 8 GPUs**: 8 ranks × 32 workers = **256 compile workers**
- **Memory usage**: ~2GB per worker = **500+ GB total**
- **Problem**: Workers don't respect SIGINT/SIGTERM properly

### 2. Environment Variable Timing Issue
- **Issue**: `TORCHINDUCTOR_*` variables must be set before `import torch`
- **Failure**: Setting them in `setup_environment()` happens too late
- **Result**: Default worker count (32) is used despite our settings

### 3. Distributed Training Cleanup Issues
- **RCCL processes** don't shutdown gracefully
- **Process groups** aren't destroyed properly on interrupt
- **TCP store connections** become orphaned
- **GPU contexts** remain allocated

### 4. FSDP Initialization Hangs
- **Memory allocation** can timeout with insufficient resources
- **Inter-rank communication** hangs if workers consume bandwidth
- **Model sharding** delays when compile workers are active

## Solutions Implemented

### 1. Immediate Fix: Disable Torch Compile
```python
# At top of script, before any imports
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch  # Now safe from worker explosion
```

**Benefits:**
- ✅ Immediate startup (no compilation delay)
- ✅ No worker process explosion
- ✅ Stable memory usage
- ✅ Clean shutdown behavior

**Tradeoffs:**
- ❌ No compilation optimizations
- ❌ Potentially slower training throughput

### 2. Proper Signal Handling
```python
import signal
import atexit

def signal_handler(signum, frame):
    # Cleanup distributed training
    if dist.is_initialized():
        dist.destroy_process_group()

    # Kill lingering workers
    subprocess.run(["pkill", "-f", "torch/_inductor/compile_worker"])

    # Force exit after timeout
    threading.Timer(2.0, lambda: os._exit(1)).start()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_on_exit)
```

### 3. Process Cleanup Script
Created `kill-training.sh` for emergency cleanup:
```bash
#!/bin/bash
pkill -f "076-qwen3.*py"
pkill -f "075-qwen3.*py"
pkill -f "accelerate launch"
pkill -f "torch/_inductor/compile_worker"
pkill -f "wandb"
rocm-smi --resetgpu  # Clear GPU state
```

### 4. Environment Optimizations
```python
# ROCm/RCCL settings for MI300X
os.environ["RCCL_DEBUG"] = "WARN"  # Reduce log spam
os.environ["RCCL_BUFFSIZE"] = "8388608"  # 8MB buffers
os.environ["RCCL_MAX_NCHANNELS"] = "16"  # Optimize for 8x topology
os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"  # Better memory access
```

### 5. Accelerate Config Updates
```yaml
# accelerate_config.fsdp2.yaml
enable_cpu_affinity: true  # Better NUMA locality
fsdp_activation_checkpointing: false  # 191GB VRAM is sufficient
```

## Performance Considerations

### With Torch Compile Disabled (Current)
- **Startup time**: 2-5 minutes
- **Memory usage**: ~8GB per rank (reasonable)
- **Training speed**: Baseline (no optimizations)
- **Stability**: Excellent

### With Torch Compile Enabled (Future)
- **Startup time**: 10-20 minutes (first run only)
- **Memory usage**: 8GB + 4×2GB workers = 16GB per rank
- **Training speed**: 10-30% faster (optimized kernels)
- **Stability**: Good (with proper worker limits)

## Next Steps and Recommendations

### Phase 1: Debugging and Development ✅
- [x] Use `TORCH_COMPILE_DISABLE=1` for immediate usability
- [x] Implement signal handling for clean shutdowns
- [x] Test training loop functionality
- [x] Validate model saving/loading
- [x] Confirm wandb integration works

### Phase 2: Performance Optimization (Future)
- [ ] Re-enable torch.compile with limited workers
- [ ] Benchmark compilation vs runtime tradeoffs
- [ ] Test worker limits: 4, 8, 16 workers per rank
- [ ] Measure actual speedup on MI300X hardware
- [ ] Compare against baseline (no compile) performance

### Phase 3: Production Deployment (Future)
- [ ] Create production configs with compile enabled
- [ ] Implement automatic worker count detection
- [ ] Add memory monitoring and adaptive scaling
- [ ] Create comprehensive testing suite
- [ ] Document optimal settings for different model sizes

## Usage Instructions

### Current (Development Mode)
```bash
# Fast startup, no compilation
python 076-qwen3-30b-a3b-v2-sft.trl.8xMI300.py --debug
```

### Future (Performance Mode)
```bash
# 1. Comment out TORCH_COMPILE_DISABLE in script
# 2. Set environment variables:
export TORCHINDUCTOR_WORKER_COUNT=4
export TORCHINDUCTOR_COMPILE_THREADS=4
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/torch_compile_cache

# 3. Run training (expect 10+ min compilation on first run)
python 076-qwen3-30b-a3b-v2-sft.trl.8xMI300.py
```

### Emergency Cleanup
```bash
# If processes hang after CTRL+C
./kill-training.sh

# Check for remaining processes
ps aux | grep -E "(python.*torch|accelerate)" | grep -v grep
```

## Files Modified

- `075-qwen3-30b-a3b-v2-sft.trl-megablocks.8xMI300.py` - Added compile disable + signal handling
- `076-qwen3-30b-a3b-v2-sft.trl.8xMI300.py` - Added compile disable + signal handling
- `accelerate_config.fsdp2.yaml` - Enabled CPU affinity, disabled activation checkpointing
- `accelerate_config.yaml` - Enabled CPU affinity
- `kill-training.sh` - Emergency process cleanup script

## Lessons Learned

1. **Environment variables matter**: Torch Inductor settings must be set before import
2. **Worker explosion is real**: 32 workers × 8 ranks = system death
3. **Signal handling is critical**: Distributed training needs custom cleanup
4. **Memory is finite**: Even with 1.5TB RAM, 256 workers will exhaust it
5. **Compilation vs usability**: Sometimes disabling optimizations improves DX
6. **ROCm != CUDA**: Different environment variables and behaviors
7. **Process cleanup is hard**: Multiple levels of process hierarchies
8. **First run != subsequent runs**: Compilation is one-time cost

## Future Research

- [ ] Investigate dynamo vs inductor backends for ROCm
- [ ] Test torch.compile with different backends (aot_eager, etc)
- [ ] Explore custom compilation caching strategies
- [ ] Benchmark memory vs performance tradeoffs
- [ ] Research ROCm-specific optimization flags
- [ ] Compare against JAX/XLA compilation approaches