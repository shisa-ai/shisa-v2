https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html

https://www.amd.com/en/developer/resources/infinity-hub.html
docker pull rocm/7.0:rocm7.0_pytorch_training_instinct_20250915

## Megatron Checkpoint Fix Patch

Training runs on ROCm PyTorch 2.5 hit a checkpoint failure inside Megatron-LM’s async saver (`_write_item() missing 1 required positional argument: 'serialization_format'`). This repository ships a small patch that forwards the new argument everywhere Megatron wraps `_write_item`.

### Files touched when the patch is applied
- `megatron/core/dist_checkpointing/strategies/filesystem_async.py`
- `tests/unit_tests/dist_checkpointing/test_async_save.py`

### Patch location
- `patch-megatron-checkpoint-save.patch` (root of this repo)

### Applying the patch
```bash
cd /workspace/Megatron-LM   # Path inside the training container
git apply /workspace/project/patch-megatron-checkpoint-save.patch
```

Rebuild or rerun training afterwards—the async checkpoint writer will now pass `serialization_format`, and checkpoint saves complete without the mid-run crash.
