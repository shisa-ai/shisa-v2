https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html

https://www.amd.com/en/developer/resources/infinity-hub.html
docker pull rocm/7.0:rocm7.0_pytorch_training_instinct_20250915

## Megatron Checkpoint Fix Patch

Training runs on ROCm PyTorch 2.5 hit a checkpoint failure inside Megatron-LM’s async saver (`_write_item() missing 1 required positional argument: 'serialization_format'`). This repository ships both a runtime monkeypatch (`megatron_checkpoint_patch.py`, automatically loaded via `run_pretrain_with_patch.py`) and a static patch file. Use whichever workflow fits your environment.

### Files touched when the patch is applied
- `megatron/core/dist_checkpointing/strategies/filesystem_async.py`
- `tests/unit_tests/dist_checkpointing/test_async_save.py`

### Runtime monkeypatch
- `megatron_checkpoint_patch.py` (imported by `run_pretrain_with_patch.py`)
- Nothing to do—just launch training via `./03-...` or `./04-...` and the fix is active.

### Optional static patch
- `patch-megatron-checkpoint-save.patch` (root of this repo) remains available if you prefer modifying Megatron sources directly (`git apply` inside `/workspace/Megatron-LM`).
