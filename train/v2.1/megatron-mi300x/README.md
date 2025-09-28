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

## Hugging Face MoE exports

The MoE conversion flow drops a custom modeling shim alongside the exported checkpoint. When loading the resulting directory with Transformers, opt in to custom code so the experts are wired up correctly:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('/path/to/moe/export', trust_remote_code=True)
```

Without `trust_remote_code=True` the loader falls back to the dense GPT-2 architecture and reports uninitialised parameter warnings.


## Qwen3-0.6B SFT

- Run `qwen3-0.6b/02-generate.sh` to mirror the latest datasets into MegaBlocks format. The script copies tokenizer assets from the cached `Qwen/Qwen3-0.6B` snapshot and calls the shared generator with the model's chat template when available.
- Launch fine-tuning with `qwen3-0.6b/03-train-dense.sh`. The wrapper feeds the new `03-megablocks-qwen3-0.6b.sh` launcher which applies the Qwen rotary/RMSNorm/SwiGLU defaults and accepts an optional `INIT_CHECKPOINT` for warm starts.


