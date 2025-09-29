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
- Launch fine-tuning with `qwen3-0.6b/03-train-dense.sh`. The wrapper feeds the updated `03-megablocks-qwen3-0.6b.sh`, which now mirrors the Llama workflow: it stages the upstream HF weights under `qwen3-0.6b/hf_snapshot/`, converts them with the custom `loader_qwen3_hf` plugin (writing `base_tp8_pp1/iter_0000001` via the mcore saver), and only then starts Megatron. Pass `BASE_CKPT_DIR`, `HF_MODEL_DIR`, or `INIT_CHECKPOINT` to override any part of the warm-start path.
- Training is iteration-based and uses Megatron’s Qwen-specific knobs out of the box (`--group-query-attention`, `--kv-channels 128`, `--qk-layernorm`). `KV_CHANNELS` and `NUM_QUERY_GROUPS` remain configurable, but default to values that keep the fused QKV projection compatible with the 0.6B checkpoint and avoid the earlier Q/K/V shape mismatch.

## Llama3.2-1B SFT

- Run `llama3.2-1b/02-generate.sh` to materialise the shared SFT dataset with the model’s native tokenizer and chat template.
- Kick off Megatron fine-tuning via `llama3.2-1b/03-train-dense.sh`; this wraps `03-megablocks-llama3.2-1b.sh` and uses GBS=128, LR=2.83e-05, rope scaling, and RMSNorm defaults from the HF config.
- Hugging Face conversions now target the mcore saver and snapshot the upstream weights locally before the first run. If `/workspace/shisa-v2.1/llama3.2-1b/base_tp8_pp1` is missing, the launcher downloads `meta-llama/Llama-3.2-1B-Instruct`, writes it to `hf_snapshot/`, and invokes `tools/checkpoint/convert.py --saver mcore` so the checkpoint layout matches Megatron-Core.
- The script auto-detects when tensor-parallel > 1 (or `CONVERT_SEQUENCE_PARALLEL=1`) and forwards `--sequence-parallel` to the converter so the rebuilt checkpoint passes Megatron’s MoE validation.
- Training is now iteration-based: we pass `--train-iters`/`--lr-decay-iters` instead of the deprecated sample-based flags and compute `--save-interval` in steps. This matches the expectations of recent Megatron releases when resuming from checkpoints.
- Llama 3.2 1B’s grouped-query attention uses 32 attention heads with 8 KV groups. We expose `NUM_QUERY_GROUPS` (default `8`) so the generated Megatron model lines up with the converted weights and avoids QKV shape mismatches when loading `base_tp8_pp1`.

## Megatron Checkpoint Conversion (ROCm build)

The ROCm fork ships converter plugins for these HF checkpoint families:
- Llama 2 (7B/13B/70B) & Llama 3 / 3.1 (8B/70B, base & instruct)
- Mistral 7B (base & instruct)
- Yi 34B
- Qwen2.5 7B / 72B (base & instruct)
- Mixtral 8×7B (via `loader_mixtral_hf.py`)

Missing plugins (`loader_hf`, etc.) mean other architectures require manual conversion or an upstream plugin drop-in before our scripts can auto-convert HF weights.
