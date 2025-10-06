# Axolotl ZeRO-2 MoE Notes (095/096)

## GPU Memory Budget Cheat Sheet

Qwen3-30B-A3B (30.53B params) on 8×MI300X — per-rank memory usage estimates:

| Component | Formula (per GPU) | ZeRO-2 | ZeRO-3 |
|-----------|-------------------|--------|--------|
| **Model weights** | 30.53B params × 2 bytes (bf16) | ≈60 GB | ≈7.5 GB |
| **Gradients** | same as weights (bf16) | ≈60 GB | ≈7.5 GB |
| **Optimizer state** | depends on optimizer | see below | see below |

Optimizer state footprint (per GPU):

| Optimizer | Precision | ZeRO-2 | ZeRO-3 |
|-----------|-----------|--------|--------|
| AdamW (fp32 moments) | 32 bit | ≈30 GB | ≈3.8 GB |
| AdamW (8-bit) | 8 bit + scales | ≈10–12 GB | ≈1.5 GB |
| AdamW (4-bit) | 4 bit + scales | ≈7–10 GB | ≈1 GB |

> With optimizer offload enabled (`device: cpu`), the state is moved off-GPU at the cost of throughput. With offload disabled (`device: none`), add the optimizer total to your on-GPU budget.

Activation + KV cache overhead scales roughly linearly with sequence length × micro batch. For DPO we run two forward passes (policy + ref) per batch:

| Sequence len | Micro batch | Approx. activation overhead |
|--------------|-------------|------------------------------|
| 2048 | 4 | ~18–20 GB |
| 2048 | 8 | ~36–40 GB |
| 3072 | 4 | ~28–30 GB |
| 4096 | 2 | ~24–26 GB |

Combine the baseline (weights + grads + optimizer) with the activation rows to estimate total VRAM. Example — ZeRO‑2, AdamW fp32, `seq=2048`, `mbs=4`:

- Base: (60 + 60 + 30) ≈ 150 GB
- Activations: ~20 GB
- Total ≈ 170 GB per GPU

Switching to 4-bit optimizer drops the base to ~127 GB, freeing room to raise micro batch or disable checkpointing.

This doc tracks the 30B Qwen3-A3B ZeRO-2 MoE runs we launched from `train/v2.1` and how to package the resulting checkpoints for downstream sharing.

## Training runs

### 095 — SFT (Axolotl + ZeRO-2 MoE)
- **Config**: `095-qwen3-30b-a3b-v2.1-sft.axolotl-dz2moe.8xMI300.yaml`
- **Base model**: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- **Data**: `sft.shisa-v2.1.jsonl` (chat template, assistant-role only)
- **Hardware / strategy**: 8× MI300X, DeepSpeed `zero2_moe.json` (128 experts, top-8, Tutel), Liger kernels on.
- **Key hyperparams**: seq 8k, packed, mbs 16, gas 1, LR 1.63e-5 (linear decay), 3 epochs, warmup 3%.
- **Outputs**: `/data/outputs/094-qwen3-30b-a3b-v2.1-sft/` (latest plus `checkpoint-1188/` after 3 epochs).

### 096 — DPO (Axolotl + ZeRO-2 MoE)
- **Config**: `096-qwen3-30b-a3b-v2.1-dpo.axolotl-dz2moe.8xMI300.yaml`
- **Base model**: `/data/outputs/094-qwen3-30b-a3b-v2.1-sft/checkpoint-1188`
- **Data**: `dpo.shisa-v2.1.jsonl` (multi-source preference merge).
- **Key hyperparams**: seq 4k / prompt cap 3584, mbs 8, gas 1, LR 2.04e-7 (constant with warmup), β=0.1, 1 epoch, Liger on, ZeRO-2 MoE state partitioning.
- **Outputs**: `/data/outputs/096-qwen3-30b-a3b-v2.1-dpo/` (ZeRO-2 shards + logs).

## Checkpoint layout vs top-level `model.safetensors`

DeepSpeed ZeRO-2 MoE writes:

```
/data/outputs/094-qwen3-30b-a3b-v2.1-sft/
├── checkpoint-1188/
│   ├── model-00001-of-00013.safetensors
│   ├── ...
│   └── zero_to_fp32.py
└── model.safetensors            # tiny (~1.8 MB)
```

That top-level `model.safetensors` is **just a stub**: each tensor inside is zero-length (`torch.Size([0])`). Loading the directory root with `AutoModelForCausalLM.from_pretrained('.')` therefore throws size-mismatch errors (the loader consumes the stub first). Always either:

1. Point consumers at a real step directory (e.g. `checkpoint-1188/`), **or**
2. Regenerate consolidated weights and replace/remove the stub (see below).

The same pattern occurs for the DPO output: use a concrete checkpoint directory unless you have already consolidated.

## Producing distribution-ready Hugging Face weights

1. **Choose the checkpoint** you want to release (e.g. `/data/outputs/096-qwen3-30b-a3b-v2.1-dpo/checkpoint-xxxx`).
2. **Run `zero_to_fp32.py` with safe serialization** to recover a full model:
   ```bash
   cd /data/outputs/096-qwen3-30b-a3b-v2.1-dpo/checkpoint-xxxx
   python zero_to_fp32.py . ../full-fp32 --safe_serialization
   ```
   - This creates `../full-fp32/pytorch_model.safetensors` (or `.bin` if you omit `--safe_serialization`).
   - Optional: rename to `model.safetensors` if you prefer the standard filename.
3. **Copy configs/tokenizer assets** into the new folder (from the checkpoint dir or the SFT root): `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `merges.txt`, `vocab.json`, `added_tokens.json`, and any custom `chat_template.jinja`.
4. **Validate locally**:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   tok = AutoTokenizer.from_pretrained("path/to/full-fp32")
   model = AutoModelForCausalLM.from_pretrained("path/to/full-fp32", torch_dtype="auto")
   ```
   Ensure no size-mismatch warnings and optionally run a short generation sanity check.
5. **(Optional) Trim ZeRO metadata**: remove the original shard files if you just need the consolidated copy.
6. **Upload/publish**: the folder is now in the standard HF layout; run `huggingface-cli upload` or `git lfs` push to your model repo.

> Tip: if you need a single-slice `model.safetensors`, pass an explicit filename with `--output`, or use the Hugging Face `safetensors` CLI to combine shards (`safetensors combine ...`). The critical piece is avoiding the zero-length stub from the ZeRO save.

## Quick reference

- Training configs live in this repo: `095-...yaml` (SFT) and `096-...yaml` (DPO).
- Use the step checkpoint directories for any downstream Axolotl runs.
- Always reconstruct consolidated weights before shipping models off the cluster.
