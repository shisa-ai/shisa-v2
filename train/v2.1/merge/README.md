# Merge Recipes

This directory tracks the configs used to explore different ways of combining the
`shisa-ai` 14B checkpoints. The goal is to keep a reproducible record of how
each blend was produced in mergekit.

## Environment Setup

```bash
mamba create -n mergekit python=3.12
conda activate mergekit
pip install mergekit

# AMD ROCm builds (optional if you have ROCm GPUs)
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
```

## Config Inventory

| Config file | Method | Intent |
|-------------|--------|--------|
| `merge-linear-152-155-149.yaml` | `linear` | Baseline weighted average of both DPO runs + original SFT |
| `merge-nuslerp-152-155-149.yaml` | `nuslerp` | NuSLERP on task vectors from the SFT base to smoothly interpolate between both DPOs |
| `merge-model_stock-152-155-149.yaml` | `model_stock` | Model Stock weight search using the SFT checkpoint as the anchor |

### Linear (merge-linear-152-155-149.yaml)
- `merge_method: linear`, `dtype: bfloat16`.
- Models and weights:
  - `155-unphi4-14b-dpo2.1-2e7`: `0.35`
  - `152-unphi4-14b-dpo2.1-1.5e7`: `0.35`
  - `149-unphi4-14b-sft2.1`: `0.30`
- Useful as a simple baseline before moving to spherical/task-vector methods.

### NuSLERP (merge-nuslerp-152-155-149.yaml)
- `merge_method: nuslerp`, `dtype: bfloat16`.
- `base_model`: `149-unphi4-14b-sft2.1` (task vectors computed relative to this SFT).
- Input models:
  - `155-unphi4-14b-dpo2.1-2e7` weight `0.5`
  - `152-unphi4-14b-dpo2.1-1.5e7` weight `0.5`
- Global params:
  - `nuslerp_flatten: true` (default behavior, SLERP over flattened tensors).
  - `nuslerp_row_wise: false` (column-wise interpolation).
- Adjusting the per-model `weight` values changes the interpolation factor `t`
  (`t = w_model2 / (w_model1 + w_model2)`). Setting `nuslerp_flatten: false`
  lets you experiment with row/column SLERP if you need finer-grained control.

### Model Stock (merge-model_stock-152-155-149.yaml)
- `merge_method: model_stock`, `dtype: bfloat16`.
- `base_model`: `149-unphi4-14b-sft2.1`.
- Other models: `155-unphi4-14b-dpo2.1-2e7`, `152-unphi4-14b-dpo2.1-1.5e7`.
- Parameters:
  - `filter_wise: false` (recommended default; set to `true` only if you want
    per-row weighting which is slower and noisier).
- Model Stock computes cosine similarities of task vectors against the base to
  pick a data-driven interpolation factor; no manual weights needed.

## Running the Merges

Use `mergekit-yaml` (or the `mergekit` CLI entrypoint) with the desired config:

```bash
mergekit-yaml \
  --config merge-nuslerp-152-155-149.yaml \
  --out-model 152-155-149-nuslerp \
  --copy-tokenizer shisa-ai/149-unphi4-14b-sft2.1
```

Swap the `--config` path to run the linear or Model Stock recipes. The output
directory name is arbitrary; pick something that reflects the experiment. Add
`--device "cuda:0"` / `--device "rocm:0"` etc. if you need to target a specific
GPU. The configs are bfloat16-safe, but you can override `dtype` via CLI if you
need a different precision (e.g. `float16` on smaller VRAM cards).
