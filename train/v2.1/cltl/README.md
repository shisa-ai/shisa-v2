# Unlikelihood Training for Multilingual Token Leakage Correction

Targeted fine-tuning approach to fix language token leakage in multilingual LLMs without full preference optimization overhead.

## Problem

Multilingual models (e.g., Qwen3-based) occasionally "leak" tokens from the wrong language mid-generation:

```
秀吉との間で勢力争いを繰り広げ、 ultimately 秀吉に敗北。
                                 ↑ English token in Japanese text
```

These leaks are typically single tokens where the model's logits favor a wrong-language token over the correct one.

## Why Not DPO?

We initially considered DPO with chosen/rejected pairs differing by one token. However:

| Approach | Drawback |
|----------|----------|
| DPO | Operates on full sequence log-probs; gradient flows through shared prefix; noisy signal for single-token fixes |
| RLHF/PPO | Heavyweight; overkill for targeted corrections |
| Full SFT | Risk of forgetting; inefficient for sparse errors |

**Unlikelihood training** directly penalizes the probability of specific bad tokens at specific positions — surgically precise for this use case.

## Approach

For each leaked token, we:

1. **Penalize the bad token**: Push down P(bad_token | context) via unlikelihood loss
2. **Reinforce the good token**: Push up P(good_token | context) via cross-entropy

Combined loss:

```
L = α_ul * L_ul + α_rf * L_rf

where:
  L_ul = -log(1 - P(bad_token | context))   # Unlikelihood loss
  L_rf = -log(P(good_token | context))      # Reinforcement loss
```

This directly "hammers down" the leaked token's logit while boosting the correct token.

## Dataset Format

```jsonl
{
  "chosen": "秀吉との間で勢力争いを繰り広げ、 結局 秀吉に敗北。",
  "rejected": "秀吉との間で勢力争いを繰り広げ、 ultimately 秀吉に敗北。",
  "leak_tokens": ["ultimately"],
  "corrected_token": ["結局"]
}
```

| Field | Description |
|-------|-------------|
| `chosen` | Correct text (used as training context) |
| `rejected` | Text with leak (optional, used as fallback for position detection) |
| `leak_tokens` | List of bad tokens to suppress (supports variants) |
| `corrected_token` | The correct token that should appear |

The script uses `leak_tokens` and `corrected_token` directly — no tokenization diffing required.

## Usage

### Installation

```bash
pip install torch transformers datasets accelerate tqdm
```

### Configure Accelerate (8x GPU)

```bash
accelerate config
# Select: multi-GPU, 8 GPUs, bf16
```

### Run Training

```bash
accelerate launch --num_processes 8 unlikelihood_train.py \
    --model_name_or_path your-org/qwen3-8b-model \
    --dataset_name ./leak_corrections.jsonl \
    --output_dir ./qwen3-8b-fixed \
    --epochs 2 \
    --lr 5e-6 \
    --alpha_ul 1.0 \
    --alpha_reinforce 0.1 \
    --verbose_diagnostics
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alpha_ul` | 1.0 | Unlikelihood loss weight (higher = more aggressive suppression) |
| `--alpha_reinforce` | 0.1 | Reinforcement loss weight (higher = stronger boost to correct token) |
| `--lr` | 5e-6 | Learning rate (keep low to avoid drift) |
| `--epochs` | 2 | Training epochs |
| `--verbose_diagnostics` | off | Print per-example token probabilities before/after |

### Tuning Guidelines

- **Leaks persist**: Increase `alpha_ul` (try 1.5–2.0)
- **Model drifts on other tasks**: Decrease `lr`, reduce `alpha_ul`, increase `alpha_reinforce`
- **Good token not rising**: Increase `alpha_reinforce` (try 0.2–0.3)

## Output

The script logs probability changes:

```
Pre-training  - Avg bad token prob: 12.34%  | Avg good token prob: 3.21%
Post-training - Avg bad token prob: 0.89%   | Avg good token prob: 45.67%
Change        - Bad: -11.45%                | Good: +42.46%
```

With `--verbose_diagnostics`, you'll see per-position top-5 predictions to verify the fix.

## Technical Details

- **Position indexing**: Logits at position `t` predict token at `t+1`. We use `position - 1` to get the relevant logits.
- **Position 0 skipped**: No logit predicts the BOS token; divergences at position 0 are excluded.
- **Numerical stability**: Uses `log1p(-p)` instead of `log(1-p)`.
- **Multiple bad tokens**: Supports penalizing multiple leak variants per example (e.g., `["ultimately", "Eventually"]`).

## References

- [Unlikelihood Training for Consistent Dialogue](https://arxiv.org/abs/1908.04319) — Welleck et al., 2019
- [Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — Holtzman et al., 2019

## License

MIT


See our repo for generation, etc: https://github.com/shisa-ai/cross-lingual-token-leakage
