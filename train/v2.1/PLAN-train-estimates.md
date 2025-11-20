# GLM4.5 Air Training Estimate (Axolotl + ZeRO-3)

## Inputs & References
- **Checkpoint inspected:** `/root/.cache/huggingface/hub/models--zai-org--GLM-4.5-Air/snapshots/a24ceef6ce4f3536971efe9b778bdaa1bab18daa/config.json`
  - `hidden_size: 4096`
  - `num_hidden_layers: 46`
  - `n_routed_experts: 128`
  - `num_experts_per_tok: 8`
  - `moe_intermediate_size: 1408`
  - `n_shared_experts: 1`
  - Reported spec: **106 B total / 12 B active**
- **Existing Axolotl configs for reference:**
  - `094-qwen3-30b-a3b-v2.1-sft.axolotl.8xMI300.yaml`
  - `110-ring-mini-2.0-v2.1-sft-megablocks.axolotl.8xMI300.yaml`
- **Observed throughput logs:**
  - Ring mini (16 B total / ~1.4 B active) @ 8× MI300X: ~2 600 tok/s/GPU (10h47m wall)
  - Qwen3-30B-A3B (active ≈3 B) @ 8× MI300X: ~700 tok/s/GPU (≈38 h wall; final epoch dipped)

## Throughput Projection (8× MI300X, AdamW-8bit, ZeRO-3, bf16)
| Model | Active Params / tok | Throughput (tok/s/GPU) | Wall clock for ≈7.6×10^8 tokens |
|-------|---------------------|------------------------|----------------------------------|
| Ring mini reference | ~1.4 B | 2 600 | 10h47m (measured) |
| Qwen3 30B reference | ~3 B | 700 | ~38 h (measured) |
| **GLM4.5 Air (estimate)** | **12 B** | **175–300** (scaled from Qwen/Ring) | **107–152 h (4.5–6.3 days)** |

### Notes
1. Scaling assumes throughput ∝ 1 / active_params. (12 B active is 4× Qwen’s 3 B → 700 / 4 ≈ 175 tok/s/GPU. Using the Ring baseline gives a slightly higher bound: 2600 / (12 B / 1.4 B) ≈ 300 tok/s/GPU.)
2. Aggregate cluster throughput @ 8 GPUs:
   - Lower bound: 175 × 8 ≈ 1.4 k tok/s
   - Upper bound: 300 × 8 ≈ 2.4 k tok/s
3. Qwen SFT consumed ≈7.6×10^8 tokens total (derived from 700 tok/s/GPU × 8 GPUs × 38 h). Reusing that curriculum at the projected aggregate rates yields:
   - 7.6e8 / 1.4e3 ≈ 5.4e5 s ≈ **152 h**
   - 7.6e8 / 2.4e3 ≈ 3.2e5 s ≈ **107 h**

## Memory Budget Per MI300X (192 GB)
- **Parameters:** 106 B × 2 bytes (bf16) ≈ 212 GB global → **~26.5 GB per GPU (ZeRO-3 shard)**.
- **Gradients:** mirrored shard, another ~26.5 GB.
- **Optimizer (AdamW 8-bit):**
  - Master weights (fp16/bf16): 2 bytes × 106 B / 8 ≈ 26.5 GB.
  - Moments (8-bit): 1 byte × 2 × 106 B / 8 ≈ 26.5 GB.
  - Combined optimizer footprint per GPU ≈ **53 GB** (conservative; fused implementations can be slightly lower).
- **Activations (sequence_len 8192, micro_batch_size 32):**
  - With checkpointing and `pad_to_sequence_len: true`, expect ~10–12 GB per GPU (attention + MoE + routing metadata).
- **Total steady-state estimate:** 26.5 (params) + 26.5 (grads) + ~53 (optimizer) + 12 (activations) ≈ **118 GB / GPU**.
  - Matches empirical note from Qwen config (“mbs=16 uses ~100 GB”). The MI300’s 192 GB leaves ~70 GB headroom for ZeRO buffers, Megablocks workspaces, NCCL staging, etc.
- **Knobs to keep inside budget:**
  1. Maintain `gradient_checkpointing: true` (non-reentrant).
  2. Keep `gradient_accumulation_steps: 1`; increasing it multiplies activations per GPU.
  3. Consider `micro_batch_size: 16` or disabling `pad_to_sequence_len` if utilization spikes >120 GB/GPU.

## Risk & Mitigation
- **Communication overhead:** ZeRO‑3 + MoE all-to-all becomes dominant below ~250 tok/s/GPU. Use Megablocks grouped kernels (`mlp_impl: megablocks`) plus NCCL overlap to avoid falling under the 175 tok/s floor.
- **Optimizer spill:** If actual moments exceed the 8-bit budget, set `adamw_torch` (bf16) and rely on ZeRO‑3 partitioning; however, that raises per-GPU optimizer memory to ~80 GB. Monitor `torch.cuda.memory_allocated()` early in the run.
- **Final epoch slowdown:** Qwen run showed throughput drop near convergence. Be prepared for an extra 5–10 h beyond the 107–152 h window if dataloader sharding leaves small batches.

## Summary
- **Expected throughput:** 175–300 tok/s/GPU (1.4–2.4 k tok/s aggregate).
- **Wall-clock for 3‑epoch SFT (~7.6×10^8 tok):** ~4.5–6.3 days on 8×MI300X.
- **Memory per GPU:** ≈118 GB (bf16 weights, AdamW‑8bit, ZeRO‑3, gradient checkpointing, `micro_batch_size<=32`).
- Config template: start from `094-qwen3-30b-a3b-v2.1-sft.axolotl.8xMI300.yaml`, swap `base_model` → `zai-org/GLM-4.5-Air`, `mlp_impl: megablocks`, and adjust `micro_batch_size` / learning rate as needed.

---

# Llama‑4 Scout 17B × 16 Experts (109 B) Training Estimate

## Inputs & References
- **Checkpoint inspected:** `/root/.cache/huggingface/hub/models--meta-llama--Llama-4-Scout-17B-16E/snapshots/*/config.json`
  - Text tower: `hidden_size: 5120`, `num_hidden_layers: 48`, `num_local_experts: 16`, `num_experts_per_tok: 1` (per layer MoE, 17 B active, 109 B total per Meta docs).
  - bf16 weights, RoPE scaling factor 16, `attention_chunk_size: 8192`.
- **Axolotl notes (`README.md`):**
  - Flex Attention recommended—Flash Attention path upstream is buggy.
  - Single H100 (4 k seq, QLoRA) → **519 tok/s @ 64.5 GB**.
  - 4× H100 (4 k seq) → **280 tok/s/GPU @ 62.8 GB** (Flex Attention + QLoRA + FSDP2).
- No in-house Axolotl YAML yet; expect to mirror the QLoRA template with `flex_attention: true` and FSDP2 if sticking to QLoRA. For full-finetune on MI300 we’ll reuse the GLM-style ZeRO‑3 config.

## Throughput Projection (8× MI300X, AdamW-8bit, ZeRO-3, Flex Attention)
| Scenario | Tokens/sec/GPU | Notes |
|----------|----------------|-------|
| Meta single H100 (QLoRA) | 519 | Seq len 4 k, 17 B active |
| Meta 4× H100 (QLoRA) | 280 | Seq len 4 k, 62.8 GB/GPU |
| **8× MI300X (estimate)** | **240–320** | MI300 throughput typically 85–115 % of H100 for Axolotl MoE; Flex Attention is the gating factor. |

### Wall-Clock Estimate (same 7.6×10^8 tokens as Shisa SFT)
- Aggregate throughput @ 8 GPUs:
  - Lower bound: 240 × 8 = 1.92 k tok/s
  - Upper bound: 320 × 8 = 2.56 k tok/s
- Runtime:
  - 7.6e8 / 1.92e3 ≈ 3.96e5 s ≈ **110 h (4.6 days)**
  - 7.6e8 / 2.56e3 ≈ 2.97e5 s ≈ **82 h (3.4 days)**
- Expect slightly longer if Flex Attention caps SM occupancy on ROCm; add ~10 % buffer → **~3.8–5.1 days**.

## Memory Budget
- Meta’s measurements (QLoRA):
  - 1× H100 full precision QLoRA: 64.5 GB @ 4 k seq.
  - 4× H100 FSDP2: 62.8 GB/GPU @ 4 k seq.
- Full finetune (ZeRO‑3, bf16 weights) on MI300X:
  - Params: 109 B × 2 bytes / 8 ≈ **27.25 GB per GPU**.
  - Grads: ~27 GB.
  - Optimizer (AdamW‑8bit): ≈54 GB (same breakdown as GLM).
  - Activations: scales with sequence. For 4 k seq we’d expect similar ~12 GB; for 8 k double to ~20 GB.
  - **Total:** 27 + 27 + 54 + 20 ≈ **128 GB / GPU** with 8 k context. 4 k context drops activations to ~12 GB, yielding ~120 GB.
  - MI300X (192 GB) still has ≥60 GB cushion for Flex Attention scratch buffers and ZeRO staging.
- **Key knobs:**
  1. Use Flex Attention (`flex_attention: true`, `flex_attention_backend: "cuda"` on H100 / `"hip"` once available). FA2 path is unreliable for Llama‑4.
  2. Keep `attention_chunk_size: 8192` aligned with config to avoid OOMs.
  3. Gradient checkpointing mandatory; consider `micro_batch_size: 16` if 32 pushes usage >140 GB.

## Risk & Mitigation
- **Flex Attention maturity on ROCm:** Need to verify the Flex kernels for MI300X (Axolotl README only cites CUDA). If Flex is missing, fall back to Flash‑Attention‑2 but expect lower throughput (<200 tok/s/GPU) and potential correctness issues.
- **Sequence length:** Meta benchmarks were 4 k. Training Shisa workloads at 8 k doubles activation memory and can stall tokens/sec; consider staged curriculum (4 k → 8 k) if VRAM spikes.
- **Expert routing:** `num_experts_per_tok = 1` (non-topk). Capacity overflow risk is lower than GLM’s 8‑expert top‑k, so ZeRO comm overhead is a bit lighter—but watch for router instabilities if Flex Attention changes kernel ordering.

## Summary
- **Expected throughput:** 240–320 tok/s/GPU on 8× MI300X (Flex Attention), giving **~3.4–4.6 days** for a ~7.6×10^8‑token SFT run; add 10 % headroom for Flex maturity.
- **Memory per GPU:** ~120 GB (4 k seq) to ~128 GB (8 k seq) using bf16 weights + AdamW‑8bit + ZeRO‑3 + checkpointing. Meta’s H100 QLoRA numbers (≈63 GB) provide a lower bound if we stay in adapter mode.
- **Config guidance:** Start from the QLoRA Flex Attention samples in Axolotl docs; for full finetunes, clone the GLM ZeRO‑3 YAML, set `base_model: meta-llama/Llama-4-Scout-17B-16E`, enable `flex_attention: true`, and keep `flash_attention: false`. Use `mlp_impl: grouped` (no MegaBlocks yet) and monitor NCCL overlap.

---

# DeepSpeed ZeRO‑2 MoE vs ZeRO‑3 (Performance & Memory)

## Qwen3‑30B empirical timings (8× MI300X, bf16)
| Run | Strategy | Duration | Notes |
|-----|----------|----------|-------|
| `095-qwen3-30b-a3b-v2.1-sft.axolotl-dz2moe` | ZeRO‑2 MoE (Tutel) | **~15 h** | seq 8k, mbs 16, AdamW‑8bit |
| `096-qwen3-30b-a3b-v2.1-dpo.axolotl-dz2moe` | ZeRO‑2 MoE | **~24 h** | seq 4k, mbs 8, β=0.1 |
| `094-qwen3-30b-a3b-v2.1-sft.axolotl` | ZeRO‑3 | **~38 h** | same dataset/epochs; throughput ≈700 tok/s/GPU |
| `165-qwen3-30b-a3b-v2.1-dpo-2e7.axolotl` | ZeRO‑3 | **≈46 h** (in flight) | same DPO pairs/β, ZeRO‑3 + Megablocks |

**Takeaway:** when optimizer state fits on each MI300X (or when the ZeRO‑2 MoE partitioning keeps expert weights off‑rank), ZeRO‑2 runs are ~2× faster than equivalent ZeRO‑3 runs. The difference is entirely communication:
- ZeRO‑2 keeps parameters/optimizer fully local; only gradients are reduced.
- ZeRO‑3 shards params + optimizer, so every layer requires an all‑gather + reduce‑scatter (in addition to MoE all‑to‑all).
- CPU offload for ZeRO‑3 optimizer eliminates the VRAM issue but erases the 2× speed advantage, so it’s only a fallback when the replicated state truly doesn’t fit.

## Per‑GPU memory budget (bf16 weights, AdamW‑8bit, optimizer on GPU)

| Model | Dense params (B) | MoE params (B) | Expert-parallel size (ZeRO‑2) | **ZeRO‑2** param+grad (GiB) | **ZeRO‑2** optim. (GiB) | **ZeRO‑2 base** (GiB) | **ZeRO‑3 base** (GiB) |
|-------|------------------|----------------|-------------------------------|----------------------------|-------------------------|-----------------------|-----------------------|
| Qwen3‑30B‑A3B | 1.54 | 28.99 | 2 (Tutel `expert_parallel_size`) | 59.7 | 44.8 | **104.5** | **24.9** |
| GLM4.5 Air | 4.13 | 101.87 | 8 (planned) | 62.8 | 47.1 | **109.9** | **86.4** |
| Llama‑4 Scout 17Bx16E | 17 (approx) | 92 (approx) | 8 (assumed) | 106.2 | 79.6 | **185.8** | **88.8** |

Notes:
- “Dense params” = attention, embeddings, shared MLPs that are still fully replicated.  
- “MoE params” = per-layer expert weights. With ZeRO‑2 MoE the experts are partitioned across dedicated expert-parallel groups, so each GPU only stores `MoE / expert_parallel_size`.  
- “Base” totals exclude activations/KV cache. Add ~10–12 GB for seq 4 k (checkpointed) or ~20 GB for seq 8 k.  
- Llama‑4 figures use Meta’s published split (17 B dense, 92 B experts). Even with ep_size=8 the ZeRO‑2 base is >185 GB, so ZeRO‑3 (or CPU offload) is still required.

### Practical implications
1. **Qwen3‑30B fits ZeRO‑2 only because the MoE variant partitions experts.** Dense weights + optimizer still amount to ~200 GB, but in practice the Tutel “zero2_moe” path replicates the shared dense blocks while distributing expert weights, so MI300X boards sit around 100–110 GB (as observed in 095/096). Keep micro batch ≤16 and activations checkpointed.
2. **GLM4.5 Air / Llama‑4 Scout cannot run pure ZeRO‑2:** even before activations they would require >390 GB per GPU. ZeRO‑3 (or ZeRO‑2 with CPU offload, which nullifies the speed win) is mandatory.
3. **Optimizer choice matters:** switching from AdamW‑8bit to fp32 moments adds ~3× more optimizer memory. Conversely, pushing to 4‑bit optimizers saves another ~20 % but costs perf/quality—only do so if ZeRO‑2 almost fits.
4. **CPU offload**: useful for staging checkpoints but not for high‑throughput training; PCIe/IF links become the bottleneck and measured speed drops back to ZeRO‑3 levels.

### Rule of thumb
- **If `param + grad + optimizer + activations` < 180 GB/GPU**, prefer ZeRO‑2 MoE for MoE models—it’s ~2× faster and still fault tolerant.
- **If it doesn’t fit, or if expert counts push the dense blocks above that budget**, stay on ZeRO‑3 (with Megablocks/A2A overlap) and budget double the wall time.

### ZeRO‑2 + MegaBlocks guidance
- Our MegaBlocks/HIP patches are orthogonal to the ZeRO stage. If the model fits under ZeRO‑2 (after accounting for expert sharding), **run ZeRO‑2 + MegaBlocks**: you keep the cheaper comms and still get grouped expert kernels. This is the recommended setup for Qwen3‑30B MoE on MI300X.
- ZeRO‑3 + MegaBlocks is the fallback when optimizer state cannot be replicated (GLM4.5 Air, Llama‑4 Scout). Expect ~2× lower tok/s/GPU compared to the ZeRO‑2 runs, even with grouped kernels.
- To quickly compare configs, watch `train/tokens_per_second_per_gpu` in W&B or the Axolotl logs. After a few hundred steps (once data loading stabilizes), the faster config will have a visibly higher steady-state curve. Use the same dataset slice + micro batch to keep comparisons fair.

---

# Dense Llama Baselines (3B / 14B / 70B)

## Observed dense runs (8× MI300X, ZeRO‑3, AdamW‑8bit, bf16)

| Run | Model | Stage | tok/s/GPU | Wall | Tokens (×10^8) | GPU mem (obs) |
|-----|-------|-------|-----------|------|----------------|---------------|
| 157 | Llama‑3.2‑3B | SFT (3 epochs, seq 8 k, mbs 16) | **8 900** | 3h04m | 7.86 | n/a (well under 192 GB) |
| 158 | Llama‑3.2‑3B | DPO (seq 4 k, mbs 8) | **8 900** | 1h42m | 4.36 | 28 % ≈ 54 GB† |
| 149 | Phi‑4 14B | SFT (3 epochs, seq 8 k, mbs 16) | **2 300** | 14h46m | 9.78 | 59 % ≈ 113 GB |
| 149 | Phi‑4 14B | DPO (seq 4 k, mbs 8) | **2 300** | 7h25m | 4.91 | n/a (logs did not record) |

† The DPO run comment references earlier RTX 5090 tests, but 28 % matches the MI300X projection (~10 GB activation budget scaled to 32 GB cards).

These runs cover the exact dense workloads we care about: ~7.9×10^8 tokens for SFT (3 epochs of `sft.shisa-v2.1`) and ~4.6×10^8 tokens for DPO (`dpo.shisa-v2.1`). They provide a clean slope for throughput + memory scaling.

## Throughput projection for `meta-llama/Llama-3.3-70B-Instruct`

- Config inspected: `~/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b/config.json` (`hidden_size: 8192`, `num_hidden_layers: 80`, `intermediate_size: 28672`, `vocab_size: 128256`).
- Empirical scaling from the 3B and 14B logs fits `tok_per_gpu ≈ k / params` with `k ≈ 3.0×10^4`. A power-law fit with exponent 0.92 provides the upper bound.

| Model | Params (B) | tok/s/GPU | tok/s @8×GPU | Tokens target | Wall (hh:mm) |
|-------|------------|-----------|--------------|---------------|--------------|
| Llama‑3.2‑3B (run 157) | 3.2 | 8 900 (measured) | 71 k | 7.86×10^8 | 3:04 |
| Phi‑4 14B (run 149) | 14 | 2 300 (measured) | 18.4 k | 9.78×10^8 | 14:46 |
| **Llama‑3.3‑70B (SFT)** | 70 | **430–530 (est.)** | 3.4–4.2 k | 7.86×10^8 | **52–64 h** |
| **Llama‑3.3‑70B (DPO)** | 70 | same as SFT | 3.4–4.2 k | 4.6×10^8 (avg of runs 158/149) | **31–37 h** |

Assumptions:
1. Sample packing + `pad_to_sequence_len: true` stabilize tokens per optimizer step, so throughput scales mainly with active parameters.
2. If we shrink the micro batch for memory reasons, we restore the same global batch with `gradient_accumulation_steps`; tokens/sec is therefore unchanged to first order (expect <5 % penalty from kernel underfill).
3. DPO uses identical per-token compute; its shorter sequences just lower the curriculum to ≈59 % of SFT, explaining the 0.5–0.6× wall times we already measured.

## Memory budget (MI300X 192 GB, ZeRO‑3, AdamW‑8bit, gradient checkpointing)

Formulas:
- Param shard per GPU = `params_B × 0.25 GB` (bf16 weights / ZeRO‑3 across 8 GPUs).
- Grad shard per GPU = same as params.
- Optimizer shard (bf16 master + 8-bit moments) = `params_B × 0.5 GB`.
- Activation/KV cache term calibrated from run 149 (99 GB at seq 8 k, mbs 16):  
  `A ≈ 6.1875 GB × micro_batch_size × (hidden/5120) × (layers/40) × (seq_len/8192)`.

### SFT (sequence 8 192, sample packing, pad-to-seq=true)

| Model | Params (B) | Param shard (GB) | Grad shard (GB) | Optimizer shard (GB) | Activation (GB) | Total (GB) | Notes |
|-------|------------|------------------|-----------------|----------------------|-----------------|------------|-------|
| Llama‑3.2‑3B | 3.2 | 0.8 | 0.8 | 1.6 | 41.6 (mbs 16) | **44.8** | Plenty of headroom (~23 % of 192 GB). |
| Phi‑4 14B | 14 | 3.5 | 3.5 | 7.0 | 99.0 (mbs 16) | **113** | Matches the 59 % telemetry from run 149. |
| **Llama‑3.3‑70B** | 70 | 17.5 | 17.5 | 35.0 | 118.8 (mbs 6) | **188.8** | mbs 16 would need 387 GB → infeasible. Use `micro_batch_size: 6` + `gradient_accumulation_steps: 3` (global batch ≈144). Drop to mbs 5 (169 GB) if extra NCCL buffer space is needed. |

### DPO (sequence 4 096, no packing)

| Model | Params (B) | Activation (GB) | Total (GB) | Suggested mbs |
|-------|------------|-----------------|------------|---------------|
| Llama‑3.2‑3B | 3.2 | 10.4 | **13.6** | 8 (matches run 158) |
| Phi‑4 14B | 14 | 24.8 | **38.8** | 8 |
| **Llama‑3.3‑70B** | 70 | 99.0 (mbs 10) | **169** | 10 keeps a comfortable 20 GB margin; mbs 12 is possible (188.8 GB) if the extra global batch is worth the tighter headroom. |

Takeaways:
1. **SFT demands a smaller micro batch for 70B.** `mbs=6` is the largest value that fits under 192 GB with the current plugins. Start the job with `torch.cuda.max_memory_allocated` logging; if startup spikes exceed 190 GB, immediately drop to `mbs=5`.
2. **DPO remains comfortable** because the 4 k context halves the activation term. `mbs=10` plus `gradient_accumulation_steps: 2` gives a 160-sample global batch with ~169 GB steady-state usage.
3. Keep `gradient_checkpointing_kwargs.use_reentrant: false`, Flash Attention 2, and MegaBlocks enabled—the activation column assumes all three optimizations remain on.
4. Padding hurts. If we ever disable `pad_to_sequence_len`, expect ~15 % lower memory at the cost of minor throughput jitter.

## Action items
- Clone `149-unphi4-14b-sft2.1.axolotl.8xMI300.yaml`, swap `base_model` → `meta-llama/Llama-3.3-70B-Instruct`, set `micro_batch_size: 6`, `gradient_accumulation_steps: 3`, and keep the same `sft.shisa-v2.1` loader (≈7.9×10^8 tokens).
- For DPO, start from `158-llama3.2-3b-v2.1-dpo-4e7.axolotl.8xMI300.yaml`, update the checkpoint path to the new 70B SFT weights, set `micro_batch_size: 10`, `gradient_accumulation_steps: 2`, and keep β=0.1.
- Reserve **~2.2–2.7 days** for the SFT run and **31–37 h** for DPO given the 430–530 tok/s/GPU window. Re-evaluate once the first 500 steps confirm the steady-state throughput curve.
