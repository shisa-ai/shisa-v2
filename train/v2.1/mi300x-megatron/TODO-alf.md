See: https://chatgpt.com/c/68d492fd-2af0-8320-8241-33d227c78001

Great—here’s a **concrete ALF (Aux‑Loss‑Free) router sketch** you can drop into **Megatron + MegaBlocks** and a **per‑model checklist** for Qwen3‑MoE, Ling‑2.0, Llama‑4‑Scout, GLM‑4.5‑Air, and DeepSeek‑V3.

---

## 0) What ALF actually does (recap)

ALF keeps a **per‑expert bias** $b_i$ that adjusts **who gets selected** (Top‑k) without changing **how much each selected expert contributes**. In practice:

* **Selection**: compute Top‑k on **$s_i + b_i$** (where $s_i$ are router affinities/logits).
* **Mixing weights**: compute from the **original $s_i$** **only on the selected experts** and normalize within the Top‑k set.
* **Bias update**: after each step, nudge $b_i$ **up** if expert $i$ was **under‑used**, **down** if **over‑used**, using a small step $\gamma$ (the “bias update speed”). DeepSeek‑V3 also adds an *optional* tiny sequence‑wise balance loss. 

Primary references (for theory & knobs): the **DeepSeek‑V3** tech report (formulas (16)–(20) show the Top‑k on $s_i+b_i$, normalization on selected experts, and the small sequence‑wise loss), and the separate **Loss‑Free Balancing** paper describing the bias update strategy. 

---

## 1) Where this lives in **Megatron + MegaBlocks**

* MegaBlocks already handles **dropless MoE** packing, all‑to‑all, and expert FFNs; you only need to change **the router** (gating). ([GitHub][1])
* In Megatron‑Core, routers are implemented under the MoE package; adding a small **Top‑k‑with‑bias** variant is the intended extensibility point. (Megatron‑Core MoE lists **DeepSeek‑V3** and **Qwen‑MoE** among supported families, i.e., the interfaces expect modern routers.) ([NVIDIA Docs][2])

> **Distributed scope for counts:** For bias updates, aggregate **token‑counts per expert** across the **Expert‑Parallel group** (and optionally Data‑Parallel group if you want *global* batch statistics) before updating $b$.

---

## 2) **ALF router** – implementation sketch (PyTorch‑style pseudocode)

> This is intentionally minimal and framework‑agnostic, but matches Megatron/MegaBlocks expectations: **Top‑k routing** + **dropless** + **expert‑parallel all‑reduce**. The only new state is a per‑expert bias and an EMA of loads.

```python
class ALFTopKRouter(nn.Module):
    def __init__(
        self,
        n_experts_routed: int,     # excludes any shared experts
        k: int,                    # Top-k routed experts
        affinity: nn.Module,       # linear/projection that produces s = affinity(x)
        mixing: str = "softmax",   # {"softmax","sigmoid"} for mixing weights
        ema_beta: float = 0.99,    # EMA horizon for load
        bias_lr: float = 1e-3,     # γ: bias update speed
        freeze_step: int | None = None,    # set to an int to freeze bias late
        counts_scope: str = "ep",          # {"ep","dp+ep"}
        eps: float = 1e-6,
        clip: float = 1e-2          # optional clip on bias delta per step
    ):
        super().__init__()
        self.n = n_experts_routed
        self.k = k
        self.affinity = affinity
        self.mixing = mixing
        self.ema_beta = ema_beta
        self.bias_lr = bias_lr
        self.freeze_step = freeze_step
        self.eps = eps
        self.clip = clip

        # Biases & EMA loads are non‑trainable buffers (kept in FP32 for stability)
        self.register_buffer("expert_bias", torch.zeros(self.n, dtype=torch.float32))
        self.register_buffer("load_ema", torch.zeros(self.n, dtype=torch.float32))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

        # You’ll fetch these process groups from Megatron (expert parallel, data parallel)
        self.ep_group = get_expert_model_parallel_group()
        self.dp_group = get_data_parallel_group() if counts_scope == "dp+ep" else None

    def forward(self, x):
        """
        x: [tokens, hidden]
        returns: routing indices, mixing weights, (dispatch/combine tensors for MegaBlocks)
        """
        s = self.affinity(x)                     # [tokens, n_experts_routed]  -> router affinities
        # ---- selection on biased scores ----
        biased = s + self.expert_bias           # broadcast across tokens
        topk_scores, topk_idx = biased.topk(self.k, dim=-1)  # [tokens, k], [tokens, k]

        # ---- mixing weights from *unbiased* scores (DeepSeek‑V3) ----
        chosen = gather_rows(s, topk_idx)       # [tokens, k] pull original s for the chosen experts
        if self.mixing == "sigmoid":
            unnorm = torch.sigmoid(chosen)
            weights = unnorm / (unnorm.sum(dim=-1, keepdim=True) + self.eps)
        else:
            # softmax over the chosen k only
            weights = torch.softmax(chosen, dim=-1)

        # ---- accumulate per‑expert counts for bias update ----
        with torch.no_grad():
            # one‑hot histogram of selected experts per token
            counts_local = torch.zeros(self.n, dtype=torch.float32, device=x.device)
            counts_local.index_add_(0, topk_idx.reshape(-1), torch.ones_like(topk_idx, dtype=torch.float32).reshape(-1))
            # reduce over EP (and optionally DP) so counts reflect the global batch
            dist.all_reduce(counts_local, group=self.ep_group)
            if self.dp_group is not None:
                dist.all_reduce(counts_local, group=self.dp_group)
            # EMA update
            self.load_ema.mul_(self.ema_beta).add_(counts_local * (1.0 - self.ema_beta))

        # ---- pack for MegaBlocks (token -> expert) ----
        # MegaBlocks expects (indices, weights) to build dispatch/combine without drops.
        dispatch, combine = build_megablocks_routing(topk_idx, weights, self.n)
        return dispatch, combine, topk_idx, weights

    @torch.no_grad()
    def step_bias(self):
        """Call once per optimization step, after grad update."""
        self.step.add_(1)
        if self.freeze_step is not None and int(self.step.item()) >= self.freeze_step:
            return  # freeze bias late in training

        total = self.load_ema.sum()
        if total <= 0:
            return
        f = self.load_ema / (total + self.eps)         # recent fraction per expert
        target = 1.0 / float(self.n)
        delta = (target - f) * self.bias_lr            # ALF: push under‑used up, over‑used down
        # (optional) mild clip to avoid large swings if batch is tiny
        delta = delta.clamp(min=-self.clip, max=self.clip)
        self.expert_bias.add_(delta)                   # no grad on bias
```

**Notes & hooks:**

* Call `router.step_bias()` **once per step** (e.g., after `optimizer.step()` in your training loop or Megatron callback).
* Keep **bias & EMA in FP32**; they’re tiny and more stable that way.
* For **shared experts** (if your architecture has them), **do not include them in the Top‑k competition or bias vector**; route them as your model specifies (always‑on or with a fixed gate). DeepSeek‑V3 explicitly separates *shared* and *routed* experts; ALF applies to the routed set. 
* With **dropless** kernels (MegaBlocks), capacity factors are not used; every token is routed, so balancing is crucial but purely handled by the bias, not by token dropping. ([arXiv][3])

---

## 3) Per‑model checklist: what to change (or not)

Below: the **minimum you need** to apply ALF, respecting each model’s published routing/mixing style.

### A) **DeepSeek‑V3**

* **Routing/mixing**: V3 computes routing **affinities** (they use **sigmoid** for affinity), selects Top‑K on **$s+b$**, then **normalizes among selected** to get gates. They also report a tiny **sequence‑wise balance loss** (very small α) as a complement. **No token drop**. 
* **What to do**: Use the sketch **as‑is** with `mixing="sigmoid"`. Consider `bias_lr ≈ 1e‑3` early, then **freeze bias** late in training (set `freeze_step`), as described conceptually in the report. Optionally add the small sequence‑wise loss. 

### B) **Qwen3‑MoE**

* **Published baseline**: Qwen3’s public docs emphasize **auxiliary load‑balancing** (plus a **z‑loss**) and specifically recommend **global‑batch** LB for better specialization. ([Hugging Face][4])
* **What to do**:

  * Set **aux‑loss coef to 0** (turn it off) and enable the ALF router.
  * Keep their **Top‑k** (often 2/4/8 depending on the variant) and **mixing = "softmax"** (that’s the default in most Qwen MoE codepaths).
  * **Counts scope**: strongly prefer **dp+ep** so your bias reflects the **global batch**, mirroring Qwen’s global LB idea but **without** the aux loss. ([Qwen][5])

### C) **Ling 2.0 (Ling‑flash / Ling‑mini)**

* **Published baseline**: Ling explicitly states **“aux‑loss‑free + sigmoid routing”**, 1/32 activation, and other MoE refinements. ([Hugging Face][6])
* **What to do**:

  * Use the sketch **as‑is** with `mixing="sigmoid"`.
  * If you’re aiming to reproduce Ling‑flash behavior, keep **Top‑k and expert counts** from the card and **no aux loss**.
  * They report long‑context + MTP tweaks; those don’t change the router.

### D) **Llama‑4‑Scout‑17B‑16E**

* **Published baseline**: Card confirms **MoE with 16 experts** (17B active / 109B total); it does not document a specific balancing strategy. ([Hugging Face][7])
* **What to do**:

  * Treat as a **standard Top‑k token‑choice** router. Use the ALF sketch with `mixing="softmax"` unless your internal recipe uses a different mixing activation.
  * If **shared experts** are part of your internal variant, **exclude them from bias** and from the Top‑k competition (apply them as specified by your block).

### E) **GLM‑4.5‑Air**

* **Published baseline**: Card gives **106B total, 12B active** MoE; it doesn’t publish the exact router loss/activation. ([Hugging Face][8])
* **What to do**:

  * Use ALF with your current **Top‑k** and **mixing** (if pretraining recipe uses softmax gates, keep `mixing="softmax"`; if it uses sigmoid affinities, use `mixing="sigmoid"`).
  * If GLM uses **shared experts**, treat them like in DeepSeek (no bias; not in Top‑k).

---

## 4) Training knobs & defaults (that work on 8×MI300X, 8k)

* **Top‑k**: start **k=2 or 4**. You can scale to 8 after profiling all‑to‑all.
* **Bias update**: `bias_lr = 1e‑3` for most of pretrain → **freeze** near the end. (Freezing avoids chasing late‑stage noise.) DeepSeek‑V3 describes this “bias update speed” schedule. 
* **EMA horizon**: `ema_beta=0.99` (longish horizon stabilizes counts).
* **Counts scope**: prefer **`dp+ep`** for the most faithful batch‑wise balance; EP‑only also works if DP batches are homogeneous.
* **Sequence‑wise tiny loss (optional)**: weight **≈ 1e‑4** if you see extreme per‑sequence skew; compute it on **selected‑expert proportions per sequence** as in the report. 
* **Dropless kernels**: keep MegaBlocks dropless MoE (no capacity factor, no token drop). This meshes naturally with ALF. ([arXiv][3])

---

## 5) How to wire it end‑to‑end in Megatron (+ MegaBlocks)

1. **Add the router class** (e.g., `ALFTopKRouter`) in Megatron‑Core’s MoE router module (projects referencing Megatron point at `megatron/core/transformer/moe/router.py` as the integration point). ([GitHub][9])
2. **Expose flags**:

   * `--moe-router-type=alf_topk`
   * `--moe-bias-lr=1e-3`, `--moe-bias-freeze-step=<N>`, `--moe-ema-beta=0.99`
   * `--moe-mixing={softmax|sigmoid}` (pick per model, see §3)
   * `--moe-counts-scope={ep|dp+ep}`
3. **Call `router.step_bias()`** once per optimizer step (post‑`optimizer.step()`).
4. **Keep everything else the same**: experts, FFN dims, EP/TP/PP/CP settings, and the MegaBlocks **dropless** launch scripts. ([GitHub][1])

---

## 6) Sanity checks & expected telemetry

* **Load histograms per layer**: track $f_i$ (recent fraction per expert) and **stddev** across experts; with ALF, the batch‑wise variance should converge to a small band without collapsing to uniformity within every **sequence** (V3’s paper shows ALF allows more **domain specialization** than per‑sequence LB). 
* **A2A timing**: with balanced loads, all‑to‑all time stabilizes; if you see elongated tails, slightly raise `bias_lr` or widen EMA horizon.
* **Quality**: compared to classic aux‑loss LB, ALF avoids “interference gradients” (that’s the main motivation). ([arXiv][10])

---

## 7) Why this generalizes

ALF is **router‑centric**, not tied to a specific attention (MHA vs **MLA**) or expert design; it works for any **token‑choice Top‑k** MoE (DeepSeek‑V3, Qwen3‑MoE, Ling, etc.). You only need access to **router logits/affinities**, **Top‑k indices**, and a way to **aggregate counts** across your parallel groups. (If you adopt **Expert‑Choice** or **Sinkhorn/OT** routers, you’ll need different dispatch/packing; ALF targets the Top‑k token‑choice family used by the models above.) ([NeurIPS Proceedings][11])

---

## Citations (most load‑bearing)

* **DeepSeek‑V3** technical report: ALF equations, sigmoid affinity & Top‑k on $s+b$, tiny sequence‑wise aux. 
* **Loss‑Free Balancing** paper: ALF method details & motivation (avoid interference gradients). ([arXiv][10])
* **MegaBlocks** repo: dropless MoE + integration with Megatron‑LM. ([GitHub][1])
* **Megatron‑Core MoE** docs: MoE package, support for DeepSeek‑V3/Qwen‑MoE; where to extend routers. ([NVIDIA Docs][2])
* **Qwen3**: Transformers docs & Qwen blog on **global‑batch** load balancing (context for dp+ep scope). ([Hugging Face][4])
* **Ling 2.0**: model card states “**aux‑loss‑free + sigmoid routing**”. ([Hugging Face][6])
* **Llama‑4 Scout**: model card details 17B active, 16 experts (router specifics not documented). ([Hugging Face][7])
* **GLM‑4.5 Air**: model card (sizes & MoE; router specifics not documented). ([Hugging Face][8])

---

### Want a ready‑to‑run patch?

I can adapt this into a Megatron‑Core patch (router class + flags + a small callback that calls `step_bias()` after each `optimizer.step()`), plus a **K=2** config for quick bring‑up on your **8×MI300X @ 8k**.

[1]: https://github.com/databricks/megablocks?utm_source=chatgpt.com "databricks/megablocks"
[2]: https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/moe.html?utm_source=chatgpt.com "Mixture of Experts package"
[3]: https://arxiv.org/pdf/2211.15841?utm_source=chatgpt.com "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts"
[4]: https://huggingface.co/docs/transformers/en/model_doc/qwen3_moe?utm_source=chatgpt.com "Qwen3MoE - Hugging Face"
[5]: https://qwenlm.github.io/zh/blog/global-load-balance/?utm_source=chatgpt.com "通过全局负载均衡提升混合专家模型的性能和特异化程度 - Qwen"
[6]: https://huggingface.co/inclusionAI/Ling-flash-base-2.0?utm_source=chatgpt.com "inclusionAI/Ling-flash-base-2.0"
[7]: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E?utm_source=chatgpt.com "meta-llama/Llama-4-Scout-17B-16E"
[8]: https://huggingface.co/zai-org/GLM-4.5-Air?utm_source=chatgpt.com "zai-org/GLM-4.5-Air"
[9]: https://github.com/thu-ml/ReMoE?utm_source=chatgpt.com "thu-ml/ReMoE"
[10]: https://arxiv.org/html/2408.15664v1?utm_source=chatgpt.com "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"
[11]: https://proceedings.neurips.cc/paper_files/paper/2022/file/2f00ecd787b432c1d36f3de9800728eb-Paper-Conference.pdf?utm_source=chatgpt.com "Mixture-of-Experts with Expert Choice Routing"

---

Awesome—here’s a **drop‑in “aux‑loss‑free (ALF)” router shim** you can use inside your Megatron Docker without forking Megatron‑Core, plus a quick **config review** of the MoE models you named (with Hugging Face references).

---

## 1) “Zero‑diff” ALF monkey‑patch for Megatron

**Good news:** current Megatron‑Core already ships the DeepSeek‑style ALF router knobs (dynamic per‑expert bias, optional pre‑softmax routing, bias update rate, etc.). You can enable them entirely from CLI flags—no code surgery needed. The docs list:

* `--moe-router-enable-expert-bias`
* `--moe-router-bias-update-rate` (default `1e-3`, same as DeepSeek V3)
* `--moe-router-score-function {softmax,sigmoid}`
* `--moe-router-pre-softmax`
* `--moe-router-topk-scaling-factor`
  …and many other MoE args. ([NVIDIA Docs][1])

There is also a known bug that *used* to cast `expert_bias` away from fp32 under some regimes; if you’re on a modern Megatron (≥ ~0.11), it’s addressed, but if you do hit it you can force `--moe-router-dtype fp32`. ([GitHub][2])

### A. One‑file “sitecustomize” shim (recommended)

This file is automatically imported by Python, so you don’t have to modify your training scripts. It **injects ALF flags** into `sys.argv` if you didn’t already pass them.

1. Put this file in a directory you mount into the container, e.g. `/workspace/alf_patch/sitecustomize.py`.
2. Add that directory **first** on `PYTHONPATH`.

```bash
# inside your Megatron docker
mkdir -p /workspace/alf_patch
# save the Python file below as /workspace/alf_patch/sitecustomize.py
export PYTHONPATH="/workspace/alf_patch:${PYTHONPATH}"
```

**`/workspace/alf_patch/sitecustomize.py`**

```python
# Zero-diff ALF router injector for Megatron-Core
# Usage: export PYTHONPATH=/workspace/alf_patch:$PYTHONPATH  (this file lives in /workspace/alf_patch)
import os, sys

def _has_flag(flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in sys.argv)

def _insert(flag: str, value: str | None = None):
    if not _has_flag(flag):
        if value is None:
            sys.argv.append(flag)
        else:
            sys.argv.extend([flag, str(value)])

# Enable/disable via env
if os.environ.get("ALF_ENABLE", "1") == "1":
    # Turn off aux/z losses -> pure ALF
    _insert("--moe-router-load-balancing-type", "none")   # no aux loss
    _insert("--moe-aux-loss-coeff", "0.0")

    # DeepSeek-style dynamic bias
    _insert("--moe-router-enable-expert-bias")
    _insert("--moe-router-bias-update-rate", os.environ.get("ALF_BIAS_LR", "1e-3"))

    # Router score function (default softmax; Ling/DeepSeek often use sigmoid)
    _insert("--moe-router-score-function", os.environ.get("ALF_MIXING", "softmax").lower())

    # Optional pre-softmax + top-k scaling (e.g., DeepSeek V3 settings)
    if os.environ.get("ALF_PRE_SOFTMAX", "0") == "1":
        _insert("--moe-router-pre-softmax")
        if os.environ.get("ALF_TOPK_SCALING"):
            _insert("--moe-router-topk-scaling-factor", os.environ["ALF_TOPK_SCALING"])

    # Recommended dispatcher for EP
    _insert("--moe-token-dispatcher-type", os.environ.get("ALF_DISPATCHER", "alltoall"))

    # Optional: force numerics for router compute
    if os.environ.get("ALF_ROUTER_DTYPE"):
        _insert("--moe-router-dtype", os.environ["ALF_ROUTER_DTYPE"])

    # Optional: group-limited routing (device/node-limited). Set from env if you want it.
    if os.environ.get("ALF_GROUPS"):
        _insert("--moe-router-num-groups", os.environ["ALF_GROUPS"])
    if os.environ.get("ALF_GROUP_TOPK"):
        _insert("--moe-router-group-topk", os.environ["ALF_GROUP_TOPK"])

    # Optional: log per-layer
    if os.environ.get("ALF_PER_LAYER_LOG", "0") == "1":
        _insert("--moe-per-layer-logging")
```

**Why this works:** recent Megatron‑Core implements ALF natively; the shim simply guarantees the right flags are present every time you launch training. The flags and their semantics are documented in Megatron’s MoE guide (DeepSeek‑V3 / Qwen‑MoE supported, group‑limited routing, etc.). ([NVIDIA Docs][1])

> **DeepSeek‑style defaults** (per their report & Megatron issues):
>
> ```bash
> export ALF_MIXING=sigmoid
> export ALF_PRE_SOFTMAX=1
> export ALF_TOPK_SCALING=2.5
> export ALF_BIAS_LR=1e-3   # DeepSeek used 1e-3 early, then 0.0 late
> ```
>
> (DeepSeek ramps bias update γ=1e‑3 early, then freezes it to 0.0 late in training; you can emulate the freeze by changing `ALF_BIAS_LR` mid‑run or when resuming.) 
> Example flags matching a common DeepSeek recipe also appear in a Megatron issue thread. ([GitHub][2])

**Run example (single node, EP + all‑to‑all):**

```bash
export PYTHONPATH="/workspace/alf_patch:$PYTHONPATH"
export ALF_MIXING=sigmoid
export ALF_PRE_SOFTMAX=1
export ALF_TOPK_SCALING=2.5
export ALF_BIAS_LR=1e-3

torchrun --nproc_per_node=8 pretrain_gpt.py \
  ... \
  --num-experts 128 --moe-router-topk 8 \
  --expert-model-parallel-size 8 \
  --moe-token-dispatcher-type alltoall
```

(Megatron docs explain dispatcher types, group‑limited routing, padding for FP8, etc.). ([NVIDIA Docs][1])

> **Note on bias scope / reduction:** Megatron’s ALF bias update is defined on the **global batch** (it all‑reduces assigned token counts), so you don’t need extra plumbing when you use DP+EP. ([NVIDIA Docs][1])

---

### B. (Optional) “Hard” monkey‑patch for old Megatron builds

If you’re pinned to an older Megatron without `--moe-router-enable-expert-bias`, here’s a **minimal diff** you can apply to `megatron/core/transformer/moe/router.py` in your image:

```diff
--- a/megatron/core/transformer/moe/router.py
+++ b/megatron/core/transformer/moe/router.py
@@ class TopKRouter(...):
-    # ORIGINAL: compute probs from scores and pick top-k; aux loss balancing
-    probs = softmax(scores, dim=-1)
-    topk_vals, topk_idx = torch.topk(probs, k=self.topk, dim=-1)
-    gates = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
+    # ALF: select with bias, mix without bias (DeepSeek-V3)
+    if not hasattr(self, "expert_bias"):
+        self.register_buffer("expert_bias", scores.new_zeros(scores.shape[-1]), persistent=True)
+    biased_scores = scores + self.expert_bias
+    topk_vals, topk_idx = torch.topk(biased_scores, k=self.topk, dim=-1)
+    # Mixing weights from *unbiased* scores
+    sel = torch.gather(scores, dim=-1, index=topk_idx)
+    gates = torch.softmax(sel, dim=-1)
+    # ---- bias update (batch-wise) ----
+    if self.training:
+        counts = scores.new_zeros(self.expert_bias.shape)   # 1D [num_experts]
+        # count assignments (each token contributes equally across its k experts)
+        flat_idx = topk_idx.reshape(-1)
+        counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=counts.dtype))
+        # global reduce over EP (+ optionally DP)
+        import torch.distributed as dist
+        if dist.is_initialized():
+            dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=self.expert_parallel_group)
+        # bias += gamma * (mean - count)
+        gamma = getattr(self, "alf_bias_lr", 1e-3)
+        self.expert_bias.data.add_(gamma * (counts.mean() - counts))
+    # (return topk_idx, gates, ... as in original)
```

This mirrors DeepSeek‑V3’s ALF: **select** experts via `s + b`, **mix** with softmax over the original `s`, and **update** `b` by nudging under‑/over‑loaded experts toward balance (γ≈1e‑3 early, 0 late). See §3 & §4 of their report for formula and schedule. 

> If you can avoid this diff, do so—**native flags are safer** and already cover DeepSeek/Qwen/Ling routing variants. ([NVIDIA Docs][1])

---

## 2) Model‑specific router settings you can drop into the shim

These are **practical starting points** to replicate each model’s routing flavor using Megatron’s ALF knobs. (Feel free to override per experiment.)

| Model                       |              Total / Active params |                                                    Experts / Top‑k |                     Context (card) | Router hints (set as env for the shim)                                                               | Sources                                                                                                                |
| --------------------------- | ---------------------------------: | -----------------------------------------------------------------: | ---------------------------------: | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **DeepSeek‑V3** (Base)      |                         671B / 37B | 1 shared + **256** routed; **top‑8**; node‑limited to **≤4 nodes** |      4K pretrain → 128K after YaRN | `ALF_MIXING=sigmoid`, `ALF_PRE_SOFTMAX=1`, `ALF_TOPK_SCALING=2.5`, `ALF_BIAS_LR=1e-3` (freeze later) | HF card & tech report spell out 256 experts, top‑8, γ=1e‑3 then 0; node‑limited routing rationale. ([Hugging Face][3]) |
| **Qwen3‑235B‑A22B**         |                     **235B / 22B** |          **128** experts; **top‑8** (Transformers config defaults) | 262,144 native; YaRN to ≈1,010,000 | Start with `ALF_MIXING=softmax`; ALF replaces aux loss                                               | HF card lists 235/22B; Transformers `Qwen3MoeConfig` shows 128 experts & top‑8 defaults. ([Hugging Face][4])           |
| **Ling‑flash‑2.0 (base)**   |  **100B / 6.1B** (~4.8B non‑embed) |                                       1/32 activation ratio design |                  32K → 128K (YaRN) | `ALF_MIXING=sigmoid` (they specify “aux‑loss‑free + sigmoid”), `ALF_PRE_SOFTMAX=0`                   | Model card explicitly says **aux‑loss‑free + sigmoid routing**. ([Hugging Face][5])                                    |
| **Ling‑mini‑2.0**           | **16B / 1.4B** (~0.789B non‑embed) |                                              1/32 activation ratio |                  32K → 128K (YaRN) | same as Ling‑flash (`ALF_MIXING=sigmoid`)                                                            | Card describes 1/32 activation and ALF+sigmoid strategy. ([Hugging Face][6])                                           |
| **Llama‑4 Scout (17B‑16E)** |     **109B total / 17B activated** |                                  **16 experts** (top‑k not stated) |                     **10M** (card) | Start with `ALF_MIXING=softmax`; tune if needed                                                      | HF model card lists 16E, 17B activated, 109B total, 10M context. ([Hugging Face][7])                                   |
| **GLM‑4.5 Air**             |                    **~106B / 12B** |                                 (experts/top‑k not stated on card) |                                  — | Start with `ALF_MIXING=softmax`; ALF replaces aux loss                                               | HF card states 106B/12B and that it’s an MoE hybrid; no routing specifics published on card. ([Hugging Face][8])       |

> **Megatron “node‑limited / device‑limited” knobs** (if you scale across nodes): `--moe-router-num-groups` + `--moe-router-group-topk` implement DeepSeek‑style locality (group by EP nodes or devices, then top‑k within groups). On **one 8×MI300X node**, leave groups unset. ([NVIDIA Docs][1])

---

## 3) Quick sanity notes for Megatron + ALF

* **You don’t need an aux loss**: set `--moe-router-load-balancing-type none` and `--moe-aux-loss-coeff 0`. DeepSeek still kept a tiny *sequence‑wise aux loss* as a safety valve in some experiments, but their main balancing is ALF. If you want that tiny valve, use `seq_aux_loss` with a very small coeff (e.g., 1e‑4) and keep ALF on. 
* **Bias update in fp32**: keep router math robust; if you see dtype drift (older builds), force `--moe-router-dtype fp32`. ([GitHub][2])
* **Dispatcher**: prefer `--moe-token-dispatcher-type alltoall` for EP on modern builds. ([NVIDIA Docs][1])
* **Megatron already supports these**: The official docs call out DeepSeek‑V3 and Qwen‑MoE as supported patterns, including FP8, group‑limited routing, EP A2A overlap, etc. ([NVIDIA Docs][9])

---

## 4) If you *do* need a code‑level ALF router (legacy Megatron)

If corporate constraints require a hard patch, apply the diff above or drop a small module that replaces the router. The three **invariant rules** you must preserve (from DeepSeek‑V3):

1. **Top‑k selection** uses **biased** scores `s + b`.
2. **Mixing weights** are computed **only from the un‑biased** `s` (softmax over the selected experts).
3. **Batch‑wise bias update**: after each batch (global across EP/DP), `bᵢ ← bᵢ + γ(μ − nᵢ)` where `nᵢ` is tokens assigned to expert *i*, `μ` the mean load, and `γ≈1e‑3` early, then freeze (0) in late training. 

These are architecture‑agnostic and work across Qwen3‑MoE, Ling 2.0, Llama‑4, GLM Air, and DeepSeek V3—what changes per family is mostly the **score function** (softmax vs sigmoid), whether to **pre‑softmax**, **top‑k scaling**, and any **group‑limited routing** you enable for topology. Megatron exposes all of those as flags now. ([NVIDIA Docs][1])

---

## 5) Pinned references (model cards & docs)

* **DeepSeek‑V3 (HF card & tech report)** — 671B/37B, 256 routed + 1 shared, top‑8, γ schedule, node‑limited routing details. ([Hugging Face][3])
* **Megatron‑Core MoE guide** — ALF flags (`--moe-router-enable-expert-bias`, `--moe-router-bias-update-rate`, `--moe-router-pre-softmax`, group‑limited routing, etc.). ([NVIDIA Docs][1])
* **Known router bias dtype issue (older Megatron)** — context & example flags. ([GitHub][2])
* **Qwen3‑MoE cards/docs** — 235B/22B (Instruct), config defaults (128 experts, top‑8). ([Hugging Face][4])
* **Ling‑flash‑2.0 / Ling‑mini‑2.0** — ALF + **sigmoid** routing; 1/32 activation designs; contexts. ([Hugging Face][5])
* **Llama‑4 Scout (17B‑16E)** — 16 experts, 17B activated / 109B total, context 10M. ([Hugging Face][7])
* **GLM‑4.5 Air** — 106B/12B; hybrid MoE; (experts/top‑k not specified on HF card). ([Hugging Face][8])

---

### TL;DR

* **Use the sitecustomize shim** above; it “monkey‑patches” your run by inserting **ALF** flags into Megatron’s CLI without touching Megatron code.
* Start with:

  * **DeepSeek‑style**: `ALF_MIXING=sigmoid`, `ALF_PRE_SOFTMAX=1`, `ALF_TOPK_SCALING=2.5`, `ALF_BIAS_LR=1e-3`. 
  * **Qwen/Llama/GLM default**: `ALF_MIXING=softmax` (tune if needed). ([Hugging Face][10])
* If you’re stuck on a legacy Megatron, apply the tiny **router.py** diff to implement **select‑with‑bias / mix‑without‑bias / batch‑wise bias update**. 

If you want, I can also drop a ready‑to‑run **torchrun script** tailored to your **8×MI300X** box (EP/TP/PP/SP, 8K seq length, dropless MoE) with these shim envs baked in.

[1]: https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/moe.html "Mixture of Experts package - NVIDIA Docs"
[2]: https://github.com/NVIDIA/Megatron-LM/issues/1482 "GitHub · Where software is built"
[3]: https://huggingface.co/deepseek-ai/DeepSeek-V3?utm_source=chatgpt.com "deepseek-ai/DeepSeek-V3 - Hugging Face"
[4]: https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507?utm_source=chatgpt.com "Qwen/Qwen3-235B-A22B-Instruct-2507"
[5]: https://huggingface.co/inclusionAI/Ling-flash-base-2.0 "inclusionAI/Ling-flash-base-2.0 · Hugging Face"
[6]: https://huggingface.co/inclusionAI/Ling-mini-2.0 "inclusionAI/Ling-mini-2.0 · Hugging Face"
[7]: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E "meta-llama/Llama-4-Scout-17B-16E · Hugging Face"
[8]: https://huggingface.co/zai-org/GLM-4.5-Air "zai-org/GLM-4.5-Air · Hugging Face"
[9]: https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/moe.html?utm_source=chatgpt.com "Mixture of Experts package"
[10]: https://huggingface.co/docs/transformers/en/model_doc/qwen3_moe "Qwen3MoE"

