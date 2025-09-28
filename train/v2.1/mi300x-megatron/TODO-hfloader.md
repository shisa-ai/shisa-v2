# TODO: General HF Loader Support

## Current Situation
- The ROCm fork of Megatron-LM we use (and NVIDIA’s current `main` branch) no longer ship a generic `loader_hf.py` plugin.
- `tools/checkpoint/convert.py` only finds the following loaders:
  - `loader_core.py`, `loader_legacy.py`: Megatron-to-Megatron conversions.
  - `loader_llama_mistral.py`: Llama 2 / Llama 3 (8B/70B), Mistral 7B, Yi 34B, Qwen2.5 7B/72B (base & instruct).
  - `loader_mixtral_hf.py`: Mixtral 8×7B.
- Calling `--loader hf` fails because the plugin doesn’t exist. Our scripts can only auto-convert for architectures handled by the shipped loaders.

## What We’d Like
- Reintroduce a general Hugging Face loader so that any HF model supported by Transformers can be converted to Megatron format without model-specific patches.

## Possible Paths
1. **Port NVIDIA’s legacy `loader_hf`**
   - Locate the old file from an earlier commit (pre-refactor) or another branch.
   - Copy it into `/workspace/Megatron-LM/tools/checkpoint/`.
   - Verify it plays nicely with the current `schema_core`/`saver_core` APIs.

2. **Implement our own bridge**
   - Load HF model with Transformers.
   - Map weights to Megatron naming and ordering.
   - Feed them to the existing `saver` module.
   - Needs support for RoPE, GQA, MoE, etc., so this is non-trivial.

3. **Extend existing loaders**
   - For each new architecture, add explicit support (e.g., add Qwen3 to `loader_llama_mistral.py`).
   - Less general, but pragmatic if we only care about a handful of models.

## Next Steps
- Decide which option we want to pursue.
- If going with (1), identify the exact commit that still contains `loader_hf.py`. It’s not present in the current `main` branch.
- Update our scripts (`03-megablocks-*.sh`) once a general loader is in place so they can auto-convert without manual intervention.

---
## ChatGPT 5 Pro research

You said:
I am trying out https://github.com/ROCm/Megatron-LM for efficient MoE training, but I want to do SFT/DPO training. This means loading existing HF models and converting them to Megatron format. There doesn't appear to be a general hf_loader HF converter? Is this something that exists upstream? https://github.com/NVIDIA/Megatron-LM/tree/main - If nothing exists, can we implement a general version that we can use. It must be possible to load the model in transformers and then save it in an aribtrary format right?
ChatGPT said:
Request for GPT-5 Pro
Reasoned for 10m 44s

Short answer:

    There isn’t a single, universal hf_loader inside the Megatron‑LM repo. Upstream Megatron‑LM ships model‑specific loaders/converters (e.g., for LLaMA/Mistral/Mixtral) under tools/checkpoint/* and you invoke them via tools/checkpoint/convert.py. You’ll see people using loaders like loader_llama_mistral or loader_mixtral_hf to import Hugging Face checkpoints and then save Megatron format.
    GitHub+1

    There is now an official “general” solution, but it lives next to (not inside) Megatron‑LM: NVIDIA’s Megatron Bridge. It provides bidirectional conversion HF ↔ Megatron‑Core, auto‑detects architectures, and includes SFT/PEFT recipes; it also handles MoE families like Qwen3‑MoE.
    GitHub+1

Below I’ll map the practical options for your SFT/DPO workflow on ROCm, and if you want to “roll your own” how to implement a robust, general converter.
What exists upstream today
1) In‑repo converters (classic Megatron‑LM)

Megatron‑LM’s tools/checkpoint/convert.py can load HF checkpoints for specific families and save a Megatron checkpoint, including repartitioning to your desired TP/PP sizes. Typical flows you’ll find in issues/PRs are:

python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader llama_mistral \                # or loader for your family
  --checkpoint-type hf \
  --saver mcore \                         # save in Megatron-Core format
  --target-tensor-parallel-size 2 \
  --target-pipeline-parallel-size 1 \
  --load-dir /path/to/hf_model_dir \
  --save-dir /path/to/out_megatron_ckpt \
  --tokenizer-model /path/to/tokenizer.model

People use loaders like loader_llama_mistral (covers LLaMA3.x/Mistral/Mixtral) and others for legacy/core formats. These are not universal; they encode architecture‑specific name/shape rules.
GitHub+1

    FYI: there’s active issue traffic around this script (e.g., converting distributed torch_dist checkpoints, PP/TP changes), so expect some rough edges across versions.
    GitHub+2GitHub+2

2) Megatron Bridge (recommended if you want “general” HF↔Megatron)

Megatron Bridge (NVIDIA‑NeMo/Megatron‑Bridge) is a PyTorch‑native layer that auto‑detects popular HF models and streams weights into Megatron‑Core models, and can export back to a proper HF directory after training. It supports many families out of the box (Llama 3/3.1/3.2/3.3, Qwen2/2.5/3, including MoE variants) and includes SFT/PEFT building blocks. Minimal usage:

from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B", trust_remote_code=True)
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.finalize()

model = provider.provide_distributed_model(wrap_with_ddp=False)
# … train …
bridge.save_hf_pretrained(model, "./exported_hf_folder")

That gets you HF → Megatron for training and Megatron → HF export again. It’s also parallelism‑aware (TP/PP/VPP/CP/EP/ETP) during conversion and supports streaming to keep memory down.
NVIDIA Docs

Caveat (ROCm): Bridge’s README lists Transformer Engine as a dependency and documents CUDA 12.8+; it’s tested and packaged for NVIDIA stacks. On AMD, ROCm has its own TransformerEngine fork with the same Python import path and similar APIs, but Bridge on ROCm isn’t officially supported; YMMV if you try to run it with ROCm TE.
GitHub+1
3) ROCm / AMD guidance

AMD’s own docs cover training LLaMA, DeepSeek‑V3 (MoE), Mixtral (MoE), Qwen, etc., with ROCm/Megatron‑LM docker images, and show the knobs you’ll care about for MoE (EP/ETP, grouped GEMM flags, etc.). If you stay inside ROCm/Megatron‑LM, use the in‑repo converter (option #1) and the provided example scripts to run SFT/finetuning.
rocm.docs.amd.com
How to do SFT and DPO with Megatron on ROCm

You’ve got two realistic paths, depending on how much you want to stick to ROCm‑first workflows vs. leverage NVIDIA’s new tooling for conversion and post‑training.
Path A (ROCm‑native path, minimal moving parts)

    Convert HF → Megatron‑Core using the in‑repo converter for your model family (e.g., LLaMA/Mistral/Mixtral):

python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader llama_mistral \
  --checkpoint-type hf \
  --saver mcore \
  --load-dir /models/llama3-8b-hf \
  --save-dir /models/llama3-8b-mcore \
  --target-tensor-parallel-size 2 --target-pipeline-parallel-size 1

(Check tools/checkpoint for the exact loader names in your clone.)
GitHub

SFT: use the LLaMA/Mixtral examples in the ROCm fork (the same train scripts work, just point to your converted checkpoint and SFT dataset). AMD’s docs show the flags for LLaMA/Mixtral/DeepSeek MoE runs and the TE/MoE tunables often needed on MI300X (e.g., MOE_USE_LEGACY_GROUPED_GEMM=true).
rocm.docs.amd.com

DPO: for a maintained DPO trainer with Megatron‑Core backend, use NeMo‑RL. It includes DPO recipes and scales up with Megatron‑Core; configs take HF model IDs or local exports. Example launch (single‑node default in docs):

    uv run examples/run_dpo.py \
      cluster.gpus_per_node=8 \
      dpo.sft_loss_weight=0.1 \
      dpo.preference_average_log_probs=True

    (Point the config to your model; NeMo‑RL docs show DPO configurations and multi‑GPU variants.)
    NVIDIA Docs+1

    If you prefer to keep everything in “HF land” for post‑training and only convert at the very end: do SFT/DPO with TRL’s DPOTrainer on the HF model directly, then convert that HF result to Megatron using the same convert.py flow.
    Hugging Face

Path B (Bridge‑centric path, simplest general conversion & SFT)

If you can tolerate using the NVIDIA stack for the conversion/training step, Megatron Bridge gives you a much smoother HF↔Megatron story (especially across different model families, including MoE like Qwen3‑MoE). You load from HF, train with Megatron‑Core (Bridge includes SFT/PEFT), and export back to HF in one code path:

from megatron.bridge import AutoBridge
import megatron.bridge.recipes.llama.llama32_1b as llama32

bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B")
model_provider = bridge.to_megatron_provider()
cfg = llama32.pretrain_config()
cfg.model = model_provider
# Run your SFT finetune loop here (Bridge includes helpers)

For DPO you’d pair this with NeMo‑RL (which runs on Megatron‑Core).
GitHub+1
“Can we implement a general converter ourselves?”

Yes—but “general” in practice means “a registry of per‑family mapping rules.” Loading any HF model and “saving in an arbitrary format” is the easy part in vanilla PyTorch. What makes Megatron conversion non‑trivial is:

    Different weight layouts and fusions (e.g., fused QKV, GQA vs MQA, rotary position embeddings, different RMSNorm/LN variants).

    Parallel sharding rules (TP/PP/CP/EP/ETP): you must slice along exact dims consistently with Megatron‑Core sharding (e.g., attention Wqkv split across head_dim×num_heads for TP; MLP weight splits).

    MoE specifics: expert ordering, router/gate weights, EP/ETP slicing, potential grouped GEMM expectations.

    Tokenizer/vocab alignment (e.g., true vocab size after adding special tokens), and RoPE base/theta/YARN options.
    GitHub

A robust design mirrors what Bridge does:

    Identify the HF architecture (config introspection) and pick a model bridge implementation.

    Define a per‑layer mapping registry that yields (source_name → dest_tensor, slicing spec).

    Implement parallelism‑aware streaming export/import so you never materialize the full state on one device.

    Validate: run a quick forward and compare activations/logits between HF and Megatron for a few random inputs.

Bridge exposes exactly this kind of API (AutoBridge, mapping registries, streaming), so if you don’t need custom formats beyond Megatron, it’s almost certainly faster to adopt it rather than re‑creating the machinery.
NVIDIA Docs
MoE (Mixtral/DeepSeek/Qwen‑MoE) gotchas to plan for

    Expert weights placement: decide on expert index ordering and shard across EP/ETP consistently with Megatron’s expectation.

    Router/logit‑scale params: HF repos sometimes differ in naming/shape vs. Megatron’s MoE modules.

    Grouped GEMM paths: on MI300X you may need flags like USE_GROUPED_GEMM, MOE_USE_LEGACY_GROUPED_GEMM, NVTE_FUSED_ATTN_CK=0 depending on your ROCm/TE versions. AMD provides these in their docs & examples.
    rocm.docs.amd.com

Quick end‑to‑end recipes
A) ROCm: HF LLaMA‑3.x → Megatron (MoE optional) → SFT → (optionally) export back to HF

    Convert HF → Megatron

    python tools/checkpoint/convert.py \
      --model-type GPT \
      --loader llama_mistral \
      --checkpoint-type hf \
      --saver mcore \
      --load-dir /models/llama-3.1-8b-hf \
      --save-dir /models/llama-3.1-8b-mcore \
      --target-tensor-parallel-size 2 \
      --target-pipeline-parallel-size 1

    GitHub

    Run SFT (use the LLaMA example script in the ROCm repo/docs; point it at your SFT dataset & converted ckpt; set TE/MoE flags as needed).
    rocm.docs.amd.com

    (Optional) DPO: run NeMo‑RL’s run_dpo.py, referencing the same model via HF export or a local checkpoint path in the config.
    NVIDIA Docs

    Export back to HF (either via convert.py in reverse or by keeping the HF copy as the “ground truth” and exporting from NeMo‑RL/Bridge—Bridge offers save_hf_pretrained).
    NVIDIA Docs

B) NVIDIA stack: Use Megatron Bridge + NeMo‑RL

    Load HF directly with AutoBridge.from_hf_pretrained(...), train SFT (Bridge has SFT/PEFT recipes), then do DPO with NeMo‑RL (Megatron‑Core backend), and export back to HF.
    GitHub+1

Final guidance for your project

    If you need a truly general converter in your own pipeline on ROCm, implement a small “bridge” layer with an architecture registry (LLaMA‑like, Qwen‑like, GPT‑like, T5‑like, and MoE variants) and copy the partitioning rules from Megatron‑Core. That’s exactly how Bridge solves it—and it’s why a single hf_loader isn’t practical for “any HF model” without per‑family logic.
    NVIDIA Docs

    If you can, adopt Megatron Bridge (best ergonomics) and NeMo‑RL (for DPO). They’re upstream‑maintained and already solve conversion, verification, and post‑training at scale.
    GitHub+1

    Sticking 100% to ROCm? Use the in‑repo converter plus AMD’s example scripts/flags for MoE. It’s the most stable route on MI300X today.
    rocm.docs.amd.com

If you want, I can sketch a minimal “HF‑to‑Megatron” mapping function for a LLaMA‑like block that slices QKV and MLP weights for TP/PP and outlines the MoE expert sharding—just say which model family (LLaMA/Qwen/Mixtral/DeepSeek‑V3) you want to target first.
