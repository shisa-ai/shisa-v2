#!/usr/bin/env python3

"""
TRL training script for Qwen3-30B-A3B-v2 SFT on the dedicated 8x MI300X box.
Converted from the old bash launcher so we can keep all knobs in one place.
"""

import os
import sys
import subprocess
import torch
import argparse
import warnings
from pathlib import Path
from typing import Any
from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# ============================================================================
# Configuration - adjust these for different systems
# ============================================================================

# Hardware configuration specific to the MI300X box
NUM_GPUS = 8  # 8x MI300X in the node (HIP/RCCL stack)

# Accelerate config used for this box
ACCELERATE_CONFIG = "accelerate_config.fsdp2.yaml"

# Model and data configuration
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DATASET_PATH = "./sft.shisa-v2.jsonl"  # Generated from generate-new-sft.py
OUT = "072-qwen3-30b-a3b-v2-sft-trl-megablocks"
CACHED_DATASET_PATH = "./cached_formatted_dataset"  # Cache for processed dataset

# Training hyperparameters - distributed
GLOBAL_BATCH_SIZE = 128
PER_DEVICE_BATCH_SIZE = 1
LR = 1.63e-5  # Based on effective batch size of 128
MAX_LEN = 8192
EPOCHS = 3

# ============================================================================
# Environment setup and validation
# ============================================================================

def setup_environment():
    """Set up environment variables"""
    os.environ["WANDB_ENTITY"] = "augmxnt"
    os.environ["WANDB_PROJECT"] = "shisa-v2.1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    # Leave device visibility and RCCL knobs to accelerate / system defaults


def _rank_prefix() -> str:
    r = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    return f"[rank{r}]" if r is not None else "[rank-]"


def log(msg: str) -> None:
    print(f"{_rank_prefix()} {msg}", flush=True)


def log_versions_and_accelerate_config() -> None:
    """Log library versions and a summary of accelerate_config.yaml if present."""
    try:
        import transformers as _tf
        import accelerate as _acc
        import trl as _trl
        log(
            f"Versions torch={torch.__version__} transformers={_tf.__version__} "
            f"accelerate={_acc.__version__} trl={_trl.__version__}"
        )
    except Exception as e:
        log(f"Version logging failed: {e}")

    cfg_path = Path(ACCELERATE_CONFIG)
    if cfg_path.exists():
        try:
            txt = cfg_path.read_text()
            keys = (
                "distributed_type",
                "mixed_precision",
                "fsdp_",
                "fsdp",
                "sharding",
                "num_processes",
                "fsdp_version",
            )
            lines = [ln.strip() for ln in txt.splitlines() if any(k in ln for k in keys)]
            log("Accelerate config summary:")
            for ln in lines:
                log(f"  {ln}")
        except Exception as e:
            log(f"Could not read {ACCELERATE_CONFIG}: {e}")
    else:
        log(f"{ACCELERATE_CONFIG} not found")


def log_cuda_mem(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    try:
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            log(f"CUDA mem {tag} dev{i}: allocated={alloc:.2f}GB reserved={reserved:.2f}GB")
    except Exception as e:
        log(f"CUDA mem log failed: {e}")


def log_model_randomness(model: Any) -> None:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        fields = [
            "attn_implementation",
            "attention_dropout",
            "attn_dropout",
            "hidden_dropout_prob",
            "resid_pdrop",
            "embedding_dropout",
            "activation_dropout",
            "rope_dropout",
            "dropout",
            "router_dropout",
            "router_jitter_noise",
            "router_noise_std",
            "router_noise",
            "moe_jitter_noise",
        ]
        present = {f: getattr(cfg, f) for f in fields if hasattr(cfg, f)}
        log(f"Model config randomness: {present}")

    # Summarize nn.Dropout modules
    try:
        import torch.nn as nn  # type: ignore

        total = 0
        nonzero = 0
        samples = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Dropout):
                total += 1
                p = getattr(mod, "p", 0.0)
                if p and p > 0:
                    nonzero += 1
                    if len(samples) < 12:
                        samples.append((name, p))
        log(f"nn.Dropout modules: total={total}, nonzero={nonzero}")
        for name, p in samples:
            log(f"  dropout>0: {name}.p={p}")
    except Exception as e:
        log(f"Dropout scan failed: {e}")

    # Router/gating attributes
    try:
        attrs = [
            "jitter_noise",
            "noise_std",
            "router_jitter_noise",
            "router_noise_std",
            "router_noise",
            "dropout",
        ]
        count = 0
        logged = 0
        for name, mod in model.named_modules():
            cls = mod.__class__.__name__.lower()
            if any(k in cls for k in ["router", "gate", "gating", "moe", "switch"]):
                count += 1
                for a in attrs:
                    if hasattr(mod, a):
                        try:
                            val = getattr(mod, a)
                            # If it's a Dropout module, report p
                            vstr = None
                            if hasattr(val, "p"):
                                vstr = f"{val.p}"
                            elif isinstance(val, (int, float)):
                                vstr = f"{val}"
                            if vstr is not None and (vstr != "0.0"):
                                log(f"  router_attr {name}.{a}={vstr}")
                                logged += 1
                                if logged >= 24:
                                    break
                        except Exception:
                            pass
        log(f"Router-like modules: {count}, attrs_logged={logged}")
    except Exception as e:
        log(f"Router attr scan failed: {e}")


def enable_reentrant_ac_monkeypatch() -> None:
    """Force Accelerate activation checkpointing to use the reentrant impl."""
    try:
        import importlib

        mod_path = "torch.distributed.algorithms._checkpoint.checkpoint_wrapper"
        ckw = importlib.import_module(mod_path)

        original_fn = getattr(ckw, "checkpoint_wrapper", None)
        impl_enum = getattr(ckw, "CheckpointImpl", None)
        if original_fn is None or impl_enum is None:
            log("AC monkeypatch skipped: checkpoint_wrapper or CheckpointImpl not found")
            return

        def _reentrant_checkpoint_wrapper(*args, **kwargs):
            try:
                kwargs["checkpoint_impl"] = impl_enum.REENTRANT
            except Exception:
                pass
            return original_fn(*args, **kwargs)

        setattr(ckw, "checkpoint_wrapper", _reentrant_checkpoint_wrapper)
        log("Applied REENTRANT AC monkeypatch (overrides Accelerate's NO_REENTRANT)")
    except Exception as e:
        log(f"Failed to apply AC monkeypatch: {e}")


def validate_batch_configuration():
    """Validate batch size configuration for multi GPU"""
    # Conservative per-device batch size for 192GB MI300X with MoE layers
    per_device_batch_size = PER_DEVICE_BATCH_SIZE
    effective_batch_size = NUM_GPUS * per_device_batch_size
    grad_accumulation_steps = GLOBAL_BATCH_SIZE // effective_batch_size

    if grad_accumulation_steps * effective_batch_size != GLOBAL_BATCH_SIZE:
        print(
            "ERROR: Global batch size ({}) is not divisible by effective batch size ({})".format(
                GLOBAL_BATCH_SIZE, effective_batch_size
            )
        )
        print(
            f"  Effective batch size = NUM_GPUS ({NUM_GPUS}) × PER_DEVICE_BATCH_SIZE ({per_device_batch_size}) = {effective_batch_size}"
        )
        sys.exit(1)

    if grad_accumulation_steps < 1:
        print(
            f"ERROR: Gradient accumulation steps ({grad_accumulation_steps}) must be at least 1"
        )
        sys.exit(1)

    print("Training Configuration:")
    print(f"  GPUs: {NUM_GPUS}")
    print(f"  Global Batch Size: {GLOBAL_BATCH_SIZE}")
    print(f"  Per Device Batch Size: {per_device_batch_size}")
    print(f"  Effective Batch Size: {effective_batch_size}")
    print(f"  Gradient Accumulation Steps: {grad_accumulation_steps}")
    print(f"  Learning Rate: {LR}")
    print(f"  Max Length: {MAX_LEN}")
    print(f"  Epochs: {EPOCHS}")
    print()

    return per_device_batch_size, grad_accumulation_steps

def load_and_format_dataset(tokenizer, debug_mode=False, refresh_cache=False):
    """Load dataset and format with caching support"""
    # Use separate cache files for debug and full mode
    if debug_mode:
        cache_path = Path(f"{CACHED_DATASET_PATH}_debug")
    else:
        cache_path = Path(CACHED_DATASET_PATH)
    dataset_file = Path(DATASET_PATH)

    # Check if cache exists and is newer than source dataset (unless refresh is forced)
    if not refresh_cache and cache_path.exists() and cache_path.stat().st_mtime > dataset_file.stat().st_mtime:
        print(f"Loading cached formatted dataset from {cache_path}")
        try:
            dataset = load_from_disk(str(cache_path))
            print(f"Cached dataset loaded: {len(dataset)} examples")

            if debug_mode:
                subset_size = min(1000, len(dataset))
                dataset = dataset.select(range(subset_size))
                print(f"Debug mode: Using subset of {len(dataset)} examples for testing")

            return dataset
        except Exception as e:
            print(f"Failed to load cached dataset: {e}")
            print("Falling back to processing raw dataset...")
    elif refresh_cache and cache_path.exists():
        print(f"Refresh flag set - ignoring existing cache at {cache_path}")

    # Load raw dataset
    print(f"Loading raw dataset from {DATASET_PATH}")
    if DATASET_PATH.endswith('.jsonl'):
        raw_dataset = load_dataset('json', data_files=DATASET_PATH, split='train')
    else:
        raw_dataset = load_dataset(DATASET_PATH, split='train')

    print(f"Raw dataset loaded: {len(raw_dataset)} examples")

    # Use subset for debug mode before processing
    if debug_mode:
        subset_size = min(1000, len(raw_dataset))
        raw_dataset = raw_dataset.select(range(subset_size))
        print(f"Debug mode: Using subset of {len(raw_dataset)} examples for processing")

    # Define formatting function using tokenizer's chat template
    def formatting_func(example):
        """Convert conversations format using model's chat template"""
        if isinstance(example["conversations"], list):
            messages = []
            for message in example["conversations"]:
                role = message.get("role", message.get("from", "unknown"))
                content = message.get("content", message.get("value", ""))
                messages.append({"role": role, "content": content})

            # Use the tokenizer's chat template
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except Exception as e:
                print(f"Warning: Failed to apply chat template: {e}")
                # Fallback to simple concatenation
                text = ""
                for message in messages:
                    text += f"{message['role']}: {message['content']}\n"
                return text
        else:
            return str(example["conversations"])

    # Format the dataset
    print("Formatting conversations using chat template...")
    formatted_texts = []
    for i, example in enumerate(raw_dataset):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(raw_dataset)} examples...")
        formatted_texts.append(formatting_func(example))

    # Create dataset from formatted texts
    dataset = Dataset.from_dict({"text": formatted_texts})

    # Cache the formatted dataset
    cache_type = "debug" if debug_mode else "full"
    print(f"Saving formatted dataset to {cache_type} cache: {cache_path}")
    try:
        dataset.save_to_disk(str(cache_path))
        print(f"{cache_type.title()} dataset cached successfully")
    except Exception as e:
        print(f"Warning: Failed to cache {cache_type} dataset: {e}")

    print(f"Dataset formatting completed: {len(dataset)} examples")
    return dataset

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TRL MegaBlocks Training Script for Qwen3-30B-A3B-v2 SFT')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode with 1000 samples for testing')
    parser.add_argument('--refresh', action='store_true',
                       help='Force refresh of cached dataset (ignore existing cache)')
    return parser.parse_args()

def main():
    """Main training function running under accelerate."""
    warnings.filterwarnings("ignore", category=FutureWarning)

    args = parse_args()

    setup_environment()
    enable_reentrant_ac_monkeypatch()
    try:
        import torch.utils.checkpoint as _ckp

        _ckp.set_checkpoint_debug_enabled(True)
        log("Enabled torch.utils.checkpoint debug globally")
    except Exception as e:
        print(f"[ckpt-debug] Failed to enable checkpoint debug: {e}")

    log_versions_and_accelerate_config()
    per_device_batch_size, grad_accumulation_steps = validate_batch_configuration()

    output_dir = f"/data/outputs/{OUT}"

    print(f"Loading tokenizer from {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log_cuda_mem("before_model_load")
    print(f"Loading model from {MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        output_router_logits=True,
        router_aux_loss_coef=0.001,
        low_cpu_mem_usage=True,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
        log(
            "Model attn_implementation effective: {}".format(
                getattr(model.config, "attn_implementation", None)
            )
        )

    log_model_randomness(model)
    log_cuda_mem("after_model_load")

    def _try_zero_attr(obj, attrs):
        for a in attrs:
            if hasattr(obj, a):
                try:
                    setattr(obj, a, 0.0)
                    print(f"Set {obj.__class__.__name__}.{a} = 0.0 for deterministic routing")
                except Exception:
                    pass

    _try_zero_attr(
        model.config if hasattr(model, "config") else model,
        [
            "router_jitter_noise",
            "router_noise_std",
            "router_jitter_std",
            "router_noise",
            "moe_jitter_noise",
            "router_dropout",
        ],
    )

    try:
        import torch.nn as nn  # type: ignore

        for mod in model.modules():
            cls = mod.__class__.__name__.lower()
            if any(k in cls for k in ["router", "gate", "gating", "moe", "switch"]):
                for a in [
                    "jitter_noise",
                    "noise_std",
                    "router_jitter_noise",
                    "router_noise_std",
                    "noise",
                ]:
                    if hasattr(mod, a):
                        try:
                            setattr(mod, a, 0.0)
                            print(f"Set {mod.__class__.__name__}.{a} = 0.0")
                        except Exception:
                            pass
                if hasattr(mod, "dropout"):
                    d = getattr(mod, "dropout")
                    try:
                        if isinstance(d, float):
                            setattr(mod, "dropout", 0.0)
                            print(f"Set {mod.__class__.__name__}.dropout = 0.0")
                        elif isinstance(d, nn.Dropout):
                            d.p = 0.0
                            print(f"Set {mod.__class__.__name__}.dropout.p = 0.0")
                    except Exception:
                        pass
    except Exception:
        pass

    dataset = load_and_format_dataset(
        tokenizer, debug_mode=args.debug, refresh_cache=args.refresh
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accumulation_steps,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=None,
        ddp_find_unused_parameters=False,
        remove_unused_columns=True,
        dataloader_drop_last=True,
        max_seq_length=MAX_LEN,
        packing=True,
        logging_steps=1,
        save_steps=500,
        save_total_limit=3,
        optim="adamw_torch_4bit",
        weight_decay=1e-4,
        report_to=["wandb"],
        run_name=OUT,
    )
    log(
        "Training args: gradient_checkpointing={}, max_seq_length={}, packing={}, per_device_train_batch_size={}, grad_accumulation_steps={}".format(
            training_args.gradient_checkpointing,
            training_args.max_seq_length,
            training_args.packing,
            training_args.per_device_train_batch_size,
            training_args.gradient_accumulation_steps,
        )
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=None,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print(f"Training completed! Model saved to {output_dir}")


def run_with_accelerate():
    print("Starting TRL MegaBlocks training with FSDP...")

    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        ACCELERATE_CONFIG,
        "--main_process_port",
        "0",
        __file__,
    ]

    cmd.extend(sys.argv[1:])

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("Training completed!")
    else:
        print(f"Training failed with return code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        main()
    else:
        run_with_accelerate()
