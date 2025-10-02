#!/usr/bin/env python3

"""
TRL training script for Llama-3.2-1B SFT with DeepSpeed ZeRO-3 on the dedicated 8x MI300X box.
Converted from the megablocks version, removing megablocks dependencies for standard training.
"""

import os

# Disable torch.compile to prevent hanging on first run with hundreds of worker processes
# This avoids 15+ minute compilation hangs and process explosion issues
# To enable compilation for performance (after debugging), comment out the line below and instead run:
# export TORCHINDUCTOR_WORKER_COUNT=4
# export TORCHINDUCTOR_COMPILE_THREADS=4
# export TORCHINDUCTOR_MAX_AUTOTUNE=1
# before running the script
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import sys
import time
import signal
import subprocess
import torch
import argparse
import warnings
import atexit
from pathlib import Path
from typing import Any, Optional
from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

try:
    from torchao.optim import AdamW8bit
except ImportError:
    AdamW8bit = None

# ============================================================================
# Configuration - adjust these for different systems
# ============================================================================

# Hardware configuration specific to the MI300X box
NUM_GPUS = 8  # 8x MI300X in the node (HIP/RCCL stack)

# Accelerate config used for this box
ACCELERATE_CONFIG = "accelerate_config.deepspeed.yaml"

# Model and data configuration
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_PATH = "./sft.shisa-v2.1.jsonl"  # Generated from generate-new-sft.py
OUT = "092-llama3.2-1b-v2.1-sft-trl-dsz3"
CACHED_DATASET_PATH = "./cached_formatted_dataset"  # Cache for processed dataset

# Training hyperparameters - distributed
GLOBAL_BATCH_SIZE = 128
PER_DEVICE_BATCH_SIZE = 16
LR = 2.83e-5  # Matches OpenRLHF SFT run for this config
MAX_LEN = 8192
EPOCHS = 3

# ============================================================================
# Environment setup and validation
# ============================================================================

def setup_environment(args: argparse.Namespace):
    """Set up environment variables"""
    os.environ["WANDB_ENTITY"] = "augmxnt"
    os.environ["WANDB_PROJECT"] = "shisa-v2.1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Torch Inductor optimizations - limit worker processes
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "4"  # Limit compile workers per process
    os.environ["TORCHINDUCTOR_WORKER_COUNT"] = "4"     # Max 4 workers instead of 32
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"     # Reduce autotuning overhead
    os.environ["TORCH_COMPILE_DEBUG"] = "0"            # Disable debug for faster compilation

    # Prevent runaway processes
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_compile_cache"
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"

    if not args.rccl_defaults:
        # ROCm/RCCL optimizations tuned for this box (optional for debugging)
        os.environ["RCCL_DEBUG"] = "WARN"
        os.environ["RCCL_NET_GDR_LEVEL"] = "3"
        os.environ["RCCL_TREE_THRESHOLD"] = "4294967296"
        os.environ["RCCL_LL_THRESHOLD"] = "0"
        os.environ["RCCL_BUFFSIZE"] = "8388608"
        os.environ["RCCL_NTHREADS"] = "512"
        os.environ["RCCL_MAX_NCHANNELS"] = "16"
        os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"

    # NCCL timeout and debugging settings
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")  # 1 hour timeout
    os.environ.setdefault("NCCL_TIMEOUT", "3600")  # 1 hour NCCL timeout

    if args.debug:
        os.environ.setdefault("TORCH_NCCL_DEBUG", "INFO")
        os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", str(1 << 20))
        os.environ.setdefault("RCCL_DEBUG", "INFO")
    else:
        # Enable flight recorder even in non-debug mode for timeout diagnosis
        os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", str(1 << 16))

    os.environ["HIP_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Explicit GPU visibility


def _rank_prefix() -> str:
    r = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    return f"[rank{r}]" if r is not None else "[rank-]"


def log(msg: str) -> None:
    print(f"{_rank_prefix()} {msg}", flush=True)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# Global variable to track training state
_training_interrupted = False


def cleanup_on_exit():
    """Cleanup function called on script exit"""
    log("Cleaning up training resources...")

    # Clear CUDA/HIP cache
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            log("Cleared GPU cache")
        except Exception as e:
            log(f"Failed to clear GPU cache: {e}")

    # Kill any lingering inductor workers
    try:
        import subprocess
        subprocess.run(["pkill", "-f", "torch/_inductor/compile_worker"],
                      capture_output=True, timeout=5)
    except Exception:
        pass


def signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM gracefully"""
    global _training_interrupted

    log(f"Received signal {signum}, initiating graceful shutdown...")
    _training_interrupted = True

    # Try to cleanup distributed training
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            log("Destroying process group...")
            dist.destroy_process_group()
    except Exception as e:
        log(f"Failed to destroy process group: {e}")

    # Cleanup and exit
    cleanup_on_exit()

    # Force exit after a brief delay
    import threading
    def force_exit():
        time.sleep(2)
        os._exit(1)

    threading.Thread(target=force_exit, daemon=True).start()

    # Normal exit
    sys.exit(0)


def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_on_exit)


def detect_local_gpu_count() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        return torch.cuda.device_count()
    except Exception:
        return 0


def detect_world_size(default: Optional[int] = None) -> int:
    for key in ("WORLD_SIZE", "ACCELERATE_NUM_PROCESSES"):
        value = os.environ.get(key)
        if value:
            try:
                return max(int(value), 1)
            except ValueError:
                continue

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return max(dist.get_world_size(), 1)
    except Exception:
        pass

    if default is not None:
        return max(default, 1)

    detected = detect_local_gpu_count()
    return max(detected, 1)


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
    # Batch sizing matches the OpenRLHF 8xMI300 configuration
    per_device_batch_size = PER_DEVICE_BATCH_SIZE
    world_size = detect_world_size(default=NUM_GPUS)
    effective_batch_size = world_size * per_device_batch_size
    grad_accumulation_steps = GLOBAL_BATCH_SIZE // effective_batch_size

    if grad_accumulation_steps * effective_batch_size != GLOBAL_BATCH_SIZE:
        print(
            "ERROR: Global batch size ({}) is not divisible by effective batch size ({})".format(
                GLOBAL_BATCH_SIZE, effective_batch_size
            )
        )
        print(
            f"  Effective batch size = WORLD_SIZE ({world_size}) × PER_DEVICE_BATCH_SIZE ({per_device_batch_size}) = {effective_batch_size}"
        )
        sys.exit(1)

    if grad_accumulation_steps < 1:
        print(
            f"ERROR: Gradient accumulation steps ({grad_accumulation_steps}) must be at least 1"
        )
        sys.exit(1)

    print("Training Configuration:")
    print(f"  Target GPUs: {NUM_GPUS}")
    print(f"  Detected world size: {world_size}")
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

    rank = int(os.environ.get("RANK", "0"))
    is_main_process = rank == 0

    def _maybe_load_cached() -> Optional[Dataset]:
        if not cache_path.exists():
            return None
        try:
            dataset = load_from_disk(str(cache_path))
            print(f"Cached dataset loaded from {cache_path}: {len(dataset)} examples")
            if debug_mode:
                subset_size = min(1000, len(dataset))
                dataset = dataset.select(range(subset_size))
                print(f"Debug mode: Using subset of {len(dataset)} examples for testing")
            return dataset
        except Exception as e:
            print(f"Failed to load cached dataset (will recompute): {e}")
            return None

    if not refresh_cache:
        if not is_main_process:
            waited = 0
            while not cache_path.exists() and waited < 600:
                time.sleep(5)
                waited += 5
            cached = _maybe_load_cached()
            if cached is not None:
                return cached
        else:
            cached = _maybe_load_cached()
            if cached is not None and cache_path.stat().st_mtime > dataset_file.stat().st_mtime:
                return cached
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
    parser = argparse.ArgumentParser(description='TRL Training Script for Llama-3.2-1B SFT')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode with 1000 samples for testing')
    parser.add_argument('--refresh', action='store_true',
                       help='Force refresh of cached dataset (ignore existing cache)')
    parser.add_argument('--rccl-defaults', action='store_true',
                       help='Skip RCCL tuning knobs and keep library defaults')
    return parser.parse_args()

def main():
    """Main training function running under accelerate."""
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Set up signal handlers first
    setup_signal_handlers()

    args = parse_args()

    setup_environment(args)

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        log(f"Set default CUDA device to {torch.cuda.current_device()}")

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
    model_kwargs = {
        'torch_dtype': torch.bfloat16,
        'attn_implementation': 'flash_attention_2',
        'trust_remote_code': True,
        'low_cpu_mem_usage': True,
    }
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        **model_kwargs,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
        if hasattr(model.config, "router_aux_loss_coef"):
            model.config.router_aux_loss_coef = 0.001
        if hasattr(model.config, "output_router_logits"):
            model.config.output_router_logits = True
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
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=None,
        ddp_find_unused_parameters=False,
        remove_unused_columns=True,
        dataloader_drop_last=True,
        max_length=MAX_LEN,
        packing=True,
        logging_steps=1,
        save_steps=500,
        save_total_limit=3,
        optim="adamw_torch",  # placeholder; overridden with torchao AdamW8bit
        weight_decay=1e-4,
        report_to=["wandb"],
        run_name=OUT,
    )
    log(
        "Training args: gradient_checkpointing={}, max_length={}, packing={}, per_device_train_batch_size={}, grad_accumulation_steps={}".format(
            training_args.gradient_checkpointing,
            training_args.max_length,
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

    if AdamW8bit is None:
        raise ImportError(
            'torchao.optim.AdamW8bit is required to mirror the OpenRLHF optimizer; install torchao.'
        )

    trainer.optimizer_cls_and_kwargs = (
        AdamW8bit,
        {
            "lr": training_args.learning_rate,
            "betas": (training_args.adam_beta1, training_args.adam_beta2),
            "eps": training_args.adam_epsilon,
        },
    )
    log(
        "Configured torchao AdamW8bit optimizer (lr={}, betas={}, eps={})".format(
            training_args.learning_rate,
            (training_args.adam_beta1, training_args.adam_beta2),
            training_args.adam_epsilon,
        )
    )

    print("Starting training...")

    # Check for interruption before training
    if _training_interrupted:
        log("Training interrupted before starting")
        return

    try:
        trainer.train()
    except KeyboardInterrupt:
        log("Training interrupted by user")
        return
    except Exception as e:
        log(f"Training failed with error: {e}")
        raise

    # Check for interruption before saving
    if _training_interrupted:
        log("Training interrupted, skipping model save")
        return

    print("Saving final model...")
    try:
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        print(f"Training completed! Model saved to {output_dir}")
    except Exception as e:
        log(f"Failed to save model: {e}")
        raise


def run_with_accelerate():
    print("Starting TRL training with DeepSpeed ZeRO-3...")

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

    print(f"Launching accelerate with config {ACCELERATE_CONFIG} (command: {' '.join(cmd)})")
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
        gpu_count = detect_local_gpu_count()
        if gpu_count > 1:
            print(
                f"Detected {gpu_count} GPUs; launching via accelerate using {ACCELERATE_CONFIG}."
            )
            run_with_accelerate()
        else:
            print("Single GPU detected; running training loop directly without accelerate.")
            main()
