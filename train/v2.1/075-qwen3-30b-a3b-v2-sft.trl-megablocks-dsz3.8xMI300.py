#!/usr/bin/env python3

"""
TRL + MegaBlocks training script for Qwen3-30B-A3B-v2 SFT with DeepSpeed ZeRO-3 on 8x MI300X.
DeepSpeed version for better NCCL stability compared to FSDP2.
"""

import os

# Disable torch.compile to prevent hanging on first run with hundreds of worker processes
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

# ============================================================================
# Configuration - adjust these for different systems
# ============================================================================

# Hardware configuration specific to the MI300X box
NUM_GPUS = 8  # 8x MI300X in the node (HIP/RCCL stack)

# Use DeepSpeed ZeRO-3 config instead of FSDP2
ACCELERATE_CONFIG = "accelerate_config.deepspeed.yaml"

# Model and data configuration
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DATASET_PATH = "./sft.shisa-v2.1.jsonl"  # Generated from generate-new-sft.py
OUT = "075-qwen3-30b-a3b-v2-sft-trl-megablocks-dsz3"
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

    # Torch Inductor optimizations - limit worker processes
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "4"  # Limit compile workers per process
    os.environ["TORCHINDUCTOR_WORKER_COUNT"] = "4"     # Max 4 workers instead of 32
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"     # Reduce autotuning overhead
    os.environ["TORCH_COMPILE_DEBUG"] = "0"            # Disable debug for faster compilation

    # Prevent runaway processes
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_compile_cache"
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"

    # ROCm/RCCL optimizations for MI300X - Conservative settings for DeepSpeed ZeRO-3
    os.environ["RCCL_DEBUG"] = "WARN"  # Less verbose than FSDP2 version
    os.environ["RCCL_NET_GDR_LEVEL"] = "3"  # Re-enable GPU Direct RDMA with DeepSpeed
    os.environ["RCCL_TREE_THRESHOLD"] = "0"  # Use ring algorithm (more stable)
    os.environ["RCCL_LL_THRESHOLD"] = "0"  # Disable low-latency for bandwidth
    os.environ["RCCL_BUFFSIZE"] = "4194304"  # 4MB buffer size for DeepSpeed
    os.environ["RCCL_NTHREADS"] = "512"  # Match MI300X compute units
    os.environ["RCCL_MAX_NCHANNELS"] = "16"  # Optimize for 8x MI300X topology
    os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"  # Fine-grain memory access
    os.environ["HIP_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Explicit GPU visibility

    # DeepSpeed specific optimizations
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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
                "deepspeed",
                "zero_stage",
                "num_processes",
            )
            lines = [ln.strip() for ln in txt.splitlines() if any(k in ln for k in keys)]
            log("Accelerate config summary:")
            for ln in lines:
                log(f"  {ln}")
        except Exception as e:
            log(f"Could not read {ACCELERATE_CONFIG}: {e}")
    else:
        log(f"{ACCELERATE_CONFIG} not found")


def register_megablocks_kernel() -> bool:
    """Register MegaBlocks kernel mapping for Qwen2 MoE blocks if possible."""
    repo_root = Path(__file__).resolve().parent / "megablocks.kernels-community"
    if not repo_root.exists():
        log(f"MegaBlocks kernel repo not found at {repo_root}")
        return False

    try:
        from kernels import LocalLayerRepository, Mode, register_kernel_mapping, use_kernel_forward_from_hub
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock  # type: ignore
    except Exception as exc:
        log(f"kernels mapping import skipped: {exc}")
        return False

    try:
        # Mark the Hugging Face block as extensible by kernels
        use_kernel_forward_from_hub("Qwen3MoeSparseMoeBlock")(Qwen3MoeSparseMoeBlock)
    except Exception as exc:
        log(f"Failed to prepare Qwen3 block for kernels: {exc}")
        return False

    device_key = "rocm" if torch.version.hip is not None else "cuda"
    repo = LocalLayerRepository(
        repo_path=repo_root,
        package_name="megablocks",
        layer_name="MegaBlocksMoeMLPWithSharedExpert",
    )

    try:
        register_kernel_mapping(
            {
                "Qwen3MoeSparseMoeBlock": {
                    device_key: {
                        Mode.TRAINING: repo,
                        Mode.TRAINING | Mode.TORCH_COMPILE: repo,
                    }
                }
            }
        )
    except Exception as exc:
        log(f"register_kernel_mapping failed: {exc}")
        return False

    log("MegaBlocks kernel mapping registered")
    return True


def kernelize_with_megablocks(model: Any) -> bool:
    """Attempt to kernelize the loaded model with MegaBlocks."""
    try:
        from kernels import Device, Mode, kernelize
    except Exception as exc:
        log(f"kernels kernelize import failed: {exc}")
        return False

    device_key = "rocm" if torch.version.hip is not None else "cuda"
    try:
        kernelize(model, mode=Mode.TRAINING, device=Device(type=device_key))
        log("Applied MegaBlocks kernels via kernelize()")
        return True
    except Exception as exc:
        log(f"kernelize() failed: {exc}")
        return False


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


def validate_batch_configuration():
    """Validate batch size configuration for multi GPU"""
    # Conservative per-device batch size for 192GB MI300X with MoE layers
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

    print("Training Configuration (DeepSpeed ZeRO-3):")
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
    parser = argparse.ArgumentParser(description='TRL MegaBlocks Training Script for Qwen3-30B-A3B-v2 SFT with DeepSpeed ZeRO-3')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode with 1000 samples for testing')
    parser.add_argument('--refresh', action='store_true',
                       help='Force refresh of cached dataset (ignore existing cache)')
    return parser.parse_args()

def main():
    """Main training function running under accelerate with DeepSpeed ZeRO-3."""
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Set up signal handlers first
    setup_signal_handlers()

    args = parse_args()

    setup_environment()

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        log(f"Set default CUDA device to {torch.cuda.current_device()}")

    # DISABLE MegaBlocks kernels to avoid torch.compile issues
    # kernels_registered = register_megablocks_kernel()
    log("MegaBlocks kernels DISABLED to avoid torch.compile CPU explosion")

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
        use_kernels=False,
    )
    # DISABLED: MegaBlocks kernels causing torch.compile explosion
    # if kernels_registered:
    #     if not kernelize_with_megablocks(model):
    #         log("MegaBlocks kernelize fallback to vanilla implementation")
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
        gradient_checkpointing=True,  # Enable for ZeRO-3
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
        optim="adamw_torch_fused",  # Better optimizer for ZeRO-3
        weight_decay=1e-4,
        report_to=["wandb"],
        run_name=OUT,
    )
    log(
        "Training args (DeepSpeed ZeRO-3): gradient_checkpointing={}, max_length={}, packing={}, per_device_train_batch_size={}, grad_accumulation_steps={}".format(
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

    print("Starting training with DeepSpeed ZeRO-3...")

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
    print("Starting TRL MegaBlocks training with DeepSpeed ZeRO-3...")

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

    print(f"Launching accelerate with DeepSpeed config {ACCELERATE_CONFIG} (command: {' '.join(cmd)})")
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
        if _env_flag("MEGABLOCKS_FORCE_ACCELERATE"):
            run_with_accelerate()
        elif _env_flag("MEGABLOCKS_NO_ACCELERATE"):
            print("MEGABLOCKS_NO_ACCELERATE set; running training loop directly without accelerate.")
            main()
        else:
            gpu_count = detect_local_gpu_count()
            if gpu_count > 1:
                print(
                    f"Detected {gpu_count} GPUs; launching via accelerate using DeepSpeed ZeRO-3 config {ACCELERATE_CONFIG}."
                )
                run_with_accelerate()
            else:
                print("Single GPU detected; running training loop directly without accelerate.")
                main()
