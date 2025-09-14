#!/usr/bin/env bash
set -eo pipefail

# Configuration - adjust these for different systems
export NUM_GPUS=2
export TOTAL_VRAM=192  # 2x 96GB PRO 6000s
export GPU_DEVICES="0,1"

# Environment setup
export WANDB_ENTITY=augmxnt
export WANDB_PROJECT="shisa-v2.1"
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

# Model and data configuration
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
DATASET_PATH="./sft.shisa-v2.jsonl"  # Generated from generate-new-sft.py
OUT="072-qwen3-30b-a3b-v2-sft-trl-megablocks"

# Training hyperparameters (scale with GPU count)
GLOBAL_BATCH_SIZE=128
LR=1.63e-5  # Based on GBS=128 from original script
MAX_LEN=8192
EPOCHS=3

# Calculate per-device batch size based on GPU count and memory
# For 96GB GPUs with MoE, conservative batch size
PER_DEVICE_BATCH_SIZE=4

# Calculate and validate gradient accumulation steps
EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * PER_DEVICE_BATCH_SIZE))
GRAD_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / EFFECTIVE_BATCH_SIZE))

# Validation checks
if [ $((GRAD_ACCUMULATION_STEPS * EFFECTIVE_BATCH_SIZE)) -ne $GLOBAL_BATCH_SIZE ]; then
    echo "ERROR: Global batch size ($GLOBAL_BATCH_SIZE) is not evenly divisible by effective batch size ($EFFECTIVE_BATCH_SIZE)"
    echo "  Effective batch size = NUM_GPUS ($NUM_GPUS) × PER_DEVICE_BATCH_SIZE ($PER_DEVICE_BATCH_SIZE) = $EFFECTIVE_BATCH_SIZE"
    echo "  Try adjusting PER_DEVICE_BATCH_SIZE or GLOBAL_BATCH_SIZE to make them compatible"
    echo ""
    echo "Suggested fixes:"
    echo "  - Change GLOBAL_BATCH_SIZE to: $((GRAD_ACCUMULATION_STEPS * EFFECTIVE_BATCH_SIZE)) (rounds down)"
    echo "  - Or change GLOBAL_BATCH_SIZE to: $(((GRAD_ACCUMULATION_STEPS + 1) * EFFECTIVE_BATCH_SIZE)) (rounds up)"
    exit 1
fi

if [ $GRAD_ACCUMULATION_STEPS -lt 1 ]; then
    echo "ERROR: Gradient accumulation steps ($GRAD_ACCUMULATION_STEPS) must be at least 1"
    echo "  Your per-device batch size ($PER_DEVICE_BATCH_SIZE) × num GPUs ($NUM_GPUS) = $EFFECTIVE_BATCH_SIZE"
    echo "  This exceeds your global batch size ($GLOBAL_BATCH_SIZE)"
    echo "  Try reducing PER_DEVICE_BATCH_SIZE or increasing GLOBAL_BATCH_SIZE"
    exit 1
fi

echo "Training Configuration:"
echo "  GPUs: $NUM_GPUS"
echo "  Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "  Per Device Batch Size: $PER_DEVICE_BATCH_SIZE"
echo "  Effective Batch Size: $EFFECTIVE_BATCH_SIZE"
echo "  Gradient Accumulation Steps: $GRAD_ACCUMULATION_STEPS"
echo "  Learning Rate: $LR"
echo ""

# Create the TRL training script
cat > train_trl_megablocks.py << 'EOF'
import os
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

def main():
    # Model configuration
    model_id = os.environ.get("MODEL", "Qwen/Qwen3-30B-A3B")
    dataset_path = os.environ.get("DATASET_PATH", "./sft.shisa-v2.jsonl")
    output_dir = f"/data/outputs/{os.environ.get('OUT', 'trl-megablocks-output')}"
    
    # Training parameters from environment
    per_device_batch_size = int(os.environ.get("PER_DEVICE_BATCH_SIZE", "4"))
    grad_accumulation_steps = int(os.environ.get("GRAD_ACCUMULATION_STEPS", "16"))
    learning_rate = float(os.environ.get("LR", "1.63e-5"))
    max_length = int(os.environ.get("MAX_LEN", "8192"))
    epochs = int(os.environ.get("EPOCHS", "3"))
    
    print(f"Loading tokenizer from {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_kernels=True,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto",
    )
    
    # Load and prepare dataset
    print(f"Loading dataset from {dataset_path}")
    if dataset_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=dataset_path, split='train')
    else:
        dataset = load_dataset(dataset_path, split='train')
    
    print(f"Dataset loaded: {len(dataset)} examples")
    
    # Configure training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        
        # Memory optimizations
        bf16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        
        # MoE specific settings
        model_init_kwargs={"output_router_logits": True},
        aux_loss_coef=0.001,  # From original config
        
        # Sequence length
        max_seq_length=max_length,
        packing=True,
        
        # Logging and saving
        logging_steps=1,
        save_steps=500,
        eval_steps=500,
        save_total_limit=3,
        
        # Optimizer
        optim="paged_adamw_8bit",
        weight_decay=1e-4,
        
        # Reporting
        report_to=["wandb"],
        run_name=os.environ.get("OUT", "trl-megablocks"),
        
        # DeepSpeed/FSDP - will be configured via accelerate
        # deepspeed="/path/to/deepspeed_config.json",
        
        # Remove deprecated arguments
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        formatting_func=None,  # Dataset already formatted
        data_collator=None,    # Use default
        dataset_text_field="conversations",  # Field containing conversation data
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Training completed! Model saved to {output_dir}")

if __name__ == "__main__":
    main()
EOF

# Export training parameters for the Python script
export MODEL="$MODEL"
export DATASET_PATH="$DATASET_PATH"
export OUT="$OUT"
export PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE"
export GRAD_ACCUMULATION_STEPS="$GRAD_ACCUMULATION_STEPS"
export LR="$LR"
export MAX_LEN="$MAX_LEN"
export EPOCHS="$EPOCHS"

# Run training with accelerate
echo "Starting TRL MegaBlocks training..."
accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    train_trl_megablocks.py

echo "Training completed!"

# Cleanup
rm -f train_trl_megablocks.py
