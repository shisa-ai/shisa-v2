#!/usr/bin/env bash
set -eo pipefail

export WANDB_ENTITY=augmxnt
export WANDB_PROJECT="shisa-v2.1"           # keep your old project
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE=0

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# MODELS
ORIG_MODEL=${ORIG_MODEL:-"Qwen/Qwen3-30B-A3B-Instruct-2507"}

SFT_CKPT=${SFT_CKPT:-"080-qwen3-30b-a3b-v2.1-sft.8xMI300X.dsz3"}
SFT_DATA="sft.shisa-v2.1.jsonl"
SFT_LR=${SFT_LR:-1.63e-5}
SFT_MBS=8

DPO_CKPT=${DPO_CKPT:-"081-qwen3-30b-a3b-v2.1-dpo.8xMI300X.dsz3"}
DPO_DATA="dpo.shisa-v2.1.jsonl"
DPO_LR=${DPO_LR:-2.04e-7}
DPO_MBS=8

main() {
  # Run training stages
  run_sft
  run_dpo
}


run_sft() {
  echo "Starting SFT training..."
  deepspeed --num_gpus 8 --module openrlhf.cli.train_sft \
    --pretrain        "$ORIG_MODEL" \
    --dataset         "$SFT_DATA" \
    --input_key conversations \
    --apply_chat_template \
    --bf16 \
    --zero_stage      3 \
    --zpg 2 \
    --aux_loss_coef  0.001 \
    --optimizer  adamw_torchao_4bit \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --use_liger_kernel \
    --train_batch_size       128 \
    --micro_train_batch_size $SFT_MBS \
    --grad_accum_dtype bf16 \
    --max_len 8192 \
    --packing_samples \
    --max_epochs      3 \
    --learning_rate   $SFT_LR \
    --lr_warmup_ratio 0.05 \
    --save_path       "/data/outputs/$SFT_CKPT" \
    --ckpt_path       "/data/checkpoint/$SFT_CKPT" \
    --save_hf_ckpt \
    --save_steps      5652 \
    --logging_steps   1 \
    --eval_steps -1 \
    --eval_dataset "" \
    --overlap_comm \
    --use_wandb       True \
    --wandb_org        "$WANDB_ENTITY" \
    --wandb_project    "$WANDB_PROJECT" \
    --wandb_run_name   "$SFT_CKPT"
}

run_dpo() {
  echo "Starting DPO training..."
  deepspeed --module openrlhf.cli.train_dpo \
    --pretrain        "/data/outputs/$SFT_CKPT" \
    --dataset         "$DPO_DATA" \
    --chosen_key      chosen \
    --rejected_key    rejected \
    --apply_chat_template \
    --bf16 \
    --zero_stage      3 \
    --zpg 2 \
    --aux_loss_coef  0.001 \
    --optimizer  adamw_torchao_4bit \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --use_liger_kernel \
    --train_batch_size       64 \
    --micro_train_batch_size $DPO_MBS \
    --grad_accum_dtype bf16 \
    --max_len 4096 \
    --packing_samples \
    --max_epochs      1 \
    --learning_rate   $DPO_LR \
    --lr_scheduler constant_with_warmup \
    --lr_warmup_ratio 0.03 \
    --save_path       "/data/outputs/$DPO_CKPT" \
    --logging_steps   1 \
    --eval_steps -1 \
    --eval_dataset "" \
    --overlap_comm \
    --use_wandb       True \
    --wandb_org        "$WANDB_ENTITY" \
    --wandb_project    "$WANDB_PROJECT" \
    --wandb_run_name   "$DPO_CKPT"
}

# Call main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
