#!/usr/bin/env bash
set -eo pipefail

export WANDB_ENTITY=augmxnt
export WANDB_PROJECT="shisa-v2.1"           # keep your old project
export HF_HUB_ENABLE_HF_TRANSFER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# MODELS
ORIG_MODEL=${ORIG_MODEL:-"google/gemma-3-270m-it"}

SFT_CKPT=${SFT_CKPT:-"070-gemma3-270m-v2new-muon-sft"}
SFT_DATA="sft.shisa-v2.jsonl"
SFT_LR=${SFT_LR:-5.44e-5}
SFT_MBS=8

DPO_CKPT=${DPO_CKPT:-"071-gemma3-270m-v2new-muon-dpo405b"}
DPO_DATA="shisa-ai/shisa-v2-dpo405b-shuffled"
DPO_LR=${DPO_LR:-6.80e-7}
DPO_MBS=4

NUM_GPUS=1

main() {
  # Run training stages
  run_sft
  run_dpo
}


run_sft() {
  echo "Starting SFT training..."
  deepspeed --num_gpus $NUM_GPUS --module openrlhf.cli.train_sft \
    --pretrain        "$ORIG_MODEL" \
    --dataset         "$SFT_DATA" \
    --input_key conversations \
    --apply_chat_template \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation eager \
    --use_liger_kernel \
    --optimizer muon_aux_adam \
    --train_batch_size       128 \
    --micro_train_batch_size $SFT_MBS \
    --max_len 8192 \
    --packing_samples \
    --max_epochs      3 \
    --learning_rate   $SFT_LR \
    --lr_warmup_ratio 0.03 \
    --save_steps      90443 \
    --save_path       "$SFT_CKPT" \
    --logging_steps   1 \
    --eval_steps   -1 \
    --overlap_comm \
    --use_wandb       True \
    --wandb_org        augmxnt \
    --wandb_project    shisa-v2.1 \
    --wandb_run_name   "$SFT_CKPT"
}

run_dpo() {
  echo "Starting DPO training..."
  deepspeed --num_gpus $NUM_GPUS --module openrlhf.cli.train_dpo \
    --pretrain        "$SFT_CKPT" \
    --dataset         "$DPO_DATA" \
    --chosen_key      chosen \
    --rejected_key    rejected \
    --apply_chat_template \
    --beta            0.1 \
    --bf16 \
    --zero_stage      3 \
    --gradient_checkpointing \
    --attn_implementation eager \
    --use_liger_kernel \
    --optimizer muon_aux_adam \
    --train_batch_size       64 \
    --micro_train_batch_size $DPO_MBS \
    --max_len         2048 \
    --packing_samples \
    --max_epochs      1 \
    --learning_rate   $DPO_LR \
    --lr_scheduler constant_with_warmup \
    --lr_warmup_ratio 0.03 \
    --save_steps      -1 \
    --save_path       "$DPO_CKPT" \
    --logging_steps   1 \
    --overlap_comm \
    --use_wandb       True \
    --wandb_org        augmxnt \
    --wandb_project    shisa-v2.1 \
    --wandb_run_name   "$DPO_CKPT"
}

# Call main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
