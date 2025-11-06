#!/usr/bin/env bash
set -eo pipefail

export WANDB_ENTITY=augmxnt
export WANDB_PROJECT="shisa-v2.1"           # keep your old project
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_ASYNC_ERROR_HANDLING=1

# 8 x MI300X
export NUM_GPUS=${NUM_GPUS:-8}

# MODELS
ORIG_MODEL=${ORIG_MODEL:-"meta-llama/Llama-3.2-1B-Instruct"}

SFT_CKPT=${SFT_CKPT:-"121-llama3.2-1b-v2.1-sft"}
SFT_DATA="sft.shisa-v2.1.jsonl"
SFT_LR=${SFT_LR:-2.53e-5}
SFT_MBS=16

DPO_CKPT=${DPO_CKPT:-"122-llama3.2-1b-v2.1-dpo"}
# DPO_DATA="shisa-ai/shisa-v2-dpo405b-shuffled"
DPO_DATA="dpo.shisa-v2.1.jsonl"
DPO_LR=${DPO_LR:-3.54e-07}
DPO_MBS=8

main() {
  # run_sft
  run_dpo
}

run_sft() {
  echo "Starting SFT training on ${NUM_GPUS} GPUs..."
  deepspeed --num_gpus ${NUM_GPUS} --module openrlhf.cli.train_sft \
    --pretrain        "${ORIG_MODEL}" \
    --dataset         "${SFT_DATA}" \
    --input_key conversations \
    --apply_chat_template \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation flash_attn \
    --use_liger_kernel \
    --optimizer adamw_torchao_4bit \
    --train_batch_size       128 \
    --micro_train_batch_size ${SFT_MBS} \
    --max_len 8192 \
    --packing_samples \
    --max_epochs      3 \
    --learning_rate   ${SFT_LR} \
    --lr_warmup_ratio 0.03 \
    --save_steps      90443 \
    --save_path       "${SFT_CKPT}" \
    --logging_steps   1 \
    --eval_steps   -1 \
    --overlap_comm \
    --use_wandb       True \
    --wandb_org        augmxnt \
    --wandb_project    shisa-v2.1 \
    --wandb_run_name   "${SFT_CKPT}-8xMI300"
}

run_dpo() {
  echo "Starting DPO training on ${NUM_GPUS} GPUs..."
  deepspeed --num_gpus ${NUM_GPUS} --module openrlhf.cli.train_dpo \
    --pretrain        "${SFT_CKPT}" \
    --dataset         "${DPO_DATA}" \
    --chosen_key      chosen \
    --rejected_key    rejected \
    --apply_chat_template \
    --beta            0.1 \
    --bf16 \
    --zero_stage      3 \
    --gradient_checkpointing \
    --attn_implementation flash_attn \
    --use_liger_kernel \
    --optimizer adamw_torchao_8bit \
    --train_batch_size       64 \
    --micro_train_batch_size ${DPO_MBS} \
    --max_len         4096 \
    --packing_samples \
    --max_epochs      1 \
    --learning_rate   ${DPO_LR} \
    --lr_scheduler constant_with_warmup \
    --lr_warmup_ratio 0.03 \
    --save_steps      -1 \
    --save_path       "${DPO_CKPT}" \
    --logging_steps   1 \
    --overlap_comm \
    --use_wandb       True \
    --wandb_org        augmxnt \
    --wandb_project    shisa-v2.1 \
    --wandb_run_name   "${DPO_CKPT}-8xMI300"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
