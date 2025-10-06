#!/usr/bin/env bash
set -eo pipefail

export WANDB_ENTITY=augmxnt
export WANDB_PROJECT="shisa-v2.1"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Hardware / launch defaults for 8× MI300X node
export NUM_GPUS=${NUM_GPUS:-8}
export DEEPSPEED_HOSTFILE=${DEEPSPEED_HOSTFILE:-}

# Models
ORIG_MODEL=${ORIG_MODEL:-"meta-llama/Llama-3.2-1B-Instruct"}

SFT_CKPT=${SFT_CKPT:-"089-llama3.2-1b-v2.1-sft"}
SFT_DATA="sft.shisa-v2.1.jsonl"
SFT_LR=${SFT_LR:-2.83e-5}
SFT_MBS=${SFT_MBS:-16}
SFT_GBS=${SFT_GBS:-128}

DPO_CKPT=${DPO_CKPT:-"090-llama3.2-1b-v2.1-dpo"}
DPO_DATA="dpo.shisa-v2.1.jsonl"
DPO_LR=${DPO_LR:-3.54e-7}
DPO_MBS=${DPO_MBS:-8}
DPO_GBS=${DPO_GBS:-64}

main() {
  run_sft
  run_dpo
}

run_sft() {
  echo "Starting SFT training on ${NUM_GPUS} GPUs (GBS=${SFT_GBS}, MBS=${SFT_MBS})..."
  deepspeed \
    --num_gpus ${NUM_GPUS} \
    ${DEEPSPEED_HOSTFILE:+--hostfile ${DEEPSPEED_HOSTFILE}} \
    --module openrlhf.cli.train_sft \
    --pretrain        "${ORIG_MODEL}" \
    --dataset         "${SFT_DATA}" \
    --input_key       conversations \
    --apply_chat_template \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --use_liger_kernel \
    --optimizer adamw_torchao_8bit \
    --train_batch_size       ${SFT_GBS} \
    --micro_train_batch_size ${SFT_MBS} \
    --max_len 8192 \
    --packing_samples \
    --max_epochs      3 \
    --learning_rate   ${SFT_LR} \
    --lr_warmup_ratio 0.03 \
    --save_steps      90443 \
    --save_path       "${SFT_CKPT}" \
    --logging_steps   1 \
    --eval_steps      -1 \
    --overlap_comm \
    --use_wandb       True \
    --wandb_org        augmxnt \
    --wandb_project    ${WANDB_PROJECT} \
    --wandb_run_name   "${SFT_CKPT}"
}

run_dpo() {
  echo "Starting DPO training on ${NUM_GPUS} GPUs (GBS=${DPO_GBS}, MBS=${DPO_MBS})..."
  deepspeed \
    --num_gpus ${NUM_GPUS} \
    ${DEEPSPEED_HOSTFILE:+--hostfile ${DEEPSPEED_HOSTFILE}} \
    --module openrlhf.cli.train_dpo \
    --pretrain        "${SFT_CKPT}" \
    --dataset         "${DPO_DATA}" \
    --chosen_key      chosen \
    --rejected_key    rejected \
    --apply_chat_template \
    --beta            0.1 \
    --bf16 \
    --zero_stage      3 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --use_liger_kernel \
    --optimizer adamw_torchao_8bit \
    --train_batch_size       ${DPO_GBS} \
    --micro_train_batch_size ${DPO_MBS} \
    --max_len         2048 \
    --packing_samples \
    --max_epochs      1 \
    --learning_rate   ${DPO_LR} \
    --lr_scheduler    constant_with_warmup \
    --lr_warmup_ratio 0.03 \
    --save_steps      -1 \
    --save_path       "${DPO_CKPT}" \
    --logging_steps   1 \
    --overlap_comm \
    --use_wandb       True \
    --wandb_org        augmxnt \
    --wandb_project    ${WANDB_PROJECT} \
    --wandb_run_name   "${DPO_CKPT}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
