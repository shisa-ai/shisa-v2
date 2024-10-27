#!/bin/bash

### Env

# vLLM -tp8 hipBLASLt workaround
ulimit -n 1031072 

# Use CK FA
export VLLM_USE_TRITON_FLASH_ATTN=0

### Define an array of models with their GPU counts

## Done already
# llm_judge_wm23_Llama-3.1-405B-Instruct_outputs.jsonl
# llm_judge_wm23_Llama-3.1-70B-Instruct_outputs.jsonl
# llm_judge_wm23_Llama-3.1-8B-Instruct_outputs.jsonl
# llm_judge_wm23_Llama-3.1-Nemotron-70B-Instruct-HF_outputs.jsonl
# llm_judge_wm23_WizardLM-2-8x22B_outputs.jsonl

declare -a models=(
    "Qwen/Qwen2.5-32B-Instruct,1"
    "Qwen/Qwen2.5-72B-Instruct,1"
    "mistralai/Mistral-Nemo-Instruct-2407,1"
    "mistralai/Mistral-Large-Instruct-2407,1"
    "NousResearch/Hermes-3-Llama-3.1-405B,4"
    "microsoft/GRIN-MoE,1"
    "mistralai/Mixtral-8x7B-Instruct-v0.1,1"
    "mgoin/Nemotron-4-340B-Instruct-hf,4"
)

# Iterate over each model and execute the Python script
for model_entry in "${models[@]}"; do
    # Split the entry into model_name and gpus
    IFS=',' read -r model_name gpus <<< "$model_entry"

    echo "Processing model: $model_name with $gpus GPU(s)"
    
    # Call the Python script with the model_name
    VLLM_USE_TRITON_FLASH_ATTN=0 time python batched-bench.py "$model_name"

    # Optionally, add a delay or check for success before proceeding
    # sleep 10
done

echo "All models have been processed."
