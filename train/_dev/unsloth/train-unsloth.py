#!/usr/bin/env python

'''
Unsloth Collab - Alpaca + Llama-3 8b Unsloth 2x faster finetuning.ipynb
https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing

Unsloth Collab - Llama-3.1 8b + Unsloth 2x faster finetuning.ipynb
https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=QmUBVEnvCDJv
'''

from unsloth import FastLanguageModel 
from unsloth import is_bfloat16_supported
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from datasets import load_dataset
max_seq_length = 4096 # Supports RoPE Scaling interally, so choose any!
url = 'https://huggingface.co/datasets/augmxnt/ultra-orca-boros-en-ja-v1/resolve/main/dataset.parquet'
dataset = load_dataset("parquet", data_files = {"train" : url}, split = "train")


# WandB setup
from datetime import datetime
import sys
import wandb
try:
    log_file = sys.argv[1]
    run_name = log_file.split('shisa-v1-llama3-8b-qlora-unsloth-')[1].split('.log')[0]
except:
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

wandb.init(
    project="shisa-v2",
    entity="augmxnt",
    name=run_name
)

    
# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = False,
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


def formatting_prompts_func(example):
    # Convert the conversations to the format expected by the chat template
    messages = []
    for conv in example['conversations']:
        role = conv['from']
        content = conv['value']
        
        # Map the roles to the appropriate template roles
        if role == 'system':
            # We are going to skip training on the system role...
            pass
            # messages.append({"role": "system", "content": content})
        elif role == 'human':
            messages.append({"role": "user", "content": content})
        elif role == 'gpt':
            messages.append({"role": "assistant", "content": content})
    
    # Apply the chat template
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    return formatted_prompt


# https://huggingface.co/docs/trl/main/en/sft_trainer#train-on-completions-only - incompatible w/ packing
# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    # dataset_text_field = "instruction",
    formatting_func=formatting_prompts_func,
    # data_collator=collator,
    packing=True,
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        num_train_epochs=1.0,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 32,
        warmup_steps = 100, # 10
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 3407,
        include_tokens_per_second=True,
        weight_decay=0.01,
        lr_scheduler_type='linear',
        # Standard
        learning_rate=2e-4,
    ),
)
trainer.train()
