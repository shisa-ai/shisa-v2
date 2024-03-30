from unsloth import FastLanguageModel
import torch
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
from transformers import AutoTokenizer, TrainingArguments
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM


max_seq_length = 4096 # Supports RoPE Scaling interally, so choose any!

# Load model in 8-bit precision
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    device_map="auto",
    llm_int8_skip_modules=["mamba"]
)

model = AutoModelForCausalLM.from_pretrained(
    "jamba",
    trust_remote_code=True,
    torch_dtype=torch.cuda.is_bf16_supported() and torch.bfloat16 or torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",
    quantization_config=quantization_config
)
# Extra bits: https://github.com/mlabonne/llm-course/blob/4dc551d702a28b94bdec8cade19110c5ed93a740/Fine_tune_a_Mistral_7b_model_with_DPO.ipynb#L471
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained("jamba")
# padding_side : "left"
# pad_token    : '<|pad|>'
# bos_token    : '<|startoftext|>'
# eos_token    : '<|endoftext|>'
# pad_token_id : 0
# bos_token_id : 1
# eos_token_id : 2

# bitsandbytes causes potential overflows w/ padding_side = 'left'?
# UserWarning: You passed a tokenizer with `padding_side` not equal to `right` to the SFTTrainer. This might lead to some unexpected behaviour due to overflow issues when training a model in half-precision. You might consider adding `tokenizer.padding_side = 'right'` to your code.
tokenizer.padding_side = 'right'

# No Chat Template, so we assign chatml
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

lora_config = LoraConfig(
    target_modules=["embed_tokens", "x_proj", "in_proj", "out_proj"],
    init_lora_weights=False
)

# https://github.com/JBAujogue/LLM-playground/blob/d72753e5825ea0599ffa6c8d3b53d6679e09f561/notebooks/7-finetune-llm-sft%E2%9C%A8.md?plain=1#L26

# Do I need this? Copied from Colab
model.add_adapter(lora_config, adapter_name="adapter_1")

from datasets import load_dataset
def convert_to_chatml_format(conversation):
    formatted_chat = []
    for message in conversation:
        role = "user" if message["from"] == "human" else "assistant"
        formatted_chat.append({"role": role, "content": message["value"]})
    return {"formatted_chat": tokenizer.apply_chat_template(formatted_chat, tokenize=False, add_generation_prompt=False)}

# See https://github.com/rodrigo-pedro/gemma-function-calling/blob/fe6416ea8de2150e2f58a8e329bf1a91935bbaf0/finetune.ipynb#L56
# https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing
# dataset = load_dataset("NTQAI/sharegpt-clean-ja", split="train")
dataset = load_dataset("augmxnt/ultra-orca-boros-en-ja-v1", split="train")
dataset = dataset.map(lambda x: convert_to_chatml_format(x["conversations"]))
print(dataset)

lora_config = LoraConfig(
    r=32,
    lora_alpha = 64,
    lora_dropout = 0,
    task_type="CAUSAL_LM",
    bias="none"
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "formatted_chat",
    max_seq_length = max_seq_length,
    neftune_noise_alpha=5,
    tokenizer = tokenizer,
    packing=True,
    args = TrainingArguments(
        num_train_epochs=3,
        learning_rate=5e-4,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        weight_decay = 0.01,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 42,
    ),
)

# Hmm by default should add special tokens?
# https://github.com/adithya-s-k/LLM-Alchemy-Chamber/blob/c50a8ac2f0078ed1e7a69427a3953fe58a825699/LLMs/Mistral-7b/notebooks_SFTTrainer%20TRL.ipynb#L394

# adamw_8bit is best: https://github.com/huggingface/transformers/issues/22101
# https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one

# https://github.com/centre-for-humanities-computing/danish-foundation-models/blob/f04f59b196ea322c6dd1f3cab036ef313eee786d/docs/blog/posts/finetune.md?plain=1#L156
# Log some GPU stats before we start the finetuning
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(
    f"You're using the {gpu_stats.name} GPU, which has {max_memory:.2f} GB of memory "
    f"in total, of which {start_gpu_memory:.2f}GB has been reserved already."
)

trainer_stats = trainer.train()

# Log some post-training GPU statistics
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(
    f"We ended up using {used_memory:.2f} GB GPU memory ({used_percentage:.2f}%), "
    f"of which {used_memory_for_lora:.2f} GB ({lora_percentage:.2f}%) "
    "was used for LoRa."
)

# Make sure we save
trainer.model.save_pretrained("final_checkpoint")
tokenizer.save_pretrained("final_checkpoint")
