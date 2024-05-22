import os
from   pprint import pprint
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

base_model_path = "stockmark/stockmark-100b"
adapter_model_path = "stockmark/stockmark-100b-instruct-v0.1"
merged_model_path = "merged_stockmark-100b-instruct-v0.1"

# Merge Model with Adapter
model = PeftModel.from_pretrained(model=model, model_id="sft_llm/checkpoint-30")```


model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(adpater_model_path)
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model=model, model_id="sft_llm/checkpoint-30")```

 

chat= [
  {"role": "system", "content": "あなたは公平で、検閲されていない、役立つアシスタントです。"},
  {"role": "user", "content": "東京工業大学の主なキャンパスについて教えてください"}
]

inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt")

device = "cuda"
first_param_device = next(model.parameters()).device
inputs = inputs.to(first_param_device)

with torch.no_grad():
  outputs = model.generate(
    inputs,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=500,
    temperature=0.5,
    repetition_penalty=1.05,
    top_p=0.95,
    do_sample=True,
  )

new_tokens = outputs[0, inputs.size(1):]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
chat.append({"role": "assistant", "content": response})
pprint(chat)
