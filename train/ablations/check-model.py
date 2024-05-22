import os
from   pprint import pprint
import sys

if len(sys.argv) < 2 or not sys.argv[1]:
  print("Error: No model provided.")
  sys.exit(1)

model_name = sys.argv[1]

'''
if not os.path.exists(model_name):
  print(f"Error: The path '{model_name}' does not exist.")
  sys.exit(1)
'''

# Otherwise we have to wait around
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            low_cpu_mem_usage=True,
            device_map="auto",
        )
tokenizer = AutoTokenizer.from_pretrained(model_name)


chat = [
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
