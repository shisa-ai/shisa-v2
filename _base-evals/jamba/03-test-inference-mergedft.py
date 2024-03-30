'''
Using the base model.
Load in 8-bit
'''

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                         llm_int8_skip_modules=["mamba"])
model = AutoModelForCausalLM.from_pretrained("merged_model",
                                             trust_remote_code=True,
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2",
                                             quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained("merged_model")

input_ids = tokenizer("In the recent Super Bowl LVIII,", return_tensors='pt').to(model.device)["input_ids"]

outputs = model.generate(input_ids, max_new_tokens=216)

print(tokenizer.batch_decode(outputs))


input_ids = tokenizer("In the recent Super Bowl LVIII,", return_tensors='pt').to(model.device)["input_ids"]

outputs = model.generate(input_ids, max_new_tokens=216)

print(tokenizer.batch_decode(outputs))



# Chatml
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

# You are an avid Pokemon fanatic.
prompt = "あなたは熱狂的なポケモンファンです。"
chat = [{"role": "system", "content": prompt}]

# Who is the most powerful Pokemon? Explain your choice.
user_input = "最強のポケモンは誰ですか？その選択理由を説明してください。"
chat.append({"role": "user", "content": user_input})

# Generate - add_generation_prompt to make sure it continues as assistant
inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt")
first_param_device = next(model.parameters()).device
inputs = inputs.to(first_param_device)
print("### apply_chat_template tokens:")
print(inputs)

with torch.no_grad():
    outputs = model.generate(
            inputs, 
            max_new_tokens=300
            )
new_tokens = outputs[0, inputs.size(1):]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
chat.append({"role": "assistant", "content": response})

print(outputs)
print(tokenizer.decode(outputs[0]))

