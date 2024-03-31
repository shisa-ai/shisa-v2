from transformers import AutoModelForCausalLM, AutoTokenizer

# Base Model
model = AutoModelForCausalLM.from_pretrained("ai21labs/Jamba-v0.1",
                                                     trust_remote_code=True)

# Load adapter
tokenizer = AutoTokenizer.from_pretrained("checkpoint-2416", trust_remote_code=True)

from safetensors import safe_open
checkpoint_dir = "checkpoint-2416"
adapter_path = checkpoint_dir + "/adapter_model.safetensors"

def load_lora_adjustments(lora_path):
    lora_adjustments = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            lora_adjustments[key] = f.get_tensor(key)
    return lora_adjustments

def apply_lora_adjustments(model, lora_adjustments):
    # Apply the LORA adjustments to the model.
    for name, param in model.named_parameters():
        if name in lora_adjustments:
            adjustment = lora_adjustments[name]
            param.data = param.data + adjustment

lora_adjustments = load_lora_adjustments(adapter_path)
apply_lora_adjustments(model, lora_adjustments)


# Chat Template (unused atm)
if tokenizer.chat_template:
	pass
else:
	# default to chatml
	tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

# System Prompt
PROMPT = 'あなたは役立つアシスタントです。'
chat = [{"role": "system", "content": PROMPT}]



# Basic inference
input_text = """<|im_start|>system 
You are GPT-4, a helpful assistant.
<|im_end|> 
<|im_start|>user 
最近、運動すれば、すぐにめっちゃくっちゃ汗かいちゃうんだけど、どうしたらいいですか？
<|im_end|> 
<|im_start|>assistant 
"""

input_ids = tokenizer(input_text, return_tensors='pt').to(model.device)["input_ids"]

outputs = model.generate(input_ids, max_new_tokens=256, temperature=0.0)

print(tokenizer.batch_decode([outputs[0][len(input_ids[0]):]]))
