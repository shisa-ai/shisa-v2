from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("augmxnt/shisa-7b-v1")
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
tokenizer = AutoTokenizer.from_pretrained("shisa-ai/shisa-llama3-8b-v1")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

messages = [
    {'role': 'system', 'content': 'This is the system prompt.'},
    {'role': 'user', 'content': 'This is the first user input.'},
    {'role': 'assistant', 'content': 'This is the first assistant response.'},
    {'role': 'user', 'content': 'This is the second user input.'},
]

tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

print()
print('Chat Template:')
print(tokenizer.chat_template)
print()
print('---')
print()

print(tokenizer.apply_chat_template(messages, tokenize=False))
