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

print()
print('Chat Template:')
print(tokenizer.chat_template)
print()
print('---')
print()

print(tokenizer.apply_chat_template(messages, tokenize=False))

print('===')

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

messages = [
    {'role': 'system', 'content': 'This is the system prompt.'},
    {'role': 'user', 'content': 'This is the first user input.'},
    {'role': 'assistant', 'content': 'This is the first assistant response.'},
    {'role': 'user', 'content': 'This is the second user input.'},
]

print()
print('Chat Template:')
print(tokenizer.chat_template)
print()
print('---')
print()

print(tokenizer.apply_chat_template(messages, tokenize=False))
