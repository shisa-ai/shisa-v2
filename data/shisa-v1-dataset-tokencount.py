from datasets import load_dataset
from transformers import AutoTokenizer
import json
import sys


# Load dataset
dataset = load_dataset('augmxnt/ultra-orca-boros-en-ja-v1')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', trust_remote_code=True)


def evaluate(tokenizer, dataset):
    n_chars = 0
    n_tokens = 0
    
    for sample in dataset:
        # print(sample['conversations'])

        messages = []
        for conv in sample['conversations']:
            messages.append({'role': conv['from'], 'content': conv['value']})

        # print(messages)

        tokens = tokenizer.apply_chat_template(messages)
        text = tokenizer.apply_chat_template(messages, tokenize=False)

        # print(tokens)
        # print(text)

        n_tokens += len(tokens)
        n_chars += len(text)

        # print(n_tokens)
        # print(n_chars)
    
    try:
        print(f"Compression rate: {n_chars / n_tokens} chars / token ({n_chars} / {n_tokens})")
        return n_chars / n_tokens, n_tokens
    except ZeroDivisionError:
        return 0, 0


# Evaluate the tokenizer
avg_chars_per_token, total_tokens = evaluate(tokenizer, dataset['train'])
print(f"Average characters per token: {avg_chars_per_token}")
print(f"Total number of tokens: {total_tokens}")

