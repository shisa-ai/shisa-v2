# Download a single sample file from CulturaX dataset.
from   datasets import Dataset
from   epochraft import CheckpointableDataset
import glob
from   huggingface_hub import snapshot_download
import json
import os
import pandas as pd
import sys
from   transformers import AutoTokenizer, LlamaTokenizer


### Settings
CULTURAX_DS = "en_part_00004"
OUTPUT = "tokenizer-eval-en"
TOKENIZERS = [
    ("augmxnt/shisa-7b-v1", AutoTokenizer, "shisa-v1"),
    ("ai21labs/Jamba-v0.1", AutoTokenizer, "Jamba"),
    ("databricks/dbrx-instruct", AutoTokenizer, "DBRX"),
    ("tokyotech-llm/Swallow-MX-8x7b-NVE-v0.1", AutoTokenizer, "Swallow MX NVE"),
    ("01-ai/Yi-34B-200K", AutoTokenizer, "Yi 34B 200K"),
    ("OrionStarAI/Orion-14B-Base", AutoTokenizer, "Orion 14B"),
    ("CohereForAI/c4ai-command-r-plus", AutoTokenizer, "Cohere Command-R+"),
    ("NousResearch/Meta-Llama-3-8B", AutoTokenizer, "Llama 3"),
    ("Rakuten/RakutenAI-7B", AutoTokenizer, "RakutenAI-7B"),
    ("01-ai/Yi-1.5-34B-Chat", AutoTokenizer, "Yi 1.5"),
    ("tiiuae/falcon-11B", AutoTokenizer, "Falcon 2"),
    ("Xenova/gpt-4", AutoTokenizer, "GPT-4"),
    ("Xenova/gpt-4o", AutoTokenizer, "GPT-4o"),
    ("google/gemma-7b", AutoTokenizer, "Gemma 7B"),
    ("stockmark/stockmark-100b", AutoTokenizer, "Stockmark 100B"),
]


### Main
def main():
    # Download a single sample file from CulturaX dataset.
    if not os.path.exists(f"{CULTURAX_DS}.jsonl"):
        print(f"Processing CulturaX: {CULTURAX_DS} ...")
        snapshot_download(repo_id='uonlp/CulturaX', local_dir='CulturaX', allow_patterns=[f'*{CULTURAX_DS}.parquet'], repo_type='dataset')
        dataset = Dataset.from_parquet(f'CulturaX/{CULTURAX_DS[:2]}/{CULTURAX_DS}.parquet')
        dataset.to_json(f'{CULTURAX_DS}.jsonl')

    cache_file = f'{OUTPUT}.json'
    cache = load_cache(cache_file)

    result = pd.DataFrame(
        [
            generate_row(*args, cache)
            for args in TOKENIZERS
        ]
    )

    print()
    print('===')
    print(result)
    result.to_markdown(f'{OUTPUT}.md')


#### FUNCTIONS ###

### Cache
def load_cache(cache_file):
    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_cache(cache_file, cache):
    with open(cache_file, 'w') as f:
        json.dump(cache, f)


def evaluate(tokenizer):
    dataset = CheckpointableDataset.from_files(f"{CULTURAX_DS}.jsonl").tokenize(tokenizer, parallel=False).take(50000)
    n_chars = 0
    n_tokens = 0
    for sample in dataset:
        n_chars += len(sample["text"])
        n_tokens += len(sample["input_ids"])
    print(f"Compression rate: {n_chars / n_tokens} chars / token ({n_chars} / {n_tokens})")
    try:
        print(f"Compression rate: {n_chars / n_tokens} chars / token ({n_chars} / {n_tokens})")
        return n_chars / n_tokens
    except:
        return 0


def generate_row(tokenizer_url, tokenizer_cls, tokenizer_name, cache):
    print()
    print(tokenizer_name)
    print('===')
    if tokenizer_url in cache:
        print("> found in cache")
        return cache[tokenizer_url]

    tokenizer = tokenizer_cls.from_pretrained(tokenizer_url, trust_remote_code=True)
    if 'custom-tokenizer' in tokenizer_url:
        tokenizer = tokenizer_cls.from_pretrained(tokenizer_url, use_fast=True, trust_remote_code=True)
    '''
    return {
        "日本語LLM": tokenizer_name,
        "トークナイザ": tokenizer_url,
        "語彙数": tokenizer.vocab_size,
        "1トークンあたりの平均文字数": evaluate(tokenizer)
    }
    '''
    result = {
        "LLM": tokenizer_name,
        "Tokenizer": tokenizer_url,
        "Vocab Size": tokenizer.vocab_size,
        "Avg Char/Token": evaluate(tokenizer)
    }

    cache[tokenizer_url] = result
    cache_file = f'{OUTPUT}.json'
    save_cache(cache_file, cache)

    return result


if __name__ == "__main__":
    main()
