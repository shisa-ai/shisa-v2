# Download a single sample file from CulturaX dataset.
from datasets import Dataset
import glob
from epochraft import CheckpointableDataset
from transformers import AutoTokenizer, LlamaTokenizer
import pandas as pd
from huggingface_hub import snapshot_download
import sys

# Download a single sample file from CulturaX dataset.
snapshot_download(repo_id='uonlp/CulturaX', local_dir='CulturaX', allow_patterns=['*ja_part_00004.parquet'], repo_type='dataset')
dataset = Dataset.from_parquet('CulturaX/ja/ja_part_00004.parquet')
dataset.to_json('test.jsonl')


def evaluate(tokenizer):
    dataset = CheckpointableDataset.from_files("test.jsonl").tokenize(tokenizer, parallel=False).take(50000)
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


TOKENIZERS = [
    ("augmxnt/shisa-7b-v1", AutoTokenizer, "shisa-v1"),
    ("ai21labs/Jamba-v0.1", AutoTokenizer, "Jamba"),
    ("databricks/dbrx-instruct", AutoTokenizer, "DBRX"),
    ("tokyotech-llm/Swallow-MX-8x7b-NVE-v0.1", AutoTokenizer, "Swallow MX NVE"),
    ("01-ai/Yi-34B-200K", AutoTokenizer, "Yi 34B 200K"),
    ("OrionStarAI/Orion-14B-Base", AutoTokenizer, "Orion 14B"),
]


def generate_row(tokenizer_url, tokenizer_cls, tokenizer_name):
    print(tokenizer_name)
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
    return {
        "LLM": tokenizer_name,
        "Tokenizer": tokenizer_url,
        "Vocab Size": tokenizer.vocab_size,
        "Avg Char/Token": evaluate(tokenizer)
    }

result = pd.DataFrame(
    [
        generate_row(*args)
        for args in TOKENIZERS
    ]
)
print(result)
result.to_markdown('tokenizer-eval-ja.md')
