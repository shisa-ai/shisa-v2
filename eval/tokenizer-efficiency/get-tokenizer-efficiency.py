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

TOKENIZERS = [
    ("augmxnt/shisa-7b-v1", AutoTokenizer, "shisa-v1"),
    ("ai21labs/Jamba-v0.1", AutoTokenizer, "Jamba"),
    ("databricks/dbrx-instruct", AutoTokenizer, "DBRX"),
    ("deepseek-ai/DeepSeek-V3-0324", AutoTokenizer, "DeepSeek V3"),
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
    ("microsoft/Phi-3-medium-128k-instruct",  AutoTokenizer, "Microsoft Phi 3"),
    ("microsoft/Phi-4",  AutoTokenizer, "Microsoft Phi 4"),
    ("mistralai/Mistral-7B-v0.3",  AutoTokenizer, "Mistral v0.3"),
    ("mistralai/Mistral-Nemo-Instruct-2407", AutoTokenizer, "Mistral Nemo (Tekken)"),
    ("mistralai/Mistral-Large-Instruct-2407", AutoTokenizer, "Mistral Large 2"),
    ("THUDM/glm-4-9b-chat", AutoTokenizer, "GLM-4"),
    ("Qwen/Qwen2-7B-Instruct", AutoTokenizer, "Qwen 2"),
    ("Qwen/Qwen3-30B-A3B", AutoTokenizer, "Qwen 3"),
    ("google/gemma-2-9b-it", AutoTokenizer, "Gemma 2"),
    ("google/gemma-3-27b-it", AutoTokenizer, "Gemma 3"),
    ("unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth", AutoTokenizer, "Llama 4"),
    ("failspy/Nemotron-4-340B-Instruct-SafeTensors", AutoTokenizer, "Nemotron 4"),
    ("llm-jp/llm-jp-13b-v2.0", AutoTokenizer, "LLM-jp v2.0"),
    ("llm-jp/llm-jp-3-172b-instruct3", AutoTokenizer, "LLM-jp 3"),
    ("cyberagent/calm3-22b-chat", AutoTokenizer, "CALM3"),
]


def main():
    TokenizerEval('en')
    TokenizerEval('ja')


class TokenizerEval:
    def __init__(self, lang):
        self.CULTURAX_DS = f"{lang}_part_00004"
        self.OUTPUT = f"tokenizer-eval-{lang}"

        # Download a single sample file from CulturaX dataset.
        if not os.path.exists(f"{self.CULTURAX_DS}.jsonl"):
            print(f"Processing CulturaX: {self.CULTURAX_DS} ...")
            snapshot_download(repo_id='uonlp/CulturaX', local_dir='CulturaX', allow_patterns=[f'*{self.CULTURAX_DS}.parquet'], repo_type='dataset')
            dataset = Dataset.from_parquet(f'CulturaX/{self.CULTURAX_DS[:2]}/{self.CULTURAX_DS}.parquet')
            dataset.to_json(f'{self.CULTURAX_DS}.jsonl')

        cache_file = f'{self.OUTPUT}.json'
        cache = self.load_cache(cache_file)

        result = pd.DataFrame(
            [
                self.generate_row(*args, cache)
                for args in TOKENIZERS
            ]
        )
        result = result.sort_values(by="Avg Char/Token", ascending=False)
        result = result.reset_index(drop=True)

        print()
        print('===')
        print(result)
        result.to_markdown(f'{self.OUTPUT}.md', index=False)


    def load_cache(self, cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}


    def save_cache(self, cache_file, cache):
        with open(cache_file, 'w') as f:
            json.dump(cache, f)


    def evaluate(self, tokenizer):
        dataset = CheckpointableDataset.from_files(f"{self.CULTURAX_DS}.jsonl").tokenize(tokenizer, parallel=False).take(50000)
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


    def generate_row(self, tokenizer_url, tokenizer_cls, tokenizer_name, cache):
        print('> ' + tokenizer_name, end=': ')
        if tokenizer_url in cache:
            print("found in cache")
            return cache[tokenizer_url]
        else:
            print('...')

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
            "Avg Char/Token": self.evaluate(tokenizer)
        }

        cache[tokenizer_url] = result
        cache_file = f'{self.OUTPUT}.json'
        self.save_cache(cache_file, cache)

        return result


if __name__ == "__main__":
    main()
