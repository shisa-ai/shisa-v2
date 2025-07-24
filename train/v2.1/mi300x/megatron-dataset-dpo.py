#!/usr/bin/env python
"""Convert OpenRLHF DPO dataset to Megatron-LM binary format."""
import argparse, tempfile, subprocess
from datasets import load_dataset
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--tokenizer", required=True)
p.add_argument("--tokenizer-json", required=True)
p.add_argument("--output-prefix", required=True)
p.add_argument("--workers", type=int, default=8)
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(args.tokenizer)

ds = load_dataset("json", data_files=args.input, split="train")

def fmt(ex):
    prmpt = ex["chosen"][:-1]
    prompt = tok.apply_chat_template(prmpt, tokenize=False, add_generation_prompt=False)
    chosen = ex["chosen"][-1]["content"]
    rejected = ex["rejected"][-1]["content"]
    return {"text": prompt + "\n### CHOSEN:\n" + chosen + "\n### REJECTED:\n" + rejected}

ds = ds.map(fmt, remove_columns=ds.column_names)

with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
    for t in ds["text"]:
        tmp.write(t + "\n")
    tmp_path = tmp.name

subprocess.check_call([
    "python", "tools/preprocess_data.py",
    "--input", tmp_path,
    "--output-prefix", args.output_prefix,
    "--vocab-file", args.tokenizer_json,
    "--tokenizer-type", "HuggingFaceTokenizer",
    "--dataset-impl", "mmap",
    "--workers", str(args.workers),
    "--append-eod",
])
