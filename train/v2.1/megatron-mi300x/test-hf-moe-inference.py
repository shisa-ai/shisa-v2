#!/usr/bin/env python3
"""Quick smoke test for Megablocks-exported MoE checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(tokenizer: AutoTokenizer, user_message: str) -> str:
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": user_message},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback for tokenizers without chat templates
    system = messages[0]["content"].strip()
    user = messages[1]["content"].strip()
    return f"<|im_start|>system\n{system}<|im_end|><|im_start|>user\n{user}<|im_end|><|im_start|>assistant\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path, help="Path to the exported Hugging Face checkpoint")
    parser.add_argument("--prompt", default="Hello, can you explain what mixture-of-experts means?", help="User message for the chat template")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum number of generated tokens")
    args = parser.parse_args()

    model_path = args.model_path.expanduser().resolve()
    if not model_path.exists():
        print(f"error: model path '{model_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    tokenizer_kwargs = {"trust_remote_code": True}
    gpt2_vocab = model_path / "gpt2-vocab.json"
    gpt2_merges = model_path / "gpt2-merges.txt"
    if gpt2_vocab.is_file() and gpt2_merges.is_file():
        tokenizer_kwargs.setdefault("vocab_file", str(gpt2_vocab))
        tokenizer_kwargs.setdefault("merges_file", str(gpt2_merges))

    print(f"Loading tokenizer from {model_path} (trust_remote_code=True)")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), **tokenizer_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Loading model from {model_path} (trust_remote_code=True)")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    model = model.to(device)

    if getattr(tokenizer, 'chat_template', None) is None:
        chat_template = getattr(model.config, 'chat_template', None)
        if chat_template:
            tokenizer.chat_template = chat_template

    prompt = build_prompt(tokenizer, args.prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("Running generation...")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0, inputs.input_ids.shape[1]:]
    completion = tokenizer.decode(generated, skip_special_tokens=True)

    separator = "=" * 40
    print(separator)
    print(completion.strip())
    print(separator)


if __name__ == "__main__":
    main()
