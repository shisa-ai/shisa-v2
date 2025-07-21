#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full‑weight SFT with Unsloth on ROCm / MI300X
"""
import argparse, torch, os
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--data", default="sft.shisa-v2-new.jsonl")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_len", type=int, default=8192)
    p.add_argument("--bs", type=int, default=8)
    return p.parse_args()

def main():
    args = parse()
    model, tok = FastLanguageModel.from_pretrained(
        model_name      = args.base_model,
        max_seq_length  = args.max_len,
        load_in_4bit    = False,          # <‑‑ dense weights
        full_finetuning = True,           # <‑‑ FFT  :contentReference[oaicite:3]{index=3}
        dtype           = torch.bfloat16, # MI300X native
    )

    # Flatten ShareGPT conversations on‑the‑fly
    def _fmt(ex): return {"text": tok.apply_chat_template(
                              ex["conversations"],
                              tokenize=False, add_generation_prompt=False)}
    ds = load_dataset("json", data_files=args.data, split="train").map(_fmt)

    global_batch_size = 128 
    micro_batch_size = args.bs
    gradient_accumulation_steps = int(global_batch_size / micro_batch_size)

    tr_args = TrainingArguments(
        per_device_train_batch_size = micro_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        learning_rate               = args.lr,
        lr_scheduler_type           = "constant_with_warmup",
        warmup_ratio                = .03,
        num_train_epochs            = 1,
        bf16                        = True,
        logging_steps               = 1,
        save_strategy               = "no",
        report_to                   = ["wandb"],
        run_name                    = args.output_dir,
        output_dir                  = args.output_dir,
    )

    trainer = SFTTrainer(
        model             = model,
        args              = tr_args,
        train_dataset     = ds,
        tokenizer         = tok,
        dataset_text_field= "text",
        max_seq_length    = args.max_len,
        packing           = True,
    )
    trainer.train()
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()

