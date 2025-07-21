#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full‑weight DPO with Unsloth on ROCm / MI300X
"""
import argparse, torch, os
from datasets import load_dataset
from unsloth import FastLanguageModel, PatchDPOTrainer; PatchDPOTrainer()
from trl import DPOTrainer
from transformers import TrainingArguments

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)   # ← SFT ckpt
    p.add_argument("--output_dir", required=True)
    p.add_argument("--data", default="dpo.shisa-v2x.jsonl")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--bs", type=int, default=8)
    return p.parse_args()

def main():
    a = parse()
    model, tok = FastLanguageModel.from_pretrained(
        model_name      = a.base_model,
        max_seq_length  = a.max_len,
        load_in_4bit    = False,
        full_finetuning = True,
        dtype           = torch.bfloat16,
    )

    def _prep(ex):
        prmpt  = ex["chosen"][:-1]
        return {
            "prompt"   : tok.apply_chat_template(prmpt, tokenize=False,
                                                 add_generation_prompt=False),
            "chosen"   : ex["chosen"][-1]["content"],
            "rejected" : ex["rejected"][-1]["content"],
        }

    ds = load_dataset("json", data_files=a.data, split="train").map(_prep)

    global_batch_size = 64 
    micro_batch_size = a.bs
    gradient_accumulation_steps = int(global_batch_size / micro_batch_size)

    tr_args = TrainingArguments(
        per_device_train_batch_size = micro_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        learning_rate               = a.lr,
        warmup_ratio                = .03,
        num_train_epochs            = 1,
        bf16                        = True,
        logging_steps               = 1,
        report_to                   = ["wandb"],
        run_name                    = a.output_dir,
        output_dir                  = a.output_dir,
    )

    trainer = DPOTrainer(
        model         = model,
        ref_model     = None,
        beta          = a.beta,
        args          = tr_args,
        train_dataset = ds,
        tokenizer     = tok,
        max_length    = a.max_len,
        max_prompt_length = a.max_len // 2,
    )
    trainer.train()
    model.save_pretrained(a.output_dir, safe_serialization=True)
    tok.save_pretrained(a.output_dir)

if __name__ == "__main__":
    main()
