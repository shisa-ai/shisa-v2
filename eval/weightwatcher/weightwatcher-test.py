#!/usr/bin/env python

import sys
import torch
import weightwatcher as ww
from   transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL = sys.argv[1]


base_model = AutoModelForCausalLM.from_pretrained(
                 "meta-llama/Meta-Llama-3-70B-Instruct",
                 torch_dtype=torch.bfloat16,
                 device_map="auto",
             )

model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

watcher = ww.WeightWatcher()
details = watcher.analyze(model=model, base_model=base_model)
print(details)
summary = watcher.get_summary(details)
details.to_feather(f'{MODEL}.feather')
