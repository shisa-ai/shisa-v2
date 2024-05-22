from   huggingface_hub import HfApi
import os
import sys
import torch
from   transformers import AutoTokenizer, AutoModelForCausalLM

try:
  path = sys.argv[1].rstrip('/')
  model_name = sys.argv[2]
except:
    print('You should run this with [existing-model-path] [model-name]')
    sys.exit(1)

# First load and convert the model
saved_model = f'{os.path.dirname(path)}/{model_name}'
if os.path.exists(saved_model):
    print(f'saved_model already exists?')
else:
    print(f'Saving {path} to {saved_model}...')
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(path)

    model.save_pretrained(saved_model)
    tokenizer.save_pretrained(saved_model)
    print('Done!')
