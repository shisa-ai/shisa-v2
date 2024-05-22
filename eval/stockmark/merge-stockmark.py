'''
### Doesn't work?

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM


# Define the paths
base_model_path = "stockmark/stockmark-100b"
adapter_model_path = "stockmark/stockmark-100b-instruct-v0.1"
merged_model_path = "merged_stockmark-100b-instruct-v0.1"

# Load the base model
tokenizer = AutoTokenizer.from_pretrained("stockmark/stockmark-100b-instruct-v0.1")
model = AutoPeftModelForCausalLM.from_pretrained("stockmark/stockmark-100b-instruct-v0.1", device_map="auto", torch_dtype=torch.bfloat16)

# Save New model
merged_model = model.merge_and_unload()
merged_model.save_pretrained(merged_model_path, save_embedding_layers=True)
tokenizer.save_pretrained(merged_model_path)
'''

'''
### Only saves the original merge oops

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "stockmark/stockmark-100b-instruct-v0.1"
merged_model_path = "merged_stockmark-100b-instruct-v0.1"

model = AutoModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

model.save_pretrained(merged_model_path)
'''

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Step 1: Load the base model
base_model_name = "stockmark/stockmark-100b"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.bfloat16)

# Step 2: Load the PEFT adapter configuration and weights
adapter_path = "stockmark/stockmark-100b-instruct-v0.1"
peft_config = PeftConfig.from_pretrained(adapter_path)
adapter_model = PeftModel.from_pretrained(base_model, adapter_path)

# Step 3: Merge the adapter with the base model
def merge_models(base_model, adapter_model):
    base_model_state_dict = base_model.state_dict()
    adapter_state_dict = adapter_model.state_dict()
    for key in adapter_state_dict.keys():
        if key in base_model_state_dict:
            base_model_state_dict[key] = adapter_state_dict[key]
    base_model.load_state_dict(base_model_state_dict)
    return base_model

merged_model = merge_models(base_model, adapter_model)

# Step 4: Ensure the correct tokenizer is used
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# Step 5: Save the merged model
merged_model_path = "merged_stockmark-100b-instruct-v0.1"
merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

