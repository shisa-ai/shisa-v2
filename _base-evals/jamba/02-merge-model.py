from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the original model
model = AutoModelForCausalLM.from_pretrained(
    "ai21labs/Jamba-v0.1",
    trust_remote_code=True,
    torch_dtype=torch.cuda.is_bf16_supported() and torch.bfloat16 or torch.float16,
)

# Load lora
adapter_path = 'final_checkpoint'
model = AutoPeftModelForCausalLM.from_pretrained(adapter_path, device_map="auto", torch_dtype=torch.bfloat16)

model = model.merge_and_unload()

# Save the merged model
merged_model_path = "merged_model"
model.save_pretrained(merged_model_path)

tokenizer = AutoTokenizer.from_pretrained("final_checkpoint")
tokenizer.padding_side = 'right'
tokenizer.save_pretrained("merged_model")

# Upload
model.push_to_hub("org/model_name", private=True)
tokenizer.push_to_hub("org/model_name", private=True)
