
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the original model
model = AutoModelForCausalLM.from_pretrained(
    "ai21labs/Jamba-v0.1",
    trust_remote_code=True,
    torch_dtype=torch.cuda.is_bf16_supported() and torch.bfloat16 or torch.float16,
)

# Load lora
from safetensors import safe_open
checkpoint_dir = "final_checkpoint"
adapter_path = checkpoint_dir + "/adapter_model.safetensors"

def load_lora_adjustments(lora_path):
    lora_adjustments = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            lora_adjustments[key] = f.get_tensor(key)
    return lora_adjustments

def apply_lora_adjustments(model, lora_adjustments):
    # Apply the LORA adjustments to the model.
    for name, param in model.named_parameters():
        if name in lora_adjustments:
            adjustment = lora_adjustments[name]
            param.data = param.data + adjustment

lora_adjustments = load_lora_adjustments(adapter_path)
apply_lora_adjustments(model, lora_adjustments)

# Save the merged model
merged_model_path = "merged_model"
model.save_pretrained(merged_model_path)

tokenizer = AutoTokenizer.from_pretrained("final_checkpoint")
tokenizer.padding_side = 'right'
tokenizer.save_pretrained("merged_model")
