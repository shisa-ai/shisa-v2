import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def merge_and_save_model(output_path, base_model_path, adapter_path):
    print(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    print(f"Loading adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging adapter with base model")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    print("Model and tokenizer saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model and save")
    parser.add_argument("output_path", type=str, help="Path to save the merged model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Path or name of the base model")
    parser.add_argument("--adapter_path", type=str, default="outputs/shisa-v1-llama3-8b-unsloth-qlora/checkpoint-343", help="Path to the LoRA adapter")
    
    args = parser.parse_args()
    
    merge_and_save_model(args.output_path, args.base_model, args.adapter_path)
