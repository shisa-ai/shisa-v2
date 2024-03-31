import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
            "shisa-jamba-v1-checkpoint-4228",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
tokenizer = AutoTokenizer.from_pretrained("shisa-jamba-v1-checkpoint-4228")
input_text = """<|im_start|>system 
You are GPT-4, a helpful assistant.
<|im_end|> 
<|im_start|>user 
最近、運動すれば、すぐにめっちゃくっちゃ汗かいちゃうんだけど、どうしたらいいですか？
<|im_end|> 
<|im_start|>assistant 
"""
input_ids = tokenizer(input_text, return_tensors='pt').to(model.device)["input_ids"]
outputs = model.generate(input_ids, max_new_tokens=256, temperature=0.1)
print(tokenizer.batch_decode([outputs[0][len(input_ids[0]):]]))
