# imports
from datasets import load_dataset
from trl import SFTTrainer

# get dataset
# dataset = load_dataset("augmxnt/ultra-orca-boros-en-ja-v1", split="train")

# get dataset
dataset = load_dataset("imdb", split="train")

'''
https://huggingface.co/docs/trl/main/en/sft_trainer#enhance-the-models-performances-using-neftune
sft_config = STFConfig(
    neftune_noise_alpha=5,
)
formatting_func = get_formatting_func_from_dataset(ds_3, self.llama_tokenizer)
'''

# get trainer
trainer = SFTTrainer(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
)

# train
trainer.train()
