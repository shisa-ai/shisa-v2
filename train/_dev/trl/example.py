# imports
from datasets import load_dataset
from trl import SFTTrainer

# get dataset
# dataset = load_dataset("imdb", split="train")
dataset = load_dataset("augmxnt/ultra-orca-boros-en-ja-v1", split="train")

# get trainer
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    dataset_text_field="conversations",
    max_seq_length=512,
)

# train
trainer.train()
