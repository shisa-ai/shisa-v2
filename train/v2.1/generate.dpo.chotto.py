import json
import random
from pathlib import Path
from datasets import load_dataset

'''
OpenRLHF's DPO wants a "chosen" and "rejected" JSON conversation
'''

# MAIN SCRIPT
def main():
    # --- Configuration ---
    final_output_file = "chotto-dpo.jsonl"
    random_seed = 42

    print("--- Starting Data Generation ---")
    random.seed(random_seed) # Set random seed globally


    # --- Process Chotto ---
    ds_name = "shisa-ai/chotto_translation_set_dpo"
    print(f"Processing {ds_name}...")
    try:
        ds_chotto = load_dataset(ds_name, split="train")

        def to_dpo_format_chotto(example):
            return {
                "chosen": example["chosen"],
                "rejected": example["rejected"]
            }

        dpo_ds_chotto = ds_chotto.map(to_dpo_format_chotto, remove_columns=ds_chotto.column_names)
        chotto_data = list(dpo_ds_chotto)
    except Exception as e:
        print(f"Error processing Ultrafeedback dataset: {e}")
        chotto_data = [] # Ensure variable exists even on error


    # --- Script 3: Combine and Shuffle ---
    print("Combining and shuffling datasets...")

    all_data = chotto_data

    if not all_data:
        print("No data loaded from intermediate files. Exiting.")
    else:
        print(f"Combined dataset has {len(all_data)} records")

        # Shuffle the data
        print("Shuffling data...")
        random.shuffle(all_data) # Seed was set at the beginning

        # Save to output file
        print(f"Saving combined and shuffled data to {final_output_file}...")
        save_jsonl(all_data, final_output_file)
        print(f"Done! Final dataset saved with {len(all_data)} records.")


# HELPER FUNCTIONS
def load_jsonl(file_path):
    """Load a JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

def save_jsonl(data, output_path):
    """Save a list of dictionaries to a JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    main()
