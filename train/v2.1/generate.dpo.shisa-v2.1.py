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
    final_output_file = "dpo.shisa-v2.1.jsonl"
    random_seed = 42

    print("--- Starting Data Generation ---")
    random.seed(random_seed) # Set random seed globally


    # --- Process UltraFeedback ---
    ds_name = "shisa-ai/shisa-v2-405b-ultrafeedback-armorm"
    print(f"Processing {ds_name}...")
    try:
        ds_ultra = load_dataset(ds_name, split="train")

        def to_dpo_format_ultra(example):
            return {
                "chosen": example["chosen"],
                "rejected": example["rejected"]
            }

        dpo_ds_ultra = ds_ultra.map(to_dpo_format_ultra, remove_columns=ds_ultra.column_names)
        ultra_data = list(dpo_ds_ultra)
    except Exception as e:
        print(f"Error processing Ultrafeedback dataset: {e}")
        ultra_data = [] # Ensure variable exists even on error


    # --- Process Shisa Roleplaying DPO ---
    print(f"Processing shisa-ai/shisa-v2-roleplaying-dpo...")
    try:
        ds_rp = load_dataset("shisa-ai/shisa-v2-roleplaying-dpo", split="train")

        def to_dpo_format_rp(example):
            conv = example["conversations"]
            chosen_msg = example["chosen"]
            rejected_msg = example["rejected"]
            return {
                "chosen": conv + [chosen_msg],
                "rejected": conv + [rejected_msg]
            }

        dpo_ds_rp = ds_rp.map(to_dpo_format_rp, remove_columns=ds_rp.column_names)

        rp_data = list(dpo_ds_rp) # Convert to list
    except Exception as e:
        print(f"Error processing RP dataset: {e}")
        rp_data = [] # Ensure variable exists even on error

    # --- Process Shisa Translation DPO ---
    print(f"Processing shisa-ai/translation-no-extra-text-dpo-dataset...")
    try:
        ds_tl = load_dataset("shisa-ai/translation-no-extra-text-dpo-dataset", split="train")

        def to_dpo_format_tl(example):
            conv = example["conversations"]
            chosen_msg = example["chosen"]
            rejected_msg = example["rejected"]
            return {
                "chosen": conv + [chosen_msg],
                "rejected": conv + [rejected_msg]
            }

        dpo_ds_tl = ds_tl.map(to_dpo_format_tl, remove_columns=ds_tl.column_names)

        tl_data = list(dpo_ds_tl) # Convert to list
    except Exception as e:
        print(f"Error processing RP dataset: {e}")
        tl_data = [] # Ensure variable exists even on error


    # --- Process Shisa Instruction Following DPO ---
    print(f"Processing shisa-ai/shisa-v2-instruction-following-dpo...")
    try:
        ds_if = load_dataset("shisa-ai/shisa-v2-instruction-following-dpo", split="train[:50%]")

        def to_dpo_format_if(example):
            conv = example["conversations"]
            chosen_msg = example["chosen"]
            rejected_msg = example["rejected"]
            return {
                "chosen": conv + [chosen_msg],
                "rejected": conv + [rejected_msg]
            }

        dpo_ds_if = ds_if.map(to_dpo_format_if, remove_columns=ds_if.column_names)

        if_data = list(dpo_ds_if) # Convert to list
    except Exception as e:
        print(f"Error processing RP dataset: {e}")
        if_data = [] # Ensure variable exists even on error


    # --- Process Chotto Translation DPO ---
    print(f"Processing shisa-ai/chotto_translation_set_dpo...")
    try:
        ds_chotto = load_dataset("shisa-ai/chotto_translation_set_dpo", split="train")

        def to_dpo_format_chotto(example):
            return {
                "chosen": example["chosen"],
                "rejected": example["rejected"]
            }

        dpo_ds_chotto = ds_chotto.map(to_dpo_format_chotto, remove_columns=ds_chotto.column_names)
        chotto_data = list(dpo_ds_chotto)
    except Exception as e:
        print(f"Error processing Chotto dataset: {e}")
        chotto_data = [] # Ensure variable exists even on error


    # --- Process Kiseki DPO ---
    print(f"Processing shisa-ai/chotto_kiseki_dpo_9_19_25...")
    try:
        ds_kiseki = load_dataset("shisa-ai/chotto_kiseki_dpo_9_19_25", split="train")

        def to_dpo_format_kiseki(example):
            return {
                "chosen": example["chosen"],
                "rejected": example["rejected"]
            }

        dpo_ds_kiseki = ds_kiseki.map(to_dpo_format_kiseki, remove_columns=ds_kiseki.column_names)
        kiseki_data = list(dpo_ds_kiseki)
    except Exception as e:
        print(f"Error processing Kiseki dataset: {e}")
        kiseki_data = [] # Ensure variable exists even on error


    # --- Script 3: Combine and Shuffle ---
    print("Combining and shuffling datasets...")

    all_data = ultra_data + rp_data + tl_data + if_data + chotto_data + kiseki_data

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
