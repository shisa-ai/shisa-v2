import os
import json
import random
import time
from datasets import Features, Sequence, Value, load_dataset, concatenate_datasets, Dataset
from tqdm import tqdm

# ==============================================================================
# Main Execution Block - Configure Datasets Here
# ==============================================================================

OUTPUT = 'sft.shisa-v2.1'

def main():
    """
    Main function to define datasets, process them, merge, shuffle, and save.
    Edit this function directly to add/configure datasets.
    """
    overall_start_time = time.time()
    processed_datasets = [] # List to hold the results of loading functions

    print("--- Starting Dataset Processing ---")

    # --- Define Datasets to Process ---
    # Each call to a load_dataset_* function should return a Dataset
    # object with (at least) a 'conversations' column, or None if failed.

    datasets_config = [
        # 2025-08-12: 1.18K rows - Proper Politeness
        {
            "dataset_path": "shisa-ai/shisa-politeness-dataset",
            "field_messages": "conversations",
            "split": "train",
        },
        # 2025-08-07: 360 rows - 10X Shisa.AI ID, hard-coded behaviors
        {
            "dataset_path": "shisa-ai/shisa-hardcoded-set",
            "field_messages": "conversations",
            "split": "train",
        },
        # 2025-06-18: 12.8K/51.2K rows of Chotto formatted multi-turn translations
        {
            "dataset_path": "shisa-ai/chotto_translation_set_sft",
            "field_messages": "conversations",
            "split": "train[:25%]",
        },
        # 2025-06-17: 181K rows - Latest (+Shisa V2 405B) rejection sampled version of primary dataset
        {
            "dataset_path": "shisa-ai/shisa-v2-2025-6-17-sharegpt",
            "field_messages": "conversations",
            "split": "train",
        },
        {
            "dataset_path": "shisa-ai/shisa-v2-roleplaying-sft",
            "field_messages": "conversations",
            "split": "train",
        },
        {

            # "dataset_path": "shisa-ai/translation_set_april_6",
            "dataset_path": "shisa-ai/translation_set_rejection_sampling",
            "field_messages": "conversations",
            "split": "train[:25%]",
        },
        {
            "dataset_path": "shisa-ai/rewild-set-deepseek-subset",
            "field_messages": "conversations",
            "split": "train[:25%]",
        },
        {
            "dataset_path": "shisa-ai/magpie-ultra-set",
            "field_messages": "conversations",
            "split": "train[:8%]",
        },
        {
            "dataset_path": "shisa-ai/magpie-advanced-questions-set",
            "field_messages": "conversations",
            "split": "train[:8%]",
        },
        {
            "dataset_path": "shisa-ai/japan-magpie-set",
            "field_messages": "conversations",
            "split": "train",
        },
        {
            "dataset_path": "shisa-ai/shisa-v2-instruction-following-sft",
            "field_messages": "conversations",
            "split": "train[:50%]",
        },
    ]

    for dsc in datasets_config:
        ds = None
        ds = load_dataset_conversation(
            dataset_path = dsc['dataset_path'],
            field_messages = dsc['field_messages'],
            split = dsc['split'],
            role_map={ # Map *source* role names to 'user', 'assistant', 'system'
                 "system": ["system"],
                 "user": ["user", "human"],
                 "assistant": ["gpt", "assistant", "model"]
            },
            # fields=['conversations', 'id'], # Optionally keep other source fields like 'id'
            # shuffle_seed=123, # Optional: shuffle only this dataset before merge
        )
        if ds:
            processed_datasets.append(ds)



    def rebuild_dataset_with_clean_schema(dataset):
        """
        Completely rebuilds a dataset with a clean schema, removing any traces of turn_identifier
        by creating an entirely new dataset.
        """
        from datasets import Dataset
        import pandas as pd

        # Extract just the data we need with the exact structure we want
        clean_data = []

        for example in dataset:
            conversations = example.get(DEFAULT_CONVERSATION_FIELD, [])
            clean_conversations = []

            for turn in conversations:
                if isinstance(turn, dict) and "role" in turn and "content" in turn:
                    # Only keep the role and content fields, nothing else
                    clean_conversations.append({
                        "role": turn["role"],
                        "content": turn["content"]
                    })

            # Only add examples with non-empty conversations
            if clean_conversations:
                clean_data.append({DEFAULT_CONVERSATION_FIELD: clean_conversations})

        # Create a brand new dataset with the clean data
        df = pd.DataFrame(clean_data)
        return Dataset.from_pandas(df)

    print("Rebuilding datasets with clean schema...")
    clean_datasets = []

    for i, ds in enumerate(processed_datasets):
        print(f"Rebuilding dataset {i+1}...")
        clean_ds = rebuild_dataset_with_clean_schema(ds)
        print(f"  Dataset {i+1} rebuilt: {len(clean_ds)} examples")
        clean_datasets.append(clean_ds)

    # Now try concatenation with the clean datasets
    print("\n--- Finalizing Dataset ---")
    print(f"Concatenating {len(clean_datasets)} clean datasets...")

    try:
        final_ds = concatenate_datasets(clean_datasets)
        print(f"Final dataset created with {len(final_ds)} examples.")
    except Exception as e:
        print(f"ERROR during concatenation: {e}")

    '''
    # --- Merge, Shuffle, Save ---
    print("\n--- Finalizing Dataset ---")

    if not processed_datasets:
        print("ERROR: No datasets were successfully processed. Exiting.")
        return

    # 1. Merge Datasets
    print(f"Concatenating {len(processed_datasets)} processed datasets...")
    try:
        # Ensure all datasets have the same features before concatenating
        # This relies on load_dataset_* functions correctly setting the 'fields'
        final_ds = concatenate_datasets(processed_datasets)
    except ValueError as e:
         print(f"\nERROR during concatenation: {e}")
         print("Ensure all loaded datasets have the same columns (check 'fields' argument).")
         print("Columns per dataset:")
         for i, ds in enumerate(processed_datasets):
             print(f"  Dataset {i+1} ({ds.builder_name if hasattr(ds, 'builder_name') else 'Unknown'}): {ds.column_names}")
         return
    print(f"Concatenated dataset size: {len(final_ds)} examples.")
    '''

    # 2. Global Shuffle (Optional but recommended)
    print(f"Shuffling the final dataset with seed {GLOBAL_SEED}...")
    final_ds = final_ds.shuffle(seed=GLOBAL_SEED)
    print("Shuffling complete.")

    # 3. Save Final Dataset
    output_filename_base = OUTPUT
    save_format = "jsonl"  # 'jsonl' or 'disk'
    num_shards = 1        # Use > 1 for multiple output files (esp. for jsonl)

    output_path = f"{output_filename_base}.{save_format}" if save_format == "jsonl" else output_filename_base
    print(f"Saving final dataset ({len(final_ds)} examples) as {save_format} to '{output_path}'...")

    try:
        if save_format == "jsonl":
            # Use datasets' built-in JSON writing with sharding support
            final_ds.to_json(
                output_path,
                lines=True,
                force_ascii=False,
                # num_shards=num_shards if num_shards > 1 else None # num_shards=None or 1 writes single file
            )
        elif save_format == "disk":
            final_ds.save_to_disk(output_path, num_shards=num_shards if num_shards > 1 else None)
        else:
            print(f"ERROR: Unknown save format: {save_format}")
            return # Exit if format is unknown
    except Exception as e:
        print(f"ERROR during saving: {e}")
        return # Exit on save error

    overall_time = time.time() - overall_start_time
    print(f"\n--- Script finished in {overall_time:.2f} seconds ---")
    print(f"Output saved to '{output_path}' {'(sharded)' if num_shards > 1 else ''}")


# ==============================================================================
# Configuration Constants
# ==============================================================================

GLOBAL_SEED = 42
DEFAULT_CONVERSATION_FIELD = "conversations" # Target field name
DEFAULT_SPLIT = "train"

# ==============================================================================
# Loading Functions for Specific Formats
# ==============================================================================

def _validate_and_select_columns(ds: Dataset, requested_fields: list, dataset_name: str) -> Dataset | None:
    """Internal helper to ensure final columns match requested fields."""
    final_columns = [f for f in requested_fields if f in ds.column_names]
    missing = set(requested_fields) - set(final_columns)
    if missing:
        print(f"  WARNING for {dataset_name}: Requested fields {missing} not found. Keeping only: {final_columns}")
    if not final_columns:
        print(f"  ERROR for {dataset_name}: No requested fields found in the processed dataset.")
        return None
    if set(final_columns) != set(ds.column_names):
        try:
            return ds.select_columns(final_columns)
        except Exception as e:
            print(f"  ERROR selecting columns {final_columns} for {dataset_name}: {e}")
            return None
    return ds # Columns already match

def load_dataset_conversation(
    dataset_path: str,
    field_messages: str = "conversations",
    role_map: dict = None, # Map source roles -> standard roles ('user', 'assistant', 'system')
    fields: list = None,   # List of *final* columns to keep (must include DEFAULT_CONVERSATION_FIELD)
    split: str = DEFAULT_SPLIT,
    shuffle_seed: int = None,
    **load_kwargs # Pass extra args like data_files to load_dataset
) -> Dataset | None:
    """
    Loads and converts a ShareGPT-like dataset to OpenAI format.
    Expected input: A field (`field_messages`) containing a list of turns.
    Each turn should have role ('role' or 'from') and content ('content' or 'value').
    """
    dataset_name = os.path.basename(dataset_path)
    print(f"Processing ShareGPT: {dataset_name} (Split: {split})")

    # Ensure the target conversation field is always included in the final output
    if fields is None:
        fields = [DEFAULT_CONVERSATION_FIELD]
    elif DEFAULT_CONVERSATION_FIELD not in fields:
        fields.append(DEFAULT_CONVERSATION_FIELD)

    # Invert role_map for efficient lookup: {'human': 'user', 'gpt': 'assistant', ...}
    if role_map is None: role_map = {} # Handle potential None
    inverted_role_map = {val.lower(): key for key, values in role_map.items() for val in values}
    # Add default mappings if not provided, useful for simple cases
    if 'user' not in inverted_role_map: inverted_role_map['user'] = 'user'
    if 'human' not in inverted_role_map: inverted_role_map['human'] = 'user'
    if 'assistant' not in inverted_role_map: inverted_role_map['assistant'] = 'assistant'
    if 'gpt' not in inverted_role_map: inverted_role_map['gpt'] = 'assistant'
    if 'system' not in inverted_role_map: inverted_role_map['system'] = 'system'


    try:
        ds = load_dataset(dataset_path, split=split, **load_kwargs)
    except Exception as e:
        print(f"  ERROR loading {dataset_name}: {e}")
        return None

    # Check if source message field exists
    if field_messages not in ds.column_names:
        print(f"  ERROR: Source field '{field_messages}' not found in {dataset_name}. Available: {ds.column_names}")
        return None

    def format_sharegpt_conversation(example):
        raw_conv = example.get(field_messages)
        processed_conv = []
        if isinstance(raw_conv, list):
            for turn in raw_conv:
                if not isinstance(turn, dict): continue # Skip invalid turns

                role_val = turn.get('role', turn.get('from'))
                content_val = turn.get('content', turn.get('value'))

                if role_val is not None and content_val is not None:
                    standard_role = inverted_role_map.get(str(role_val).lower())
                    if standard_role:
                        processed_conv.append({"role": standard_role, "content": str(content_val)})
                    # else: # Optionally warn about unmapped roles
                    #    print(f"  WARNING: Unmapped role '{role_val}' in {dataset_name}")

        result = {DEFAULT_CONVERSATION_FIELD: processed_conv}
        # Add other requested fields from the original example if they exist
        for f in fields:
            if f != DEFAULT_CONVERSATION_FIELD and f in example:
                result[f] = example[f]
        return result

    # Determine columns to remove: all except the source messages field and any other fields requested to be kept
    columns_to_remove = [
        col for col in ds.column_names
        if col != field_messages and col not in fields
    ]
    # Handle edge case: if field_messages is same as DEFAULT_CONVERSATION_FIELD and not in fields list initially
    if field_messages == DEFAULT_CONVERSATION_FIELD and field_messages not in fields:
         columns_to_remove = [col for col in ds.column_names if col not in fields] # Keep only final fields

    print(f"  Columns to remove for {dataset_name}: {columns_to_remove}")

    # columns_to_potentially_keep = [field_messages] + [f for f in fields if f != DEFAULT_CONVERSATION_FIELD]
    # columns_to_remove = [col for col in ds.column_names if col not in columns_to_potentially_keep]

    ds = ds.map(
        format_sharegpt_conversation,
        remove_columns=columns_to_remove,
        desc=f"Formatting {dataset_name}",
        # load_from_cache_file=False,
    )

    # Filter out examples with empty conversations after processing
    initial_count = len(ds)
    ds = ds.filter(lambda ex: len(ex.get(DEFAULT_CONVERSATION_FIELD, [])) > 0)
    if len(ds) < initial_count:
        print(f"  Filtered out {initial_count - len(ds)} examples with empty conversations.")

    if shuffle_seed is not None:
        print(f"  Shuffling {dataset_name} with seed {shuffle_seed}")
        ds = ds.shuffle(seed=shuffle_seed)

    # Validate and select final columns
    ds = _validate_and_select_columns(ds, fields, dataset_name)
    if ds is None: return None # Error during column selection

    print(f"  Finished {dataset_name}. Resulting examples: {len(ds)}")
    return ds



# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    # Set global random seed for reproducibility if needed by other libraries
    random.seed(GLOBAL_SEED)
    main()


