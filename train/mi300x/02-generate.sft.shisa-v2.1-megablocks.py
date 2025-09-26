import os
import sys
import json
import random
import time
import multiprocessing
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from datasets import Features, Sequence, Value, load_dataset, concatenate_datasets, Dataset
from tqdm import tqdm

# Add Megatron-LM to Python path when running in the container
if '/workspace/Megatron-LM' not in sys.path:
    sys.path.insert(0, '/workspace/Megatron-LM')

try:
    from megatron.training.tokenizer import build_tokenizer
    from megatron.training.arguments import _add_tokenizer_args
    from megatron.core.datasets import indexed_dataset
    import argparse
    MEGATRON_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Megatron imports failed: {e}")
    MEGATRON_AVAILABLE = False

# ==============================================================================
# Main Execution Block - Configure Datasets Here
# ==============================================================================

OUTPUT = 'sft.shisa-v2.1'
DATA_DIR = './data'

def apply_chat_template(conversations):
    """Convert conversation format to chat template format"""
    if not conversations or not isinstance(conversations, list):
        return ""

    text_parts = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue

        role = turn.get('role', '').strip()
        content = turn.get('content', '').strip()

        if not role or not content:
            continue

        # Use ChatML format (compatible with many models)
        if role == 'system':
            text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == 'user':
            text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == 'assistant':
            text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        else:
            # Fallback for unknown roles
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    # Join with newlines and add final newline
    if text_parts:
        return "\n".join(text_parts) + "\n"
    return ""

def setup_megatron_tokenizer():
    """Setup Megatron tokenizer for preprocessing"""
    if not MEGATRON_AVAILABLE:
        raise RuntimeError("Megatron is not available. Cannot create tokenizer.")

    # Create minimal args for tokenizer
    parser = argparse.ArgumentParser()
    parser = _add_tokenizer_args(parser)

    # Get vocab and merge files
    vocab_file = os.path.join(DATA_DIR, 'gpt2-vocab.json')
    merge_file = os.path.join(DATA_DIR, 'gpt2-merges.txt')

    # Download if not exists using urllib instead of wget
    if not os.path.exists(vocab_file):
        print("Downloading GPT-2 vocab.json...")
        import urllib.request
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve('https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json', vocab_file)
        print(f"Downloaded to {vocab_file}")

    if not os.path.exists(merge_file):
        print("Downloading GPT-2 merges.txt...")
        import urllib.request
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve('https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt', merge_file)
        print(f"Downloaded to {merge_file}")

    args = parser.parse_args([
        '--vocab-file', vocab_file,
        '--merge-file', merge_file,
        '--tokenizer-type', 'GPT2BPETokenizer'
    ])

    # Megatron's tokenizer builder expects several runtime attributes that are
    # normally injected by its argument parser in distributed runs. Set sane
    # defaults here so the builder can operate in this standalone script.
    args.rank = getattr(args, 'rank', 0)
    args.make_vocab_size_divisible_by = getattr(args, 'make_vocab_size_divisible_by', 128)
    args.tensor_model_parallel_size = getattr(args, 'tensor_model_parallel_size', 1)
    args.vocab_extra_ids = getattr(args, 'vocab_extra_ids', 0)

    return build_tokenizer(args), args

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate SFT dataset for MegaBlocks training')
    parser.add_argument('--count', action='store_true',
                       help='Only count samples without generating full dataset')
    parser.add_argument('--rerun', action='store_true',
                       help='Regenerate dataset even if files already exist')
    parser.add_argument('--workers', type=int, default=min(8, multiprocessing.cpu_count()//2),
                       help=f'Number of worker processes (default: min(8, {multiprocessing.cpu_count()//2}))')
    return parser.parse_args()

def count_dataset_samples(dataset_config):
    """Count samples in a dataset configuration without loading full data"""
    try:
        print(f"Counting samples in {dataset_config['dataset_path']} ({dataset_config['split']})...")
        ds = load_dataset(dataset_config['dataset_path'], split=dataset_config['split'])
        count = len(ds)
        print(f"  -> {count:,} samples")
        return dataset_config['dataset_path'], count
    except Exception as e:
        print(f"  ERROR counting {dataset_config['dataset_path']}: {e}")
        return dataset_config['dataset_path'], 0

def main():
    """
    Main function to define datasets, process them, merge, shuffle, and save as Megatron binary format.
    """
    args = parse_args()
    overall_start_time = time.time()
    processed_datasets = [] # List to hold the results of loading functions

    # Check if files already exist and handle --rerun flag
    output_prefix = os.path.join(DATA_DIR, OUTPUT)
    output_bin_file = f"{output_prefix}_text_document.bin"
    output_idx_file = f"{output_prefix}_text_document.idx"

    if os.path.exists(output_bin_file) and os.path.exists(output_idx_file) and not args.rerun:
        print(f"Dataset files already exist:")
        print(f"  - {output_bin_file}")
        print(f"  - {output_idx_file}")
        print("Use --rerun to regenerate or --count to just count samples.")
        if not args.count:
            return

    print("--- Starting Dataset Processing for MegaBlocks ---")
    print(f"Workers: {args.workers}")
    print(f"Count only: {args.count}")
    print(f"Rerun: {args.rerun}")

    # Setup output directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Define Datasets to Process ---
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
            "dataset_path": "shisa-ai/translation_set_april_6",
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

    # Handle --count mode
    if args.count:
        print("\n--- Counting Samples in Each Dataset ---")
        total_samples = 0

        # Use multiprocessing for faster counting
        with ProcessPoolExecutor(max_workers=min(args.workers, len(datasets_config))) as executor:
            future_to_config = {executor.submit(count_dataset_samples, dsc): dsc for dsc in datasets_config}

            for future in as_completed(future_to_config):
                dataset_path, count = future.result()
                total_samples += count

        print(f"\n=== Total Sample Count ===")
        print(f"Total samples across all datasets: {total_samples:,}")
        print(f"Estimated training steps (3 epochs, batch size 512): {(total_samples * 3) // 512:,}")
        return

    # Regular processing mode
    for dsc in datasets_config:
        print(f"\nProcessing {dsc['dataset_path']} ({dsc['split']})...")
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
        )
        if ds:
            print(f"  Loaded {len(ds):,} examples")
            processed_datasets.append(ds)
        else:
            print(f"  Failed to load dataset")

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
        return

    # 2. Global Shuffle (Optional but recommended)
    print(f"Shuffling the final dataset with seed {GLOBAL_SEED}...")
    final_ds = final_ds.shuffle(seed=GLOBAL_SEED)
    print("Shuffling complete.")

    # 3. Setup Megatron tokenization and save as binary format
    print("\n--- Setting up Megatron tokenizer ---")
    try:
        tokenizer, tokenizer_args = setup_megatron_tokenizer()
        print("Tokenizer initialized successfully.")
    except Exception as e:
        print(f"ERROR setting up tokenizer: {e}")
        return

    # 4. Process conversations and create binary dataset
    print(f"\n--- Processing {len(final_ds)} conversations for MegaBlocks ---")

    output_prefix = os.path.join(DATA_DIR, OUTPUT)
    output_bin_file = f"{output_prefix}_text_document.bin"
    output_idx_file = f"{output_prefix}_text_document.idx"

    builder = indexed_dataset.IndexedDatasetBuilder(
        output_bin_file,
        dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
    )

    processed_count = 0
    skipped_count = 0
    total_tokens = 0

    print("Tokenizing and writing binary data...")
    for i, example in enumerate(tqdm(final_ds)):
        conversations = example.get(DEFAULT_CONVERSATION_FIELD, [])

        if not conversations:
            skipped_count += 1
            continue

        # Apply chat template
        formatted_text = apply_chat_template(conversations)

        if not formatted_text:
            skipped_count += 1
            continue

        # Tokenize the formatted text
        token_ids = tokenizer.tokenize(formatted_text)

        if len(token_ids) > 0:
            # Add end-of-document token
            token_ids.append(tokenizer.eod)

            # Add to builder
            builder.add_document(token_ids, [len(token_ids)])
            total_tokens += len(token_ids)
            processed_count += 1
        else:
            skipped_count += 1

    # Finalize the binary dataset
    print("Finalizing binary dataset...")
    builder.finalize(output_idx_file)

    overall_time = time.time() - overall_start_time
    print(f"\n--- MegaBlocks Dataset Generation Complete ---")
    print(f"Processing time: {overall_time:.2f} seconds")
    print(f"Conversations processed: {processed_count}")
    print(f"Conversations skipped: {skipped_count}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Generated files:")
    print(f"  - {output_bin_file}")
    print(f"  - {output_idx_file}")
    print(f"  - Vocab: {os.path.join(DATA_DIR, 'gpt2-vocab.json')}")
    print(f"  - Merge: {os.path.join(DATA_DIR, 'gpt2-merges.txt')}")

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

    ds = ds.map(
        format_sharegpt_conversation,
        remove_columns=columns_to_remove,
        desc=f"Formatting {dataset_name}",
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
