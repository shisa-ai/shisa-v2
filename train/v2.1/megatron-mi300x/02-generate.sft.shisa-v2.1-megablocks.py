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

TOKENIZER_VOCAB_PATH = None
TOKENIZER_MERGE_PATH = None
TOKENIZER_MODEL_PATH = None
TOKENIZER_TYPE = 'GPT2BPETokenizer'
CHAT_TEMPLATE_TOKENIZER = None
CHAT_TEMPLATE_WARNING_PRINTED = False

def apply_chat_template(conversations):
    """Return formatted chat text using tokenizer template when available."""
    global CHAT_TEMPLATE_WARNING_PRINTED

    if not conversations or not isinstance(conversations, list):
        return ""

    if CHAT_TEMPLATE_TOKENIZER is not None:
        try:
            formatted = CHAT_TEMPLATE_TOKENIZER.apply_chat_template(
                conversations, tokenize=False, add_generation_prompt=False
            )
            if formatted:
                if not formatted.endswith("\n"):
                    formatted += "\n"
                return formatted
        except Exception as exc:
            if not CHAT_TEMPLATE_WARNING_PRINTED:
                print(f"Warning: chat_template application failed ({exc}); falling back to ChatML format.")
                CHAT_TEMPLATE_WARNING_PRINTED = True

    text_parts = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue

        role = (turn.get('role') or '').strip()
        content = turn.get('content')
        if isinstance(content, list):
            content = '\n'.join(str(item) for item in content if item)
        content = (content or '').strip()

        if not role or not content:
            continue

        if role == 'system':
            text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == 'user':
            text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == 'assistant':
            text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        else:
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    if text_parts:
        return "\n".join(text_parts) + "\n"
    return ""

def setup_megatron_tokenizer():
    """Setup Megatron tokenizer for preprocessing"""
    if not MEGATRON_AVAILABLE:
        raise RuntimeError("Megatron is not available. Cannot create tokenizer.")

    tokenizer_type = TOKENIZER_TYPE or 'GPT2BPETokenizer'

    parser = argparse.ArgumentParser()
    parser = _add_tokenizer_args(parser)
    args_list = ['--tokenizer-type', tokenizer_type]

    if tokenizer_type == 'HuggingFaceTokenizer':
        if not TOKENIZER_MODEL_PATH:
            raise ValueError('tokenizer-model must be specified for HuggingFaceTokenizer')
        if not os.path.exists(TOKENIZER_MODEL_PATH):
            raise FileNotFoundError(f"Tokenizer model path not found at {TOKENIZER_MODEL_PATH}")
        args_list += ['--tokenizer-model', TOKENIZER_MODEL_PATH]
    else:
        vocab_file = TOKENIZER_VOCAB_PATH or os.path.join(DATA_DIR, 'gpt2-vocab.json')
        merge_file = TOKENIZER_MERGE_PATH or os.path.join(DATA_DIR, 'gpt2-merges.txt')

        default_vocab = os.path.join(DATA_DIR, 'gpt2-vocab.json')
        default_merges = os.path.join(DATA_DIR, 'gpt2-merges.txt')

        if not os.path.exists(vocab_file):
            if vocab_file == default_vocab:
                print('Downloading GPT-2 vocab.json...')
                import urllib.request
                os.makedirs(os.path.dirname(default_vocab), exist_ok=True)
                urllib.request.urlretrieve('https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json', default_vocab)
                print(f"Downloaded to {default_vocab}")
            else:
                raise FileNotFoundError(f"Tokenizer vocab file not found at {vocab_file}")

        if not os.path.exists(merge_file):
            if merge_file == default_merges:
                print('Downloading GPT-2 merges.txt...')
                import urllib.request
                os.makedirs(os.path.dirname(default_merges), exist_ok=True)
                urllib.request.urlretrieve('https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt', default_merges)
                print(f"Downloaded to {default_merges}")
            else:
                raise FileNotFoundError(f"Tokenizer merges file not found at {merge_file}")

        args_list += ['--vocab-file', vocab_file, '--merge-file', merge_file]

    args = parser.parse_args(args_list)

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
    parser.add_argument('--workers', type=int, default=min(80, multiprocessing.cpu_count()),
                       help=f'Number of worker processes (default: min(64, {multiprocessing.cpu_count()}))')
    parser.add_argument('--debug', type=int, default=0,
                       help='Debug mode: only process first N conversations (0 = all, >0 = debug with N conversations)')
    parser.add_argument('--tokenizer-vocab', type=str, default=None,
                       help='Path to tokenizer vocab.json (defaults to ./data/gpt2-vocab.json)')
    parser.add_argument('--tokenizer-merges', type=str, default=None,
                       help='Path to tokenizer merges.txt (defaults to ./data/gpt2-merges.txt)')
    parser.add_argument('--tokenizer-model', type=str, default=None,
                       help='Path to a Hugging Face tokenizer directory (required for HuggingFaceTokenizer)')
    parser.add_argument('--tokenizer-type', type=str, default=None,
                       choices=['GPT2BPETokenizer', 'HuggingFaceTokenizer'],
                       help='Tokenizer implementation to use for Megatron preprocessing (auto-detect if omitted)')
    parser.add_argument('--chat-template-model', type=str, default=None,
                       help='Hugging Face model ID or path providing a chat template (optional)')
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

    global TOKENIZER_VOCAB_PATH, TOKENIZER_MERGE_PATH, TOKENIZER_MODEL_PATH, TOKENIZER_TYPE, CHAT_TEMPLATE_TOKENIZER

    tokenizer_type = args.tokenizer_type
    if tokenizer_type is None:
        if args.tokenizer_model:
            tokenizer_type = 'HuggingFaceTokenizer'
        elif os.path.exists(os.path.join(DATA_DIR, 'tokenizer.json')):
            tokenizer_type = 'HuggingFaceTokenizer'
        else:
            tokenizer_type = 'GPT2BPETokenizer'
    TOKENIZER_TYPE = tokenizer_type

    if TOKENIZER_TYPE == 'HuggingFaceTokenizer':
        if args.tokenizer_model:
            TOKENIZER_MODEL_PATH = os.path.abspath(args.tokenizer_model)
        elif os.path.exists(os.path.join(DATA_DIR, 'tokenizer.json')):
            TOKENIZER_MODEL_PATH = os.path.abspath(DATA_DIR)
        elif args.chat_template_model:
            TOKENIZER_MODEL_PATH = args.chat_template_model
        else:
            TOKENIZER_MODEL_PATH = None
        if not TOKENIZER_MODEL_PATH:
            raise ValueError('HuggingFaceTokenizer selected but no tokenizer model path provided (use --tokenizer-model).')
    else:
        TOKENIZER_MODEL_PATH = os.path.abspath(args.tokenizer_model) if args.tokenizer_model else None

    if args.tokenizer_vocab:
        TOKENIZER_VOCAB_PATH = os.path.abspath(args.tokenizer_vocab)
    else:
        TOKENIZER_VOCAB_PATH = os.path.abspath(os.path.join(DATA_DIR, 'gpt2-vocab.json'))
    if args.tokenizer_merges:
        TOKENIZER_MERGE_PATH = os.path.abspath(args.tokenizer_merges)
    else:
        TOKENIZER_MERGE_PATH = os.path.abspath(os.path.join(DATA_DIR, 'gpt2-merges.txt'))

    CHAT_TEMPLATE_TOKENIZER = None
    if args.chat_template_model:
        try:
            from transformers import AutoTokenizer
            CHAT_TEMPLATE_TOKENIZER = AutoTokenizer.from_pretrained(args.chat_template_model, trust_remote_code=True)
            if getattr(CHAT_TEMPLATE_TOKENIZER, 'chat_template', None) is None:
                print(f"Warning: model {args.chat_template_model!r} does not define a chat_template; using fallback ChatML format.")
                CHAT_TEMPLATE_TOKENIZER = None
            else:
                print(f"Loaded chat template from {args.chat_template_model}.")
        except Exception as exc:
            print(f"Warning: unable to load chat template from {args.chat_template_model}: {exc}. Using fallback ChatML format.")
            CHAT_TEMPLATE_TOKENIZER = None
    else:
        print('No chat template model specified; using fallback ChatML format.')

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

    # Determine if we're in debug mode
    debug_mode = args.debug > 0

    print("--- Starting Dataset Processing for MegaBlocks ---")
    print(f"Workers: {args.workers}")
    print(f"Count only: {args.count}")
    print(f"Rerun: {args.rerun}")
    if debug_mode:
        print(f"🐛 DEBUG MODE: Processing only {args.debug} conversations")
    else:
        print("Processing FULL dataset (all 376k+ conversations)")

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
        for dsc in datasets_config:
            _, count = count_dataset_samples(dsc)
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

    # Debug mode: limit dataset size for testing
    if debug_mode:
        print(f"\n🐛 DEBUG MODE: Limiting to first {args.debug} conversations")
        final_ds = final_ds.select(range(min(args.debug, len(final_ds))))
        print(f"Debug dataset size: {len(final_ds)} examples")

    # 2. Global Shuffle (Optional but recommended)
    if not debug_mode:  # Skip shuffle in debug mode for speed
        print(f"Shuffling the final dataset with seed {GLOBAL_SEED}...")
        final_ds = final_ds.shuffle(seed=GLOBAL_SEED)
        print("Shuffling complete.")
    else:
        print("Skipping shuffle in debug mode")

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

    # Process in batches with multiprocessing for tokenization

    print("Tokenizing conversations in parallel batches...")
    # Calculate optimal batch size for high-thread systems
    # With 160 threads available, we want smaller batches for better distribution
    target_conversations_per_batch = min(200, max(50, len(final_ds) // (args.workers * 10)))
    batch_size = max(1, target_conversations_per_batch)
    total_batches = (len(final_ds) + batch_size - 1) // batch_size
    print(f"Using batch size: {batch_size}, workers: {args.workers}, total batches: {total_batches}")
    estimated_throughput = args.workers * 150  # Conservative estimate
    print(f"Expected throughput: ~{estimated_throughput:,} conversations/second (vs 150 sequential)")
    expected_time_minutes = len(final_ds) / estimated_throughput / 60
    print(f"Estimated completion time: ~{expected_time_minutes:.1f} minutes (vs ~{len(final_ds)/150/60:.1f} minutes sequential)")

    # Prepare batches
    batches = []
    for i in range(0, len(final_ds), batch_size):
        batch_data = []
        for j in range(i, min(i + batch_size, len(final_ds))):
            conversations = final_ds[j].get(DEFAULT_CONVERSATION_FIELD, [])
            batch_data.append((j, conversations))
        batches.append(batch_data)

    print(f"Created {len(batches)} batches for processing")

    # Process batches in parallel
    from concurrent.futures import ProcessPoolExecutor, as_completed

    all_results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all batches
        future_to_batch = {executor.submit(tokenize_batch, batch): i for i, batch in enumerate(batches)}

        # Collect results with progress bar
        with tqdm(total=len(batches), desc="Processing batches") as pbar:
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    # Store results by original index
                    for orig_idx, token_ids, status in batch_results:
                        all_results[orig_idx] = (token_ids, status)
                except Exception as e:
                    print(f"Batch {batch_idx} failed with error: {e}")
                pbar.update(1)

    # Write results to binary file in original order
    print("Writing tokenized data to binary file...")
    for i in tqdm(range(len(final_ds)), desc="Writing to binary"):
        if i in all_results:
            token_ids, status = all_results[i]
            if status == "success" and token_ids:
                builder.add_document(token_ids, [len(token_ids)])
                total_tokens += len(token_ids)
                processed_count += 1
            else:
                skipped_count += 1
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
# Multiprocessing Helper Functions
# ==============================================================================

def tokenize_batch(batch_data):
    """Tokenize a batch of conversations in parallel worker"""
    import os
    import sys
    import io
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer warnings in multiprocessing

    batch_results = []
    # Setup tokenizer in each worker process (suppress verbose output)
    try:
        # Suppress tokenizer build messages in workers
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        local_tokenizer, _ = setup_megatron_tokenizer()

        # Restore output
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    except Exception as e:
        # Make sure to restore output even if there's an error
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        return [(idx, None, f"tokenizer_setup_error: {str(e)}") for idx, _ in batch_data]

    for idx, conversations in batch_data:
        if not conversations:
            batch_results.append((idx, None, "empty_conversations"))
            continue

        # Apply chat template
        formatted_text = apply_chat_template(conversations)
        if not formatted_text:
            batch_results.append((idx, None, "empty_formatted_text"))
            continue

        # Tokenize the formatted text
        try:
            token_ids = local_tokenizer.tokenize(formatted_text)
            if len(token_ids) > 0:
                # Add end-of-document token
                token_ids.append(local_tokenizer.eod)
                batch_results.append((idx, token_ids, "success"))
            else:
                batch_results.append((idx, None, "empty_tokens"))
        except Exception as e:
            batch_results.append((idx, None, f"tokenization_error: {str(e)}"))

    return batch_results

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    # Set global random seed for reproducibility if needed by other libraries
    random.seed(GLOBAL_SEED)
    main()
