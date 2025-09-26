#!/usr/bin/env python3
"""
Comprehensive dataset analysis script for MegaBlocks training.
This script analyzes the dataset files and provides training configuration insights.
"""

import struct
import os
import sys
import json
from pathlib import Path

def analyze_idx_file(idx_file_path):
    """Analyze Megatron .idx file to extract dataset statistics"""
    print(f"=== Analyzing .idx file ===")
    print(f"File path: {idx_file_path}")

    if not os.path.exists(idx_file_path):
        print(f"❌ ERROR: File does not exist: {idx_file_path}")
        return None

    try:
        file_size = os.path.getsize(idx_file_path)
        print(f"File size: {file_size:,} bytes")

        with open(idx_file_path, 'rb') as f:
            # Read the header of the .idx file
            # Format:
            # - 8 bytes: magic number (MMIDIDX\x00)
            # - 1 byte: version
            # - 8 bytes: number of documents
            # - 8 bytes: number of tokens

            magic = f.read(9)
            version_bytes = f.read(8)
            dtype_code_bytes = f.read(1)

            print(f"Magic number: {magic}")
            version = struct.unpack('<Q', version_bytes)[0]
            print(f"Version: {version}")

            # Read number of sequences and documents
            sequence_count_bytes = f.read(8)
            document_count_bytes = f.read(8)
            if len(sequence_count_bytes) != 8 or len(document_count_bytes) != 8:
                print(f"❌ ERROR: Could not read sequence/document counts")
                return None

            sequence_count = struct.unpack('<Q', sequence_count_bytes)[0]
            document_count = struct.unpack('<Q', document_count_bytes)[0]

            # Document array includes an end sentinel, so subtract one when positive
            num_samples = document_count - 1 if document_count > 0 else sequence_count

            # Read number of tokens (optional, may not exist in older formats)
            num_tokens_bytes = f.read(8)
            if len(num_tokens_bytes) == 8:
                num_tokens = struct.unpack('<Q', num_tokens_bytes)[0]
                print(f"Number of tokens: {num_tokens:,}")
            else:
                num_tokens = None
                print("Number of tokens: Not available")

            print(f"Sequence count: {sequence_count:,}")
            print(f"Document count (including sentinel): {document_count:,}")
            print(f"Number of samples: {num_samples:,}")

            return {
                'num_samples': num_samples,
                'num_tokens': num_tokens,
                'file_size': file_size,
                'magic': magic,
                'version': version
            }

    except Exception as e:
        print(f"❌ ERROR parsing .idx file: {e}")
        return None

def analyze_bin_file(bin_file_path):
    """Analyze the .bin file"""
    print(f"\n=== Analyzing .bin file ===")
    print(f"File path: {bin_file_path}")

    if not os.path.exists(bin_file_path):
        print(f"❌ ERROR: File does not exist: {bin_file_path}")
        return None

    try:
        file_size = os.path.getsize(bin_file_path)
        print(f"File size: {file_size:,} bytes ({file_size / (1024**2):.1f} MB)")

        # Try to estimate number of tokens from file size
        # Assuming 2 bytes per token (int16) or 4 bytes per token (int32)
        estimated_tokens_int16 = file_size // 2
        estimated_tokens_int32 = file_size // 4

        print(f"Estimated tokens (if int16): {estimated_tokens_int16:,}")
        print(f"Estimated tokens (if int32): {estimated_tokens_int32:,}")

        return {
            'file_size': file_size,
            'estimated_tokens_int16': estimated_tokens_int16,
            'estimated_tokens_int32': estimated_tokens_int32
        }

    except Exception as e:
        print(f"❌ ERROR analyzing .bin file: {e}")
        return None

def analyze_vocab_files(data_dir):
    """Analyze vocabulary and merge files"""
    print(f"\n=== Analyzing vocabulary files ===")

    vocab_file = os.path.join(data_dir, 'gpt2-vocab.json')
    merge_file = os.path.join(data_dir, 'gpt2-merges.txt')

    vocab_info = {}

    # Analyze vocab file
    if os.path.exists(vocab_file):
        try:
            with open(vocab_file, 'r') as f:
                vocab = json.load(f)
            vocab_size = len(vocab)
            vocab_info['vocab_size'] = vocab_size
            vocab_info['vocab_file_exists'] = True
            print(f"✅ Vocabulary file found: {vocab_file}")
            print(f"   Vocabulary size: {vocab_size:,} tokens")
        except Exception as e:
            print(f"❌ Error reading vocab file: {e}")
            vocab_info['vocab_file_exists'] = False
    else:
        print(f"❌ Vocabulary file not found: {vocab_file}")
        vocab_info['vocab_file_exists'] = False

    # Analyze merge file
    if os.path.exists(merge_file):
        try:
            with open(merge_file, 'r') as f:
                lines = f.readlines()
            merge_count = len([line for line in lines if line.strip() and not line.startswith('#')])
            vocab_info['merge_count'] = merge_count
            vocab_info['merge_file_exists'] = True
            print(f"✅ Merge file found: {merge_file}")
            print(f"   Number of merges: {merge_count:,}")
        except Exception as e:
            print(f"❌ Error reading merge file: {e}")
            vocab_info['merge_file_exists'] = False
    else:
        print(f"❌ Merge file not found: {merge_file}")
        vocab_info['merge_file_exists'] = False

    return vocab_info

def calculate_training_configurations(num_samples, num_tokens=None):
    """Calculate training configurations for different scenarios"""
    print(f"\n=== Training Configuration Analysis ===")

    if num_samples <= 0:
        print("❌ ERROR: Invalid number of samples")
        return

    print(f"Dataset has {num_samples:,} samples")
    if num_tokens:
        print(f"Dataset has {num_tokens:,} tokens")
        avg_tokens_per_sample = num_tokens / num_samples if num_samples > 0 else 0
        print(f"Average tokens per sample: {avg_tokens_per_sample:.1f}")

    # Test different configurations
    configurations = [
        {"epochs": 1, "batch_size": 32, "name": "Quick test (1 epoch, small batch)"},
        {"epochs": 3, "batch_size": 32, "name": "Small batch training"},
        {"epochs": 3, "batch_size": 128, "name": "Medium batch training"},
        {"epochs": 3, "batch_size": 512, "name": "Default configuration (current scripts)"},
        {"epochs": 5, "batch_size": 512, "name": "Extended training"},
        {"epochs": 1, "batch_size": 1, "name": "Single sample debugging"},
    ]

    print(f"\n{'Configuration':<40} {'Steps':<8} {'Save Interval':<12} {'Status'}")
    print("-" * 80)

    for config in configurations:
        epochs = config["epochs"]
        batch_size = config["batch_size"]
        name = config["name"]

        total_samples = num_samples * epochs
        training_steps = total_samples // batch_size

        # Calculate save interval (save at end of each epoch, minimum 1)
        if training_steps > 0:
            save_interval = max(training_steps // epochs, 1)
        else:
            save_interval = 1

        # Status
        if training_steps == 0:
            status = "❌ NO TRAINING"
        elif training_steps < 10:
            status = "⚠️  Very short"
        elif training_steps < 100:
            status = "⚠️  Short"
        else:
            status = "✅ Good"

        print(f"{name:<40} {training_steps:<8} {save_interval:<12} {status}")

def analyze_data_completeness(data_dir, data_prefix):
    """Check if all required data files exist"""
    print(f"\n=== Data Completeness Check ===")

    required_files = [
        f"{data_prefix}.idx",
        f"{data_prefix}.bin",
        "gpt2-vocab.json",
        "gpt2-merges.txt"
    ]

    all_present = True
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✅ {filename} (size: {size:,} bytes)")
        else:
            print(f"❌ {filename} - MISSING")
            all_present = False

    if all_present:
        print("✅ All required files are present")
    else:
        print("❌ Some required files are missing")
        print("\nTo generate missing files, run:")
        print("   python 02-generate.sft.shisa-v2.1-megablocks.py")

    return all_present

def provide_recommendations(idx_info, vocab_info):
    """Provide recommendations based on analysis"""
    print(f"\n=== Recommendations ===")

    if not idx_info:
        print("❌ Cannot analyze dataset - .idx file issues")
        return

    num_samples = idx_info['num_samples']

    # Sample count recommendations
    if num_samples == 0:
        print("❌ CRITICAL: Dataset has 0 samples")
        print("   → Regenerate dataset with 02-generate.sft.shisa-v2.1-megablocks.py")
    elif num_samples < 100:
        print(f"⚠️  WARNING: Very small dataset ({num_samples} samples)")
        print("   → Consider reducing batch size to 1-32 for testing")
        print("   → Training steps will be minimal with default batch size (512)")
    elif num_samples < 1000:
        print(f"⚠️  Small dataset ({num_samples:,} samples)")
        print("   → Consider reducing batch size to 32-128")
        print("   → May work with default settings but training will be short")
    elif num_samples < 10000:
        print(f"✅ Moderate dataset ({num_samples:,} samples)")
        print("   → Should work well with batch sizes 128-512")
    else:
        print(f"✅ Large dataset ({num_samples:,} samples)")
        print("   → Excellent for training with default batch size 512")

    # Vocabulary recommendations
    if not vocab_info.get('vocab_file_exists') or not vocab_info.get('merge_file_exists'):
        print("⚠️  Missing vocabulary files")
        print("   → Will be downloaded automatically during training")

    # Batch size recommendations
    print(f"\n📋 Recommended batch sizes for {num_samples:,} samples:")
    recommended_batches = [1, 8, 16, 32, 64, 128, 256, 512]
    for batch_size in recommended_batches:
        steps = (num_samples * 3) // batch_size  # 3 epochs
        if steps > 0:
            print(f"   Batch size {batch_size:>3}: {steps:>4} training steps")
        else:
            print(f"   Batch size {batch_size:>3}: ❌ No training steps")

def main():
    """Main analysis function"""
    # Determine paths
    script_dir = Path(__file__).parent.absolute()
    data_dir = script_dir / 'data'
    data_prefix = 'sft.shisa-v2.1_text_document'

    print("🔍 MegaBlocks Dataset Analysis Tool")
    print("=" * 50)
    print(f"Script directory: {script_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Data prefix: {data_prefix}")

    # Check data completeness
    completeness = analyze_data_completeness(data_dir, data_prefix)

    # Analyze .idx file
    idx_file_path = data_dir / f"{data_prefix}.idx"
    idx_info = analyze_idx_file(str(idx_file_path))

    # Analyze .bin file
    bin_file_path = data_dir / f"{data_prefix}.bin"
    bin_info = analyze_bin_file(str(bin_file_path))

    # Analyze vocabulary files
    vocab_info = analyze_vocab_files(str(data_dir))

    # Calculate training configurations
    if idx_info:
        calculate_training_configurations(
            idx_info['num_samples'],
            idx_info.get('num_tokens')
        )

    # Provide recommendations
    provide_recommendations(idx_info, vocab_info)

    # Summary
    print(f"\n" + "=" * 50)
    print("🏁 Analysis Complete")

    if idx_info:
        print(f"📊 Dataset: {idx_info['num_samples']:,} samples")
        if idx_info.get('num_tokens'):
            print(f"🔤 Tokens: {idx_info['num_tokens']:,} tokens")

    if completeness and idx_info and idx_info['num_samples'] > 0:
        print("✅ Ready for training!")
    else:
        print("❌ Dataset needs attention before training")

if __name__ == "__main__":
    main()
