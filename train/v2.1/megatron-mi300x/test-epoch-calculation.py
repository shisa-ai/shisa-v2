#!/usr/bin/env python3
"""
Test script for epoch calculation logic used in training scripts.
This script tests the .idx file parsing to extract number of samples.
"""

import struct
import os
import sys

def test_idx_file_parsing(idx_file_path):
    """Test parsing of Megatron .idx file to extract number of documents"""
    print(f"Testing .idx file: {idx_file_path}")

    if not os.path.exists(idx_file_path):
        print(f"ERROR: File does not exist: {idx_file_path}")
        return None

    try:
        with open(idx_file_path, 'rb') as f:
            # Read the header of the .idx file
            # Format:
            # - 8 bytes: magic number
            # - 1 byte: version
            # - 8 bytes: number of documents
            # - 8 bytes: number of tokens

            # Skip magic number (8 bytes) and version (1 byte)
            magic = f.read(8)
            version = f.read(1)

            print(f"Magic: {magic}")
            print(f"Version: {version}")

            # Read number of documents (8 bytes, little-endian unsigned long long)
            num_docs_bytes = f.read(8)
            if len(num_docs_bytes) != 8:
                print(f"ERROR: Could not read 8 bytes for number of documents")
                return None

            num_samples = struct.unpack('<Q', num_docs_bytes)[0]

            # Read number of tokens (optional, for info)
            num_tokens_bytes = f.read(8)
            if len(num_tokens_bytes) == 8:
                num_tokens = struct.unpack('<Q', num_tokens_bytes)[0]
                print(f"Number of tokens: {num_tokens:,}")

            print(f"Number of samples: {num_samples:,}")
            return num_samples

    except Exception as e:
        print(f"ERROR parsing .idx file: {e}")
        return None

def test_training_calculation(num_samples, epochs=3, global_batch_size=512):
    """Test the training steps calculation"""
    print(f"\n=== Training Calculation Test ===")
    print(f"Number of samples: {num_samples:,}")
    print(f"Epochs: {epochs}")
    print(f"Global batch size: {global_batch_size}")

    if num_samples <= 0:
        print("ERROR: Invalid number of samples")
        return 0

    training_steps = (num_samples * epochs) // global_batch_size
    print(f"Training steps: {training_steps}")

    # Calculate save interval (save at end of each epoch)
    save_interval = max(training_steps // epochs, 100)  # Minimum 100
    print(f"Save interval: {save_interval}")

    # Additional stats
    samples_per_step = global_batch_size
    total_samples_processed = num_samples * epochs
    print(f"Samples per step: {samples_per_step}")
    print(f"Total samples to process: {total_samples_processed:,}")

    return training_steps

def main():
    """Main test function"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')

    # Test files
    test_files = [
        'sft.shisa-v2.1_text_document.idx',
        # Add more test files if needed
    ]

    print("=== Epoch Calculation Test Script ===")
    print(f"Script directory: {script_dir}")
    print(f"Data directory: {data_dir}")

    for test_file in test_files:
        idx_file_path = os.path.join(data_dir, test_file)
        print(f"\n--- Testing {test_file} ---")

        num_samples = test_idx_file_parsing(idx_file_path)

        if num_samples is not None and num_samples > 0:
            test_training_calculation(num_samples)
        else:
            print("Skipping training calculation due to parsing error")

    # Test with some example values even if files don't exist
    print(f"\n--- Testing with Example Values ---")
    example_samples = [100, 1000, 10000, 100000, 200000]

    for samples in example_samples:
        print(f"\n** Example: {samples:,} samples **")
        test_training_calculation(samples)

if __name__ == "__main__":
    main()