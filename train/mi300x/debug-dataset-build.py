#!/usr/bin/env python3
"""
Debug script to understand why dataset build results in only 1 sample
"""

import os
import sys
import struct

def analyze_idx_file(idx_file_path):
    """Analyze the structure of an .idx file"""
    if not os.path.exists(idx_file_path):
        print(f"File does not exist: {idx_file_path}")
        return

    with open(idx_file_path, 'rb') as f:
        file_size = os.path.getsize(idx_file_path)
        print(f"File size: {file_size:,} bytes")

        # Read header
        magic = f.read(8)
        version = f.read(1)
        num_docs = struct.unpack('<Q', f.read(8))[0]
        num_tokens = struct.unpack('<Q', f.read(8))[0]

        print(f"Magic: {magic}")
        print(f"Version: {ord(version)}")
        print(f"Number of documents: {num_docs:,}")
        print(f"Number of tokens: {num_tokens:,}")

        # The rest should be document pointers
        remaining_bytes = file_size - 25
        num_pointers = remaining_bytes // 8
        print(f"Document pointers expected: {num_pointers:,}")

        if num_pointers > 0:
            print("\nFirst 10 document lengths:")
            for i in range(min(10, num_pointers)):
                doc_len_bytes = f.read(8)
                if len(doc_len_bytes) == 8:
                    doc_len = struct.unpack('<Q', doc_len_bytes)[0]
                    print(f"  Document {i}: {doc_len:,} tokens")
                else:
                    break

        # Key insight: if num_docs is 1 but we have many pointers,
        # then multiple conversations got combined into one document

def analyze_bin_file(bin_file_path, max_tokens=100):
    """Analyze the binary file structure"""
    if not os.path.exists(bin_file_path):
        print(f"File does not exist: {bin_file_path}")
        return

    file_size = os.path.getsize(bin_file_path)
    print(f"Binary file size: {file_size:,} bytes")

    # Each token is stored as int16 (2 bytes) or int32 (4 bytes)
    # Let's assume int32 for now
    expected_tokens = file_size // 4
    print(f"Expected tokens (assuming int32): {expected_tokens:,}")

    with open(bin_file_path, 'rb') as f:
        print(f"\nFirst {max_tokens} tokens:")
        for i in range(max_tokens):
            token_bytes = f.read(4)
            if len(token_bytes) == 4:
                token = struct.unpack('<I', token_bytes)[0]  # unsigned int32
                print(f"  Token {i}: {token}")
            else:
                break

def main():
    data_dir = './data'
    base_name = 'sft.shisa-v2.1_text_document'

    idx_file = os.path.join(data_dir, f'{base_name}.idx')
    bin_file = os.path.join(data_dir, f'{base_name}.bin')

    print("=== Dataset Debug Analysis ===")
    print(f"Analyzing: {base_name}")

    print("\n--- Index File Analysis ---")
    analyze_idx_file(idx_file)

    print("\n--- Binary File Analysis ---")
    analyze_bin_file(bin_file, max_tokens=20)

if __name__ == "__main__":
    main()