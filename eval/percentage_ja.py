import json
import os
import string

def is_japanese_char(char):
    # Check if the character falls within the Japanese Unicode codepoint ranges
    return any([
        0x3040 <= ord(char) <= 0x309F,  # Hiragana
        0x30A0 <= ord(char) <= 0x30FF,  # Katakana
        0x4E00 <= ord(char) <= 0x9FFF   # Kanji
    ])

def is_acceptable_char(char):
    # Check if the character is a common punctuation or numeral
    return char in string.punctuation or char.isdigit()

def calculate_non_japanese_percentage(text):
    total_chars = len(text)
    japanese_chars = sum(is_japanese_char(char) or is_acceptable_char(char) for char in text)
    non_japanese_chars = total_chars - japanese_chars
    percentage = (non_japanese_chars / total_chars) * 100 if total_chars > 0 else 0
    return percentage

def analyze_jsonl_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(directory, filename)
            total_text = ""
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    data = json.loads(line)
                    for choice in data["choices"]:
                        for turn in choice["turns"]:
                            total_text += turn
            percentage = calculate_non_japanese_percentage(total_text)
            percentage = 100-percentage
            print(f"File: {filename}, Japanese Percentage: {percentage:.2f}%")

# Usage
directory = "./"
analyze_jsonl_files(directory)
