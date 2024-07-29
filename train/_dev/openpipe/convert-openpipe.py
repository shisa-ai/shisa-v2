# https://docs.openpipe.ai/features/uploading-data

import json
import sys

def convert_conversation(conversation):
    role_mapping = {
        'system': 'system',
        'human': 'user',
        'gpt': 'assistant'
    }
    
    messages = []
    for msg in conversation:
        role = role_mapping.get(msg['from'], msg['from'])
        content = msg['value']
        messages.append({"role": role, "content": content})
    
    return {"messages": messages}

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            converted = convert_conversation(data['conversations'])
            json.dump(converted, outfile)
            outfile.write('\n')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file.jsonl>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = input_file.rsplit('.', 1)[0] + '-openpipe.jsonl'
    
    process_file(input_file, output_file)
    print(f"Conversion complete. Output saved to {output_file}")
