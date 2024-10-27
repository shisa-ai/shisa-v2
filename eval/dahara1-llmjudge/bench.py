import os
import gc
import re
import xml.etree.ElementTree as ET
import time
import json
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM


def clean_xml(xml_string):
    # Remove code block markers and language indicators
    clean = re.sub(r'```(?:XML)?\n?', '', xml_string)

    # Remove standalone 'xml' lines
    clean = re.sub(r'^xml\s*\n', '', clean, flags=re.MULTILINE)

    # Remove XML declaration line
    clean = re.sub(r'<\?xml[^>]+\?>\s*', '', clean)

    # Extract content between <result> and </result> tags if present
    result_match = re.search(r'<result>.*?</result>', clean, re.DOTALL)
    if result_match:
        clean = result_match.group(0)
    else:
        # Wrap content in result tags if it doesn't start with '<'
        if not clean.startswith('<'):
            clean = f'<result>\n{clean}\n</result>'

    # Remove leading/trailing whitespace
    clean = clean.strip()

    return clean


def parse_verdict(verdict):
    verdict_scores = {
        'A is much better': 1.5,
        'A is better': 1.0,
        'A is slightly better': 0.5,
        'same': 0.0,
        'B is slightly better': -0.5,
        'B is better': -1.0,
        'B is much better': -1.5
    }
    return verdict_scores.get(verdict.strip(), 0.0)
    

def generate_prompt(inst, response_a, response_b):
    prompt = f"""You will be given a user question and two responses, Response A and Response B, provided by two AI assistants.
Your task is to act as a judge by determining which response is answering the user's question better.

When you are evaluating, you can consider the following criteria:
- Does the response fully answer the user's question?
- Does the response address the key points in the question?
- Is the response clearly written and avoiding unnecessary information?
- Is the response creative, especially when the question is asking for generating creative content?
- Does the response contain factual information?
- Does the response NOT contain any harmful, unsafe, dangerous, or sexually explicit content?
- Does the response refuse to answer to the question that asks for harmful, unsafe, dangerous, or sexually explicit content?

You will provide a short explanation and your final rating (verdict) in the following XML format.

<result>
  <explanation>YOUR EXPLANATION GOES HERE.</explanation>
  <verdict>A is slightly better</verdict>
</result>

Your explanation can compare the two responses and describe your rationale behind the rating.
It should be about two or three sentences.
Your final rating (verdict) must be in 7-point Likert and must be exactly one of the following:
['A is much better', 'A is better', 'A is slightly better', 'same', 'B is slightly better', 'B is better', 'B is much better'].

[User Question]
{inst}

[The Start of Response A]
{response_a}
[The End of Response A]

[The Start of Response B]
{response_b}
[The End of Response B]

[Result with explanation and verdict in the above XML format]
    """

    return prompt 


def check_bench(model, prompt):
    system_prompt = ""

    message = []
    if "gemma" in model.config.name_or_path:
         messages = [
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
        ]

    if os.getenv('DEBUG'):
        """
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        """
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        input_length = inputs.shape[1]
        response_only = tokenizer.decode(inputs[0][input_length:], skip_special_tokens=True)
        if response_only == "":
            return """<result>
  <explanation>TEST.</explanation>
  <verdict>A is slightly better</verdict>
</result>"""
        else:
            print("dame!")
            import sys
            sys.exit()
    else:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        attention_mask = torch.ones_like(inputs)
        input_length = inputs.shape[1]

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=3, max_new_tokens=600, do_sample=False,
                top_p=None,
                temperature=None,
                repetition_penalty=1.0
            )
        response_only = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True)
        return response_only


def process_jsonl_files(model, tokenizer, input_file, output_file, checkpoint_file, start_line=0):
    examples = []
    last_processed_line = start_line

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            examples = checkpoint_data['examples']
            last_processed_line = checkpoint_data['last_processed_line']
            print(f"Resuming from line {last_processed_line}")

    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if i <= last_processed_line:
                continue

            try:
                data = json.loads(line)
                inst = data['prompt']
                response_a = data['response_a']
                response_b = data['response_b']

                prompt = generate_prompt(inst, response_a, response_b)
                judge_xml = check_bench(model, prompt)
                clean_judge_xml = clean_xml(judge_xml)
                try:
                    root = ET.fromstring(clean_judge_xml)
                    explanation = root.find('explanation').text
                    verdict = root.find('verdict').text
                except ET.ParseError as e:
                    print(f"Error parsing XML: {e}")
                    print(f"Problematic XML:\n[{clean_judge_xml}]")
                    explanation = "XML parsing failed"
                    verdict = ""

                tag = "Japanese to English" if "Japanese to English" in inst else "English to Japanese"
                score = parse_verdict(verdict)

                example = {
                    "input_text": inst,
                    "tags": [tag],
                    "output_text_a": response_a,
                    "output_text_b": response_b,
                    "score": score,
                    "individual_rater_scores": [],
                    "custom_fields": {"explanation": explanation}
                }
                examples.append(example)

                if i % 10 == 0:
                    save_checkpoint(checkpoint_file, examples, i)
                    print(f"Processed {i} lines")

            except Exception as e:
                print(f"Error processing line {i}: {str(e)}")
                write_error_to_output(output_file, i, str(e))

            last_processed_line = i

    save_final_result(output_file, examples)
    
    return examples

def save_checkpoint(checkpoint_file, examples, last_processed_line):
    checkpoint_data = {
        'examples': examples,
        'last_processed_line': last_processed_line
    }
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

def write_error_to_output(output_file, line_number, error_message):
    error_data = {"error": f"Error processing line {line_number}: {error_message}"}
    with open(output_file, 'a', encoding='utf-8') as f:
        json.dump(error_data, f, ensure_ascii=False)
        f.write('\n')

def save_final_result(output_file, examples):
    final_json = {
        "metadata": {
            "source_path": model_name,
            "custom_fields_schema": []
        },
        "models": [
            {"name": "thinking version"},
            {"name": "standard version"}
        ],
        "examples": examples
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":

    all_models = (
        "alpindale/WizardLM-2-8x22B",
        "CohereForAI/c4ai-command-r-plus-08-2024",
        "deepseek-ai/DeepSeek-V2.5",
        "meta-llama/Llama-3.1-405B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mgoin/Nemotron-4-340B-Instruct-hf",
        "microsoft/GRIN-MoE",
        "mistralai/Mistral-Large-Instruct-2407",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "NousResearch/Hermes-3-Llama-3.1-405B",

    )

    for model_name in all_models:
        tokenizer_name = model_name

        if os.getenv('DEBUG'):
            tokenizer_name = tokenizer_name.split("models/")[1]
            tokenizer_name = tokenizer_name.replace("/", "/")

        try:
            model_filename = model_name.split("/")[-1]
            print(f"model_filename: {model_filename}")

            input_file = 'llm_judge_wm23_inputs.jsonl'
            output_file = f'llm_judge_wm23_{model_filename}_outputs.jsonl'
            checkpoint_file = f'processing_wm23_{model_filename}_checkpoint.json'
            if os.path.exists(output_file):
                continue

            print(f"start {model_name}")
            start_time = time.time()

            model = ""
            if os.getenv('DEBUG'):
                pass
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    use_cache=True,
                )

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id  # モデルにpad_token_idを設定

            examples = process_jsonl_files(model, tokenizer, input_file, output_file, checkpoint_file, 0)

        finally:
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()
        checkpoint_file
        os.remove(checkpoint_file)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"end {model_name} : elapsed_time : {elapsed_time:.3f} second")



