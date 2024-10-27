import os
import gc
import re
import xml.etree.ElementTree as ET
import time
import json
import torch
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from tqdm import tqdm
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log'),
        logging.StreamHandler()
    ]
)

class InferenceBackend(ABC):
    """Abstract base class for inference backends"""
    @abstractmethod
    def generate_batch(self, prompts: List[str]) -> List[str]:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass

class VLLMBackend(InferenceBackend):
    """vLLM implementation of inference backend"""
    def __init__(self, model_name: str, batch_size: int):
        from vllm import LLM, SamplingParams
        self.model_name = model_name
        self.batch_size = batch_size
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_num_batched_tokens=4096
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=600,
            presence_penalty=0.0,
            frequency_penalty=0.0
        )

    def generate_batch(self, prompts: List[str]) -> List[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def get_name(self) -> str:
        return "vLLM"

class HFBackend(InferenceBackend):
    """HuggingFace Transformers implementation of inference backend"""
    def __init__(self, model_name: str, batch_size: int):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_cache=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def generate_batch(self, prompts: List[str]) -> List[str]:
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=4096
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=600,
                num_beams=3,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        return self.tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    def get_name(self) -> str:
        return "HuggingFace"

def create_inference_backend(model_name: str, batch_size: int) -> InferenceBackend:
    """Factory function that tries to create vLLM backend first, falls back to HF"""
    try:
        import vllm
        logging.info(f"Attempting to load {model_name} with vLLM backend...")
        try:
            backend = VLLMBackend(model_name, batch_size)
            logging.info("Successfully created vLLM backend")
            return backend
        except Exception as e:
            logging.warning(f"Failed to create vLLM backend: {str(e)}")
            raise
    except ImportError:
        logging.info("vLLM not available")
    except Exception as e:
        logging.warning(f"vLLM initialization failed: {str(e)}")

    logging.info(f"Falling back to HuggingFace backend for {model_name}")
    return HFBackend(model_name, batch_size)

@dataclass
class ProcessingStats:
    total_examples: int
    processing_time: float
    examples_per_second: float
    total_tokens: int
    tokens_per_second: float
    batch_size: int
    model_name: str
    backend: str

def read_existing_output(output_file: str) -> List[Dict]:
    """Read the output file and return processed examples."""
    if not os.path.exists(output_file):
        return []

    with open(output_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return data.get('examples', [])
        except json.JSONDecodeError as e:
            logging.error(f"Failed to read {output_file}: {str(e)}")
            return []

def save_final_result(output_file: str, examples: List[Dict], model_name: str, stats: ProcessingStats):
    """Save the final results including metadata to output_file."""
    final_json = {
        "metadata": {
            "source_path": model_name,
            "custom_fields_schema": []
        },
        "models": [
            {"name": model_name}
        ],
        "examples": examples
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

def load_jsonl_batch(input_file: str, start_idx: int, batch_size: int) -> (List[Dict], bool):
    """Load a batch of examples from a JSONL input file."""
    examples = []
    is_end = False

    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx < start_idx:
                continue
            if len(examples) >= batch_size:
                break
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing JSON on line {idx + 1}: {str(e)}")

        if idx + 1 == start_idx + len(examples):
            is_end = True

    return examples, is_end

def save_checkpoint(checkpoint_file: str, examples: List[Dict], last_processed_line: int):
    """Save a checkpoint to resume processing if interrupted."""
    checkpoint_data = {
        'examples': examples,
        'last_processed_line': last_processed_line
    }
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)


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


def generate_prompt(inst: str, response_a: str, response_b: str) -> str:
    """Generate a prompt for comparing two responses."""
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

### XML Cleanup
def clean_xml(xml_string: str) -> str:
    """Clean and normalize XML string with multiple fallback mechanisms."""
    # Remove code block markers and language indicators
    clean = re.sub(r'```(?:XML)?\n?', '', xml_string)
    clean = re.sub(r'^xml\s*\n', '', clean, flags=re.MULTILINE)
    clean = re.sub(r'<\?xml[^>]+\?>\s*', '', clean)
    
    # Fix common XML issues
    clean = re.sub(r'&(?!amp;|lt;|gt;|apos;|quot;)', '&amp;', clean)  # Fix unescaped ampersands
    clean = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', clean)   # Remove invalid XML characters
    
    # Try to extract content between <result> tags
    result_match = re.search(r'<result>.*?</result>', clean, re.DOTALL)
    if result_match:
        clean = result_match.group(0)
    else:
        # If no <result> tags found, try to extract explanation and verdict using regex
        explanation_match = re.search(r'<explanation>(.*?)</explanation>', clean, re.DOTALL)
        verdict_match = re.search(r'<verdict>(.*?)</verdict>', clean, re.DOTALL)
        
        if explanation_match and verdict_match:
            clean = f"<result>\n  <explanation>{explanation_match.group(1)}</explanation>\n  <verdict>{verdict_match.group(1)}</verdict>\n</result>"
        elif not clean.startswith('<'):
            # Last resort: wrap everything in result tags
            clean = f'<result>\n{clean}\n</result>'
    
    # Ensure proper XML structure
    clean = clean.replace('<<', '<').replace('>>', '>')  # Fix double brackets
    clean = re.sub(r'\s+', ' ', clean)  # Normalize whitespace
    clean = clean.strip()
    
    return clean

def extract_explanation_verdict(xml_string: str) -> tuple[str, str]:
    """Extract explanation and verdict with multiple parsing strategies."""
    try:
        # First try: Standard XML parsing
        root = ET.fromstring(xml_string)
        explanation = root.find('explanation')
        verdict = root.find('verdict')
        
        if explanation is not None and verdict is not None:
            return explanation.text.strip(), verdict.text.strip()
        
        # Second try: Direct regex matching if XML parsing succeeds but elements not found
        explanation_text = ""
        verdict_text = ""
        
        explanation_match = re.search(r'<explanation>(.*?)</explanation>', xml_string, re.DOTALL)
        if explanation_match:
            explanation_text = explanation_match.group(1).strip()
        
        verdict_match = re.search(r'<verdict>(.*?)</verdict>', xml_string, re.DOTALL)
        if verdict_match:
            verdict_text = verdict_match.group(1).strip()
        
        if explanation_text and verdict_text:
            return explanation_text, verdict_text
        
    except ET.ParseError as e:
        print(f"XML parsing failed: {str(e)}")
        # Third try: Fallback to regex with more flexible patterns
        explanation_text = ""
        verdict_text = ""
        
        # Try to find explanation
        explanation_patterns = [
            r'<explanation>(.*?)</explanation>',
            r'explanation:\s*(.*?)\s*(?:<verdict>|$)',
            r'explanation"?\s*:\s*(.*?)\s*(?:verdict|$)'
        ]
        
        for pattern in explanation_patterns:
            match = re.search(pattern, xml_string, re.DOTALL | re.IGNORECASE)
            if match:
                explanation_text = match.group(1).strip()
                break
        
        # Try to find verdict
        verdict_patterns = [
            r'<verdict>(.*?)</verdict>',
            r'verdict:\s*((?:A|B)\s+is\s+(?:much\s+)?(?:slightly\s+)?better|same)',
            r'verdict"?\s*:\s*((?:A|B)\s+is\s+(?:much\s+)?(?:slightly\s+)?better|same)'
        ]
        
        for pattern in verdict_patterns:
            match = re.search(pattern, xml_string, re.DOTALL | re.IGNORECASE)
            if match:
                verdict_text = match.group(1).strip()
                break
        
        if explanation_text or verdict_text:
            return (
                explanation_text or "Failed to parse explanation",
                verdict_text or "same"  # Default to 'same' if no verdict found
            )
    
    # Final fallback
    return "Failed to parse explanation", "same"

def validate_verdict(verdict: str) -> str:
    """Validate and normalize verdict string."""
    valid_verdicts = {
        'a is much better': 'A is much better',
        'a is better': 'A is better',
        'a is slightly better': 'A is slightly better',
        'same': 'same',
        'b is slightly better': 'B is slightly better',
        'b is better': 'B is better',
        'b is much better': 'B is much better'
    }
    
    normalized = verdict.lower().strip()
    return valid_verdicts.get(normalized, 'same')


### Processing
def process_batch(backend: InferenceBackend, batch_examples: List[Dict], stats: Dict) -> List[Dict]:
    """Process a batch of examples with the given backend."""
    prompts = [
        generate_prompt(ex['prompt'], ex['response_a'], ex['response_b'])
        for ex in batch_examples
    ]
    responses = backend.generate_batch(prompts)

    processed_batch = []
    for i, response in enumerate(responses):
        try:
            clean_response = clean_xml(response)
            explanation, verdict = extract_explanation_verdict(clean_response)
            verdict = validate_verdict(verdict)

            tag = "Japanese to English" if "Japanese to English" in batch_examples[i]['prompt'] else "English to Japanese"
            score = parse_verdict(verdict)

            processed_example = {
                "input_text": batch_examples[i]['prompt'],
                "tags": [tag],
                "output_text_a": batch_examples[i]['response_a'],
                "output_text_b": batch_examples[i]['response_b'],
                "score": score,
                "individual_rater_scores": [],
                "custom_fields": {
                    "explanation": explanation,
                    "raw_response": response,  # Optional: keep raw response for debugging
                    "cleaned_xml": clean_response  # Optional: keep cleaned XML for debugging
                }
            }
            processed_batch.append(processed_example)
        except Exception as e:
            logging.error(f"Error processing example {i}: {str(e)}")

    stats['total_examples'] += len(batch_examples)
    return processed_batch

def process_model(
    model_name: str,
    input_file: str,
    output_file: str,
    checkpoint_file: str,
    batch_size: int
) -> Optional[ProcessingStats]:
    """Process a single model with timing statistics, resuming if needed."""
    if os.path.exists(output_file):
        logging.info(f"Loading existing output from {output_file}")
        processed_examples = read_existing_output(output_file)
        current_idx = len(processed_examples)
        if current_idx > 0:
            logging.info(f"Resuming {model_name} from line {current_idx}")
    else:
        processed_examples = []
        current_idx = 0

    start_time = time.time()
    stats = {
        'total_batches': 0,
        'total_examples': 0,
        'total_time': 0,
        'total_tokens': 0  # If you want to track tokens, need to add token counting logic
    }

    try:
        backend = create_inference_backend(model_name, batch_size)
        logging.info(f"Processing {model_name} using {backend.get_name()} backend")

        pbar = tqdm(desc=f"Processing {model_name}", unit="examples", initial=current_idx, dynamic_ncols=True)

        while True:
            batch_examples, is_end = load_jsonl_batch(input_file, current_idx, batch_size)
            if not batch_examples:
                break

            processed_batch = process_batch(backend, batch_examples, stats)
            processed_examples.extend(processed_batch)

            current_idx += len(batch_examples)
            pbar.update(len(batch_examples))

            # Save intermediate checkpoint
            if current_idx % (batch_size * 2) == 0:
                save_checkpoint(checkpoint_file, processed_examples, current_idx)

                # Intermediate logging
                elapsed = time.time() - start_time
                examples_per_sec = stats['total_examples'] / elapsed if elapsed > 0 else 0
                logging.info(
                    f"Progress: {current_idx} examples, "
                    f"Speed: {examples_per_sec:.2f} examples/sec, "
                    f"Backend: {backend.get_name()}"
                )

            if is_end:
                break

        pbar.close()
        end_time = time.time()
        total_time = end_time - start_time

        # Calculate final statistics
        processing_stats = ProcessingStats(
            total_examples=len(processed_examples),
            processing_time=total_time,
            examples_per_second=len(processed_examples) / total_time if total_time > 0 else 0,
            total_tokens=stats['total_tokens'],
            tokens_per_second=stats['total_tokens'] / total_time if total_time > 0 else 0,
            batch_size=batch_size,
            model_name=model_name,
            backend=backend.get_name()
        )

        # Save final results
        save_final_result(output_file, processed_examples, model_name, processing_stats)

        # Clean up checkpoint file
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        return processing_stats

    except Exception as e:
        logging.error(f"Error processing {model_name}: {str(e)}")
        return None

    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def main():
    BATCH_SIZE = 32  # Adjust based on your GPU memory
    
    all_models = [
        "meta-llama/Llama-3.1-8B-Instruct",
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
    ]

    results = []
    for model_name in all_models:
        model_filename = model_name.split("/")[-1]
        input_file = 'llm_judge_wm23_inputs.jsonl'
        output_file = f'llm_judge_wm23_{model_filename}_outputs.jsonl'
        checkpoint_file = f'processing_wm23_{model_filename}_checkpoint.json'

        stats = process_model(
            model_name=model_name,
            input_file=input_file,
            output_file=output_file,
            checkpoint_file=checkpoint_file,
            batch_size=BATCH_SIZE
        )

        if stats:
            results.append({
                'model': model_name,
                'backend': stats.backend,
                'examples_per_second': stats.examples_per_second,
                'total_examples': stats.total_examples,
                'processing_time': stats.processing_time,
                'batch_size': stats.batch_size
            })

            # Print summary after each model
            logging.info(f"\nResults for {model_name}:")
            logging.info(f"Backend: {stats.backend}")
            logging.info(f"Processing time: {stats.processing_time:.2f} seconds")
            logging.info(f"Speed: {stats.examples_per_second:.2f} examples/second")
            logging.info(f"Total examples: {stats.total_examples}")
            logging.info(f"Batch size: {stats.batch_size}")
            logging.info("-" * 50)

    # Save overall results summary
    with open('processing_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
