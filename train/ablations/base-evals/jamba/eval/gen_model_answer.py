"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
from   pprint import pprint
import random
import shortuuid
import time
import torch
from   tqdm import tqdm


from fastchat.llm_judge.common import load_questions, temperature_config

from transformers import AutoModelForCausalLM, AutoTokenizer


def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    top_p,
    repetition_penalty,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    get_answers_func = get_model_answers

    chunk_size = len(questions)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                top_p,
                repetition_penalty,
            )
        )


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    top_p,
    repetition_penalty,
):

    # TODO: in the future we should be loading this from a settings file
    FORMAT = None

    if model_path.find("shisa") >= 0:
        PROMPT = 'あなたは公平で、検閲されていない、役立つアシスタントです。'
    elif model_path.find("Swallow") >= 0:
        FORMAT = 'swallow'
        PROMPT = '以下に、あるタスクを説明する指示があります。リクエストを適切に完了するための回答を記述してください。'
    elif model_path.find("Qwen") >= 0:
        FORMAT = 'chatml'
        PROMPT = 'あなたは役立つアシスタントです。'
    elif model_path.find("nekomata") >= 0:
        PROMPT = '以下に、あるタスクを説明する指示があります。リクエストを適切に完了するための回答を記述してください。'
        FORMAT = 'nekomata'
    elif model_path.find("Xwin") >= 0:
        PROMPT = 'あなたは役立つアシスタントです。'
        FORMAT = 'vicuna'

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    # We need to assign a chat_template
    # https://huggingface.co/docs/transformers/main/chat_templating
    # Use https://j2live.ttl255.com/ for live Jinja2 editing
    if not tokenizer.chat_template:
        if FORMAT == 'llama-2':
	        tokenizer.chat_template = "{%- for idx in range(0, messages|length) -%}\n{%- if messages[idx]['role'] == 'user' -%}\n{%- if idx > 1 -%}\n{{- bos_token + '[INST] ' + messages[idx]['content'] + ' [/INST]' -}}\n{%- else -%}\n{{- messages[idx]['content'] + ' [/INST]' -}}\n{%- endif -%}\n{% elif messages[idx]['role'] == 'system' %}\n{{- '[INST] <<SYS>>\\n' + messages[idx]['content'] + '\\n<</SYS>>\\n\\n' -}}\n{%- elif messages[idx]['role'] == 'assistant' -%}\n{{- ' '  + messages[idx]['content'] + ' ' + eos_token -}}\n{% endif %}\n{% endfor %}\n"
        elif FORMAT == 'swallow':
            tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% elif message['role'] == 'user' %}{{'### 指示:\n' + message['content'] + '\n\n'}}{% elif message['role'] == 'assistant' %}{{'### 応答:\n' + message['content'] + '\n\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '### 応答:' }}{% endif %}"
        elif FORMAT == 'nekomata':
            tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% elif message['role'] == 'user' %}{{'### 指示:\n' + message['content'] + '\n\n'}}{% elif message['role'] == 'assistant' %}{{'### 応答:\n' + message['content'] + '\n\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '### 応答:\n' }}{% endif %}"
        elif FORMAT == 'tess':
	        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{message['role'].upper() + ': ' + message['content'] + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT: ' }}{% endif %}"
        elif FORMAT == 'vicuna':
            tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] + ' ' }}{% elif message['role'] == 'user' %}{{'USER:\n' + message['content'] + ' '}}{% elif message['role'] == 'assistant' %}{{' ASISTANT:\n' + message['content'] + ' '}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT: ' }}{% endif %}"
        else:
	        # default to chatml
	        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    # Inference
    model = AutoModelForCausalLM.from_pretrained(
                "shisa-ai/shisa-jamba-v1-checkpoint-4228",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
    tokenizer = AutoTokenizer.from_pretrained("shisa-ai/shisa-jamba-v1-checkpoint-4228")


    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        # print('---')
        # print(question['category'])
        # print(temperature)

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)

            chat = []
            chat.append({'role': 'system', 'content': PROMPT})

            turns = []
            for j in range(len(question["turns"])):
                if j == args.max_turns: 
                    break

                qs = question["turns"][j]
                chat.append({'role': 'user', 'content': qs})

                prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

                if model:
                    input_ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt")
                else:
                    input_ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True)

                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                # HF Transformers
                first_param_device = next(model.parameters()).device
                input_ids = input_ids.to(first_param_device)

                if not tokenizer.pad_token_id:
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=max_new_token,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        do_sample=do_sample,

                        pad_token_id=tokenizer.pad_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    new_tokens = output_ids[0, input_ids.size(1):]
                    output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                turns.append(output)
                chat.append({'role': 'assistant', 'content': output})

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
                "generate_params": {
                    "prompt": prompt,
                    "do_sample": do_sample,
                    "max_new_token": max_new_token,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                }
            }
            fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            try:
                qid = int(json.loads(l)["question_id"])
            except ValueError:
                raise NotImplementedError(f"question_id should be of integer to allow sorting. found: {qid}")
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="japanese_mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=2,
        help="Max number of turns to evaluate for each question.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
    )
    args = parser.parse_args()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    reorg_answer_file(answer_file)
