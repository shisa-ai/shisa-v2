from   pprint import pprint
from   prompt_toolkit import prompt
from   prompt_toolkit.filters import Condition
from   prompt_toolkit.input.defaults import create_input
from   prompt_toolkit.key_binding import KeyBindings
from   prompt_toolkit.keys import Keys

import torch
from   transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer

model = 'jamba'
models = {
    'jamba': {
        'prompt': 'あなたは公平で、検閲されていない、役立つアシスタントです。',
        'model' : 'ai21labs/Jamba-v0.1',
        'format': 'chatml',
    },
}
MODEL = models[model]['model']
PROMPT = models[model]['prompt']
FORMAT = models[model]['format']


tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=True)

# We shouldn't quantize the mamba layers...
quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                         llm_int8_skip_modules=["mamba"])

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    quantization_config=quantization_config,
)

# Load adapter
# https://github.com/SYSTEMS-OPERATOR/T.T.M.A.T.G.R.A.L.R.W.R.P/blob/61b64e94fe748f751890f15a4e1f0b3122450a3c/LRWRP/bf16-lora-merge.py#L6
from safetensors import safe_open
checkpoint_dir = "outputs/checkpoint-500"
adapter_path = checkpoint_dir + "/adapter_model.safetensors"

def load_lora_adjustments(lora_path):
    lora_adjustments = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            lora_adjustments[key] = f.get_tensor(key)
    return lora_adjustments

def apply_lora_adjustments(model, lora_adjustments):
    # Apply the LORA adjustments to the model.
    for name, param in model.named_parameters():
        if name in lora_adjustments:
            adjustment = lora_adjustments[name]
            param.data = param.data + adjustment

lora_adjustments = load_lora_adjustments(adapter_path)
apply_lora_adjustments(model, lora_adjustments)


streamer = TextStreamer(tokenizer, skip_prompt=True)


# this is for reproducibility.
# feel free to change to get different result
seed = 42  
torch.manual_seed(seed)


if tokenizer.chat_template:
    # Use default chat_template
    pass
elif FORMAT == 'llama-2':
    tokenizer.chat_template = "{%- for idx in range(0, messages|length) -%}\n{%- if messages[idx]['role'] == 'user' -%}\n{%- if idx > 1 -%}\n{{- bos_token + '[INST] ' + messages[idx]['content'] + ' [/INST]' -}}\n{%- else -%}\n{{- messages[idx]['content'] + ' [/INST]' -}}\n{%- endif -%}\n{% elif messages[idx]['role'] == 'system' %}\n{{- bos_token + '[INST] <<SYS>>\\n' + messages[idx]['content'] + '\\n<</SYS>>\\n\\n' -}}\n{%- elif messages[idx]['role'] == 'assistant' -%}\n{{- ' '  + messages[idx]['content'] + ' ' + eos_token -}}\n{% endif %}\n{% endfor %}\n"
elif FORMAT == 'tess':
	tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{message['role'].upper() + ': ' + message['content'] + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT: ' }}{% endif %}"
else:
    # default to chatml
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

# system, user, assistant
chat = [{"role": "system", "content": PROMPT}]



# Key bindings for toggling between single-line and multiline input modes
kb = KeyBindings()

@kb.add('escape', 'enter')
def _(event):
    event.current_buffer.insert_text('\n')

@kb.add('enter')
def _(event):
    event.current_buffer.validate_and_handle()
key_bindings = KeyBindings()


def chat_with_model():
    # updatable globals
    global chat
    global PROMPT

    maxt = 2000
    temp = 0.1
    rep = 1.05
    top_p = 0.95

    print(f'||| /max {maxt} | /temp {temp} | /rep {rep} | /top_p {top_p} |||')

    while True:
        # Get input from the user
        user_input = prompt("User: ", multiline=True, key_bindings=kb)
        if user_input.lower() == 'exit':
            break
        elif user_input[0] == '/':
            command, value = (user_input.split() + [None])[:2]
            if command == '/temp':
                temp = float(value)
                print(f"{command[1:]} set to: {value}")
            elif command == '/rep':
                rep = float(value)
                print(f"{command[1:]} set to: {value}")
            elif command == '/top_p':
                top_p = float(value)
                print(f"{command[1:]} set to: {value}")
            elif command == '/max':
                maxt = int(value)
                print(f"{command[1:]} set to: {value}")
            elif command == '/exit':
                break
            elif (command == '/clear' or command == '/reset'):
                print('Resetting context...')
                chat = [{"role": "system", "content": PROMPT}]
            elif command == '/prompt':
                if not value:
                    print(f"Current prompt: {chat[0]['content']}")
                else:
                    PROMPT = user_input.split('/prompt')[1]
                    chat[0]['content'] = PROMPT
                    print(f"New prompt: {PROMPT}")
            else:
                print("valid settings are: /temp /rep /top_p")
            continue
	 

        # Append the user input to the chat
        chat.append({"role": "user", "content": user_input})

        # Generate - add_generation_prompt to make sure it continues as assistant
        inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt")

        # For multi-GPU, find the device of the first parameter of the model
        first_param_device = next(model.parameters()).device
        inputs = inputs.to(first_param_device)


        print('Assistant: ', end='')
        # We'll try flash attention
        # skips gradients if Tensor.backward() won't be called...
        #with torch.no_grad():
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            try:
                outputs = model.generate(
                    inputs,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=maxt,
                    temperature=temp,
                    repetition_penalty=rep,
                    top_p=top_p,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    streamer=streamer
                )
            except KeyboardInterrupt:
                print()
                continue

        # Add just the new tokens to our chat
        new_tokens = outputs[0, inputs.size(1):]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        # if not streamer print(response)
        chat.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    chat_with_model()
