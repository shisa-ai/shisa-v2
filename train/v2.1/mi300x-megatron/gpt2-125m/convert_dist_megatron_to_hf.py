#!/usr/bin/env python3

"""Convert Megatron-LM distributed checkpoints (dist_cp) to Hugging Face format.

This script is tailored for GPT-style checkpoints produced by Megatron-LM using the
new distributed checkpoint API (``torch.distributed.checkpoint``) and outputs a
single-process Hugging Face checkpoint directory.

The implementation reuses helper utilities from ``transformers`` but adapts them
for the new checkpoint naming conventions (e.g., ``input_norm``) found in recent
Megatron versions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, GPT2Config
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils import check_torch_load_is_safe

from transformers.models.megatron_gpt2.checkpoint_reshaping_and_interoperability import (
    get_megatron_sharded_states,
    get_element_from_dict_by_path,
    megatron_to_transformers_fix_query_key_value_ordering,
    tensor_parallel_params,
)

# Allow pickled argparse.Namespace objects stored inside Megatron checkpoints
from torch.serialization import add_safe_globals

try:
    from megatron.core.transformer.enums import AttnBackend
except Exception:
    AttnBackend = None

extra_globals = [argparse.Namespace]
if AttnBackend is not None:
    extra_globals.append(AttnBackend)
add_safe_globals(extra_globals)

# Force torch.load to allow object pickles stored in Megatron checkpoints
_orig_torch_load = torch.load

def _torch_load_allow_pickle(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = _torch_load_allow_pickle


def _normalize_block(block: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rename keys to legacy names expected by the Transformers converter."""

    renamed: Dict[str, torch.Tensor] = {}
    for key, value in block.items():
        if value is None:
            continue
        if key.endswith("._extra_state"):
            continue
        new_key = key
        new_key = new_key.replace("input_norm", "input_layernorm")
        new_key = new_key.replace("post_attention_norm", "post_attention_layernorm")
        new_key = new_key.replace("final_norm", "final_layernorm")
        renamed[new_key] = value
    return renamed


def _normalize_state_dict(state: Dict) -> Dict:
    """Apply key normalization in-place to match legacy naming."""

    model = state.get("model")
    if not isinstance(model, dict):
        return state
    language_model = model.get("language_model")
    if not isinstance(language_model, dict):
        return state

    for block_name in ("encoder", "transformer"):
        block = language_model.get(block_name)
        if isinstance(block, dict):
            language_model[block_name] = _normalize_block(block)

    return state


def _load_megatron_args(load_path: Path) -> Tuple[argparse.Namespace, Dict]:
    """Load the Megatron arguments and state dict metadata from rank 0."""

    sub_dirs = os.listdir(load_path)
    rank0_dir = None
    for candidate in ("mp_rank_00", "mp_rank_00_000"):
        if candidate in sub_dirs:
            rank0_dir = load_path / candidate
            break
    if rank0_dir is None:
        raise FileNotFoundError(f"Could not locate rank-0 directory under {load_path}")

    # Prefer model_optim_rng.pt but fall back to model_rng.pt if necessary.
    checkpoint_file = None
    for candidate in ("model_optim_rng.pt", "model_rng.pt"):
        candidate_path = rank0_dir / candidate
        if candidate_path.is_file():
            checkpoint_file = candidate_path
            break
    if checkpoint_file is None:
        raise FileNotFoundError(f"No model checkpoint found under {rank0_dir}")

    print(f"Loading Megatron-LM checkpoint arguments from: {checkpoint_file}")
    check_torch_load_is_safe()
    rank0_state = torch.load(str(checkpoint_file), map_location="cpu", weights_only=False)
    megatron_args = rank0_state.get("args")
    if megatron_args is None:
        raise ValueError("Unable to find Megatron arguments in checkpoint; conversion requires them")
    return megatron_args, rank0_state


LAYER_RE = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z_]+)")


def convert_checkpoint(load_path: Path, save_path: Path, tokenizer_path: Path | None, max_shard_size: str) -> None:
    megatron_args, _ = _load_megatron_args(load_path)

    if megatron_args.bias_gelu_fusion:
        activation_function = "gelu_fast"
    elif getattr(megatron_args, "openai_gelu", False):
        activation_function = "gelu_new"
    else:
        activation_function = "gelu"

    vocab_size = (
        megatron_args.padded_vocab_size
        if getattr(megatron_args, "orig_vocab_size", None) is None
        else megatron_args.orig_vocab_size
    )

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=megatron_args.max_position_embeddings,
        n_embd=megatron_args.hidden_size,
        n_layer=megatron_args.num_layers,
        n_head=megatron_args.num_attention_heads,
        n_inner=megatron_args.ffn_hidden_size,
        activation_function=activation_function,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=vocab_size - 1,
        eos_token_id=vocab_size - 1,
        architectures=["GPT2LMHeadModel"],
    )

    config.chat_template = """{% for message in messages %}
{% set role = message['role'] | lower %}
{% if role == 'system' %}<|im_start|>system
{{ message['content'] | trim }}<|im_end|>
{% elif role == 'user' %}<|im_start|>user
{{ message['content'] | trim }}<|im_end|>
{% elif role == 'assistant' %}<|im_start|>assistant
{{ message['content'] | trim }}<|im_end|>
{% else %}<|im_start|>{{ message['role'] }}
{{ message['content'] | trim }}<|im_end|>
{% endif %}
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""


    num_experts = int(getattr(megatron_args, "num_experts", 0) or 0)
    expert_model_parallel_size = int(getattr(megatron_args, "expert_model_parallel_size", 1) or 1)
    num_local_experts = num_experts // expert_model_parallel_size if num_experts else 0
    moe_router_topk = int(getattr(megatron_args, "moe_router_topk", 1) or 1)
    moe_router_pre_softmax = bool(getattr(megatron_args, "moe_router_pre_softmax", False))
    moe_aux_loss_coeff = float(getattr(megatron_args, "moe_aux_loss_coeff", 0.0) or 0.0)
    disable_bias_linear = bool(getattr(megatron_args, "disable_bias_linear", False))
    add_bias_linear = bool(getattr(megatron_args, "add_bias_linear", True))
    add_qkv_bias = bool(getattr(megatron_args, "add_qkv_bias", True))

    is_moe = num_local_experts > 0
    config.is_moe = is_moe
    config.disable_bias_linear = disable_bias_linear
    config.add_bias_linear = add_bias_linear
    config.add_qkv_bias = add_qkv_bias
    if is_moe:
        config.architectures = ["MegablocksMoEGPT2ForCausalLM"]
        config.num_experts = num_experts
        config.num_local_experts = num_local_experts
        config.expert_model_parallel_size = expert_model_parallel_size
        config.moe_router_topk = moe_router_topk
        config.moe_router_pre_softmax = moe_router_pre_softmax
        config.moe_aux_loss_coeff = moe_aux_loss_coeff
        config.auto_map = {
            "AutoModelForCausalLM": "modeling_megablocks_moe_gpt2.MegablocksMoEGPT2ForCausalLM",
        }

    tp_size = megatron_args.tensor_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size
    dtype = torch.float32

    output_state_dict: Dict[str, torch.Tensor] = {}

    print("Converting embeddings")
    loader_args = argparse.Namespace(load_path=str(load_path))
    tp_state_dicts = [_normalize_state_dict(sd) for sd in get_megatron_sharded_states(loader_args, tp_size, pp_size, 0)]

    position_embeddings = get_element_from_dict_by_path(
        tp_state_dicts[0], "model.language_model.embedding.position_embeddings.weight"
    )
    output_state_dict["transformer.wpe.weight"] = position_embeddings.to(dtype).contiguous()

    word_embeddings = torch.cat(
        [
            get_element_from_dict_by_path(sd, "model.language_model.embedding.word_embeddings.weight")
            for sd in tp_state_dicts
        ],
        dim=0,
    ).contiguous()
    word_embeddings = word_embeddings[:vocab_size].to(dtype).contiguous()
    output_state_dict["transformer.wte.weight"] = word_embeddings


    print("Converting transformer layers")
    hidden_size_per_head = config.n_embd // config.n_head
    n_positions = config.n_positions
    layers_per_pp = config.num_hidden_layers // pp_size

    for pp_rank in range(pp_size):
        print(f"  Pipeline rank {pp_rank}")
        tp_state_dicts = [_normalize_state_dict(sd) for sd in get_megatron_sharded_states(loader_args, tp_size, pp_size, pp_rank)]

        root_parent = get_element_from_dict_by_path(tp_state_dicts[0], "model.language_model")
        root_key = "transformer" if "transformer" in root_parent else "encoder"
        root_path = f"model.language_model.{root_key}"
        tp_root_dicts = [get_element_from_dict_by_path(sd, root_path) for sd in tp_state_dicts]
        root_dict = tp_root_dicts[0]

        for key, val in root_dict.items():
            match = LAYER_RE.match(key)
            if match is None:
                continue

            layer_idx = int(match.group(1)) + pp_rank * layers_per_pp
            op_name = match.group(2)
            weight_or_bias = match.group(3)
            layer_name = f"transformer.h.{layer_idx}"

            if weight_or_bias == '_extra_state':
                continue

            if op_name in {"input_layernorm", "input_norm"}:
                target = "ln_1"
                params = val.to(dtype).contiguous()
                output_state_dict[f"{layer_name}.{target}.{weight_or_bias}"] = params
                continue
            if op_name in {"post_attention_layernorm", "post_attention_norm"}:
                target = "ln_2"
                params = val.to(dtype).contiguous()
                output_state_dict[f"{layer_name}.{target}.{weight_or_bias}"] = params
                continue

            if is_moe and op_name.startswith("mlp.local_experts"):
                if weight_or_bias != "weight":
                    continue
                parts = op_name.split('.')
                if len(parts) < 4:
                    raise KeyError(f"Unexpected MoE key '{op_name}'")
                expert_idx = int(parts[2])
                expert_submodule = parts[3]
                shard_values = [tp_root_dicts[tp_rank][key] for tp_rank in range(tp_size)]
                if expert_submodule == "dense_h_to_4h":
                    params = torch.cat(shard_values, dim=0).to(dtype).contiguous()
                    output_state_dict[f"{layer_name}.mlp.experts.{expert_idx}.w1.weight"] = params
                elif expert_submodule == "dense_4h_to_h":
                    params = torch.cat(shard_values, dim=1).to(dtype).contiguous()
                    output_state_dict[f"{layer_name}.mlp.experts.{expert_idx}.w2.weight"] = params
                else:
                    raise KeyError(f"Unsupported MoE expert submodule '{expert_submodule}'")
                continue

            if is_moe and op_name == "mlp.router" and weight_or_bias == "weight":
                shard_values = [tp_root_dicts[tp_rank][key] for tp_rank in range(tp_size)]
                params = torch.cat(shard_values, dim=1).to(dtype).contiguous() if tp_size > 1 else shard_values[0].to(dtype).contiguous()
                output_state_dict[f"{layer_name}.mlp.router.weight"] = params
                continue

            if op_name + "." + weight_or_bias not in tensor_parallel_params:
                params = val.to(dtype).contiguous()
            else:
                dim = 1 if op_name in ["self_attention.dense", "attention.dense", "mlp.dense_4h_to_h"] else 0
                slices = [val] + [get_element_from_dict_by_path(tp_state_dicts[tp_rank], key) for tp_rank in range(1, tp_size)]
                params = torch.cat(slices, dim=dim).to(dtype).contiguous()

            if op_name in {"self_attention.query_key_value", "attention.query_key_value"} and weight_or_bias == "weight":
                causal_mask = torch.tril(torch.ones((n_positions, n_positions), dtype=dtype)).view(1, 1, n_positions, n_positions)
                output_state_dict[f"{layer_name}.attn.bias"] = causal_mask
                output_state_dict[f"{layer_name}.attn.masked_bias"] = torch.tensor(-1e4, dtype=dtype)

                out_val = megatron_to_transformers_fix_query_key_value_ordering(
                    params, 2.0, 3, config.n_head, hidden_size_per_head
                )
                out_val = out_val.transpose(0, 1).contiguous()
                output_state_dict[f"{layer_name}.attn.c_attn.weight"] = out_val
                continue

            if op_name in {"self_attention.query_key_value", "attention.query_key_value"} and weight_or_bias == "bias":
                out_val = megatron_to_transformers_fix_query_key_value_ordering(
                    params, 2.0, 3, config.n_head, hidden_size_per_head
                )
                output_state_dict[f"{layer_name}.attn.c_attn.bias"] = out_val
                continue

            mapping = {
                "self_attention.dense": ".attn.c_proj.",
                "attention.dense": ".attn.c_proj.",
                "mlp.dense_h_to_4h": ".mlp.c_fc.",
                "mlp.dense_4h_to_h": ".mlp.c_proj.",
            }
            if op_name not in mapping:
                raise KeyError(f"Unexpected operation name '{op_name}' in checkpoint")

            target = mapping[op_name]
            if weight_or_bias == "weight":
                output_state_dict[f"{layer_name}{target}weight"] = params.transpose(0, 1).contiguous()
            else:
                output_state_dict[f"{layer_name}{target}bias"] = params

    if not add_bias_linear:
        for layer_idx in range(config.num_hidden_layers):
            layer_name = f"transformer.h.{layer_idx}"
            attn_bias_key = f"{layer_name}.attn.c_attn.bias"
            proj_bias_key = f"{layer_name}.attn.c_proj.bias"
            if attn_bias_key not in output_state_dict:
                output_state_dict[attn_bias_key] = torch.zeros(3 * config.n_embd, dtype=dtype)
            if proj_bias_key not in output_state_dict:
                output_state_dict[proj_bias_key] = torch.zeros(config.n_embd, dtype=dtype)
            if not is_moe:
                fc_bias_key = f"{layer_name}.mlp.c_fc.bias"
                proj_mlp_bias_key = f"{layer_name}.mlp.c_proj.bias"
                if fc_bias_key not in output_state_dict:
                    output_state_dict[fc_bias_key] = torch.zeros(config.n_inner, dtype=dtype)
                if proj_mlp_bias_key not in output_state_dict:
                    output_state_dict[proj_mlp_bias_key] = torch.zeros(config.n_embd, dtype=dtype)

    print("Converting final layernorm")
    final_dict = get_element_from_dict_by_path(tp_state_dicts[0], root_path)
    if "final_layernorm.weight" not in final_dict:
        raise KeyError(f"Could not locate final_layernorm in path {root_path}")
    output_state_dict["transformer.ln_f.weight"] = final_dict["final_layernorm.weight"].to(dtype)
    output_state_dict["transformer.ln_f.bias"] = final_dict["final_layernorm.bias"].to(dtype)

    print("Converting LM head")
    lm_head = word_embeddings
    output_state_dict["lm_head.weight"] = lm_head

    tied_base = output_state_dict.get("transformer.wte.weight")
    if tied_base is not None and lm_head.data_ptr() == tied_base.data_ptr():
        output_state_dict["lm_head.weight"] = lm_head.clone()

    print("Saving Hugging Face files")
    save_path.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(save_path)

    if tokenizer_path:
        print(f"Copying tokenizer files from {tokenizer_path}")
        tokenizer_files = ["tokenizer.json", "tokenizer.model", "merges.txt", "vocab.json", "gpt2-merges.txt", "gpt2-vocab.json"]
        for name in tokenizer_files:
            src = tokenizer_path / name
            if src.is_file():
                dst = save_path / src.name
                dst.write_bytes(src.read_bytes())
        vocab_aliases = [("gpt2-vocab.json", "vocab.json"), ("gpt2-merges.txt", "merges.txt")]
        for src_name, dst_name in vocab_aliases:
            src = tokenizer_path / src_name
            dst = save_path / dst_name
            if src.is_file() and not dst.exists():
                dst.write_bytes(src.read_bytes())
    else:
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.save_pretrained(save_path)

    if is_moe:
        modeling_src = Path(__file__).resolve().parent / 'modeling_megablocks_moe_gpt2.py'
        if modeling_src.is_file():
            shutil.copy(modeling_src, save_path / modeling_src.name)
        else:
            print(f"Warning: expected MoE modeling file at {modeling_src}, but it was not found.")

    weights_file = save_path / "model.safetensors"
    save_file(output_state_dict, str(weights_file))
    print(f"Model weights saved in {weights_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("load_path", type=Path, help="Path to Megatron checkpoint iteration (e.g. iter_0002203)")
    parser.add_argument("save_path", type=Path, help="Destination directory for Hugging Face checkpoint")
    parser.add_argument("--tokenizer-path", type=Path, default=None, help="Directory containing tokenizer files to copy")
    parser.add_argument("--max-shard-size", default="10GB", help="Maximum shard size passed to Transformers sharding utility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_checkpoint(args.load_path.resolve(), args.save_path.resolve(), args.tokenizer_path, args.max_shard_size)


if __name__ == "__main__":
    main()
