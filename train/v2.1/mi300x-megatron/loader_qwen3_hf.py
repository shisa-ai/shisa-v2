import os
import sys
import types
from pathlib import Path

import torch
from tqdm import tqdm

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - conversion requires transformers
    raise ImportError("The 'transformers' package is required for loader_qwen3_hf.") from exc


def add_arguments(parser):
    group = parser.add_argument_group(title='Qwen3 HF loader.')

    group.add_argument('--tokenizer-model', required=True,
                       help='Directory containing the tokenizer assets used for conversion.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of the Megatron repository.')
    group.add_argument('--loader-transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Transformer implementation to instantiate when rebuilding the Megatron model.')
    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='Optional explicit vocab size to store in checkpoint metadata.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Optional vocab file used to compute the true vocab size when not provided.')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=None,
                       help='Optional padding multiple for the vocab size reported to the saver.')


def _resolve_vocab_size(args):
    if getattr(args, 'true_vocabsized_provided', False):
        return args.true_vocab_size
    if args.vocab_file:
        with open(args.vocab_file, 'r', encoding='utf-8') as handle:
            return sum(1 for _ in handle)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=True)
    return tokenizer.vocab_size + len(tokenizer.get_added_vocab())


def load_args_from_checkpoint(args):
    """Fill Megatron args using the upstream Hugging Face config."""

    config = AutoConfig.from_pretrained(args.load_dir, trust_remote_code=True)

    args.iteration = 1
    args.hidden_size = config.hidden_size
    args.ffn_hidden_size = config.intermediate_size
    args.num_layers = config.num_hidden_layers
    args.num_attention_heads = config.num_attention_heads
    args.num_query_groups = config.num_key_value_heads or config.num_attention_heads
    args.group_query_attention = (config.num_key_value_heads is not None
                                  and config.num_key_value_heads != config.num_attention_heads)
    args.kv_channels = getattr(config, 'head_dim', args.hidden_size // args.num_attention_heads)
    args.seq_length = config.max_position_embeddings
    args.max_position_embeddings = config.max_position_embeddings
    args.position_embedding_type = 'rope'
    args.rotary_base = getattr(config, 'rope_theta', 10000)
    args.rotary_percent = 1.0
    args.use_rotary_position_embeddings = True
    args.rotary_interleaved = False
    args.bf16 = True
    args.fp16 = False
    args.params_dtype = torch.bfloat16
    args.norm_epsilon = getattr(config, 'rms_norm_eps', 1e-5)
    args.normalization = 'RMSNorm'
    args.swiglu = True
    args.add_qkv_bias = False
    args.add_bias_linear = False
    args.disable_bias_linear = True
    args.untie_embeddings_and_output_weights = True
    args.tokenizer_type = 'HuggingFaceTokenizer'
    args.vocab_size = config.vocab_size
    args.padded_vocab_size = config.vocab_size
    args.attention_dropout = getattr(config, 'attention_dropout', 0.0)
    args.qk_layernorm = True
    args.qk_head_dim = args.kv_channels
    args.qk_pos_emb_head_dim = 0
    args.global_batch_size = getattr(args, 'global_batch_size', 128)


def set_preprocess_state(model, hf_model):
    with torch.no_grad():
        model.embedding.word_embeddings.weight.copy_(hf_model.model.embed_tokens.weight)


def set_postprocess_state(model, hf_model):
    with torch.no_grad():
        model.decoder.final_layernorm.weight.copy_(hf_model.model.norm.weight)
        model.output_layer.weight.copy_(hf_model.lm_head.weight)


def _reshape_qkv(args, hf_attn):
    num_query_groups = args.num_query_groups
    num_heads = args.num_attention_heads
    kv_channels = args.kv_channels
    heads_per_group = num_heads // num_query_groups

    q_weight = hf_attn.q_proj.weight.view(num_query_groups, heads_per_group * kv_channels, -1)
    k_weight = hf_attn.k_proj.weight.view(num_query_groups, kv_channels, -1)
    v_weight = hf_attn.v_proj.weight.view(num_query_groups, kv_channels, -1)

    stacked = torch.cat([q_weight, k_weight, v_weight], dim=1)
    return stacked.reshape(-1, args.hidden_size)


def set_attn_state(args, layer, hf_layer):
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn

    with torch.no_grad():
        attn.linear_qkv.weight.copy_(_reshape_qkv(args, hf_attn))
        if getattr(attn.linear_qkv, 'bias', None) is not None:
            attn.linear_qkv.bias.zero_()

        attn.linear_proj.weight.copy_(hf_attn.o_proj.weight)
        if getattr(attn.linear_proj, 'bias', None) is not None:
            attn.linear_proj.bias.zero_()

        if hasattr(attn, 'q_layernorm') and getattr(attn.q_layernorm, 'weight', None) is not None:
            attn.q_layernorm.weight.copy_(hf_attn.q_norm.weight)
            if getattr(attn.q_layernorm, 'bias', None) is not None:
                attn.q_layernorm.bias.zero_()
        if hasattr(attn, 'k_layernorm') and getattr(attn.k_layernorm, 'weight', None) is not None:
            attn.k_layernorm.weight.copy_(hf_attn.k_norm.weight)
            if getattr(attn.k_layernorm, 'bias', None) is not None:
                attn.k_layernorm.bias.zero_()

        # Megatron stores the pre-attention norm on the fused linear module.
        if hasattr(attn.linear_qkv, 'layer_norm_weight'):
            attn.linear_qkv.layer_norm_weight.copy_(hf_layer.input_layernorm.weight)
        if hasattr(attn.linear_qkv, 'layer_norm_bias') and attn.linear_qkv.layer_norm_bias is not None:
            attn.linear_qkv.layer_norm_bias.zero_()


def set_mlp_state(layer, hf_layer):
    mlp = layer.mlp
    hf_mlp = hf_layer.mlp

    with torch.no_grad():
        gate = hf_mlp.gate_proj.weight
        up = hf_mlp.up_proj.weight
        mlp.linear_fc1.weight.copy_(torch.cat([gate, up], dim=0))
        if getattr(mlp.linear_fc1, 'bias', None) is not None:
            mlp.linear_fc1.bias.zero_()

        mlp.linear_fc2.weight.copy_(hf_mlp.down_proj.weight)
        if getattr(mlp.linear_fc2, 'bias', None) is not None:
            mlp.linear_fc2.bias.zero_()

    if hasattr(mlp, 'linear_fc1') and hasattr(mlp.linear_fc1, 'layer_norm_weight'):
        mlp.linear_fc1.layer_norm_weight.copy_(hf_layer.post_attention_layernorm.weight)
    if hasattr(mlp, 'linear_fc1') and getattr(mlp.linear_fc1, 'layer_norm_bias', None) is not None:
        mlp.linear_fc1.layer_norm_bias.zero_()

    if hasattr(layer, 'pre_mlp_layernorm') and getattr(layer.pre_mlp_layernorm, 'weight', None) is not None:
        with torch.no_grad():
            layer.pre_mlp_layernorm.weight.copy_(hf_layer.post_attention_layernorm.weight)
            if getattr(layer.pre_mlp_layernorm, 'bias', None) is not None:
                layer.pre_mlp_layernorm.bias.zero_()


def set_layer_state(args, model, hf_model, idx):
    layer = model.decoder.layers[idx]
    hf_layer = hf_model.model.layers[idx]

    set_attn_state(args, layer, hf_layer)
    set_mlp_state(layer, hf_layer)


def load_checkpoint_to_model(args):
    from pretrain_gpt import model_provider

    torch_dtype = args.params_dtype
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.load_dir,
        device_map='cpu',
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model = model_provider(True, True).to(torch_dtype)
    model.eval()

    set_preprocess_state(model, hf_model)
    set_postprocess_state(model, hf_model)
    for layer_idx in tqdm(range(args.num_layers), desc='set layer states'):
        set_layer_state(args, model, hf_model, layer_idx)

    del hf_model
    return model


def _load_checkpoint(queue, args):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.global_vars import set_global_variables
    from megatron.core import mpu
    from megatron.core.enums import ModelType
    from megatron.legacy import fused_kernels
    from megatron.legacy.model import module

    sys.argv = [
        'convert.py',
        '--use-mcore-models',
        '--disable-bias-linear',
        '--no-masked-softmax-fusion',
        '--no-bias-gelu-fusion',
        '--no-bias-dropout-fusion',
        '--no-async-tensor-model-parallel-allreduce',
        '--no-gradient-accumulation-fusion',
        '--use-cpu-initialization',
        '--micro-batch-size', '1',
        '--no-load-optim',
        '--no-load-rng',
        '--no-save-optim',
        '--no-save-rng',
        '--no-initialization',
        '--mock-data',
        '--transformer-impl', args.loader_transformer_impl,
        '--load', args.load_dir,
        '--no-one-logger',
    ]

    margs = parse_args()
    margs.tokenizer_model = args.tokenizer_model
    load_args_from_checkpoint(margs)

    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size
    margs = validate_args(margs)

    margs.use_legacy_models = False
    margs.transformer_impl = args.loader_transformer_impl
    margs.model_type = ModelType.encoder_or_decoder
    margs.params_dtype = torch.bfloat16

    module.MegatronModule.embedding_warning_printed = True
    set_global_variables(margs, build_tokenizer=False)
    mpu.initialize_model_parallel(margs.tensor_model_parallel_size, margs.pipeline_model_parallel_size)
    fused_kernels.load(margs)

    metadata = types.SimpleNamespace()
    metadata.model_type = 'GPT'
    metadata.num_layers = margs.num_layers
    metadata.hidden_size = margs.hidden_size
    metadata.seq_length = margs.seq_length
    metadata.num_attention_heads = margs.num_attention_heads
    metadata.max_position_embeddings = margs.max_position_embeddings
    metadata.tokenizer_type = margs.tokenizer_type
    metadata.iteration = margs.iteration
    metadata.params_dtype = margs.params_dtype
    metadata.bert_binary_head = False
    metadata.output_layer = True
    metadata.position_embedding_type = margs.position_embedding_type
    metadata.linear_bias = False
    metadata.qkv_bias = False
    metadata.norm_has_bias = False
    metadata.swiglu = True
    metadata.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    metadata.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    metadata.make_vocab_size_divisible_by = getattr(args, 'make_vocab_size_divisible_by', None)
    metadata.checkpoint_args = margs
    metadata.consumed_train_samples = 0
    metadata.consumed_valid_samples = 0
    metadata.num_experts = None

    if args.true_vocab_size is not None:
        metadata.true_vocab_size = args.true_vocab_size
    else:
        metadata.true_vocab_size = _resolve_vocab_size(args)

    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    model = load_checkpoint_to_model(margs)

    queue.put(metadata)

    def queue_put(name, payload):
        payload['name'] = name
        queue.put(payload)

    with torch.no_grad():
        queue_put('embeddings', {
            'word embeddings': model.embedding.word_embeddings.weight.clone(),
        })

        for layer_idx in range(margs.num_layers):
            layer = model.decoder.layers[layer_idx]

            message = {
                'input norm weight': layer.self_attention.linear_qkv.layer_norm_weight.clone(),
                'qkv weight': layer.self_attention.linear_qkv.weight.clone(),
                'dense weight': layer.self_attention.linear_proj.weight.clone(),
                'post norm weight': layer.mlp.linear_fc1.layer_norm_weight.clone(),
            }

            fc1 = layer.mlp.linear_fc1.weight
            gate, up = torch.chunk(fc1, 2, dim=0)
            message['mlp l0 weight W'] = gate.clone()
            message['mlp l0 weight V'] = up.clone()
            message['mlp l1 weight'] = layer.mlp.linear_fc2.weight.clone()

            queue_put(f'transformer layer {layer_idx}', message)

        queue_put('final norm', {
            'weight': model.decoder.final_layernorm.weight.clone(),
        })
        queue_put('output layer', {
            'weight': model.output_layer.weight.clone(),
        })

    queue.put('done')


def load_checkpoint(queue, args):
    # Ensure helper flag exists before usage.
    args.true_vocabsized_provided = args.true_vocab_size is not None
    try:
        _load_checkpoint(queue, args)
    except Exception:
        queue.put('exit')
        raise
