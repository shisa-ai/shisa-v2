"""Minimal GPT-2 MoE module compatible with Hugging Face AutoModel."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model


class MegablocksMoEExpert(nn.Module):
    """Single expert feed-forward network matching Megatron's ParallelMLP layout."""

    def __init__(self, hidden_size: int, intermediate_size: int, activation: str, bias: bool) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.activation = ACT2FN[activation]
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.w1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.w2(hidden_states)
        return hidden_states


class MegablocksMoEMLP(nn.Module):
    """MoE MLP block using top-1 routing identical to Megatron SwitchMLP inference."""

    def __init__(self, config) -> None:
        super().__init__()
        if not getattr(config, "num_local_experts", 0):
            raise ValueError("MegablocksMoEMLP requires num_local_experts > 0")

        self.hidden_size = config.n_embd
        self.num_experts = config.num_local_experts
        self.top_k = max(1, getattr(config, "moe_router_topk", 1))
        self.disable_bias_linear = getattr(config, "disable_bias_linear", False)
        self.add_bias_linear = getattr(config, "add_bias_linear", True)

        bias = bool(self.add_bias_linear) and not self.disable_bias_linear
        intermediate_size = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MegablocksMoEExpert(self.hidden_size, intermediate_size, config.activation_function, bias) for _ in range(self.num_experts)]
        )
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = hidden_states.shape
        flat_states = hidden_states.view(-1, hidden)

        router_logits = self.router(flat_states)
        router_scores = torch.sigmoid(router_logits)

        if self.top_k == 1:
            top_probs, top_indices = router_scores.max(dim=-1)
            expert_outputs = torch.zeros_like(flat_states)

            for expert_idx, expert in enumerate(self.experts):
                mask = top_indices == expert_idx
                if mask.any():
                    expert_out = expert(flat_states[mask])
                    expert_outputs[mask] = expert_out

            expert_outputs *= top_probs.unsqueeze(-1)
        else:
            top_probs, top_indices = torch.topk(router_scores, k=self.top_k, dim=-1)
            expert_outputs = torch.zeros_like(flat_states)
            for k in range(self.top_k):
                probs = top_probs[:, k]
                indices = top_indices[:, k]
                for expert_idx, expert in enumerate(self.experts):
                    mask = indices == expert_idx
                    if mask.any():
                        expert_out = expert(flat_states[mask])
                        expert_outputs[mask] += expert_out * probs[mask].unsqueeze(-1)

        expert_outputs = expert_outputs.view(batch, seq_len, hidden)
        expert_outputs = self.dropout(expert_outputs)
        return expert_outputs


class MegablocksMoEGPT2Model(GPT2Model):
    """GPT-2 backbone with Megablocks-style MoE feed-forward layers."""

    def __init__(self, config):
        super().__init__(config)
        if not getattr(config, "is_moe", False):
            raise ValueError("MegablocksMoEGPT2Model expects config.is_moe = True")

        for block in self.h:
            block.mlp = MegablocksMoEMLP(config)


class MegablocksMoEGPT2ForCausalLM(GPT2LMHeadModel):
    """Causal LM head that swaps GPT-2 MLPs for MoE experts."""

    def __init__(self, config):
        super().__init__(config)
        if not getattr(config, "is_moe", False):
            raise ValueError("MegablocksMoEGPT2ForCausalLM expects config.is_moe = True")
        self.transformer = MegablocksMoEGPT2Model(config)
        self.post_init()
