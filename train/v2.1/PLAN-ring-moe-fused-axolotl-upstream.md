# Implementation Plan: Ring/Bailing MoE Fused Kernels for Axolotl

## Executive Summary

This document outlines the requirements and implementation plan for adding fused MoE kernel support for Ring-mini-2.0 (bailing_moe_v2 architecture) to Axolotl, suitable for upstream contribution.

**Key Challenge**: Ring uses hierarchical group-based expert routing (256 experts → 8 groups → topk selection) which differs fundamentally from simple top-k routing used by Mixtral/Qwen3-MoE.

**Expected Performance Gain**: ~10x training speedup based on Megatron vs transformers benchmarks.

---

## Part 1: Technical Requirements

### 1.1 Architecture Analysis

**Ring/Bailing MoE V2 Routing (Hierarchical)**
```python
# Two-stage selection process:
# Stage 1: Group selection
routing_weights = F.softmax(router_logits, dim=1)  # [batch*seq, num_experts]
group_logits = routing_weights.view(-1, n_group, num_experts // n_group).sum(dim=-1)
topk_groups = torch.topk(group_logits, topk_group)  # Select 4 groups from 8

# Stage 2: Expert selection within groups
experts_in_groups = gather_experts_from_selected_groups(topk_groups)
selected_experts = torch.topk(experts_in_groups, num_experts_per_tok)  # Select 8 experts

# Plus: Shared experts (always active)
# Plus: routed_scaling_factor for load balancing
```

**Qwen3-MoE Routing (Simple Top-K)**
```python
# Single-stage selection:
routing_weights = F.softmax(router_logits, dim=1)
selected_experts = torch.topk(routing_weights, num_experts_per_tok)
```

### 1.2 Custom Kernel Requirements

**New Triton Kernel Needed**: `bailing_moe_fused_linear`

Must implement:
1. **Hierarchical routing** with group selection
2. **Shared experts** always-active path
3. **Mixed expert processing** (shared + routed)
4. **Grouped GEMM** with expert sorting for memory coalescence
5. **Load balancing** via routed_scaling_factor

**Base it on**:
- PyTorch triton-kernels (MIT license, not Unsloth's AGPLv3)
- Reference: https://github.com/triton-lang/triton/tree/main/python/triton_kernels

---

## Part 2: Axolotl Integration Plan

### 2.1 File Structure (Following Axolotl Conventions)

```
src/axolotl/
├── monkeypatch/
│   ├── models/
│   │   └── bailing_moe_v2/           # NEW
│   │       ├── __init__.py
│   │       ├── modeling.py           # Patched BailingMoeV2SparseMoeBlock
│   │       └── functional.py         # Fused kernels
│   └── bailing_moe_v2/               # NEW (if needed for other patches)
│       └── __init__.py
└── loaders/
    └── patch_manager.py              # MODIFY: Add bailing_moe_v2 detection

docs/
└── bailing_moe_v2_fused.md          # NEW: Documentation

tests/
└── e2e/
    └── test_bailing_moe_v2_fused.py # NEW: E2E tests
```

### 2.2 Core Implementation Files

#### File 1: `src/axolotl/monkeypatch/models/bailing_moe_v2/functional.py`

```python
"""
Fused MoE operations for Bailing MoE V2 architecture.
Based on PyTorch triton-kernels (MIT licensed).
"""

import torch
import triton
import triton.language as tl

def bailing_moe_fused_linear(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    shared_expert_weight: torch.Tensor,
    config: BailingMoeV2Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused MoE forward pass for Bailing V2 architecture.

    Implements:
    1. Hierarchical group-based expert routing
    2. Shared expert processing
    3. Grouped GEMM with expert sorting

    Args:
        hidden_states: [batch*seq, hidden_dim]
        gate_weight: [num_experts, moe_intermediate_size, hidden_dim]
        up_weight: [num_experts, moe_intermediate_size, hidden_dim]
        down_weight: [num_experts, hidden_dim, moe_intermediate_size]
        shared_expert_weight: Weights for shared experts
        config: Model configuration

    Returns:
        output: [batch*seq, hidden_dim]
        router_logits: [batch*seq, num_experts]
    """
    # TODO: Implement hierarchical routing + grouped GEMM
    # See qwen3_moe_fused/functional.py for grouped GEMM reference
    # Add group selection logic before expert selection
    pass

@triton.jit
def _bailing_moe_kernel(...):
    """Triton kernel for fused MoE computation with group-based routing."""
    # TODO: Implement Triton kernel
    pass
```

#### File 2: `src/axolotl/monkeypatch/models/bailing_moe_v2/modeling.py`

```python
"""
Patched modeling components for Bailing MoE V2 with fused kernels.
"""

import torch
import torch.nn as nn
from transformers import BailingMoeV2Config

from .functional import bailing_moe_fused_linear


class BailingMoeV2FusedSparseMoeBlock(nn.Module):
    """Fused version of BailingMoeV2SparseMoeBlock."""

    def __init__(self, config: BailingMoeV2Config):
        super().__init__()
        self.config = config
        # Store expert weights in fused format [num_experts, out, in]
        # Similar to qwen3_moe_fused/modular_qwen3_moe_fused.py

    def forward(self, hidden_states: torch.Tensor):
        # Call bailing_moe_fused_linear
        pass


def patch_bailing_moe_v2_with_fused_kernels():
    """
    Monkey-patch transformers BailingMoeV2SparseMoeBlock with fused version.
    Called by PatchManager.
    """
    try:
        from transformers.models.bailing_moe_v2.modeling_bailing_moe_v2 import (
            BailingMoeV2SparseMoeBlock,
        )
    except ImportError:
        # Ring models use custom modeling, need trust_remote_code
        # Try loading from model's modeling file
        pass

    # Replace forward method
    BailingMoeV2SparseMoeBlock.forward = BailingMoeV2FusedSparseMoeBlock.forward
```

#### File 3: Modify `src/axolotl/loaders/patch_manager.py`

```python
# Add to PatchManager class:

def _apply_bailing_moe_v2_fused_patches(self):
    """Apply fused kernel patches for Bailing MoE V2 models."""
    if not self.cfg.get("use_fused_moe_kernels", False):
        return

    model_type = self.model_config.model_type if hasattr(self.model_config, "model_type") else None

    if model_type == "bailing_moe_v2" or self._is_ring_model():
        LOG.info("Patching Bailing MoE V2 with fused kernels...")
        from axolotl.monkeypatch.models.bailing_moe_v2.modeling import (
            patch_bailing_moe_v2_with_fused_kernels,
        )
        patch_bailing_moe_v2_with_fused_kernels()

def _is_ring_model(self) -> bool:
    """Detect if model is Ring/Bailing architecture."""
    # Check for bailing-specific config attributes
    return (
        hasattr(self.model_config, "n_group") and
        hasattr(self.model_config, "topk_group") and
        hasattr(self.model_config, "num_shared_experts")
    )

# Add to apply_pre_model_load_patches:
def apply_pre_model_load_patches(self):
    # ... existing patches ...
    self._apply_bailing_moe_v2_fused_patches()  # ADD THIS
```

### 2.3 Configuration Support

#### Add to YAML config schema:

```yaml
# Enable fused MoE kernels (10x faster for MoE models)
use_fused_moe_kernels: true  # Default: false

# Bailing MoE V2 specific settings (optional, auto-detected from model)
bailing_moe_config:
  use_shared_experts: true  # Include shared experts in computation
  routed_scaling_factor: 1.0  # Load balancing factor
```

---

## Part 3: Upstream Contribution Requirements

### 3.1 Code Quality Requirements

**Must Have:**
1. ✅ **Clean code** following Axolotl style guide
2. ✅ **Type hints** for all functions
3. ✅ **Docstrings** (Google style) for all public APIs
4. ✅ **Error handling** with informative messages
5. ✅ **Logging** using Axolotl's logger
6. ✅ **No breaking changes** to existing functionality

### 3.2 Testing Requirements

**Unit Tests** (`tests/unit/test_bailing_moe_fused.py`):
```python
def test_bailing_moe_fused_forward():
    """Test fused forward pass matches unfused output."""
    # Compare fused vs unfused outputs
    pass

def test_bailing_moe_fused_backward():
    """Test gradients match unfused version."""
    pass

def test_hierarchical_routing():
    """Test group-based expert selection."""
    pass
```

**E2E Tests** (`tests/e2e/test_bailing_moe_v2_fused.py`):
```python
def test_ring_mini_2_0_training():
    """Test full training run with Ring-mini-2.0."""
    # Use small model or mocked version
    pass

def test_config_validation():
    """Test config detection and validation."""
    pass
```

**Benchmarking** (`tests/benchmark/bench_bailing_moe_fused.py`):
```python
def benchmark_speedup():
    """Measure speedup vs unfused (target: ~10x)."""
    pass
```

### 3.3 Documentation Requirements

**File: `docs/bailing_moe_v2_fused.md`**

Must include:
1. **Overview** - What is Bailing MoE V2, why fused kernels
2. **Installation** - Dependencies (triton, etc.)
3. **Configuration** - How to enable in YAML
4. **Supported Models** - Ring-mini-2.0, Ring-flash-2.0, etc.
5. **Performance** - Benchmarks and expected speedup
6. **Limitations** - Known issues, compatibility
7. **Troubleshooting** - Common errors and solutions
8. **References** - Links to papers, repos

**Update Existing Docs:**
- `docs/config.qmd` - Add use_fused_moe_kernels option
- `docs/special_model_types.md` - Add Ring/Bailing section
- `README.md` - Add to supported optimizations list

### 3.4 Licensing Requirements

**Critical Issue**: qwen3-moe-fused uses AGPLv3-licensed kernels from Unsloth.

**Solutions:**
1. ✅ **Recommended**: Use MIT-licensed [PyTorch triton-kernels](https://github.com/triton-lang/triton/tree/main/python/triton_kernels) instead
2. ❌ **Not recommended**: Use Unsloth kernels (incompatible with Axolotl's Apache 2.0)

**License Header** for new files:
```python
# Copyright 2025 Axolotl Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ...
```

### 3.5 PR Requirements

**PR Title**: `feat: Add fused MoE kernel support for Bailing MoE V2 (Ring models)`

**PR Description Template**:
```markdown
## What does this PR do?

Adds fused MoE kernel support for Bailing MoE V2 architecture (used in Ring-mini-2.0, Ring-flash-2.0, etc.), providing ~10x training speedup.

## Motivation

MoE models in transformers are notoriously slow due to for-loop expert iteration. This PR adds optimized Triton kernels using grouped GEMM.

## Implementation

- Custom fused kernels for hierarchical group-based routing
- Automatic model detection and patching
- Backward compatible with existing configs
- Based on MIT-licensed PyTorch triton-kernels

## Benchmarks

- Ring-mini-2.0 (16.8B total, 1.4B active): 0.21 → 2.1 samples/sec (~10x)
- Memory: Same as unfused version
- Accuracy: Validated against unfused (max diff < 1e-5)

## Testing

- [x] Unit tests for fused operations
- [x] E2E training tests
- [x] Backward pass validation
- [x] Benchmarking
- [x] Documentation

## Checklist

- [x] Tests pass
- [x] Documentation updated
- [x] Follows Axolotl code style
- [x] No breaking changes
- [x] Apache 2.0 compatible (uses MIT triton-kernels)
```

---

## Part 4: Implementation Phases

### Phase 1: Proof of Concept (1-2 weeks)
- [ ] Implement basic `bailing_moe_fused_linear` without hierarchical routing
- [ ] Test with simple top-k routing first (like Qwen3)
- [ ] Validate forward/backward pass correctness
- [ ] Benchmark basic speedup

### Phase 2: Hierarchical Routing (2-3 weeks)
- [ ] Implement group-based routing in Triton kernel
- [ ] Add shared expert processing
- [ ] Handle routed_scaling_factor
- [ ] Test with Ring-mini-2.0

### Phase 3: Integration (1 week)
- [ ] Add PatchManager integration
- [ ] Add config detection
- [ ] Create comprehensive tests
- [ ] Write documentation

### Phase 4: Optimization (1-2 weeks)
- [ ] Optimize Triton kernel performance
- [ ] Auto-tune configurations
- [ ] Add support for 4-bit quantization (optional)
- [ ] Multi-GPU support (optional)

### Phase 5: Upstream Contribution (1-2 weeks)
- [ ] Code review and cleanup
- [ ] Final testing
- [ ] Create PR
- [ ] Address reviewer feedback

**Total Estimated Time**: 6-10 weeks

---

## Part 5: Alternative Approaches

### Option A: Wait for Transformers v5 (Recommended for Low Effort)
- **Timeline**: Unknown, likely Q2-Q3 2025
- **Effort**: None
- **Pros**: Official support, well-tested
- **Cons**: Uncertain timeline, may not support hierarchical routing

### Option B: Use Megatron (Current Alternative)
- **Pros**: Already 10x faster, battle-tested
- **Cons**: Bad numerics (your experience), not good for DPO

### Option C: Contribute to transformers-qwen3-moe-fused (Parallel Path)
- Add Ring/Bailing support there first
- Then integrate with Axolotl
- **Pros**: Smaller scope, faster iteration
- **Cons**: Two separate contributions

---

## Part 6: Minimal Viable Implementation

For fastest path to working solution (without upstream):

**Single File Approach** (`ring_moe_fused_patch.py`):
```python
"""
Drop-in patch for Ring MoE models in Axolotl.
Usage: import this file before loading model.
"""

def patch_ring_moe():
    # Implement minimal fused kernel
    # Monkey-patch BailingMoeV2SparseMoeBlock
    pass

# Auto-apply on import
patch_ring_moe()
```

**Use in training**:
```python
# In your training script:
import ring_moe_fused_patch  # Apply patch
# Then proceed with normal Axolotl training
```

**Effort**: 2-3 weeks (just kernel + basic patch, no tests/docs)

---

## Recommendations

### For Quick Solution (2-3 weeks):
1. Implement minimal fused kernel as standalone patch
2. Test with Ring-mini-2.0 training
3. Keep it internal for now

### For Upstream Contribution (2-3 months):
1. Start with proof of concept using MIT-licensed triton-kernels
2. Validate ~10x speedup
3. Follow full Phase 1-5 plan above
4. Work with Axolotl maintainers early (open issue first)

### Hybrid Approach (Recommended):
1. **Week 1-2**: Build minimal working patch (Phase 1)
2. **Week 3-4**: Test and validate speedup (Phase 2)
3. **Week 5-6**: Decide based on results:
   - If good: Continue to full upstream contribution
   - If issues: Keep as internal patch, wait for Transformers v5

---

## Key Success Factors

1. ✅ **License compliance**: Use MIT triton-kernels, not AGPLv3 Unsloth
2. ✅ **Performance validation**: Must achieve ≥5x speedup to justify complexity
3. ✅ **Correctness**: Rigorous testing of hierarchical routing logic
4. ✅ **Maintainability**: Clean code, good docs, comprehensive tests
5. ✅ **Community engagement**: Work with Axolotl maintainers from start

---

## Questions for Consideration

1. **Is 10x speedup worth 6-10 weeks of development?**
2. **Can you wait for Transformers v5?** (May not support hierarchical routing)
3. **Would a minimal internal patch suffice?** (2-3 weeks, no upstream)
4. **Do you need LoRA support?** (Adds complexity)
5. **Single GPU only or multi-GPU?** (Multi-GPU adds 2-3 weeks)

---

## Next Steps

1. **Validate approach**: Test basic grouped GEMM with Ring model
2. **Open Axolotl issue**: Gauge maintainer interest
3. **Prototype kernel**: Implement Phase 1 (basic fused linear)
4. **Measure speedup**: Validate 10x improvement is achievable
5. **Decision point**: Go/no-go for full upstream contribution
