from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch isn't available in the lightweight dev env
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for MuonClip attention instrumentation. "
            "Install torch before calling these helpers."
        )


def _distributed_max(values: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
    if torch is None:
        return values
    if not torch.distributed.is_available():
        return values
    if not torch.distributed.is_initialized():
        return values
    clone = values.clone()
    torch.distributed.all_reduce(clone, op=torch.distributed.ReduceOp.MAX)
    return clone


def _per_head_max(logits: "torch.Tensor", num_heads: int) -> "torch.Tensor":  # type: ignore[name-defined]
    """
    Collapse an attention score tensor down to shape [num_heads], tracking the
    maximum logit per head over batch and sequence dimensions.
    """

    _require_torch()
    if logits.ndim == 4:
        # [batch, heads, q_len, k_len]
        per_head = logits.detach().amax(dim=-1).amax(dim=-1)  # -> [batch, heads]
        per_head = per_head.max(dim=0).values  # -> [heads]
    elif logits.ndim == 3:
        # [batch*heads, q_len, k_len] common when heads already merged.
        batch_heads = logits.shape[0]
        if batch_heads % num_heads != 0:
            raise ValueError(f"Cannot reshape logits of shape {tuple(logits.shape)} for {num_heads} heads.")
        per_head = (
            logits.detach()
            .view(-1, num_heads, logits.shape[-2], logits.shape[-1])
            .amax(dim=-1)
            .amax(dim=-1)
            .amax(dim=0)
        )
    else:
        raise ValueError(f"Unsupported logits rank {logits.ndim}; expected 3 or 4.")

    return per_head


@dataclass
class AttentionModuleSpec:
    """
    Metadata describing an attention module we want to track.
    """

    name: str
    module: Any
    num_heads: int
    head_dim: Optional[int] = None
    gather: str = "auto"


class AttentionLogitTracker:
    """
    Tracks per-head attention logits for a single module. Modules should call
    `observe(logits)` whenever they compute attention scores (before Softmax).
    """

    def __init__(
        self,
        name: str,
        num_heads: int,
        *,
        device: Optional["torch.device"] = None,  # type: ignore[name-defined]
        dtype: Optional["torch.dtype"] = None,  # type: ignore[name-defined]
        reducer: Optional[Callable[["torch.Tensor"], "torch.Tensor"]] = None,  # type: ignore[name-defined]
    ) -> None:
        _require_torch()
        self.name = name
        self.num_heads = num_heads
        self.reducer = reducer or _distributed_max
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32
        self.max_logits = torch.zeros(num_heads, device=self.device, dtype=self.dtype)
        self.total_updates = 0
        self.last_global_max: float = 0.0

    def observe(self, logits: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        _require_torch()
        with torch.no_grad():
            per_head = _per_head_max(logits, self.num_heads)
            if per_head.device != self.device:
                per_head = per_head.to(self.device)
            reduced = self.reducer(per_head)
            torch.maximum(self.max_logits, reduced, out=self.max_logits)
            self.total_updates += 1
            self.last_global_max = float(torch.max(reduced).item())
            return reduced

    def reset(self) -> None:
        self.max_logits.zero_()
        self.total_updates = 0
        self.last_global_max = 0.0


class AttentionRegistry:
    """
    Book-keeping helper that keeps track of all attention modules we instrumented.
    """

    def __init__(self) -> None:
        self._trackers: Dict[str, AttentionLogitTracker] = {}

    def register(self, spec: AttentionModuleSpec) -> AttentionLogitTracker:
        tracker = AttentionLogitTracker(
            spec.name,
            spec.num_heads,
            device=_infer_device(spec.module),
            dtype=_infer_dtype(spec.module),
        )
        setattr(spec.module, "_muonclip_tracker", tracker)
        self._trackers[spec.name] = tracker
        return tracker

    def tracker(self, name: str) -> Optional[AttentionLogitTracker]:
        return self._trackers.get(name)

    def all_trackers(self) -> Dict[str, AttentionLogitTracker]:
        return dict(self._trackers)

    def reset(self) -> None:
        for tracker in self._trackers.values():
            tracker.reset()

    def stats(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "max_logit": float(tracker.last_global_max),
                "updates": tracker.total_updates,
            }
            for name, tracker in self._trackers.items()
        }


def _infer_device(module: Any) -> "torch.device":  # type: ignore[name-defined]
    _require_torch()
    for param in module.parameters(recurse=False) if hasattr(module, "parameters") else []:
        return param.device
    return torch.device("cpu")


def _infer_dtype(module: Any) -> "torch.dtype":  # type: ignore[name-defined]
    _require_torch()
    for param in module.parameters(recurse=False) if hasattr(module, "parameters") else []:
        return param.dtype
    return torch.float32


def observe_attention_logits(module: Any, logits: "torch.Tensor") -> Optional["torch.Tensor"]:  # type: ignore[name-defined]
    """
    Convenience wrapper modules can call after computing their attention scores.
    Handles missing trackers gracefully so instrumentation is opt-in.
    """

    tracker: Optional[AttentionLogitTracker] = getattr(module, "_muonclip_tracker", None)
    if tracker is None:
        return None
    return tracker.observe(logits)


@contextlib.contextmanager
def maybe_zero3_full_params(param: Any):
    """
    Placeholder context manager that will gather ZeRO-3 shards when DeepSpeed is
    available. For now it simply yields the parameter so the controller code can
    rely on the same API during unit tests.
    """

    ds_context = getattr(param, "ds_tensor", None)
    if ds_context is None:
        yield param
        return

    try:
        import deepspeed

        with deepspeed.zero.GatheredParameters([param], modifier_rank=None):
            yield param
    except Exception:
        # DeepSpeed is optional in this dev environment; fall back to the raw param.
        yield param
