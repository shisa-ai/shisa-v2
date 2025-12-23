from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - torch is not available in this environment yet
    torch = None
    nn = None  # type: ignore[assignment]

NamedParam = Tuple[str, Any]
ParamPredicate = Callable[[str, Any], bool]

_PARAM_METADATA: dict[int, dict[str, Any]] = {}
_PARAM_FINALIZERS: dict[int, weakref.finalize] = {}


def _cleanup_param_metadata(param_id: int) -> None:
    _PARAM_METADATA.pop(param_id, None)
    finalizer = _PARAM_FINALIZERS.pop(param_id, None)
    if finalizer is not None:
        finalizer.detach()


def _remember_param(param: Any) -> None:
    param_id = id(param)
    if param_id in _PARAM_FINALIZERS:
        return
    try:
        _PARAM_FINALIZERS[param_id] = weakref.finalize(param, _cleanup_param_metadata, param_id)
    except TypeError:
        # Some tensor subclasses (e.g., torch.Parameter in stripped-down builds) may not support weakrefs.
        # Fallback: no automatic cleanup; metadata will live for the process lifetime which is fine for trainers.
        pass


def _set_param_metadata(param: Any, key: str, value: Any) -> None:
    param_id = id(param)
    bucket = _PARAM_METADATA.setdefault(param_id, {})
    bucket[key] = value
    _remember_param(param)


def get_param_metadata(param: Any, key: str, default: Any = None) -> Any:
    if hasattr(param, key):
        return getattr(param, key)
    param_id = id(param)
    return _PARAM_METADATA.get(param_id, {}).get(key, default)


@dataclass
class ParameterTagSummary:
    """
    Diagnostics reported after running tag_parameters_for_muon.
    """

    total: int = 0
    muon: int = 0
    adam: int = 0
    frozen: int = 0
    skipped: List[str] = field(default_factory=list)
    muon_parameters: List[str] = field(default_factory=list)
    adam_parameters: List[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "muon": self.muon,
            "adam": self.adam,
            "frozen": self.frozen,
            "skipped": list(self.skipped),
        }


def _iter_named_parameters(target: Any) -> Iterator[NamedParam]:
    if target is None:
        return iter(())

    if hasattr(target, "named_parameters"):
        return target.named_parameters()  # type: ignore[return-value]

    if isinstance(target, dict):
        return iter(target.items())

    if isinstance(target, Sequence):
        return iter(target)  # type: ignore[return-value]

    raise TypeError("Expected nn.Module, Mapping, or iterable of (name, param) pairs.")


def default_muon_predicate(name: str, param: Any) -> bool:
    """
    Default heuristic for assigning parameters to Muon:
    - requires_grad True
    - has at least 2 dimensions
    - not a bias or LayerNorm weight
    - not an embedding matrix marked as `param.muon_exempt = True`
    """

    if not getattr(param, "requires_grad", True):
        return False

    ndim = getattr(param, "ndim", None)
    if ndim is None:
        return False

    if getattr(param, "muon_exempt", False):
        return False

    if ndim < 2:
        return False

    name_lower = name.lower()
    if name_lower.endswith("bias") or "layernorm" in name_lower:
        return False

    return True


def tag_parameters_for_muon(
    model_or_params: Any,
    predicate: Optional[ParamPredicate] = None,
    attribute_name: str = "use_muon",
) -> ParameterTagSummary:
    """
    Annotate parameters with a boolean flag describing whether MuonClip should
    orthogonalize them or leave them to the Adam fallback. The predicate can be
    customized for architecture-specific quirks.
    """

    predicate = predicate or default_muon_predicate
    summary = ParameterTagSummary()

    named_params = list(_iter_named_parameters(model_or_params))
    for name, param in named_params:
        summary.total += 1

        if not getattr(param, "requires_grad", True):
            summary.frozen += 1
            _set_param_metadata(param, attribute_name, False)
            continue

        try:
            use_muon = bool(predicate(name, param))
        except Exception as exc:
            logger.warning("Muon param predicate failed for %s: %s", name, exc)
            summary.skipped.append(name)
            continue

        if use_muon:
            summary.muon += 1
            summary.muon_parameters.append(name)
        else:
            summary.adam += 1
            summary.adam_parameters.append(name)

        if not _assign_attribute(param, attribute_name, use_muon):
            _set_param_metadata(param, attribute_name, use_muon)

    return summary


def tag_parameters_for_optimizer_split(
    model_or_params: Any,
    predicate: Optional[ParamPredicate] = None,
    attribute_name: str = "use_muon",
) -> Tuple[ParameterTagSummary, List[Any], List[Any]]:
    """
    Convenience wrapper that tags parameters and returns two lists that can be
    fed directly into optimizer parameter groups.
    """

    summary = tag_parameters_for_muon(model_or_params, predicate=predicate, attribute_name=attribute_name)
    muon_params: List[Any] = []
    adam_params: List[Any] = []

    for _, param in _iter_named_parameters(model_or_params):
        flag = get_param_metadata(param, attribute_name, default=None)
        if flag is None:
            flag = getattr(param, attribute_name, False)
        if flag:
            muon_params.append(param)
        else:
            adam_params.append(param)

    return summary, muon_params, adam_params


def _assign_attribute(param: Any, key: str, value: Any) -> bool:
    try:
        setattr(param, key, value)
        return True
    except Exception:
        return False
