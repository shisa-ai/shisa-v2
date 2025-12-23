from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence


@dataclass(slots=True)
class OrthogonalizerConfig:
    """
    Configuration for the Muon orthogonalization controller.

    Attributes:
        enabled: Gate for the Newton–Schulz update logic.
        momentum: Momentum coefficient μ used for Muon velocity tracking.
        ns_steps: Number of Newton–Schulz iterations per update.
        rms_scale: RMS normalization factor applied after orthogonalization.
        weight_decay: Optional extra weight decay applied when Muon writes back.
        gather: Strategy name describing how full-parameter materialization is handled.
        state_dtype: dtype to use for momentum/velocity buffers; defaults to bf16.
    """

    enabled: bool = True
    momentum: float = 0.95
    ns_steps: int = 5
    rms_scale: float = 0.2
    weight_decay: float = 0.0
    gather: str = "auto"
    state_dtype: str = "bf16"


@dataclass(slots=True)
class QKClipConfig:
    """
    Configuration for the per-head attention logit clipping controller.

    Attributes:
        enabled: Gate for QK-Clip execution.
        tau: Threshold used to cap logits.
        alpha: Relaxation factor; values <1.0 lazily shrink logits back to τ.
        target_modules: Projection names that should be rescaled (default q/k).
        sync_metrics: Whether to emit distributed logging metrics by default.
        cooldown_steps: Number of steps with sub-threshold logits before disabling.
    """

    enabled: bool = False
    tau: float = 120.0
    alpha: float = 1.0
    target_modules: tuple[str, ...] = ("q_proj", "k_proj")
    sync_metrics: bool = True
    cooldown_steps: int = 64


@dataclass(slots=True)
class MuonClipConfig:
    """
    Root MuonClip configuration block exposed to Axolotl YAML configs.

    This mirrors the MX configuration format we described in
    IMPLEMENTATION-muonclip.md and keeps the defaults backwards compatible:
    when the block is absent or disabled existing Muon configs continue to
    route through the legacy optimizer pathways.
    """

    enabled: bool = False
    orthogonalizer: OrthogonalizerConfig = field(default_factory=OrthogonalizerConfig)
    qk_clip: QKClipConfig = field(default_factory=QKClipConfig)
    state_offload: str = "auto"
    metrics_prefix: str = "train/muonclip"
    allow_zero3: bool = True
    allow_fsdp: bool = True

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, object]]) -> "MuonClipConfig":
        """
        Build a MuonClipConfig from user-provided dictionaries. Nested dicts for
        orthogonalizer and qk_clip are converted into their respective dataclasses.
        """

        if not data:
            return cls()
        if not isinstance(data, Mapping):
            # Allow boolean shorthand such as `muonclip: true`.
            return cls(enabled=bool(data))

        def build_nested(target_cls, source: Optional[Mapping[str, object]]):
            if not source:
                return target_cls()
            defaults = target_cls()
            kwargs = {}
            for key in defaults.__dataclass_fields__.keys():  # type: ignore[attr-defined]
                if key in source:
                    kwargs[key] = source[key]
                else:
                    kwargs[key] = getattr(defaults, key)
            return target_cls(**kwargs)

        ortho = build_nested(OrthogonalizerConfig, data.get("orthogonalizer") if isinstance(data, Mapping) else None)
        qk_cfg = build_nested(QKClipConfig, data.get("qk_clip") if isinstance(data, Mapping) else None)

        kwargs = {k: v for k, v in data.items() if k not in {"orthogonalizer", "qk_clip"}}
        return cls(orthogonalizer=ortho, qk_clip=qk_cfg, **kwargs)

    def as_dict(self) -> dict[str, object]:
        """
        Convert the dataclass to a plain dict that is convenient for JSON
        logging or checkpoint metadata.
        """

        return {
            "enabled": self.enabled,
            "orthogonalizer": self.orthogonalizer.__dict__,
            "qk_clip": self.qk_clip.__dict__,
            "state_offload": self.state_offload,
            "metrics_prefix": self.metrics_prefix,
            "allow_zero3": self.allow_zero3,
            "allow_fsdp": self.allow_fsdp,
        }

    def should_activate(self) -> bool:
        """
        Returns True when any MuonClip controller is active. This lets callers
        short-circuit expensive instrumentation when the block exists but the
        user disabled both orthogonalizer and QK-Clip.
        """

        if not self.enabled:
            return False
        return self.orthogonalizer.enabled or self.qk_clip.enabled


def normalize_target_modules(modules: Optional[Sequence[str]]) -> tuple[str, ...]:
    """
    Normalise module name lists to a stable tuple for deterministic configs.
    """

    if not modules:
        return ("q_proj", "k_proj")
    if isinstance(modules, tuple):
        return modules
    return tuple(dict.fromkeys(modules))  # deduplicate while keeping order
