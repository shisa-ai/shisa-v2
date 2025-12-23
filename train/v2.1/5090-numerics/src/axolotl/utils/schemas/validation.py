from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from .muonclip import MuonClipConfig


class ConfigError(ValueError):
    """
    Raised when a training configuration violates constraints documented in
    IMPLEMENTATION-muonclip.md.
    """


@dataclass(slots=True)
class TrainingConfig:
    """
    Minimal subset of the Axolotl training config that we need for MuonClip.
    """

    optimizer: str = "adamw"
    muonclip: Optional[MuonClipConfig] = None
    deepspeed: Optional[Union[str, Path, Mapping[str, Any]]] = None
    fsdp: Optional[Mapping[str, Any]] = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TrainingConfig":
        muonclip_cfg = None
        if "muonclip" in raw:
            muonclip_cfg = MuonClipConfig.from_dict(raw.get("muonclip"))
        fsdp_cfg = raw.get("fsdp_config") or raw.get("fsdp")
        return cls(
            optimizer=raw.get("optimizer", "adamw"),
            muonclip=muonclip_cfg,
            deepspeed=raw.get("deepspeed"),
            fsdp=fsdp_cfg if isinstance(fsdp_cfg, Mapping) else None,
        )

    @property
    def optimizer_normalized(self) -> str:
        return self.optimizer.lower().replace("-", "_")

    def muonclip_or_default(self) -> MuonClipConfig:
        return self.muonclip or MuonClipConfig()

    @property
    def fsdp_enabled(self) -> bool:
        return bool(self.fsdp)


@dataclass(slots=True)
class OptimizerPlan:
    """
    Summarises how the trainer should instantiate optimizers/controllers.
    """

    primary: str
    muonclip: Optional[MuonClipConfig] = None
    use_post_step_controller: bool = False
    zero_stage: Optional[int] = None
    fsdp_enabled: bool = False
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        controller = " + MuonClip" if self.use_post_step_controller and self.muonclip else ""
        stage = f", ZeRO-{self.zero_stage}" if self.zero_stage else ""
        fsdp = ", FSDP" if self.fsdp_enabled else ""
        return f"{self.primary}{controller}{stage}{fsdp}"

    def requires_muonclip(self) -> bool:
        return self.use_post_step_controller and bool(self.muonclip)


def detect_zero_stage(
    deepspeed_cfg: Optional[Union[str, Path, Mapping[str, Any]]],
) -> Optional[int]:
    """
    Best-effort detection of the ZeRO stage value.
    """

    if deepspeed_cfg is None:
        return None

    if isinstance(deepspeed_cfg, Mapping):
        zero_cfg = deepspeed_cfg.get("zero_optimization") or {}
        try:
            stage = zero_cfg.get("stage")
            return int(stage) if stage is not None else None
        except (TypeError, ValueError):
            return None

    path = Path(deepspeed_cfg)
    if not path.is_file():
        return None

    with path.open("r", encoding="utf-8") as handle:
        try:
            ds_cfg = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Failed to parse DeepSpeed config '{path}': {exc}") from exc

    zero_cfg = ds_cfg.get("zero_optimization") or {}
    stage_value = zero_cfg.get("stage")

    try:
        return int(stage_value) if stage_value is not None else None
    except (TypeError, ValueError):
        return None


def plan_optimizer(cfg: TrainingConfig) -> OptimizerPlan:
    """
    Decide whether Axolotl should use legacy Muon, DeepSpeed Muon, or the new
    MuonClip controller pathway. The decision is based on the requested
    optimizer string, the detected ZeRO stage, and whether FSDP is enabled.
    """

    zero_stage = detect_zero_stage(cfg.deepspeed)
    muon_requested = cfg.optimizer_normalized == "muon"
    muonclip_cfg = cfg.muonclip_or_default()
    muonclip_active = muonclip_cfg.should_activate()

    notes: list[str] = []

    if muon_requested:
        if zero_stage == 3 or cfg.fsdp_enabled:
            if not muonclip_cfg.enabled:
                raise ConfigError(
                    "optimizer: muon under ZeRO-3/FSDP now routes through the MuonClip "
                    "controller. Please add `muonclip.enabled: true` to your config."
                )
            if not muonclip_active:
                notes.append("MuonClip block present but both controllers disabled; enabling orthogonalizer by default.")
                muonclip_cfg.orthogonalizer.enabled = True
            notes.append("Route Muon -> AdamW + MuonClip post-step controller")
            return OptimizerPlan(
                primary="adamw_torch",
                muonclip=muonclip_cfg,
                use_post_step_controller=True,
                zero_stage=zero_stage,
                fsdp_enabled=cfg.fsdp_enabled,
                notes=notes,
            )

        if zero_stage is not None and zero_stage <= 2 and not muonclip_active:
            notes.append("Use DeepSpeed's native Muon optimizer for ZeRO-2")
            return OptimizerPlan(
                primary="deepspeed_muon",
                zero_stage=zero_stage,
                fsdp_enabled=cfg.fsdp_enabled,
                notes=notes,
            )

        if muonclip_active:
            notes.append("Apply MuonClip controllers even without ZeRO-3")
            return OptimizerPlan(
                primary="adamw_torch",
                muonclip=muonclip_cfg,
                use_post_step_controller=True,
                zero_stage=zero_stage,
                fsdp_enabled=cfg.fsdp_enabled,
                notes=notes,
            )

        notes.append("Fallback to legacy torch.optim.Muon path")
        return OptimizerPlan(primary="torch_muon", zero_stage=zero_stage, fsdp_enabled=cfg.fsdp_enabled, notes=notes)

    if muonclip_active:
        notes.append("MuonClip enabled while using non-muon optimizer")
        return OptimizerPlan(
            primary=cfg.optimizer_normalized,
            muonclip=muonclip_cfg,
            use_post_step_controller=True,
            zero_stage=zero_stage,
            fsdp_enabled=cfg.fsdp_enabled,
            notes=notes,
        )

    return OptimizerPlan(primary=cfg.optimizer_normalized, zero_stage=zero_stage, fsdp_enabled=cfg.fsdp_enabled, notes=notes)


def validate_training_config(raw_cfg: Mapping[str, Any]) -> OptimizerPlan:
    """
    Entry point used by tooling/tests. Accepts raw config dictionaries,
    performs schema expansion, and returns the derived optimizer plan.
    """

    cfg = raw_cfg if isinstance(raw_cfg, TrainingConfig) else TrainingConfig.from_dict(raw_cfg)
    plan = plan_optimizer(cfg)

    # Guard rails for combinations that are explicitly unsupported.
    if plan.primary == "torch_muon" and (plan.zero_stage == 3 or plan.fsdp_enabled):
        raise ConfigError(
            "torch.optim.Muon cannot run under ZeRO-3 or FSDP. Enable the MuonClip controller "
            "or switch DeepSpeed to ZeRO stage <= 2."
        )

    return plan
