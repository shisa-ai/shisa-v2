"""
Schema helpers that mirror the real Axolotl structures closely enough for
MuonClip planning and validation exercises.
"""

from .muonclip import MuonClipConfig, OrthogonalizerConfig, QKClipConfig
from .validation import (
    ConfigError,
    OptimizerPlan,
    TrainingConfig,
    detect_zero_stage,
    plan_optimizer,
    validate_training_config,
)

__all__ = [
    "ConfigError",
    "OptimizerPlan",
    "TrainingConfig",
    "detect_zero_stage",
    "plan_optimizer",
    "validate_training_config",
    "MuonClipConfig",
    "OrthogonalizerConfig",
    "QKClipConfig",
]
