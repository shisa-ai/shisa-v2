"""
Axolotl lightweight scaffolding used for MuonClip development tasks.

The real Axolotl package is substantially larger; this namespace only
contains the configuration and validation helpers we need for the
MuonClip integration experiments tracked in IMPLEMENTATION-muonclip.md.
"""

from .utils.schemas.validation import (
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
]
