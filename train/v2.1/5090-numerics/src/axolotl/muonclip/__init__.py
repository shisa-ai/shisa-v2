"""
MuonClip integration helpers for Axolotl.

Only a very small subset of the eventual trainer wiring lives here: parameter
tagging utilities, attention instrumentation stubs, and shared context managers
that will be used by the MuonClip controller once it lands.
"""

from .parameters import (
    ParameterTagSummary,
    get_param_metadata,
    tag_parameters_for_muon,
    tag_parameters_for_optimizer_split,
)
from .attention import (
    AttentionLogitTracker,
    AttentionModuleSpec,
    AttentionRegistry,
)

__all__ = [
    "ParameterTagSummary",
    "tag_parameters_for_muon",
    "tag_parameters_for_optimizer_split",
    "get_param_metadata",
    "AttentionLogitTracker",
    "AttentionModuleSpec",
    "AttentionRegistry",
]
