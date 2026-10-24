"""Algorithms and diagnostics for the isolated RBLA problem experiments."""

from .diagnostics import RblaReferenceDiagnostics, run_reparameterization_stress_test
from .coverage import (
    SlotCoverageDiagnostics,
    support_scaled_discrepancy,
    support_scaled_discrepancy_metrics,
)
from .loss import StrongAConfig, StrongAProximalLoss
from .support_scaling import (
    SCALING_TYPES,
    aggregate_scaled_lora_b,
    canonical_scaling_type,
    coefficient_for_eligible,
    normalise_weights,
)

__all__ = [
    "RblaReferenceDiagnostics",
    "SlotCoverageDiagnostics",
    "StrongAConfig",
    "StrongAProximalLoss",
    "run_reparameterization_stress_test",
    "support_scaled_discrepancy",
    "support_scaled_discrepancy_metrics",
    "SCALING_TYPES",
    "aggregate_scaled_lora_b",
    "canonical_scaling_type",
    "coefficient_for_eligible",
    "normalise_weights",
]
