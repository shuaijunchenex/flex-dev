"""Isolated RBLA aggregation registrations for reference-frame experiments."""
from __future__ import annotations

from .._fed_aggregator_rbla import FedAggregator_RBLA
from ...fed_aggregator_args import FedAggregatorArgs


class _FedAggregator_RBLAReferenceBase(FedAggregator_RBLA):
    method_name = "rbla_reference"

    def __init__(self, args: FedAggregatorArgs | None = None):
        super().__init__(args)
        self._aggregation_method = self.method_name


class FedAggregator_RBLARefDiag(_FedAggregator_RBLAReferenceBase):
    method_name = "rbla_refdiag"


class FedAggregator_RBLAFreezeA(_FedAggregator_RBLAReferenceBase):
    method_name = "rbla_freeze_a"


class FedAggregator_RBLAStrongA(_FedAggregator_RBLAReferenceBase):
    method_name = "rbla_strong_a"


class FedAggregator_RBLARefDiagAnalysis(_FedAggregator_RBLAReferenceBase):
    method_name = "rbla_refdiag_analysis"


class FedAggregator_RBLAFreezeAAnalysis(_FedAggregator_RBLAReferenceBase):
    method_name = "rbla_freeze_a_analysis"


class FedAggregator_RBLAStrongAAnalysis(_FedAggregator_RBLAReferenceBase):
    method_name = "rbla_strong_a_analysis"
