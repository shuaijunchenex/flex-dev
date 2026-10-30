"""New RBLA server strategies with reference-frame diagnostics."""
from __future__ import annotations

from .._rbla_server import RblaServerStrategy
from ....ml_algorithms.rbla_problem import RblaReferenceDiagnostics


class _RblaReferenceServerStrategy(RblaServerStrategy):
    strategy_name = "rbla_reference"

    def __init__(self, args, server_node) -> None:
        super().__init__(args, server_node)
        self._strategy_type = self.strategy_name
        cfg = server_node.node_var.config_dict.get("reference_diagnostics", {})
        self._diagnostic = RblaReferenceDiagnostics(
            eps=float(cfg.get("eps", 1e-8)),
            compute_pinv=bool(cfg.get("compute_pinv", True)),
        )
        self._round_reference_metrics: dict[str, float] = {}

    def aggregation(self) -> dict:
        updates = self._obj.node_var.client_updates or []
        state_dicts = [entry["updated_weights"] for entry in updates]
        weights = [
            float(entry["train_record"].get("data_sample_num", 1.0))
            for entry in updates
        ]
        self._round_reference_metrics = self._diagnostic.compute(
            state_dicts,
            weights,
            self._obj.node_var.model_weight,
        )
        return super().aggregation()

    def evaluate(self) -> None:
        super().evaluate()
        self._obj.eval_results.update(self._round_reference_metrics)


class RblaRefDiagServerStrategy(_RblaReferenceServerStrategy):
    strategy_name = "rbla_refdiag"


class RblaFreezeAServerStrategy(_RblaReferenceServerStrategy):
    strategy_name = "rbla_freeze_a"


class RblaStrongAServerStrategy(_RblaReferenceServerStrategy):
    strategy_name = "rbla_strong_a"
