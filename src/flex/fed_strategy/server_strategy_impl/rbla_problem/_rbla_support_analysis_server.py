"""Isolated checkpoint/coverage servers for P4--P7 RBLA experiments."""
from __future__ import annotations

from ._rbla_reference_server import _RblaReferenceServerStrategy
from ....ml_algorithms.rbla_problem import (
    SlotCoverageDiagnostics,
    support_scaled_discrepancy_metrics,
)


class _RblaSupportAnalysisServer(_RblaReferenceServerStrategy):
    strategy_name = "rbla_support_analysis"

    def __init__(self, args, server_node) -> None:
        super().__init__(args, server_node)
        config = server_node.node_var.config_dict
        eps = float(config.get("reference_diagnostics", {}).get("eps", 1e-8))
        self._coverage = SlotCoverageDiagnostics.from_config(config, eps=eps)
        self._analysis_round = 0
        self._support_gamma = float(config.get("support_scaling", {}).get("gamma", 0.0))
        self._support_scaling_type = str(
            config.get("support_scaling", {}).get("scaling_type", "q_power")
        )

    def aggregation(self) -> dict:
        result = super().aggregation()
        updates = self._obj.node_var.client_updates or []
        state_dicts = [entry["updated_weights"] for entry in updates]
        weights = [float(entry["train_record"].get("data_sample_num", 1.0)) for entry in updates]
        client_ids = [str(entry.get("client_id", f"client.{i + 1}")) for i, entry in enumerate(updates)]
        client_indices = [int(entry.get("client_index", i)) for i, entry in enumerate(updates)]
        aggregated = self._obj.node_var.aggregated_weight
        self._round_reference_metrics.update(support_scaled_discrepancy_metrics(
            state_dicts,
            weights,
            aggregated,
            gamma=self._support_gamma,
            scaling_type=self._support_scaling_type,
            eps=self._coverage.eps,
        ))
        self._round_reference_metrics.update(self._coverage.compute(
            round_idx=self._analysis_round,
            client_state_dicts=state_dicts,
            weights=weights,
            aggregated_state_dict=aggregated,
            client_ids=client_ids,
            client_indices=client_indices,
        ))
        self._round_reference_metrics["ref_support_gamma"] = self._support_gamma
        self._analysis_round += 1
        return result

    def finalize_analysis(self, checkpoint_path: str) -> dict[str, str]:
        return self._coverage.write_artifacts(checkpoint_path)


class RblaRefDiagAnalysisServerStrategy(_RblaSupportAnalysisServer):
    strategy_name = "rbla_refdiag_analysis"


class RblaFreezeAAnalysisServerStrategy(_RblaSupportAnalysisServer):
    strategy_name = "rbla_freeze_a_analysis"


class RblaStrongAAnalysisServerStrategy(_RblaSupportAnalysisServer):
    strategy_name = "rbla_strong_a_analysis"


class RblaFreezeASupportGammaServerStrategy(_RblaSupportAnalysisServer):
    strategy_name = "rbla_freeze_a_support_gamma"


class RblaP8RefDiagSupportScalingServerStrategy(_RblaSupportAnalysisServer):
    strategy_name = "rbla_p8_refdiag_support_scaling"


class RblaP8StrongASupportScalingServerStrategy(_RblaSupportAnalysisServer):
    strategy_name = "rbla_p8_strong_a_support_scaling"


class RblaP10FreezeASupportScalingServerStrategy(_RblaSupportAnalysisServer):
    strategy_name = "rbla_p10_freeze_a_support_scaling"


class RblaP9FreezeASupportScalingServerStrategy(_RblaSupportAnalysisServer):
    strategy_name = "rbla_p9_freeze_a_support_scaling"
