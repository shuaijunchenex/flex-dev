"""Isolated RBLA variants with one reusable B-side support coefficient."""
from __future__ import annotations

from collections import OrderedDict

import torch

from flex.ml_algorithms.rbla_problem.support_scaling import aggregate_scaled_lora_b

from .._fed_aggregator_rbla import FedAggregator_RBLA
from ...fed_aggregator_args import FedAggregatorArgs


class _FedAggregator_RBLASupportScaling(FedAggregator_RBLA):
    method_name = "rbla_support_scaling"

    def __init__(self, args: FedAggregatorArgs | None = None):
        super().__init__(args)
        self._aggregation_method = self.method_name
        self.gamma = float(args.get("gamma", 0.0)) if args is not None else 0.0
        self.scaling_type = str(args.get("scaling_type", "q_power")) if args is not None else "q_power"

    @staticmethod
    def aggregate_lora_b(
        tensors: list[torch.Tensor],
        weights: list[float],
        gamma: float,
        scaling_type: str = "q_power",
    ) -> torch.Tensor:
        return aggregate_scaled_lora_b(
            tensors,
            weights,
            scaling_type=scaling_type,
            gamma=gamma,
        )

    @classmethod
    def aggregate_state_dicts_scaled(
        cls,
        state_dicts: list[dict],
        weights: list[float] | None,
        *,
        gamma: float,
        scaling_type: str,
        lora_suffixes: set[str] = {"lora_A", "lora_B"},
        lora_only: bool = False,
    ) -> dict:
        if not state_dicts:
            raise ValueError("aggregate_state_dicts_scaled: empty state_dicts")
        raw = weights or [1.0] * len(state_dicts)
        total = float(sum(raw))
        norm = [float(value) / total for value in raw] if total > 0 else [1.0 / len(raw)] * len(raw)
        result: dict[str, torch.Tensor] = {}
        first = state_dicts[0]
        for key in first:
            values = [sd[key] for sd in state_dicts]
            lora_type = FedAggregator_RBLA.get_lora_type(key, lora_suffixes)
            if lora_type == "lora_A":
                result[key] = FedAggregator_RBLA.aggregate_lora_tensors(values, norm, pad_mode="nan")
            elif lora_type == "lora_B":
                result[key] = cls.aggregate_lora_b(values, norm, gamma, scaling_type)
            elif lora_only or not torch.is_floating_point(values[0]):
                result[key] = first[key].clone()
            else:
                stacked = torch.stack(values)
                shape = (len(norm),) + (1,) * (stacked.dim() - 1)
                wt = torch.as_tensor(norm, dtype=stacked.dtype, device=stacked.device).view(shape)
                result[key] = (stacked * wt).sum(dim=0)
        if any(not torch.isfinite(value).all() for value in result.values() if torch.is_floating_point(value)):
            raise FloatingPointError("support-scaled state dict contains NaN/Inf")
        return result

    @classmethod
    def aggregate_state_dicts_gamma(
        cls,
        state_dicts: list[dict],
        weights: list[float] | None,
        gamma: float,
        lora_suffixes: set[str] = {"lora_A", "lora_B"},
        lora_only: bool = False,
    ) -> dict:
        """Backward-compatible P6 API."""
        return cls.aggregate_state_dicts_scaled(
            state_dicts,
            weights,
            gamma=gamma,
            scaling_type="q_power",
            lora_suffixes=lora_suffixes,
            lora_only=lora_only,
        )

    def _do_aggregation(self) -> None:
        if hasattr(self, "_aggregation_data_list") and self._aggregation_data_list:
            pairs = self._aggregation_data_list
        elif hasattr(self, "_aggregation_data_dict") and isinstance(self._aggregation_data_dict, list):
            pairs = self._aggregation_data_dict
        elif hasattr(self, "_aggregation_data_dict") and isinstance(self._aggregation_data_dict, dict):
            state_dicts = self._aggregation_data_dict["state_dicts"]
            weights = self._aggregation_data_dict.get("weights")
            pairs = list(zip(state_dicts, weights or [1.0] * len(state_dicts)))
        else:
            raise ValueError("No aggregation data for support-scaled RBLA")
        state_dicts = [state for state, _weight in pairs]
        weights = [float(weight) for _state, weight in pairs]
        moved = [{key: value.to(self._device) for key, value in state.items()} for state in state_dicts]
        aggregated = self.aggregate_state_dicts_scaled(
            moved,
            weights,
            gamma=self.gamma,
            scaling_type=self.scaling_type,
            lora_suffixes=self._lora_suffixes,
            lora_only=self._lora_only,
        )
        self._aggregated_weight = OrderedDict((key, aggregated[key]) for key in state_dicts[0])


class FedAggregator_RBLAFreezeASupportGamma(_FedAggregator_RBLASupportScaling):
    """Existing P6 method; behaviour remains q_s ** gamma on B only."""

    method_name = "rbla_freeze_a_support_gamma"


class FedAggregator_RBLARefDiagSupportScaling(_FedAggregator_RBLASupportScaling):
    method_name = "rbla_p8_refdiag_support_scaling"


class FedAggregator_RBLAStrongASupportScaling(_FedAggregator_RBLASupportScaling):
    method_name = "rbla_p8_strong_a_support_scaling"


class FedAggregator_RBLAP10FreezeASupportScaling(_FedAggregator_RBLASupportScaling):
    method_name = "rbla_p10_freeze_a_support_scaling"


class FedAggregator_RBLAP9FreezeASupportScaling(_FedAggregator_RBLASupportScaling):
    method_name = "rbla_p9_freeze_a_support_scaling"
