"""FedSA-LoRA weighted averaging of the shared A factors."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping
from numbers import Real

import torch

from ..fed_aggregator_abc import AbstractFedAggregator


def validate_shared_a(payload, template) -> None:
    if not isinstance(payload, Mapping) or not payload or payload.keys() != template.keys():
        raise ValueError("FedSA-LoRA payload must contain exactly the shared A factors")
    for key, tensor in payload.items():
        if not isinstance(key, str) or key.split(".")[-1] != "lora_A":
            raise ValueError(f"FedSA-LoRA payload contains a non-A key {key!r}")
        if (not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point()
                or tensor.layout != torch.strided or tensor.ndim != 2
                or tensor.shape != template[key].shape or tensor.numel() == 0):
            raise ValueError(f"Invalid FedSA-LoRA shared A shape or dtype for '{key}'")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Nonfinite FedSA-LoRA shared A factor '{key}'")


def validate_sample_count(count) -> None:
    if isinstance(count, bool) or not isinstance(count, Real) or not math.isfinite(count) or count < 0:
        raise ValueError("FedSA-LoRA sample counts must be finite and nonnegative")


class FedAggregator_FedSAShareA(AbstractFedAggregator):
    """Compute the shared A factors without access to client-private B."""

    def __init__(self, args=None):
        super().__init__(args)
        self._aggregation_method = "fedsa_lora"
        self._weighting_mode = self.args.get("weighting_mode", "samples")
        if self._weighting_mode not in ("uniform", "samples"):
            raise ValueError("FedSA-LoRA weighting_mode must be 'uniform' or 'samples'")

    def _before_aggregation(self) -> None:
        self._aggregated_weight = None
        if not self._aggregation_data_dict:
            raise ValueError("FedSA-LoRA requires at least one client update")

    @torch.no_grad()
    def _do_aggregation(self) -> None:
        pairs = self._aggregation_data_dict
        template = pairs[0][0]
        if not isinstance(template, Mapping):
            raise ValueError("FedSA-LoRA expected a mapping of shared A factors")
        active = []
        for state, count in pairs:
            validate_shared_a(state, template)
            validate_sample_count(count)
            if count:
                weight = 1.0 if self._weighting_mode == "uniform" else float(count)
                active.append((state, weight))
        try:
            total = math.fsum(weight for _, weight in active)
        except OverflowError as error:
            raise ValueError("FedSA-LoRA total weight must be finite") from error
        if total <= 0:
            raise ValueError("FedSA-LoRA requires a positive total sample count")

        result = OrderedDict()
        for key, first in template.items():
            dtype = torch.float64 if any(state[key].dtype == torch.float64 for state, _ in active) else torch.float32
            average = torch.zeros(first.shape, dtype=dtype, device=self._device)
            for state, weight in active:
                average.add_(state[key].to(average), alpha=weight / total)
            result[key] = average.to(dtype=first.dtype)
        validate_shared_a(result, template)
        self._aggregated_weight = result

    def _after_aggregation(self) -> None:
        return
