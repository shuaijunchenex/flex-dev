"""Server-side factor means and effective residuals for FedEx-LoRA."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import torch
from torch import Tensor

from ....ml_algorithms.lora.fedex_lora import FedExLayerResult
from ....ml_algorithms.lora.rank_scale_alignment import LoRAScaleProfile
from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs


class FedAggregator_FedExLoRA(AbstractFedAggregator):
    def __init__(self, args: FedAggregatorArgs | None = None):
        super().__init__(args)
        self._aggregation_method = "fedex_lora"
        self._weighting_mode = self.args.get("weighting_mode", "uniform")
        if self._weighting_mode not in ("uniform", "samples"):
            raise ValueError("FedEx-LoRA weighting_mode must be 'uniform' or 'samples'")
        self._layers: LoRAScaleProfile = {}

    def configure_layers(self, layers: LoRAScaleProfile) -> None:
        self._layers = dict(layers)

    @staticmethod
    def validate_sample_count(count) -> None:
        if isinstance(count, bool) or not isinstance(count, Real) or not math.isfinite(count) or count < 0:
            raise ValueError("FedEx-LoRA sample counts must be finite and nonnegative")

    @staticmethod
    def compute_fedex_layer(
        a_list: Sequence[Tensor],
        b_list: Sequence[Tensor],
        weights: Sequence[float],
        *,
        scaling: float,
        device: torch.device | str = "cpu",
    ) -> FedExLayerResult:
        """Average the factors and compute the correction in effective weight units."""
        if not weights or len(a_list) != len(weights) or len(b_list) != len(weights):
            raise ValueError("FedEx-LoRA requires one A/B pair and weight for each client")
        if not math.isfinite(scaling) or scaling <= 0:
            raise ValueError("FedEx-LoRA requires a positive finite scaling")
        total = sum(weights)
        if not math.isfinite(total) or total <= 0 or any(weight < 0 for weight in weights):
            raise ValueError("FedEx-LoRA requires nonnegative finite weights with a positive sum")

        factors = (*a_list, *b_list)
        if any(
            not isinstance(value, Tensor) or value.ndim != 2 or not value.is_floating_point()
            for value in factors
        ):
            raise ValueError("FedEx-LoRA factors must be floating-point matrices")
        a_shape, b_shape = a_list[0].shape, b_list[0].shape
        if (
            a_shape[0] == 0
            or a_shape[0] != b_shape[1]
            or any(a.shape != a_shape for a in a_list)
            or any(b.shape != b_shape for b in b_list)
        ):
            raise ValueError("FedEx-LoRA requires the same positive factor rank and shape across clients")

        dtype = torch.float64 if any(value.dtype == torch.float64 for value in factors) else torch.float32
        mean_a = torch.zeros(a_shape, device=device, dtype=dtype)
        mean_b = torch.zeros(b_shape, device=device, dtype=dtype)
        mean_product = torch.zeros((b_shape[0], a_shape[1]), device=device, dtype=dtype)
        for a, b, weight in zip(a_list, b_list, weights):
            if weight == 0:
                continue
            a = a.detach().to(device=device, dtype=dtype)
            b = b.detach().to(device=device, dtype=dtype)
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("FedEx-LoRA factors must be finite")
            probability = weight / total
            mean_a.add_(a, alpha=probability)
            mean_b.add_(b, alpha=probability)
            mean_product.addmm_(b, a, alpha=probability)
        residual = mean_product.addmm_(mean_b, mean_a, alpha=-1).mul_(scaling)
        return FedExLayerResult(mean_a, mean_b, residual)

    def _before_aggregation(self) -> None:
        if not self._layers:
            raise ValueError("FedEx-LoRA must be configured with the server's LoRA scale profile")
        if not self._aggregation_data_dict:
            raise ValueError("FedEx-LoRA requires at least one client update")

    def _do_aggregation(self) -> None:
        active = []
        for state, count in self._aggregation_data_dict:
            self.validate_sample_count(count)
            if count > 0:
                active.append((state, count))
        if not active:
            raise ValueError("FedEx-LoRA requires at least one client with positive sample count")
        states, counts = zip(*active)
        weights = (
            [1.0] * len(states)
            if self._weighting_mode == "uniform"
            else [float(count) for count in counts]
        )
        result: dict[str, FedExLayerResult] = {}
        for prefix, layer in self._layers.items():
            a_list = [state[layer.key_A] for state in states]
            b_list = [state[layer.key_B] for state in states]
            result[prefix] = self.compute_fedex_layer(
                a_list, b_list, weights, scaling=layer.scaling, device=self._device
            )
            if result[prefix].mean_a.shape[0] != layer.rank:
                raise ValueError(f"FedEx-LoRA upload rank differs from server rank at '{prefix}'")
        self._aggregated_weight = result

    def _after_aggregation(self) -> None:
        pass
