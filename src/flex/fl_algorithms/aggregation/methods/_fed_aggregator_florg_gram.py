"""Server-side FLoRG Gram aggregation for compact factor payloads."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import torch

from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs

GRAM_FACTOR_NAME = "gram_factor"


@dataclass(frozen=True)
class GramAggregationResult:
    factor: torch.Tensor
    gram_rank: int
    eigenvalues: torch.Tensor
    gram_projection_residual: float
    factor_drift_before: float
    factor_drift_after: float



def validated_sample_count(value) -> float:
    try:
        count = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("FLoRG sample counts must be finite and nonnegative") from error
    if not math.isfinite(count) or count < 0:
        raise ValueError("FLoRG sample counts must be finite and nonnegative")
    return count


def validate_factor_payload(
    payload: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
) -> None:
    if not isinstance(payload, Mapping) or payload.keys() != reference.keys():
        raise ValueError("FLoRG payload must contain exactly the configured Gram factor keys")
    for key, value in payload.items():
        expected = reference[key]
        if not torch.is_tensor(value) or value.shape != expected.shape or not value.is_floating_point():
            raise ValueError(f"FLoRG factor {key!r} must be a floating-point tensor of shape {tuple(expected.shape)}")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"FLoRG factor {key!r} contains non-finite values")


class FedAggregator_FLoRGGram(AbstractFedAggregator):
    def __init__(self, args: FedAggregatorArgs | None = None):
        super().__init__(args)
        self._aggregation_method = "florg_gram"
        self.weighting_mode = self.args.get("weighting_mode", "uniform")
        self.align = self.args.get("align", True)
        if self.weighting_mode not in ("uniform", "samples"):
            raise ValueError("FLoRG weighting_mode must be 'uniform' or 'samples'")
        if not isinstance(self.align, bool):
            raise ValueError("FLoRG align must be boolean")
        self._reference = {}
        self.last_stats = {}

    @staticmethod
    @torch.no_grad()
    def aggregate_gram_factors(
        factors: Sequence[torch.Tensor],
        weights: Sequence[float],
        previous_factor: torch.Tensor,
        *,
        align: bool = True,
    ) -> GramAggregationResult:
        """Average client Gram matrices and fit the shape of the last broadcast.

        The weighted vertical stack T satisfies T.T @ T = sum_i p_i C_i.T @ C_i.
        Keeping its leading right singular directions yields the requested PSD
        rank projection. Procrustes aligns the fitted factor to the prior factor
        while preserving that projection.
        """
        if not factors or len(factors) != len(weights):
            raise ValueError("Gram aggregation requires a nonempty set of factors and matching weights")
        if previous_factor.ndim != 2 or not previous_factor.is_floating_point():
            raise ValueError("The previous Gram factor must be a floating-point matrix")
        rank, width = previous_factor.shape
        if not 1 <= rank <= width:
            raise ValueError("Gram factor rank must be between 1 and its feature width")
        if not torch.isfinite(previous_factor).all().item():
            raise ValueError("The previous Gram factor contains non-finite values")
        if any(
            value.shape != previous_factor.shape
            or not value.is_floating_point()
            or not torch.isfinite(value).all().item()
            for value in factors
        ):
            raise ValueError("Client Gram factors must be finite matrices of the broadcast shape")

        weights = [float(weight) for weight in weights]
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Gram aggregation weights must be finite and nonnegative")
        max_weight = max(weights)
        if max_weight <= 0:
            raise ValueError("Gram aggregation requires a positive total weight")
        scaled_weights = [weight / max_weight for weight in weights]
        total = sum(scaled_weights)

        work_dtype = torch.float64 if any(
            value.dtype == torch.float64 for value in (*factors, previous_factor)
        ) else torch.float32
        previous = previous_factor.to(dtype=work_dtype)
        stacked = torch.cat([
            value.to(device=previous.device, dtype=work_dtype) * math.sqrt(weight / total)
            for value, weight in zip(factors, scaled_weights)
            if weight > 0
        ])
        _, singular_values, right = torch.linalg.svd(stacked, full_matrices=False)
        eigenvalues = singular_values.square()
        if not torch.isfinite(eigenvalues).all().item():
            raise ValueError("FLoRG Gram spectrum is non-finite")
        candidate = singular_values[:rank, None] * right[:rank]
        before = torch.linalg.vector_norm(candidate - previous).item()

        if align:
            cross = previous @ candidate.T
            if torch.count_nonzero(cross).item():
                left, _, right = torch.linalg.svd(cross, full_matrices=False)
                candidate = (left @ right) @ candidate

        tolerance = torch.finfo(work_dtype).eps * max(stacked.shape) * singular_values[0]
        return GramAggregationResult(
            factor=candidate.to(previous_factor),
            gram_rank=int(torch.count_nonzero(singular_values > tolerance)),
            eigenvalues=eigenvalues,
            gram_projection_residual=torch.linalg.vector_norm(eigenvalues[rank:]).item(),
            factor_drift_before=before,
            factor_drift_after=torch.linalg.vector_norm(candidate - previous).item(),
        )

    def set_reference(self, reference: Mapping[str, torch.Tensor]) -> None:
        if not reference or any(key.rsplit(".", 1)[-1] != GRAM_FACTOR_NAME for key in reference):
            raise ValueError("FLoRG reference must contain the previous broadcast's Gram factors")
        validate_factor_payload(reference, reference)
        self._reference = {
            key: value.detach().to(device=self._device, copy=True) for key, value in reference.items()
        }

    def _before_aggregation(self) -> None:
        if not self._reference:
            raise RuntimeError("Set the previous FLoRG broadcast before aggregation")
        active = []
        for payload, volume in self._aggregation_data_dict:
            count = validated_sample_count(volume)
            if count > 0:
                validate_factor_payload(payload, self._reference)
                active.append((payload, count))
        if not active:
            raise ValueError("FLoRG requires at least one positive-sample upload")
        self._aggregation_data_dict = active

    def _do_aggregation(self) -> None:
        payloads, sample_counts = zip(*self._aggregation_data_dict)
        weights = [1.0] * len(payloads) if self.weighting_mode == "uniform" else sample_counts
        output = {}
        stats = {}
        for key, previous in self._reference.items():
            result = self.aggregate_gram_factors(
                [payload[key] for payload in payloads], weights, previous, align=self.align,
            )
            # C=0 is a stationary point of the Gram parameterization. If the
            # mean provides no direction, keep the last broadcast trainable.
            keep_previous = result.gram_rank == 0
            output[key] = previous if keep_previous else result.factor
            if keep_previous:
                work_dtype = torch.float64 if previous.dtype == torch.float64 else torch.float32
                work = previous.to(work_dtype)
                residual = torch.linalg.vector_norm(work @ work.T).item()
            else:
                residual = result.gram_projection_residual
            stats[key] = {
                "gram_rank": result.gram_rank,
                "target_rank": previous.shape[0],
                "eigenvalues": result.eigenvalues.tolist(),
                "gram_projection_residual": result.gram_projection_residual,
                "broadcast_gram_residual": residual,
                "factor_drift_before": result.factor_drift_before,
                "factor_drift_after": 0.0 if keep_previous else result.factor_drift_after,
                "kept_previous_factor": keep_previous,
            }
        self._aggregated_weight = output
        self.last_stats = stats

    def _after_aggregation(self) -> None:
        return
