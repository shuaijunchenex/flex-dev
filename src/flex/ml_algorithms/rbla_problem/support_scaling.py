"""Reusable slot-support scaling primitives for isolated RBLA experiments."""
from __future__ import annotations

import math
from typing import Sequence

import torch


SCALING_TYPES = frozenset({"conditional", "q_power", "effective_support", "population"})


def normalise_weights(weights: Sequence[float]) -> list[float]:
    if not weights:
        raise ValueError("weights must not be empty")
    total = float(sum(weights))
    if total <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [float(weight) / total for weight in weights]


def canonical_scaling_type(scaling_type: str, gamma: float) -> str:
    name = str(scaling_type or "q_power").lower()
    aliases = {
        "gamma": "q_power",
        "q": "population",
        "q_sqrt": "q_power",
        "none": "conditional",
        "effective": "effective_support",
    }
    name = aliases.get(name, name)
    if name not in SCALING_TYPES:
        raise ValueError(f"Unsupported support scaling type: {scaling_type}")
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError(f"support gamma must be in [0, 1], got {gamma}")
    return name


def coefficient_for_eligible(
    normalised_weights: Sequence[float],
    eligible_indices: Sequence[int],
    *,
    scaling_type: str = "q_power",
    gamma: float = 0.0,
) -> tuple[float, float, float, float]:
    """Return ``(coefficient, q_s, N_eff,s, N_eff,full)`` for one slot."""
    name = canonical_scaling_type(scaling_type, gamma)
    q_s = float(sum(normalised_weights[index] for index in eligible_indices))
    n_eff_full = 1.0 / sum(float(weight) ** 2 for weight in normalised_weights)
    if q_s <= 0:
        return 0.0, 0.0, 0.0, n_eff_full
    alphas = [float(normalised_weights[index]) / q_s for index in eligible_indices]
    n_eff = 1.0 / sum(alpha * alpha for alpha in alphas)
    if name == "conditional":
        coefficient = 1.0
    elif name == "population":
        coefficient = q_s
    elif name == "effective_support":
        coefficient = math.sqrt(n_eff / n_eff_full)
    else:
        coefficient = q_s ** float(gamma)
    return float(coefficient), q_s, float(n_eff), float(n_eff_full)


def _pad_with_nan(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        raise ValueError("tensors must not be empty")
    ndim = tensors[0].dim()
    if any(tensor.dim() != ndim for tensor in tensors):
        raise ValueError("all tensors must have the same dimensionality")
    max_shape = tuple(max(int(tensor.shape[axis]) for tensor in tensors) for axis in range(ndim))
    padded = []
    for tensor in tensors:
        target = torch.full(max_shape, float("nan"), dtype=tensor.dtype, device=tensor.device)
        target[tuple(slice(0, int(size)) for size in tensor.shape)] = tensor
        padded.append(target)
    return torch.stack(padded)


def aggregate_scaled_lora_b(
    tensors: Sequence[torch.Tensor],
    weights: Sequence[float],
    *,
    scaling_type: str = "q_power",
    gamma: float = 0.0,
) -> torch.Tensor:
    """Conditionally aggregate B then apply exactly one slot coefficient."""
    if len(tensors) != len(weights):
        raise ValueError("tensors and weights must have the same length")
    name = canonical_scaling_type(scaling_type, gamma)
    normalised = normalise_weights(weights)
    padded = _pad_with_nan(tensors)
    valid = ~torch.isnan(padded)
    clean = torch.nan_to_num(padded, nan=0.0)
    view_shape = (len(normalised),) + (1,) * (padded.dim() - 1)
    weight_tensor = torch.as_tensor(
        normalised, dtype=clean.dtype, device=clean.device
    ).view(view_shape)
    weighted_sum = (clean * weight_tensor).sum(dim=0)
    q_s = (valid * weight_tensor).sum(dim=0)
    safe_q = torch.where(q_s > 0, q_s, torch.ones_like(q_s))
    conditional = weighted_sum / safe_q

    if name == "conditional":
        coefficient = torch.ones_like(safe_q)
    elif name == "population":
        coefficient = safe_q
    elif name == "effective_support":
        alpha_squared_sum = (valid * (weight_tensor / safe_q) ** 2).sum(dim=0)
        n_eff = torch.where(
            alpha_squared_sum > 0,
            alpha_squared_sum.reciprocal(),
            torch.zeros_like(alpha_squared_sum),
        )
        n_eff_full = 1.0 / sum(weight * weight for weight in normalised)
        coefficient = torch.sqrt(n_eff / float(n_eff_full))
    else:
        coefficient = safe_q.pow(float(gamma))
    coefficient = torch.where(q_s > 0, coefficient, torch.zeros_like(coefficient))
    result = conditional * coefficient
    if not torch.isfinite(result).all():
        raise FloatingPointError("support-scaled B aggregation produced NaN/Inf")
    return result
