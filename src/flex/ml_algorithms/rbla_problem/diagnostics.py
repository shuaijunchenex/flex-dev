"""Mechanism-level diagnostics for RBLA reference-frame experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

from ...fl_algorithms.aggregation.methods._fed_aggregator_rbla import FedAggregator_RBLA


def _lora_pairs(state_dict: Dict[str, torch.Tensor]) -> List[Tuple[str, str, str]]:
    """Return ``(prefix, A_key, B_key)`` pairs with exact state-dict keys."""
    pairs: List[Tuple[str, str, str]] = []
    for key in state_dict:
        parts = key.split(".")
        if "lora_A" not in parts:
            continue
        index = parts.index("lora_A")
        b_parts = list(parts)
        b_parts[index] = "lora_B"
        b_key = ".".join(b_parts)
        if b_key in state_dict:
            prefix = ".".join(parts[:index] + parts[index + 1 :])
            pairs.append((prefix, key, b_key))
    return pairs


def _weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    value_list = list(values)
    weight_list = list(weights)
    total = float(sum(weight_list))
    if not value_list:
        return 0.0
    if total <= 0:
        return float(sum(value_list) / len(value_list))
    return float(sum(v * w for v, w in zip(value_list, weight_list)) / total)


@dataclass
class RblaReferenceDiagnostics:
    """Compute round-level summaries without changing RBLA aggregation."""

    eps: float = 1e-8
    compute_pinv: bool = True

    def compute(
        self,
        client_state_dicts: List[Dict[str, torch.Tensor]],
        weights: List[float],
        global_state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        if not client_state_dicts:
            return {}
        if len(client_state_dicts) != len(weights):
            raise ValueError("client_state_dicts and weights must have the same length")

        pair_specs = _lora_pairs(global_state_dict)
        if not pair_specs:
            return {}

        drift_cos: List[float] = []
        drift_norm: List[float] = []
        drift_prox: List[float] = []
        frame_transform: List[float] = []
        frame_residual: List[float] = []
        layer_discrepancies: List[float] = []
        tail_ratios: List[float] = []
        support_min: List[float] = []
        support_max: List[float] = []

        normalized_weights = self._normalize_weights(weights)

        for _prefix, a_key, b_key in pair_specs:
            available = [
                (sd[a_key].detach().float().cpu(), sd[b_key].detach().float().cpu(), weight)
                for sd, weight in zip(client_state_dicts, normalized_weights)
                if a_key in sd and b_key in sd
            ]
            if not available:
                continue

            global_a = global_state_dict[a_key].detach().float().cpu()
            tensors_a = [item[0] for item in available]
            tensors_b = [item[1] for item in available]
            layer_weights = [item[2] for item in available]

            for local_a, _local_b, client_weight in available:
                rows = min(int(local_a.shape[0]), int(global_a.shape[0]))
                local_prefix = local_a[:rows]
                global_prefix = global_a[:rows]
                cos = F.cosine_similarity(local_prefix, global_prefix, dim=1, eps=self.eps)
                norm_ratio = torch.log(
                    (local_prefix.norm(dim=1) + self.eps)
                    / (global_prefix.norm(dim=1) + self.eps)
                ).abs()
                prox = (
                    (local_prefix - global_prefix).pow(2).sum(dim=1)
                    / (global_prefix.pow(2).sum(dim=1) + self.eps)
                )
                drift_cos.append(float((1.0 - cos).mean().item()) * client_weight)
                drift_norm.append(float(norm_ratio.mean().item()) * client_weight)
                drift_prox.append(float(prox.mean().item()) * client_weight)

                if self.compute_pinv and rows > 0:
                    # Full-row-rank ridge form of A_i A_g^dagger.  Solving the
                    # r x r Gram system is substantially cheaper than running
                    # an SVD-based pinv on every r x d MNIST matrix.
                    gram = global_prefix @ global_prefix.T
                    ridge = self.eps * torch.eye(rows, dtype=gram.dtype)
                    cross = local_prefix @ global_prefix.T
                    transform = torch.linalg.solve(gram + ridge, cross.T).T
                    identity = torch.eye(rows, dtype=transform.dtype)
                    transform_error = torch.linalg.norm(transform - identity) / max(rows ** 0.5, 1.0)
                    residual = local_prefix - transform @ global_prefix
                    residual_error = torch.linalg.norm(residual) / (torch.linalg.norm(local_prefix) + self.eps)
                    frame_transform.append(float(transform_error.item()) * client_weight)
                    frame_residual.append(float(residual_error.item()) * client_weight)

            a_bar = FedAggregator_RBLA.aggregate_lora_tensors(tensors_a, layer_weights, pad_mode="nan")
            b_bar = FedAggregator_RBLA.aggregate_lora_tensors(tensors_b, layer_weights, pad_mode="nan")
            factor_aggregate = b_bar @ a_bar
            direct_aggregate, supports = self._conditional_dense_average(
                tensors_a, tensors_b, layer_weights
            )
            discrepancy = torch.linalg.norm(factor_aggregate - direct_aggregate) / (
                torch.linalg.norm(direct_aggregate) + self.eps
            )
            layer_discrepancies.append(float(discrepancy.item()))

            component_energy = b_bar.norm(dim=0) * a_bar.norm(dim=1)
            split = max(1, int(component_energy.numel() // 2))
            tail_energy = component_energy[split:].sum()
            tail_ratios.append(float((tail_energy / (component_energy.sum() + self.eps)).item()))
            support_min.append(float(min(supports)))
            support_max.append(float(max(supports)))

        result = {
            "ref_a_cos_drift": float(sum(drift_cos) / max(len(pair_specs), 1)),
            "ref_a_norm_drift": float(sum(drift_norm) / max(len(pair_specs), 1)),
            "ref_a_prox_drift": float(sum(drift_prox) / max(len(pair_specs), 1)),
            "ref_agg_discrepancy": self._mean(layer_discrepancies),
            "ref_tail_energy_ratio": self._mean(tail_ratios),
            "ref_slot_support_min": self._mean(support_min),
            "ref_slot_support_max": self._mean(support_max),
        }
        if self.compute_pinv:
            result["ref_frame_transform_drift"] = float(sum(frame_transform) / max(len(pair_specs), 1))
            result["ref_frame_residual"] = float(sum(frame_residual) / max(len(pair_specs), 1))
        return result

    @staticmethod
    def _normalize_weights(weights: List[float]) -> List[float]:
        total = float(sum(weights))
        if total <= 0:
            return [1.0 / len(weights)] * len(weights)
        return [float(weight) / total for weight in weights]

    @staticmethod
    def _mean(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    def _conditional_dense_average(
        self,
        tensors_a: List[torch.Tensor],
        tensors_b: List[torch.Tensor],
        weights: List[float],
    ) -> Tuple[torch.Tensor, List[int]]:
        max_rank = max(int(a.shape[0]) for a in tensors_a)
        out_dim = int(tensors_b[0].shape[0])
        in_dim = int(tensors_a[0].shape[1])
        dense = torch.zeros(out_dim, in_dim, dtype=tensors_a[0].dtype)
        supports: List[int] = []
        for slot in range(max_rank):
            eligible = [
                index
                for index, (a, b) in enumerate(zip(tensors_a, tensors_b))
                if slot < a.shape[0] and slot < b.shape[1]
            ]
            supports.append(len(eligible))
            slot_weight = sum(weights[index] for index in eligible)
            if slot_weight <= 0:
                continue
            for index in eligible:
                alpha = weights[index] / slot_weight
                dense += alpha * torch.outer(tensors_b[index][:, slot], tensors_a[index][slot, :])
        return dense, supports


def run_reparameterization_stress_test(
    *,
    seed: int = 42,
    client_ranks: Tuple[int, ...] = (2, 4, 6),
    in_dim: int = 12,
    out_dim: int = 10,
) -> Dict[str, float]:
    """Apply client-specific orthogonal gauges while preserving every ``B @ A``."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    factors = []
    transformed = []
    for rank in client_ranks:
        a = torch.randn(rank, in_dim, generator=generator)
        b = torch.randn(out_dim, rank, generator=generator)
        q_raw = torch.randn(rank, rank, generator=generator)
        q, _ = torch.linalg.qr(q_raw)
        factors.append((a, b))
        transformed.append((q.T @ a, b @ q))

    weights = [1.0 / len(factors)] * len(factors)
    dense_before = sum(weight * (b @ a) for weight, (a, b) in zip(weights, factors))
    dense_after = sum(weight * (b @ a) for weight, (a, b) in zip(weights, transformed))

    def factor_average(items: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        a_bar = FedAggregator_RBLA.aggregate_lora_tensors([a for a, _ in items], weights, "nan")
        b_bar = FedAggregator_RBLA.aggregate_lora_tensors([b for _, b in items], weights, "nan")
        return b_bar @ a_bar

    rbla_before = factor_average(factors)
    rbla_after = factor_average(transformed)
    dense_error = torch.linalg.norm(dense_before - dense_after) / (torch.linalg.norm(dense_before) + 1e-8)
    rbla_sensitivity = torch.linalg.norm(rbla_before - rbla_after) / (torch.linalg.norm(rbla_before) + 1e-8)
    per_client_error = max(
        float((torch.linalg.norm(b @ a - bt @ at) / (torch.linalg.norm(b @ a) + 1e-8)).item())
        for (a, b), (at, bt) in zip(factors, transformed)
    )
    return {
        "per_client_function_error": per_client_error,
        "dense_update_invariance_error": float(dense_error.item()),
        "rbla_reparameterization_sensitivity": float(rbla_sensitivity.item()),
    }
