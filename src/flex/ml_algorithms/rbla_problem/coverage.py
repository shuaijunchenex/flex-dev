"""Slot-level support and label-coverage diagnostics for isolated RBLA studies."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from ...fl_algorithms.aggregation.methods._fed_aggregator_rbla import FedAggregator_RBLA
from .support_scaling import coefficient_for_eligible, normalise_weights


def lora_pairs(state_dict: Dict[str, torch.Tensor]) -> List[Tuple[str, str, str]]:
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


def _normalise(weights: Sequence[float]) -> List[float]:
    total = float(sum(weights))
    if total <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [float(weight) / total for weight in weights]


def support_scaled_discrepancy(
    client_state_dicts: List[Dict[str, torch.Tensor]],
    weights: Sequence[float],
    aggregated_state_dict: Dict[str, torch.Tensor],
    *,
    gamma: float = 0.0,
    scaling_type: str = "q_power",
    eps: float = 1e-8,
) -> float:
    """Compare actual factor aggregation with the matching gamma-weighted dense sum."""
    return support_scaled_discrepancy_metrics(
        client_state_dicts,
        weights,
        aggregated_state_dict,
        gamma=gamma,
        scaling_type=scaling_type,
        eps=eps,
    )["ref_agg_discrepancy"]


def support_scaled_discrepancy_metrics(
    client_state_dicts: List[Dict[str, torch.Tensor]],
    weights: Sequence[float],
    aggregated_state_dict: Dict[str, torch.Tensor],
    *,
    gamma: float = 0.0,
    scaling_type: str = "q_power",
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Return matching relative/absolute discrepancy and global factor norms."""
    normalised = normalise_weights(weights)
    discrepancies: List[float] = []
    numerators: List[float] = []
    denominators: List[float] = []
    a_norm_squared = 0.0
    b_norm_squared = 0.0
    delta_norm_squared = 0.0
    for _prefix, a_key, b_key in lora_pairs(aggregated_state_dict):
        available = [
            (sd[a_key].detach().float().cpu(), sd[b_key].detach().float().cpu(), normalised[i])
            for i, sd in enumerate(client_state_dicts)
            if a_key in sd and b_key in sd
        ]
        if not available:
            continue
        max_rank = max(int(a.shape[0]) for a, _b, _w in available)
        out_dim = int(available[0][1].shape[0])
        in_dim = int(available[0][0].shape[1])
        direct = torch.zeros(out_dim, in_dim, dtype=available[0][0].dtype)
        for slot in range(max_rank):
            eligible_indices = [
                index for index, (a, b, _weight) in enumerate(available)
                if slot < a.shape[0] and slot < b.shape[1]
            ]
            q_s = float(sum(available[index][2] for index in eligible_indices))
            if q_s <= 0:
                continue
            scale, _q, _n_eff, _n_eff_full = coefficient_for_eligible(
                [item[2] for item in available],
                eligible_indices,
                scaling_type=scaling_type,
                gamma=gamma,
            )
            for index in eligible_indices:
                a, b, weight = available[index]
                direct += scale * (weight / q_s) * torch.outer(b[:, slot], a[slot, :])

        a_bar = aggregated_state_dict[a_key].detach().float().cpu()
        b_bar = aggregated_state_dict[b_key].detach().float().cpu()
        factor = b_bar @ a_bar
        numerator = torch.linalg.norm(factor - direct)
        denominator = torch.linalg.norm(direct)
        value = numerator / (denominator + eps)
        discrepancies.append(float(value.item()))
        numerators.append(float(numerator.item()))
        denominators.append(float(denominator.item()))
        a_norm_squared += float(torch.linalg.norm(a_bar).item()) ** 2
        b_norm_squared += float(torch.linalg.norm(b_bar).item()) ** 2
        delta_norm_squared += float(torch.linalg.norm(factor).item()) ** 2
    return {
        "ref_agg_discrepancy": float(sum(discrepancies) / len(discrepancies)) if discrepancies else 0.0,
        "ref_agg_discrepancy_abs_numerator": math.sqrt(sum(value * value for value in numerators)),
        "ref_agg_discrepancy_abs_denominator": math.sqrt(sum(value * value for value in denominators)),
        "ref_global_a_norm": math.sqrt(a_norm_squared),
        "ref_global_b_norm": math.sqrt(b_norm_squared),
        "ref_global_delta_w_norm": math.sqrt(delta_norm_squared),
    }


@dataclass
class SlotCoverageDiagnostics:
    """Compute P7 statistics and retain per-round slot records."""

    class_counts: List[List[float]]
    num_classes: int = 10
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self.records: List[dict] = []

    @classmethod
    def from_config(cls, config: dict, eps: float = 1e-8) -> "SlotCoverageDiagnostics":
        distribution = config.get("data_distribution", {})
        name = distribution.get("use", "")
        matrix = distribution.get("custom_define", {}).get(name, [])
        return cls(
            class_counts=[[float(value) for value in row] for row in matrix],
            num_classes=max((len(row) for row in matrix), default=10),
            eps=eps,
        )

    def compute(
        self,
        *,
        round_idx: int,
        client_state_dicts: List[Dict[str, torch.Tensor]],
        weights: Sequence[float],
        aggregated_state_dict: Dict[str, torch.Tensor],
        client_ids: Sequence[str] | None = None,
        client_indices: Sequence[int] | None = None,
    ) -> Dict[str, float]:
        normalised = _normalise(weights)
        n_eff_full = 1.0 / sum(weight * weight for weight in normalised)
        ids = list(client_ids or [f"client.{i + 1}" for i in range(len(client_state_dicts))])
        global_indices = list(client_indices or range(len(client_state_dicts)))
        round_rows: List[dict] = []

        for layer_index, (prefix, a_key, b_key) in enumerate(lora_pairs(aggregated_state_dict)):
            a_bar = aggregated_state_dict[a_key].detach().float().cpu()
            b_bar = aggregated_state_dict[b_key].detach().float().cpu()
            max_rank = min(int(a_bar.shape[0]), int(b_bar.shape[1]))
            tail_start = max(1, max_rank // 2)
            for slot in range(max_rank):
                eligible = [
                    i for i, sd in enumerate(client_state_dicts)
                    if a_key in sd and b_key in sd
                    and slot < sd[a_key].shape[0] and slot < sd[b_key].shape[1]
                ]
                q_s = float(sum(normalised[i] for i in eligible))
                alphas = [normalised[i] / q_s for i in eligible] if q_s > 0 else []
                n_eff = 1.0 / sum(alpha * alpha for alpha in alphas) if alphas else 0.0
                q_sqrt = math.sqrt(q_s) if q_s > 0 else 0.0
                effective_scale = math.sqrt(n_eff / n_eff_full) if n_eff > 0 else 0.0

                label_mass = [0.0] * self.num_classes
                visible_labels = set()
                for alpha, update_index in zip(alphas, eligible):
                    client_index = global_indices[update_index]
                    counts = self.class_counts[client_index] if client_index < len(self.class_counts) else []
                    client_total = float(sum(counts))
                    if client_total <= 0:
                        continue
                    for class_index in range(min(len(counts), self.num_classes)):
                        count = float(counts[class_index])
                        if count > 0:
                            visible_labels.add(class_index)
                        label_mass[class_index] += alpha * count / client_total
                mass_total = sum(label_mass)
                pi = [value / mass_total for value in label_mass] if mass_total > 0 else label_mass
                entropy = -sum(value * math.log(value) for value in pi if value > 0)
                entropy /= math.log(self.num_classes) if self.num_classes > 1 else 1.0
                energy = float((b_bar[:, slot].norm() * a_bar[slot, :].norm()).item())

                round_rows.append({
                    "round": int(round_idx),
                    "layer_index": int(layer_index),
                    "layer": prefix,
                    "slot": int(slot),
                    "relative_slot": float((slot + 1) / max_rank),
                    "is_tail": int(slot >= tail_start),
                    "raw_support": int(len(eligible)),
                    "support_weight": q_s,
                    "effective_client_count": float(n_eff),
                    "effective_client_count_full": float(n_eff_full),
                    "q_sqrt_coefficient": float(q_sqrt),
                    "effective_support_coefficient": float(effective_scale),
                    "effective_minus_q_sqrt": float(effective_scale - q_sqrt),
                    "class_coverage": int(len(visible_labels)),
                    "class_coverage_ratio": float(len(visible_labels) / max(self.num_classes, 1)),
                    "label_entropy": float(entropy),
                    "slot_energy": energy,
                    "eligible_client_ids": ";".join(ids[i] for i in eligible),
                    "eligible_client_indices": ";".join(str(global_indices[i]) for i in eligible),
                    "eligible_labels": ";".join(str(i) for i in sorted(visible_labels)),
                    "weighted_class_distribution": json.dumps(pi),
                })

        self.records.extend(round_rows)
        if not round_rows:
            return {}
        energies = [row["slot_energy"] for row in round_rows]
        tail_rows = [row for row in round_rows if row["is_tail"]]
        total_energy = sum(energies) + self.eps
        tail_risk = sum(row["slot_energy"] * (1.0 - row["label_entropy"]) for row in tail_rows) / total_energy
        tail_energy = sum(row["slot_energy"] for row in tail_rows)
        tail_entropy = (
            sum(row["slot_energy"] * row["label_entropy"] for row in tail_rows) / (tail_energy + self.eps)
            if tail_rows else 0.0
        )
        return {
            "ref_support_weight_min": min(row["support_weight"] for row in round_rows),
            "ref_support_weight_mean": sum(row["support_weight"] for row in round_rows) / len(round_rows),
            "ref_effective_clients_min": min(row["effective_client_count"] for row in round_rows),
            "ref_effective_clients_mean": sum(row["effective_client_count"] for row in round_rows) / len(round_rows),
            "ref_effective_support_gap_mean": sum(abs(row["effective_minus_q_sqrt"]) for row in round_rows) / len(round_rows),
            "ref_effective_support_gap_max": max(abs(row["effective_minus_q_sqrt"]) for row in round_rows),
            "ref_class_coverage_min": min(row["class_coverage_ratio"] for row in round_rows),
            "ref_class_coverage_mean": sum(row["class_coverage_ratio"] for row in round_rows) / len(round_rows),
            "ref_label_entropy_mean": sum(row["label_entropy"] for row in round_rows) / len(round_rows),
            "ref_tail_label_entropy": float(tail_entropy),
            "ref_tail_risk": float(tail_risk),
        }

    def write_artifacts(self, checkpoint_path: str) -> Dict[str, str]:
        """Write all-round CSV, final JSON, and four requested P7 figures."""
        checkpoint = Path(checkpoint_path)
        stem = checkpoint.with_suffix("")
        csv_path = Path(f"{stem}_slot_coverage.csv")
        json_path = Path(f"{stem}_slot_coverage_final.json")
        if not self.records:
            return {}

        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.records[0]))
            writer.writeheader()
            writer.writerows(self.records)

        final_round = max(int(row["round"]) for row in self.records)
        final_rows = [row for row in self.records if int(row["round"]) == final_round]
        tail_rows = [row for row in final_rows if row["is_tail"]]
        payload = {
            "checkpoint": str(checkpoint),
            "final_round": final_round,
            "tail_slots": tail_rows,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        figures = self._write_plots(final_rows, stem)
        return {"slot_csv": str(csv_path), "slot_json": str(json_path), **figures}

    @staticmethod
    def _write_plots(rows: List[dict], stem: Path) -> Dict[str, str]:
        from .simple_plots import write_line_plot, write_scatter_plot

        outputs: Dict[str, str] = {}
        specs = [
            ("raw_support", "Raw support", "slot_support"),
            ("label_entropy", "Normalized label entropy", "slot_entropy"),
            ("slot_energy", "Slot energy", "slot_energy"),
        ]
        for key, ylabel, suffix in specs:
            layers = sorted({row["layer"] for row in rows})
            series = {
                layer: [(row["relative_slot"], row[key]) for row in rows if row["layer"] == layer]
                for layer in layers
            }
            path = Path(f"{stem}_{suffix}.svg")
            write_line_plot(series, path, xlabel="Relative slot index", ylabel=ylabel, title=ylabel)
            outputs[suffix] = str(path)

        path = Path(f"{stem}_energy_vs_entropy.svg")
        write_scatter_plot(
            [(row["label_entropy"], row["slot_energy"], row["relative_slot"]) for row in rows],
            path,
            xlabel="Normalized label entropy",
            ylabel="Slot energy",
            title="Slot energy vs label entropy",
        )
        outputs["energy_vs_entropy"] = str(path)
        return outputs
