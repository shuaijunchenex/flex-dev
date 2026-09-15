"""
RBLA-SASG aggregator.

Performs per-semantic-slot weighted averaging of client LoRA factors.
Does NOT reconstruct dense ΔW = B·A.  Does NOT perform SVD.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs
from ....ml_utils import console
from ....ml_algorithms.rblasa.semantic_grid import inverse_slot_index, semantic_grid_mapping


class FedAggregator_RBLA_SASG(AbstractFedAggregator):
    """
    RBLA-SASG: per-semantic-slot aggregation.

    Expected client_updates format (list of dicts):
        {
            "updated_weights": state_dict,   # client model state_dict (has lora_A/lora_B)
            "train_record": {...},
            "data_sample_num": int,
            "r_i": int,                      # local LoRA rank
            "Phi_i": list[int],              # semantic slot mapping (1-indexed)
        }

    The aggregator identifies lora_A / lora_B keys in state_dicts,
    maps each local rank index to its global semantic slot, and
    performs a weighted average per semantic slot across all clients
    that cover that slot.

    Non-LoRA keys are averaged with standard FedAvg weighting.

    If a semantic slot is not covered by any client in the current
    round, the previous round's global value for that slot is kept
    (pass-through).
    """

    def __init__(self, args: Optional[FedAggregatorArgs] = None):
        super().__init__(args)
        self._aggregation_method = "rbla_sasg"
        self._max_rank: int = int(args.get("max_rank", 0)) if args is not None else 0
        self._lora_suffixes: set[str] = {"lora_A", "lora_B"}
        # Per-prefix global slot tensors {prefix: {"A": [R_max, in], "B": [out, R_max]}}
        self._global_slots: Dict[str, Dict[str, torch.Tensor]] = {}

    # ---------- Public config ----------
    def set_max_rank(self, max_rank: int) -> None:
        self._max_rank = max_rank

    def set_lora_suffixes(self, suffixes: set[str]) -> None:
        self._lora_suffixes = suffixes

    # ---------- Data building ----------
    def build_data_list(self, aggregation_data_dict: dict) -> None:
        self._aggregation_data_list = list(aggregation_data_dict.values())

    def build_data_dict(self, aggregation_data_dict: Any) -> None:
        self._aggregation_data_dict = aggregation_data_dict

    # ---------- Aggregation lifecycle ----------
    def aggregate(self, client_data_dict):
        """
        Override base aggregate() to preserve full client_update dicts
        (which carry r_i, Phi_i, etc.) instead of stripping to tuples.
        """
        self._aggregation_data_list = list(client_data_dict)  # list of full dicts
        self._before_aggregation()
        self._do_aggregation()
        self._after_aggregation()
        return self._aggregated_weight

    def _before_aggregation(self) -> None:
        return

    def _do_aggregation(self) -> None:
        updates: List[dict] = self._aggregation_data_list

        if not updates:
            return

        # ── First pass: discover per-prefix R_max from client updates ──
        # Prefer client-reported Phi_by_prefix; fall back to tensor shapes.
        prefix_R_max: Dict[str, int] = {}
        for up in updates:
            if not isinstance(up, dict):
                continue
            # Try per-prefix metadata first
            pb = up.get("Phi_by_prefix") or {}
            for prefix, phi_list in pb.items():
                if phi_list:
                    prefix_R_max[prefix] = max(prefix_R_max.get(prefix, 0), max(phi_list))
            # Also scan tensor shapes for any prefix not covered
            sd = up.get("updated_weights") or up.get("state_dict") or {}
            for key, tensor in sd.items():
                suf = self._suffix_of(key)
                if suf not in self._lora_suffixes:
                    continue
                prefix = self._prefix_of(key)
                if prefix in prefix_R_max:
                    continue  # already known from metadata
                r_actual = tensor.shape[0] if suf == "lora_A" else tensor.shape[1]
                prefix_R_max[prefix] = max(prefix_R_max.get(prefix, 0), r_actual)

        if not prefix_R_max:
            console.warn("[RBLA-SASG] No LoRA keys found in client updates.")
            self._aggregated_weight = OrderedDict()
            return

        self._max_rank = max(prefix_R_max.values())
        console.debug(
            f"\n[RBLA-SASG] Aggregating {len(updates)} clients "
            f"(max_rank={self._max_rank}, prefixes={list(prefix_R_max.keys())})"
        )

        # ── Second pass: group per semantic slot using per-prefix Phi ────
        # slot_data[prefix][slot] = {"A": [(tensor, weight), ...], "B": [...]}
        slot_data: Dict[str, Dict[int, Dict[str, List[Tuple[torch.Tensor, float]]]]] = {}

        total_samples = 0.0
        for up in updates:
            if not isinstance(up, dict):
                continue
            sd = up.get("updated_weights") or up.get("state_dict") or {}
            n_i = float(up.get("data_sample_num", 1))
            total_samples += n_i

            # Per-prefix metadata from this client
            pb: Dict[str, List[int]] = up.get("Phi_by_prefix") or {}
            rb: Dict[str, int] = up.get("rank_by_prefix") or {}

            for key, tensor in sd.items():
                suf = self._suffix_of(key)
                if suf not in self._lora_suffixes:
                    continue
                prefix = self._prefix_of(key)
                R_max_p = prefix_R_max.get(prefix, 0)
                if R_max_p <= 0:
                    continue

                # Get per-prefix Phi (client-reported or derived from shape)
                Phi_prefix = pb.get(prefix)
                if Phi_prefix is None:
                    r_actual = tensor.shape[0] if suf == "lora_A" else tensor.shape[1]
                    r_client = rb.get(prefix, r_actual)
                    Phi_prefix = semantic_grid_mapping(
                        min(r_client, R_max_p), R_max_p
                    )

                slot_data.setdefault(prefix, {})
                for k_local, s_global in enumerate(Phi_prefix):
                    if suf == "lora_A" and k_local >= tensor.shape[0]:
                        break
                    if suf == "lora_B" and k_local >= tensor.shape[1]:
                        break

                    slot_data[prefix].setdefault(s_global, {"A": [], "B": []})

                    if suf == "lora_A":
                        slot_data[prefix][s_global]["A"].append(
                            (tensor[k_local, :].detach().to(self._device), n_i)
                        )
                    elif suf == "lora_B":
                        slot_data[prefix][s_global]["B"].append(
                            (tensor[:, k_local].detach().to(self._device), n_i)
                        )

        # ── Per-slot weighted average ────────────────────────────────────
        aggregated: Dict[str, torch.Tensor] = {}

        # Phase 1: Non-LoRA keys — standard FedAvg
        first_sd = None
        for up in updates:
            if isinstance(up, dict):
                sd = up.get("updated_weights") or up.get("state_dict") or {}
                if sd:
                    first_sd = sd
                    break
        if first_sd is not None:
            for key in first_sd:
                if self._suffix_of(key) in self._lora_suffixes:
                    continue
                values: List[torch.Tensor] = []
                aligned_ws: List[float] = []
                for up in updates:
                    sd = up.get("updated_weights") or up.get("state_dict") or {}
                    if key in sd and torch.is_tensor(sd[key]):
                        values.append(sd[key].to(self._device))
                        n_i = float(up.get("data_sample_num", 1))
                        aligned_ws.append(n_i)
                if values:
                    stacked = torch.stack(values, dim=0)
                    w = torch.as_tensor(aligned_ws, dtype=stacked.dtype, device=self._device)
                    w = w.view(-1, *([1] * (stacked.dim() - 1)))
                    aggregated[key] = (stacked * w).sum(dim=0) / w.sum()

        # Phase 2: LoRA keys — per-slot aggregation (per-prefix R_max)
        for prefix, slots in slot_data.items():
            a_key = f"{prefix}.lora_A"
            b_key = f"{prefix}.lora_B"

            # Determine per-prefix max_rank from actual slot coverage
            R_prefix = max(slots.keys()) if slots else 0
            if R_prefix <= 0:
                continue

            # Determine shapes from first available slot
            # (slots store 1-D slices: A-row [d_in], B-col [d_out])
            a_shape, b_shape = None, None
            a_dtype, b_dtype = None, None
            for s in slots:
                if slots[s]["A"]:
                    t = slots[s]["A"][0][0]       # 1-D [d_in]
                    a_shape = (R_prefix, t.shape[0])
                    a_dtype = t.dtype
                    break
            for s in slots:
                if slots[s]["B"]:
                    t = slots[s]["B"][0][0]       # 1-D [d_out]
                    b_shape = (t.shape[0], R_prefix)
                    b_dtype = t.dtype
                    break

            if a_shape is None or b_shape is None:
                continue

            A_agg = torch.zeros(a_shape, dtype=a_dtype, device=self._device)
            B_agg = torch.zeros(b_shape, dtype=b_dtype, device=self._device)

            for s in range(1, R_prefix + 1):
                if s not in slots:
                    continue  # slot not covered → keep zero (caller handles)

                # Aggregate A slot
                a_list = slots[s]["A"]
                if a_list:
                    a_total_w = sum(w for _, w in a_list)
                    if a_total_w > 0:
                        a_sum = sum(t * w for t, w in a_list)
                        A_agg[s - 1, :] = a_sum / a_total_w

                # Aggregate B slot
                b_list = slots[s]["B"]
                if b_list:
                    b_total_w = sum(w for _, w in b_list)
                    if b_total_w > 0:
                        b_sum = sum(t * w for t, w in b_list)
                        B_agg[:, s - 1] = b_sum / b_total_w

            aggregated[a_key] = A_agg
            aggregated[b_key] = B_agg

        # ── Build ordered output ─────────────────────────────────────────
        if first_sd is not None:
            ordered = OrderedDict()
            for k in first_sd:
                if k in aggregated:
                    ordered[k] = aggregated[k]
            for k, v in aggregated.items():
                if k not in ordered:
                    ordered[k] = v
        else:
            ordered = OrderedDict(aggregated)

        self._aggregated_weight = ordered

        # Store per-slot A/B for use in broadcast (server strategy reads these)
        self._slot_A: Dict[str, torch.Tensor] = {}
        self._slot_B: Dict[str, torch.Tensor] = {}
        for key, tensor in aggregated.items():
            suf = self._suffix_of(key)
            prefix = self._prefix_of(key)
            if suf == "lora_A":
                self._slot_A[prefix] = tensor
            elif suf == "lora_B":
                self._slot_B[prefix] = tensor

        console.debug(f"[RBLA-SASG] Aggregation complete. {len(aggregated)} keys.")

    def _after_aggregation(self) -> None:
        return

    # ---------- Slot accessors (used by server strategy) ------------------
    @property
    def slot_A(self) -> Dict[str, torch.Tensor]:
        """Per-prefix A matrices [R_max, d_in]."""
        return getattr(self, "_slot_A", {})

    @property
    def slot_B(self) -> Dict[str, torch.Tensor]:
        """Per-prefix B matrices [d_out, R_max]."""
        return getattr(self, "_slot_B", {})

    @property
    def max_rank(self) -> int:
        return self._max_rank

    # ---------- Helpers ---------------------------------------------------
    @staticmethod
    def _suffix_of(key: str) -> str:
        return key.rsplit(".", 1)[-1]

    @staticmethod
    def _prefix_of(key: str) -> str:
        return key.rsplit(".", 1)[0]
