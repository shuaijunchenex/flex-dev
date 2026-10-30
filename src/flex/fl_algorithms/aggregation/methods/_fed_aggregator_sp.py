import torch
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional, Any

from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs
from ....ml_utils import console

class FedAggregator_SP(AbstractFedAggregator):
    """
    Sum-Product aggregator (SP):
    - For each LoRA prefix (prefix.lora_A / prefix.lora_B), compute per-client ΔW_i = B_i @ A_i.
    - Aggregate ΔW_i across clients via a weighted sum to obtain ΔW_agg.
    - For base weights (non-LoRA keys), compute their weighted average across clients.
    - The final output writes base W (weighted) and, for each LoRA prefix, an additional key:
    f"{prefix}.sp_aggregated" = base_W_agg + ΔW_agg (or just ΔW_agg if base is missing).
    - The output dict does not include lora_A / lora_B tensors.
    """

    def __init__(self, args: Optional[FedAggregatorArgs] = None):
        super().__init__(args)
        self._aggregation_method = "sp"
        self._lora_suffix_A = "lora_A"
        self._lora_suffix_B = "lora_B"
        # If True, normalize client weights so they sum to 1.0
        self._normalize_weights: bool = True

    # ---------- Public config ----------
    def set_lora_suffixes(self, suffix_A: str = "lora_A", suffix_B: str = "lora_B") -> None:
        """Customize the suffixes used to detect LoRA A/B keys."""
        self._lora_suffix_A = suffix_A
        self._lora_suffix_B = suffix_B

    def set_normalize_weights(self, normalize: bool = True) -> None:
        """Enable/disable weight normalization (default: True)."""
        self._normalize_weights = normalize

    # ---------- FedAvg-style data building ----------
    def build_data_list(self, aggregation_data_dict: dict) -> None:
        """
        Accept an internal mapping like:
            {client_id: (state_dict, data_volume), ...}
        and convert it into a list of (state_dict, weight).
        """
        self._aggregation_data_list = list(aggregation_data_dict.values())

    def build_data_dict(self, aggregation_data_dict: Any) -> None:
        """
        Accept either:
        - list[(state_dict, weight)]
        - dict{'state_dicts': [...], 'weights': [...]}
        Store it directly for later use.
        """
        self._aggregation_data_dict = aggregation_data_dict

    # ---------- Helpers ----------
    @staticmethod
    def _suffix_of(key: str) -> str:
        """Return the last dotted component (e.g., 'layer.lora_A' -> 'lora_A')."""
        return key.rsplit(".", 1)[-1]

    @staticmethod
    def _prefix_of(key: str) -> str:
        """Return the prefix before the last dot (e.g., 'layer.lora_A' -> 'layer')."""
        return key.rsplit(".", 1)[0]

    @staticmethod
    def _weighted_sum(tensors: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
        """
        Weighted sum of tensors with identical shape/device/dtype.
        weights[i] is aligned with tensors[i].
        """
        if len(tensors) == 0:
            raise ValueError("Weighted sum received an empty tensor list.")
        if len(tensors) != len(weights):
            raise ValueError(f"Tensor/weight length mismatch: {len(tensors)} vs {len(weights)}")

        stacked = torch.stack(tensors, dim=0)
        w = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device)
        # Broadcast weights to stacked shape
        view_shape = (len(weights),) + (1,) * (stacked.dim() - 1)
        w = w.view(view_shape)
        return (stacked * w).sum(dim=0)

    def _collect_inputs(self) -> Tuple[List[Dict[str, torch.Tensor]], List[float]]:
        """
        Collect (state_dicts, weights) from the stored data structure.
        Optionally normalize weights to sum to 1.0.
        """
        if hasattr(self, "_aggregation_data_list") and self._aggregation_data_list:
            pairs = self._aggregation_data_list
            sds = [sd for sd, vol in pairs]
            ws = [float(vol) for _, vol in pairs]
        elif hasattr(self, "_aggregation_data_dict") and isinstance(self._aggregation_data_dict, list):
            pairs = self._aggregation_data_dict
            sds = [sd for sd, vol in pairs]
            ws = [float(vol) for _, vol in pairs]
        else:
            agg = self._aggregation_data_dict
            sds = agg["state_dicts"]
            ws = agg.get("weights", [1.0] * len(sds))

        if len(sds) != len(ws):
            raise ValueError(f"Number of state_dicts and weights must match: {len(sds)} vs {len(ws)}")

        if self._normalize_weights:
            tw = float(sum(ws))
            ws = [w / tw for w in ws] if tw > 0 else [1.0 / len(ws)] * len(ws)

        return sds, ws

    # ---------- Aggregation core ----------
    def _before_aggregation(self) -> None:
        """Hook for pre-aggregation logic (no-op by default)."""
        return

    def _do_aggregation(self) -> None:
        """
        Produce an OrderedDict that contains only base-weight keys plus optional
        '{prefix}.sp_aggregated' entries for layers where LoRA A/B pairs exist.

        Rules:
        - For non-LoRA keys (and tensor values): compute a weighted sum across clients,
            aligning each present value with its client weight.
        - For each LoRA prefix where both A and B are present in a client:
            ΔW_i = B_i @ A_i  (shape: out × in)
            Then compute ΔW_agg as a weighted sum across those clients that provide the pair.
        - If a base weight 'prefix' exists, write:
                f"{prefix}.sp_aggregated" = base_agg + ΔW_agg
            Otherwise:
                f"{prefix}.sp_aggregated" = ΔW_agg
        """
        state_dicts, weights = self._collect_inputs()
        dev = self._device
        sds_on_device: List[Dict[str, torch.Tensor]] = [{k: v.to(dev) for k, v in sd.items()} for sd in state_dicts]

        sample_keys = list(state_dicts[0].keys())
        aggregated: Dict[str, torch.Tensor] = {}

        console.debug(f"\n[Sum-Product] Aggregating {len(state_dicts)} clients...")
        total_data_vol = sum(vol for _, vol in self._aggregation_data_dict)
        for i, (_, vol) in enumerate(self._aggregation_data_dict):
            console.debug(f"  Client {i}: {vol} samples ({vol / total_data_vol * 100:.1f}%)")

        # 1) Weighted sum for all base (non-LoRA) tensor keys.
        #    Align values with the correct client weights (skip missing keys per client).
        for key in sample_keys:
            suf = self._suffix_of(key)
            if suf in {self._lora_suffix_A, self._lora_suffix_B}:
                continue
            values: List[torch.Tensor] = []
            aligned_ws: List[float] = []
            for ci, sd in enumerate(sds_on_device):
                if key in sd and torch.is_tensor(sd[key]):
                    values.append(sd[key])
                    aligned_ws.append(weights[ci])
            if values:
                aggregated[key] = self._weighted_sum(values, aligned_ws)

        # 2) Group per-client LoRA (A, B) by prefix; then aggregate ΔW = B@A.
        pair_groups: Dict[str, List[Tuple[int, torch.Tensor, torch.Tensor]]] = {}
        for ci, sd in enumerate(sds_on_device):
            per_client_pairs: Dict[str, Dict[str, str]] = {}
            for k in sd.keys():
                suf = self._suffix_of(k)
                if suf == self._lora_suffix_A or suf == self._lora_suffix_B:
                    prefix = self._prefix_of(k)
                    per_client_pairs.setdefault(prefix, {})
                    per_client_pairs[prefix][suf] = k
            for prefix, d in per_client_pairs.items():
                if self._lora_suffix_A in d and self._lora_suffix_B in d:
                    kA, kB = d[self._lora_suffix_A], d[self._lora_suffix_B]
                    pair_groups.setdefault(prefix, []).append((ci, sd[kA], sd[kB]))

        # Compute ΔW_agg per prefix and write '{prefix}.sp_aggregated'
        for prefix, entries in pair_groups.items():
            if not entries:
                continue
            dWs: List[torch.Tensor] = []
            local_ws: List[float] = []
            for ci, A, B in entries:
                # ΔW_i = B @ A -> shape [out, in]
                dWs.append(B @ A)
                local_ws.append(weights[ci])

            dW_agg = self._weighted_sum(dWs, local_ws)

            base_key = prefix
            out_key = f"{prefix}.sp_aggregated"
            if base_key in aggregated:
                aggregated[out_key] = aggregated[base_key] + dW_agg
            else:
                aggregated[out_key] = dW_agg

        # 3) Build an OrderedDict: keep original (sample_keys) order for base keys, then append extras.
        ordered = OrderedDict(
            (k, aggregated[k])
            for k in sample_keys
            if k in aggregated and self._suffix_of(k) not in {self._lora_suffix_A, self._lora_suffix_B}
        )
        for k, v in aggregated.items():
            if k not in ordered:
                ordered[k] = v

        self._aggregated_weight = ordered

    def _after_aggregation(self) -> None:
        """Hook for post-aggregation logic (no-op by default)."""
        return
