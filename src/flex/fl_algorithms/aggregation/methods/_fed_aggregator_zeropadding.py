import torch
from collections import OrderedDict

from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs
from ....ml_utils import console


class FedAggregator_ZeroPadding(AbstractFedAggregator):
    """Zero-Padding LoRA aggregation.

    Pad all LoRA matrices (lora_A, lora_B) to the maximum rank across clients
    with zeros, then perform standard weighted FedAvg.  Non-LoRA parameters
    are averaged directly without any padding.

    This is the simplest heterogeneous-rank aggregation baseline: every client
    trains its own rank, the server pads smaller matrices to match the largest
    one, and averages element-wise.
    """

    def __init__(self, args: FedAggregatorArgs | None = None):
        super().__init__(args)
        self._aggregation_method = "zeropadding"
        self._lora_suffixes: set[str] = {"lora_A", "lora_B"}
        return

    def _before_aggregation(self) -> None:
        return

    def _do_aggregation(self) -> None:
        pairs = self._aggregation_data_dict
        if not isinstance(pairs, list) or len(pairs) == 0:
            raise ValueError("[ZeroPadding] Expected list of (state_dict, data_volume).")

        state_dicts = [sd for sd, _vol in pairs]
        weights     = [float(vol) for _sd, vol in pairs]

        console.debug(f"\n[ZeroPadding] Aggregating {len(state_dicts)} clients...")
        total_data_vol = sum(weights)
        for i, (_sd, vol) in enumerate(pairs):
            console.debug(f"  Client {i}: {vol} samples ({vol / total_data_vol * 100:.1f}%)")

        dev = self._device
        sds_on_device = [{k: v.to(dev) for k, v in sd.items()} for sd in state_dicts]
        aggregated = self._aggregate_state_dicts(sds_on_device, weights)

        sample_keys = list(state_dicts[0].keys())
        ordered = OrderedDict((k, aggregated[k]) for k in sample_keys)
        self._aggregated_weight = ordered

        first_param_name = next(iter(ordered.keys()))
        console.debug(f"[ZeroPadding] Aggregated first param mean: {ordered[first_param_name].mean():.6f}")

    def _after_aggregation(self) -> None:
        console.debug("[ZeroPadding] Aggregation completed.")

    # ---------- helpers ----------
    @staticmethod
    def _is_lora(key: str) -> bool:
        parts = key.split(".")
        return any(s in parts for s in {"lora_A", "lora_B"})

    @staticmethod
    def _pad_2d_to_max(tensors: list[torch.Tensor]) -> torch.Tensor:
        """Pad 2-D tensors to (max_rows, max_cols) with zeros; return (N, max_r, max_c)."""
        max_rows = max(t.shape[0] for t in tensors)
        max_cols = max(t.shape[1] for t in tensors)
        out = []
        for t in tensors:
            p = torch.zeros(max_rows, max_cols, dtype=t.dtype, device=t.device)
            p[:t.shape[0], :t.shape[1]] = t
            out.append(p)
        return torch.stack(out, dim=0)

    @classmethod
    def _aggregate_state_dicts(cls, state_dicts: list[dict], weights: list[float]) -> dict:
        tw = float(sum(weights))
        weights = [w / tw for w in weights] if tw > 0 else [1.0 / len(weights)] * len(weights)

        keys = list(state_dicts[0].keys())
        aggregated: dict[str, torch.Tensor] = {}

        for key in keys:
            values = [sd[key] for sd in state_dicts]
            if cls._is_lora(key) and values[0].dim() == 2:
                padded = cls._pad_2d_to_max(values)               # (N, max_r, max_c)
                w = torch.as_tensor(weights, dtype=padded.dtype, device=padded.device).view(-1, 1, 1)
                aggregated[key] = (padded * w).sum(dim=0)
            else:
                stacked = torch.stack(values, dim=0)
                view_shape = (len(weights),) + (1,) * (stacked.dim() - 1)
                w = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device).view(*view_shape)
                aggregated[key] = (stacked * w).sum(dim=0)

        return aggregated
