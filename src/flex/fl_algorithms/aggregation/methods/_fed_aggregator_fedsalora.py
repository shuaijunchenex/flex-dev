import torch
from collections import OrderedDict

from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs
from ....ml_utils import console


class FedAggregator_FedSALoRA(AbstractFedAggregator):
    """FedSA-LoRA: Federated Sparse-Alternating LoRA aggregation.

    Key ideas:
      - lora_A is **frozen** (same as FFA-LoRA): server takes it from the first
        client and does not aggregate it.
      - lora_B is trained locally and aggregated, optionally with top-k
        sparsification applied before averaging.
      - Non-LoRA parameters: standard FedAvg.

    Top-k sparsification (``top_k_ratio``):
      Before averaging, each client's lora_B is sparsified server-side by
      zeroing out all but the top-k fraction of elements by absolute magnitude.
      Set to 1.0 (default) to disable.  This simulates the communication
      compression described in the FedSA-LoRA paper.

    Heterogeneous-rank lora_B (``lora_B_mode``):
      - ``"zeropadding"``  — pad to max rank with zeros (default).
      - ``"truncate"``     — truncate to min rank.
      - ``"fedavg"``       — direct average, requires same rank.
    """

    def __init__(self, args: FedAggregatorArgs | None = None):
        super().__init__(args)
        self._aggregation_method = "fedsalora"
        self._top_k_ratio: float = 1.0      # 1.0 = no sparsification
        self._lora_B_mode: str = "zeropadding"
        return

    def set_top_k_ratio(self, ratio: float) -> None:
        assert 0.0 < ratio <= 1.0, "top_k_ratio must be in (0, 1]"
        self._top_k_ratio = ratio

    def set_lora_B_mode(self, mode: str) -> None:
        assert mode in {"fedavg", "zeropadding", "truncate"}, f"Unknown lora_B mode: {mode}"
        self._lora_B_mode = mode

    def _before_aggregation(self) -> None:
        return

    def _do_aggregation(self) -> None:
        pairs = self._aggregation_data_dict
        if not isinstance(pairs, list) or len(pairs) == 0:
            raise ValueError("[FedSA-LoRA] Expected list of (state_dict, data_volume).")

        state_dicts = [sd for sd, _vol in pairs]
        weights     = [float(vol) for _sd, vol in pairs]

        console.debug(f"\n[FedSA-LoRA] Aggregating {len(state_dicts)} clients, "
                      f"top_k={self._top_k_ratio:.2f}, mode={self._lora_B_mode}")
        total = sum(weights)
        for i, (_sd, vol) in enumerate(pairs):
            console.debug(f"  Client {i}: {vol} samples ({vol / total * 100:.1f}%)")

        dev = self._device
        sds_on_device = [{k: v.to(dev) for k, v in sd.items()} for sd in state_dicts]
        aggregated = self._aggregate(sds_on_device, weights,
                                     self._top_k_ratio, self._lora_B_mode)

        sample_keys = list(state_dicts[0].keys())
        ordered = OrderedDict((k, aggregated[k]) for k in sample_keys)
        self._aggregated_weight = ordered
        first = next(iter(ordered.keys()))
        console.debug(f"[FedSA-LoRA] Aggregated first param mean: {ordered[first].mean():.6f}")

    def _after_aggregation(self) -> None:
        console.debug("[FedSA-LoRA] Aggregation completed.")

    # ---------- helpers ----------
    @staticmethod
    def _is_lora_A(key: str) -> bool:
        return "lora_A" in key.split(".")

    @staticmethod
    def _is_lora_B(key: str) -> bool:
        return "lora_B" in key.split(".")

    @staticmethod
    def _topk_sparsify(t: torch.Tensor, ratio: float) -> torch.Tensor:
        """Zero out all but the top-k fraction of elements (by abs magnitude)."""
        if ratio >= 1.0:
            return t
        flat = t.reshape(-1)
        k = max(1, int(flat.numel() * ratio))
        threshold = flat.abs().topk(k, largest=True, sorted=False).values.min()
        mask = t.abs() >= threshold
        return t * mask

    @classmethod
    def _aggregate(
        cls,
        state_dicts: list[dict],
        weights: list[float],
        top_k_ratio: float,
        lora_B_mode: str,
    ) -> dict:
        tw = float(sum(weights))
        weights = [w / tw for w in weights] if tw > 0 else [1.0 / len(weights)] * len(weights)

        keys = list(state_dicts[0].keys())
        aggregated: dict[str, torch.Tensor] = {}

        for key in keys:
            values = [sd[key] for sd in state_dicts]

            if cls._is_lora_A(key):
                # lora_A frozen & shared — take from first client
                aggregated[key] = values[0].clone()

            elif cls._is_lora_B(key):
                # Optional sparsification of each B_i before averaging
                values = [cls._topk_sparsify(v, top_k_ratio) for v in values]
                aggregated[key] = cls._aggregate_lora_B(values, weights, lora_B_mode)

            else:
                stacked = torch.stack(values, dim=0)
                view_shape = (len(weights),) + (1,) * (stacked.dim() - 1)
                w = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device).view(*view_shape)
                aggregated[key] = (stacked * w).sum(dim=0)

        return aggregated

    @staticmethod
    def _aggregate_lora_B(
        values: list[torch.Tensor],
        weights: list[float],
        mode: str,
    ) -> torch.Tensor:
        """Same heterogeneous-rank handling as FFA-LoRA."""
        shapes = [v.shape for v in values]
        if all(s == shapes[0] for s in shapes):
            mode = "fedavg"

        if mode == "fedavg":
            stacked = torch.stack(values, dim=0)
            view_shape = (len(weights),) + (1,) * (stacked.dim() - 1)
            w = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device).view(*view_shape)
            return (stacked * w).sum(dim=0)

        out_dim = shapes[0][0]
        device  = values[0].device
        dtype   = values[0].dtype

        if mode == "zeropadding":
            r_max  = max(s[1] for s in shapes)
            result = torch.zeros(out_dim, r_max, dtype=dtype, device=device)
            for v, w in zip(values, weights):
                result[:, : v.shape[1]] += w * v
            return result

        if mode == "truncate":
            r_min    = min(s[1] for s in shapes)
            stacked  = torch.stack([v[:, :r_min] for v in values], dim=0)
            w = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device).view(-1, 1, 1)
            return (stacked * w).sum(dim=0)

        raise ValueError(f"Unknown lora_B mode: {mode}")
