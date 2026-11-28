import torch
from collections import OrderedDict

from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs
from ....ml_utils import console


class FedAggregator_FFALoRA(AbstractFedAggregator):
    """FFA-LoRA aggregation: Federated Freeze-A LoRA.

    Core idea (from FFA-LoRA):
      - lora_A is **frozen** during local training and **shared** across all
        clients.  The server broadcasts the same global lora_A every round.
      - lora_B is trained locally and aggregated.
      - Non-LoRA parameters (base model) are aggregated via standard FedAvg.

    Heterogeneous-rank lora_B handling (``lora_B_aggregation``):
      - ``"fedavg"``      — same rank across clients, direct weighted average.
      - ``"zeropadding"`` — pad smaller B to max rank with **zeros** before
                           weighted average.  Correct because lora_B is always
                           zero-initialised; untrained columns are truly zero,
                           not missing.
      - ``"truncate"``    — truncate all B matrices to the **minimum** rank
                           across clients.  Conservative: only uses the portion
                           that every client has trained.

    Note: RBLA-style NaN-masking is intentionally *not* offered here because
    the zero value for untrained columns is meaningful (B_init = 0), not
    "unknown".  Using masked averaging would over-weight larger-rank clients in
    the shared columns.

    lora_A is always taken from the first client (frozen & shared).
    """

    def __init__(self, args: FedAggregatorArgs | None = None):
        super().__init__(args)
        self._aggregation_method = "ffalora"
        self._lora_B_mode: str = "zeropadding"  # "fedavg" | "zeropadding" | "truncate"
        return

    def set_lora_B_mode(self, mode: str) -> None:
        assert mode in {"fedavg", "zeropadding", "truncate"}, f"Unknown lora_B mode: {mode}"
        self._lora_B_mode = mode

    def _before_aggregation(self) -> None:
        return

    def _do_aggregation(self) -> None:
        pairs = self._aggregation_data_dict
        if not isinstance(pairs, list) or len(pairs) == 0:
            raise ValueError("[FFA-LoRA] Expected list of (state_dict, data_volume).")

        state_dicts = [sd for sd, _vol in pairs]
        weights     = [float(vol) for _sd, vol in pairs]

        console.debug(f"\n[FFA-LoRA] Aggregating {len(state_dicts)} clients...")
        total_data_vol = sum(weights)
        for i, (_sd, vol) in enumerate(pairs):
            console.debug(f"  Client {i}: {vol} samples ({vol / total_data_vol * 100:.1f}%)")

        dev = self._device
        sds_on_device = [{k: v.to(dev) for k, v in sd.items()} for sd in state_dicts]
        aggregated = self._aggregate_state_dicts(sds_on_device, weights, self._lora_B_mode)

        sample_keys = list(state_dicts[0].keys())
        ordered = OrderedDict((k, aggregated[k]) for k in sample_keys)
        self._aggregated_weight = ordered

        first_param_name = next(iter(ordered.keys()))
        console.debug(f"[FFA-LoRA] Aggregated first param mean: {ordered[first_param_name].mean():.6f}")

    def _after_aggregation(self) -> None:
        console.debug("[FFA-LoRA] Aggregation completed.")

    # ---------- helpers ----------
    @staticmethod
    def _is_lora_A(key: str) -> bool:
        return "lora_A" in key.split(".")

    @staticmethod
    def _is_lora_B(key: str) -> bool:
        return "lora_B" in key.split(".")

    @classmethod
    def _aggregate_state_dicts(
        cls,
        state_dicts: list[dict],
        weights: list[float],
        lora_B_mode: str,
    ) -> dict:
        tw = float(sum(weights))
        weights = [w / tw for w in weights] if tw > 0 else [1.0 / len(weights)] * len(weights)

        keys = list(state_dicts[0].keys())
        aggregated: dict[str, torch.Tensor] = {}

        for key in keys:
            values = [sd[key] for sd in state_dicts]
            if cls._is_lora_A(key):
                # lora_A is frozen & shared → take the first one
                aggregated[key] = values[0].clone()
            elif cls._is_lora_B(key):
                aggregated[key] = cls._aggregate_lora_B(values, weights, lora_B_mode)
            else:
                # Non-LoRA: standard FedAvg
                stacked = torch.stack(values, dim=0)
                view_shape = (len(weights),) + (1,) * (stacked.dim() - 1)
                w = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device).view(*view_shape)
                aggregated[key] = (stacked * w).sum(dim=0)

        return aggregated

    # ---- lora_B heterogeneous-rank aggregation ----
    @staticmethod
    def _aggregate_lora_B(
        values: list[torch.Tensor],
        weights: list[float],
        mode: str,
    ) -> torch.Tensor:
        """Aggregate lora_B matrices possibly of different ranks.

        lora_B shape: (out_dim, r_i).  When r_i differ across clients:

        ``"fedavg"``
            Requires all r_i identical.  Plain weighted average.

        ``"zeropadding"``
            Pad each B_i to (out_dim, r_max) with **zeros**, then weighted
            average.  Correct because lora_B is zero-initialised in LoRA:
            columns beyond a client's trained rank genuinely are zero.

        ``"truncate"``
            Truncate all B_i to (out_dim, r_min) and compute weighted average.
            Conservative: only uses the rank range that *every* client trained.
        """
        shapes = [v.shape for v in values]
        if all(s == shapes[0] for s in shapes):
            mode = "fedavg"  # same rank → plain FedAvg regardless of mode

        if mode == "fedavg":
            stacked = torch.stack(values, dim=0)
            view_shape = (len(weights),) + (1,) * (stacked.dim() - 1)
            w = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device).view(*view_shape)
            return (stacked * w).sum(dim=0)

        out_dim = shapes[0][0]
        device = values[0].device
        dtype = values[0].dtype

        if mode == "zeropadding":
            # Zero-pad to (out_dim, r_max) — untrained columns are legitimately 0
            r_max = max(s[1] for s in shapes)
            result = torch.zeros(out_dim, r_max, dtype=dtype, device=device)
            for v, w in zip(values, weights):
                result[:, : v.shape[1]] += w * v
            return result

        if mode == "truncate":
            # Truncate to shared min rank — only average what everyone trained
            r_min = min(s[1] for s in shapes)
            truncated = [v[:, :r_min] for v in values]
            stacked = torch.stack(truncated, dim=0)           # (N, out_dim, r_min)
            w = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device).view(-1, 1, 1)
            return (stacked * w).sum(dim=0)

        raise ValueError(f"Unknown lora_B mode: {mode}")
