import torch
from collections import OrderedDict

from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs
from ....ml_utils import console


class FedAggregator_RoLoRA(AbstractFedAggregator):
    """RoLoRA: Rotation-aligned LoRA aggregation.

    Core idea: during training, each client regularises ``lora_A`` towards
    orthonormal rows (``A @ A^T ≈ I``).  Because every client's A matrix
    lives in the same canonical orthogonal frame, the server can aggregate
    them via a **simple weighted average** — no SVD, QR, or alignment step
    is needed at aggregation time.

    This aggregator is therefore equivalent to plain FedAvg over all
    parameters (LoRA and non-LoRA alike).

    References
    ----------
    RoLoRA enforces the orthogonal constraint during local training; the
    server-side aggregation simply averages the resulting parameters.
    """

    def __init__(self, args: FedAggregatorArgs | None = None):
        super().__init__(args)
        self._aggregation_method = "rolora"
        return

    def _before_aggregation(self) -> None:
        return

    def _do_aggregation(self) -> None:
        pairs = self._aggregation_data_dict
        if not isinstance(pairs, list) or len(pairs) == 0:
            raise ValueError("[RoLoRA] Expected list of (state_dict, data_volume).")

        state_dicts = [sd for sd, _vol in pairs]
        weights     = [float(vol) for _sd, vol in pairs]

        console.debug(f"\n[RoLoRA] Aggregating {len(state_dicts)} clients...")
        total = sum(weights)
        for i, (_sd, vol) in enumerate(pairs):
            console.debug(f"  Client {i}: {vol} samples ({vol / total * 100:.1f}%)")

        dev = self._device
        sds_on_device = [{k: v.to(dev) for k, v in sd.items()} for sd in state_dicts]
        aggregated = self._weighted_average(sds_on_device, weights)

        sample_keys = list(state_dicts[0].keys())
        ordered = OrderedDict((k, aggregated[k]) for k in sample_keys)
        self._aggregated_weight = ordered
        first = next(iter(ordered.keys()))
        console.debug(f"[RoLoRA] Aggregated first param mean: {ordered[first].mean():.6f}")

    def _after_aggregation(self) -> None:
        console.debug("[RoLoRA] Aggregation completed.")

    # ------------------------------------------------------------------
    # Weighted average over all parameters (identical to FedAvg)
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_average(
        state_dicts: list[dict],
        weights: list[float],
    ) -> dict:
        """Per-key weighted average of *state_dicts*."""
        tw = float(sum(weights))
        weights = [w / tw for w in weights] if tw > 0 else [1.0 / len(weights)] * len(weights)

        aggregated: dict[str, torch.Tensor] = {}
        keys = list(state_dicts[0].keys())
        for key in keys:
            values = [sd[key] for sd in state_dicts]
            stacked = torch.stack(values, dim=0)             # (N, *shape)
            view_shape = (len(weights),) + (1,) * (stacked.dim() - 1)
            w = torch.as_tensor(weights, dtype=stacked.dtype,
                                device=stacked.device).view(*view_shape)
            aggregated[key] = (stacked * w).sum(dim=0)

        return aggregated
