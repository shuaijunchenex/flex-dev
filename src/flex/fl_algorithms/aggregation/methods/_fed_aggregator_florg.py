import torch
from collections import OrderedDict

from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs
from ....ml_utils import console


class FedAggregator_FLoRG(AbstractFedAggregator):
    """FLoRG: Federated LoRA with Gate aggregation.

    Instead of weighting clients purely by data volume, FLoRG uses a gate
    mechanism that scores each client's LoRA adapter by the magnitude of its
    effective update in the full-rank product space:

        g_i = ‖ B_i @ A_i ‖_F                    (Frobenius norm of the product)

    The gate score is then softmax-normalised with a temperature T:

        gate_i = softmax( g_i / T )_i

    The final aggregation weight blends the gate score with the data-volume
    weight via a mixing coefficient α:

        final_w_i = α · gate_i + (1 − α) · data_w_i

    Parameters
    ----------
    temperature : float
        Softmax temperature.  Lower T sharpens the gate (winner-take-most);
        higher T flattens toward uniform.  Default 1.0.
    gate_alpha : float
        Blending coefficient α ∈ [0, 1].  0 = pure data-volume FedAvg,
        1 = pure gate weights.  Default 0.5.
    lora_B_mode : str
        Heterogeneous-rank handling for LoRA tensors: "zeropadding" (default)
        or "truncate".  (All modes fall back to "fedavg" when ranks are equal.)

    Notes
    -----
    This is a stateless, inference-time approximation of the full FLoRG gate.
    The full paper trains a small MLP gate server-side across rounds; here we
    derive the gate implicitly from the current round's update norms, which
    is equivalent to a one-shot gate with no learnable parameters.
    """

    def __init__(self, args: FedAggregatorArgs | None = None):
        super().__init__(args)
        self._aggregation_method = "florg"
        self._temperature: float = 1.0
        self._gate_alpha:  float = 0.5
        self._lora_B_mode: str  = "zeropadding"
        return

    def set_temperature(self, t: float) -> None:
        assert t > 0, "temperature must be positive"
        self._temperature = t

    def set_gate_alpha(self, alpha: float) -> None:
        assert 0.0 <= alpha <= 1.0, "gate_alpha must be in [0, 1]"
        self._gate_alpha = alpha

    def set_lora_B_mode(self, mode: str) -> None:
        assert mode in {"fedavg", "zeropadding", "truncate"}, f"Unknown mode: {mode}"
        self._lora_B_mode = mode

    def _before_aggregation(self) -> None:
        return

    def _do_aggregation(self) -> None:
        pairs = self._aggregation_data_dict
        if not isinstance(pairs, list) or len(pairs) == 0:
            raise ValueError("[FLoRG] Expected list of (state_dict, data_volume).")

        state_dicts = [sd for sd, _vol in pairs]
        data_vols   = [float(vol) for _sd, vol in pairs]

        console.debug(f"\n[FLoRG] Aggregating {len(state_dicts)} clients, "
                      f"T={self._temperature}, alpha={self._gate_alpha}")
        total = sum(data_vols)
        for i, (_sd, vol) in enumerate(pairs):
            console.debug(f"  Client {i}: {vol} samples ({vol / total * 100:.1f}%)")

        dev = self._device
        sds_on_device = [{k: v.to(dev) for k, v in sd.items()} for sd in state_dicts]

        # Compute gate weights
        final_weights = self._compute_gate_weights(
            sds_on_device, data_vols,
            self._temperature, self._gate_alpha,
        )
        console.debug(f"[FLoRG] Final weights: {[f'{w:.4f}' for w in final_weights]}")

        aggregated = self._aggregate(sds_on_device, final_weights, self._lora_B_mode)

        sample_keys = list(state_dicts[0].keys())
        ordered = OrderedDict((k, aggregated[k]) for k in sample_keys)
        self._aggregated_weight = ordered
        first = next(iter(ordered.keys()))
        console.debug(f"[FLoRG] Aggregated first param mean: {ordered[first].mean():.6f}")

    def _after_aggregation(self) -> None:
        console.debug("[FLoRG] Aggregation completed.")

    # ---------- gate computation ----------
    @staticmethod
    def _compute_gate_weights(
        state_dicts: list[dict],
        data_vols: list[float],
        temperature: float,
        alpha: float,
    ) -> list[float]:
        """Blend data-volume weights with update-magnitude gate scores."""
        from collections import defaultdict

        n = len(state_dicts)

        # Map each lora_B key → corresponding lora_A key (by replacing suffix)
        def a_key_for_b(b_key: str) -> str | None:
            parts = b_key.split(".")
            try:
                idx = parts.index("lora_B")
                parts[idx] = "lora_A"
                candidate = ".".join(parts)
                return candidate if candidate in state_dicts[0] else None
            except ValueError:
                return None

        # g_i = Σ_layers ‖ B_i @ A_i ‖_F
        g = [0.0] * n
        for key in state_dicts[0]:
            if "lora_B" not in key.split("."):
                continue
            ak = a_key_for_b(key)
            if ak is None:
                continue
            for i, sd in enumerate(state_dicts):
                B = sd[key].float()
                A = sd[ak].float()
                # handle rank mismatch by min-truncation
                r = min(B.shape[1], A.shape[0])
                g[i] += float(torch.norm(B[:, :r] @ A[:r, :], p="fro").item())

        # Softmax gate
        g_t = torch.tensor(g, dtype=torch.float64) / max(temperature, 1e-9)
        gate = torch.softmax(g_t, dim=0).tolist()

        # Data-volume weight
        total = sum(data_vols) or 1.0
        data_w = [v / total for v in data_vols]

        # Blend
        final = [alpha * g_i + (1.0 - alpha) * d_i for g_i, d_i in zip(gate, data_w)]
        # Re-normalise to sum = 1
        s = sum(final)
        return [w / s for w in final]

    # ---------- aggregation ----------
    @staticmethod
    def _is_lora(key: str) -> bool:
        parts = key.split(".")
        return "lora_A" in parts or "lora_B" in parts

    @classmethod
    def _aggregate(
        cls,
        state_dicts: list[dict],
        weights: list[float],
        lora_B_mode: str,
    ) -> dict:
        keys = list(state_dicts[0].keys())
        aggregated: dict[str, torch.Tensor] = {}

        for key in keys:
            values = [sd[key] for sd in state_dicts]
            shapes = [v.shape for v in values]
            same_shape = all(s == shapes[0] for s in shapes)

            if same_shape:
                # All same shape — standard weighted average
                stacked = torch.stack(values, dim=0)
                view_shape = (len(weights),) + (1,) * (stacked.dim() - 1)
                w = torch.as_tensor(weights, dtype=stacked.dtype,
                                    device=stacked.device).view(*view_shape)
                aggregated[key] = (stacked * w).sum(dim=0)

            elif cls._is_lora(key) and values[0].dim() == 2:
                # Heterogeneous rank LoRA tensor
                aggregated[key] = cls._aggregate_het(values, weights, lora_B_mode)

            else:
                # Fallback: use first client
                console.warn(f"[FLoRG] Shape mismatch for '{key}' — using first client.")
                aggregated[key] = values[0].clone()

        return aggregated

    @staticmethod
    def _aggregate_het(
        values: list[torch.Tensor],
        weights: list[float],
        mode: str,
    ) -> torch.Tensor:
        shapes = [v.shape for v in values]
        device = values[0].device
        dtype  = values[0].dtype

        if mode == "truncate":
            # truncate along each dim to min size
            min_rows = min(s[0] for s in shapes)
            min_cols = min(s[1] for s in shapes)
            stacked  = torch.stack([v[:min_rows, :min_cols] for v in values], dim=0)
            w = torch.as_tensor(weights, dtype=stacked.dtype, device=device).view(-1, 1, 1)
            return (stacked * w).sum(dim=0)

        # default: zeropadding
        max_rows = max(s[0] for s in shapes)
        max_cols = max(s[1] for s in shapes)
        result = torch.zeros(max_rows, max_cols, dtype=dtype, device=device)
        for v, w in zip(values, weights):
            result[: v.shape[0], : v.shape[1]] += w * v
        return result
