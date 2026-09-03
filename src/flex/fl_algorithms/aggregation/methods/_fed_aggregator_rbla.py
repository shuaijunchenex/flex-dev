import torch
from collections import OrderedDict
from collections.abc import Mapping

from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs
from ....ml_utils import console
from ....ml_algorithms.lora.canonicalization import (
    CanonicalizationConfig,
    canonicalize_lora_state_dict,
)


class FedAggregator_RBLA(AbstractFedAggregator):
    """
    RBLA aggregation that is API-compatible with FedAggregator_FedAvg:
      - build_data_list(dict_like) takes values of (state_dict, data_volume)
      - _do_aggregation() aggregates into self._aggregated_weight (OrderedDict)
    """

    def __init__(self, args: FedAggregatorArgs | None = None):
        super().__init__(args)
        self._aggregation_method = "rbla"
        self._lora_suffixes: set[str] = {"lora_A", "lora_B"}
        self._pad_mode: str = str(args.get("pad_mode", "nan")) if args is not None else "nan"
        self.set_pad_mode(self._pad_mode)
        # lora_only: when True, non-LoRA params are copied from first client
        # instead of averaged.  Makes CNN aggregation behave identically to MLP
        # (whose non-LoRA params are frozen & identical → averaging is a no-op).
        # Can be set via YAML: aggregation.lora_only: true
        self._lora_only: bool = bool(args.get("lora_only", False)) if args is not None else False
        canonicalization_args = args.get("canonicalization", {}) if args is not None else {}
        self._canonicalization_config = CanonicalizationConfig.from_mapping(canonicalization_args)
        self._aggregation_round = 0
        self._canonicalization_applied_last_round = False
        self._canonicalization_singular_values: dict[str, torch.Tensor] = {}
        self._canonicalization_diagnostics: dict[str, dict] = {}
        self._canonicalization_summary: dict[str, float] = {}
        self._canonicalization_activation_inputs: dict[str, torch.Tensor] | None = None
        return

    # ---------- Public config ----------
    def set_lora_suffixes(self, lora_suffixes: set[str]) -> None:
        self._lora_suffixes = lora_suffixes

    def set_pad_mode(self, pad_mode: str) -> None:
        assert pad_mode in {"nan", "zero"}, f"Unsupported pad_mode: {pad_mode}"
        self._pad_mode = pad_mode

    def set_lora_only(self, lora_only: bool) -> None:
        """
        When True, non-LoRA keys are NOT averaged across clients — they are copied
        directly from the first state_dict.  This mirrors how MLP works in RBLA:
        all non-LoRA params are either frozen (identical) or absent.
        """
        self._lora_only = lora_only

    def set_canonicalization_activation_inputs(
        self,
        activation_inputs: Mapping[str, torch.Tensor] | None,
    ) -> None:
        """Inject one global calibration activation tensor per complete LoRA A key.

        Keys are exact state-dict keys such as ``layer.lora_A`` or
        ``layer.lora_A.default.weight``. The mapping persists across rounds until
        replaced or cleared with ``None``. Callers are responsible for combining
        any client-specific statistics before injection so every client receives
        prefixes from the same global ordering.
        """

        if activation_inputs is None:
            self._canonicalization_activation_inputs = None
            return
        if not isinstance(activation_inputs, Mapping):
            raise TypeError("activation_inputs must be a mapping keyed by full lora_A key")
        copied: dict[str, torch.Tensor] = {}
        for key, activation in activation_inputs.items():
            if not isinstance(key, str):
                raise TypeError("activation_inputs keys must be strings")
            if not isinstance(activation, torch.Tensor):
                raise TypeError(f"activation_inputs['{key}'] must be a torch.Tensor")
            copied[key] = activation
        self._canonicalization_activation_inputs = copied

    # ---------- FedAvg-style data building ----------
    def build_data_list(self, aggregation_data_dict: dict) -> None:
        """
        Make internal list like: [(state_dict, data_volume), ...]
        (Compatible with FedAggregator_FedAvg expectation)
        """
        self._aggregation_data_list = list(aggregation_data_dict.values())
        return

    def build_data_dict(self, aggregation_data_dict: dict) -> None:
        """If you still pass {'state_dicts': [...], 'weights': [...]}, keep it."""
        self._aggregation_data_dict = aggregation_data_dict

    # ---------- Aggregation lifecycle ----------
    def _before_aggregation(self) -> None:
        # console.debug(f"[RBLA] Starting aggregation with {len(self._aggregation_data_list)} clients...")
        return

    def _do_aggregation(self) -> None:
        """
        Aggregate using RBLA. Accept inputs as:
        1) self._aggregation_data_list = [(state_dict, data_volume), ...]
        2) self._aggregation_data_dict = [(state_dict, data_volume), ...]
        3) self._aggregation_data_dict = {'state_dicts': [...], 'weights': [...]}
        """
        state_dicts, weights = None, None

        # 1) Preferred list-on-list
        if hasattr(self, "_aggregation_data_list") and self._aggregation_data_list:
            pairs = self._aggregation_data_list
            state_dicts = [sd for sd, vol in pairs]
            weights    = [float(vol) for sd, vol in pairs]

        # 2) Your current case: _aggregation_data_dict is actually a list of (sd, vol)
        elif hasattr(self, "_aggregation_data_dict") and isinstance(self._aggregation_data_dict, list):
            pairs = self._aggregation_data_dict
            state_dicts = [sd for sd, vol in pairs]
            weights    = [float(vol) for sd, vol in pairs]

        # 3) Legacy dict form
        elif hasattr(self, "_aggregation_data_dict") and isinstance(self._aggregation_data_dict, dict):
            state_dicts = self._aggregation_data_dict["state_dicts"]
            weights     = self._aggregation_data_dict.get("weights", None)

        else:
            raise ValueError("[RBLA] No aggregation data found. Provide a list of (state_dict, data_volume) "
                            "or dict {'state_dicts': [...], 'weights': [...]}.")

        console.debug(f"\n[RBLA] Aggregating {len(state_dicts)} clients...")
        total_data_vol = sum(vol for _, vol in self._aggregation_data_dict)
        for i, (_, vol) in enumerate(self._aggregation_data_dict):
            console.debug(f"  Client {i}: {vol} samples ({vol / total_data_vol * 100:.1f}%)")

        # move to device
        dev = self._device
        sds_on_device = [{k: v.to(dev) for k, v in sd.items()} for sd in state_dicts]

        aggregated = self.aggregate_state_dicts(
            sds_on_device,
            weights=weights,
            lora_suffixes=self._lora_suffixes,
            pad_mode=self._pad_mode,
            lora_only=self._lora_only,
        )

        # keep key order like the first state_dict
        from collections import OrderedDict
        sample_keys = list(state_dicts[0].keys())
        ordered = OrderedDict((k, aggregated[k]) for k in sample_keys)
        self._aggregated_weight = ordered

        first_param_name = next(iter(ordered.keys()))
        console.debug(f"[RBLA] Aggregated first param mean: {ordered[first_param_name].mean():.6f}")


    def _after_aggregation(self) -> None:
        self._canonicalization_applied_last_round = False
        self._canonicalization_singular_values = {}
        self._canonicalization_diagnostics = {}
        self._canonicalization_summary = {}
        if self._canonicalization_config.should_run(self._aggregation_round):
            result = canonicalize_lora_state_dict(
                self._aggregated_weight,
                compute_dtype=self._canonicalization_config.compute_dtype,
                deterministic_sign=self._canonicalization_config.deterministic_sign,
                svd_fallback=self._canonicalization_config.svd_fallback,
                eps=self._canonicalization_config.eps,
                ordering=self._canonicalization_config.ordering,
                activation_inputs=self._canonicalization_activation_inputs,
                activation_chunk_size=self._canonicalization_config.activation_chunk_size,
                activation_fallback=self._canonicalization_config.activation_fallback,
                overcomplete_policy=self._canonicalization_config.overcomplete_policy,
            )
            self._aggregated_weight = result.state_dict
            self._canonicalization_singular_values = {
                key: values.detach().cpu()
                for key, values in result.singular_values.items()
            }
            self._canonicalization_diagnostics = result.diagnostics
            if result.diagnostics:
                layer_diagnostics = list(result.diagnostics.values())
                self._canonicalization_summary = {
                    "layer_count": float(len(layer_diagnostics)),
                    "mean_effective_rank": sum(
                        float(item["effective_rank"]) for item in layer_diagnostics
                    ) / len(layer_diagnostics),
                    "maximum_singular_value": max(
                        float(item["maximum_singular_value"]) for item in layer_diagnostics
                    ),
                    "minimum_singular_value": min(
                        float(item["minimum_singular_value"]) for item in layer_diagnostics
                    ),
                    "maximum_core_reconstruction_error": max(
                        float(item["core_reconstruction_error"]) for item in layer_diagnostics
                    ),
                    "maximum_factor_balance_error": max(
                        float(item["factor_balance_error"]) for item in layer_diagnostics
                    ),
                }
                if self._canonicalization_config.ordering == "activation_aware":
                    applied_count = sum(
                        item.get("ordering_applied") == "activation_aware"
                        for item in layer_diagnostics
                    )
                    self._canonicalization_summary.update(
                        {
                            "activation_aware_layer_count": float(applied_count),
                            "activation_fallback_layer_count": float(
                                len(layer_diagnostics) - applied_count
                            ),
                        }
                    )
            self._canonicalization_applied_last_round = True
            if self._canonicalization_config.log_diagnostics:
                summary = self._canonicalization_summary
                console.debug(
                    f"[RBLA canonicalization] round={self._aggregation_round}, "
                    f"layers={int(summary.get('layer_count', 0))}, "
                    f"mean_effective_rank={summary.get('mean_effective_rank', 0.0):.4f}, "
                    f"max_core_error={summary.get('maximum_core_reconstruction_error', 0.0):.3e}, "
                    f"max_balance_error={summary.get('maximum_factor_balance_error', 0.0):.3e}"
                )
        self._aggregation_round += 1
        console.debug("[RBLA] Aggregation completed.")

    @property
    def canonicalization_applied_last_round(self) -> bool:
        return self._canonicalization_applied_last_round

    @property
    def canonicalization_diagnostics(self) -> dict[str, dict]:
        return self._canonicalization_diagnostics

    @property
    def canonicalization_singular_values(self) -> dict[str, torch.Tensor]:
        return self._canonicalization_singular_values

    @property
    def canonicalization_summary(self) -> dict[str, float]:
        return self._canonicalization_summary

    @property
    def canonicalization_log_diagnostics(self) -> bool:
        return self._canonicalization_config.log_diagnostics

    # ---------- Core RBLA ops ----------
    @staticmethod
    def get_lora_type(key: str, lora_suffixes: set[str]) -> str | None:
        """
        Check if key contains a LoRA suffix component (e.g. 'lora_A').
        Returns the suffix if found (e.g. 'lora_A'), else None.
        Compatible with HuggingFace PEFT keys like 'base_model.model...lora_A.default.weight'.
        """
        parts = key.split(".")
        for suffix in lora_suffixes:
            if suffix in parts:
                return suffix
        return None

    @staticmethod
    def pad_tensors_to_max_shape(tensors: list[torch.Tensor], pad_mode: str = "nan") -> torch.Tensor:
        """
        Pad 2D tensors to a common shape; return stacked 3D tensor: (N, max_rows, max_cols).
        """
        assert pad_mode in {"nan", "zero"}, f"Unsupported pad_mode: {pad_mode}"
        if len(tensors) == 0:
            raise ValueError("pad_tensors_to_max_shape: empty tensor list")

        # Ensure 2D for LoRA matrices
        for t in tensors:
            if t.dim() != 2:
                raise ValueError(f"LoRA tensor must be 2D, got {t.dim()}D for shape {tuple(t.shape)}")

        max_rows = max(t.shape[0] for t in tensors)
        max_cols = max(t.shape[1] for t in tensors)
        device = tensors[0].device
        dtype = tensors[0].dtype

        fill_val = float("nan") if pad_mode == "nan" else 0.0
        padded_list = []
        for t in tensors:
            pad = torch.full((max_rows, max_cols), fill_val, dtype=dtype, device=device)
            pad[: t.shape[0], : t.shape[1]] = t
            padded_list.append(pad)
        return torch.stack(padded_list, dim=0)

    @staticmethod
    def aggregate_lora_tensors(
        tensors: list[torch.Tensor],
        weights: list[float],
        pad_mode: str = "nan",
    ) -> torch.Tensor:
        """
        Weighted average with padding-aware handling for LoRA matrices.
        """
        if len(tensors) == 0:
            raise ValueError("aggregate_lora_tensors: empty tensor list")

        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=tensors[0].device).view(-1, 1, 1)
        padded = FedAggregator_RBLA.pad_tensors_to_max_shape(tensors, pad_mode=pad_mode)

        if pad_mode == "nan":
            valid_mask = ~torch.isnan(padded)
            padded = torch.nan_to_num(padded, nan=0.0)
            weighted_sum = (padded * weights_tensor).sum(dim=0)
            weight_mask = valid_mask * weights_tensor
            total_weight = weight_mask.sum(dim=0)
            total_weight[total_weight == 0] = 1.0  # avoid div-by-zero
            return weighted_sum / total_weight
        else:  # zero padding
            weighted_sum = (padded * weights_tensor).sum(dim=0)
            total_weight = sum(weights)
            return weighted_sum / total_weight

    @staticmethod
    def aggregate_state_dicts(
        state_dicts: list[dict],
        weights: list[float] | None = None,
        lora_suffixes: set[str] = {"lora_A", "lora_B"},
        pad_mode: str = "nan",
        lora_only: bool = False,
    ) -> dict:
        """
        Aggregate multiple state_dicts with LoRA-aware averaging.

        Args:
            lora_only: If True, non-LoRA keys are copied directly from the first
                state_dict (no averaging).  This mirrors MLP behaviour (whose
                non-LoRA params are frozen & identical, so averaging is a no-op).
                Also fixes the integer-tensor bug (e.g. BatchNorm
                ``num_batches_tracked`` becoming 0 after float-weight averaging).
        """
        if len(state_dicts) == 0:
            raise ValueError("aggregate_state_dicts: empty state_dicts")

        if weights is None:
            weights = [1.0] * len(state_dicts)

        # normalize weights to sum=1 for stability
        tw = float(sum(weights))
        weights = [w / tw for w in weights] if tw > 0 else [1.0 / len(weights)] * len(weights)

        keys = list(state_dicts[0].keys())
        aggregated: dict[str, torch.Tensor] = {}
        first_sd = state_dicts[0]  # reference for lora_only / integer fallback

        for key in keys:
            values = [sd[key] for sd in state_dicts]
            lora_type = FedAggregator_RBLA.get_lora_type(key, lora_suffixes)

            if lora_type is not None:
                # ── LoRA A/B: NaN-padded weighted average (RBLA core algorithm) ──
                aggregated[key] = FedAggregator_RBLA.aggregate_lora_tensors(
                    values, weights, pad_mode=pad_mode
                )
            elif lora_only:
                # ── Non-LoRA, lora_only mode: copy from first client (no averaging) ──
                aggregated[key] = first_sd[key].clone()
            elif not torch.is_floating_point(values[0]):
                # ── Non-LoRA, integer tensor (e.g. num_batches_tracked): copy from
                #     first client to avoid float→int truncation to 0 ──
                aggregated[key] = first_sd[key].clone()
            else:
                # ── Non-LoRA, float tensor: standard weighted average (FedAvg) ──
                stacked = torch.stack(values, dim=0)  # (N, ...)
                # weights reshape: (N, 1, 1, ..., 1)
                view_shape = (len(weights),) + (1,) * (stacked.dim() - 1)
                weight_tensor = torch.as_tensor(
                    weights, dtype=stacked.dtype, device=stacked.device
                ).view(*view_shape)
                weighted_sum = (stacked * weight_tensor).sum(dim=0)
                aggregated[key] = weighted_sum  # weights已归一化

        return aggregated

    @staticmethod
    def broadcast_lora_state_dict(global_sd: dict, local_sd: dict, lora_suffixes: set[str] = {"lora_A", "lora_B"}) -> dict:
        """
        Slice/pad global LoRA matrices back to each client's local tensor shape,
        and copy non-LoRA tensors directly.

        When compact canonicalization is enabled, these leading rows/columns are
        global singular-value prefixes. They must not be interpreted as the
        pre-canonicalization client slot identities.

        Works for both MLP (``lora_A=[r, in]``, ``lora_B=[out, r]``) and CNN
        (``lora_A=[r·k, C_in·k]``, ``lora_B=[C_out·k, r·k]``) because the
        "rank" dimension is always dim 0 for A and dim 1 for B.

        When the global rank is lower than the local rank (e.g. after SVD-based
        aggregation truncates thin layers), the result is zero-padded to match
        the local tensor shape so that ``load_state_dict(strict=True)`` succeeds.
        The implementation uses shape-based fitting rather than hard-coding an
        A/B rank dimension.
        """
        def fit_tensor_to_local_shape(global_tensor: torch.Tensor, local_tensor: torch.Tensor) -> torch.Tensor:
            if tuple(global_tensor.shape) == tuple(local_tensor.shape):
                return global_tensor.clone()

            fitted = local_tensor.new_zeros(local_tensor.shape)
            common_slices = tuple(
                slice(0, min(g_dim, l_dim))
                for g_dim, l_dim in zip(global_tensor.shape, local_tensor.shape)
            )
            fitted[common_slices] = global_tensor[common_slices].to(
                dtype=local_tensor.dtype,
                device=local_tensor.device,
            )
            return fitted

        new_local_sd = {}
        for key, local_tensor in local_sd.items():
            global_tensor = global_sd[key]
            lora_type = FedAggregator_RBLA.get_lora_type(key, lora_suffixes)

            if lora_type is None:
                new_local_sd[key] = global_tensor.clone()
            elif lora_type in lora_suffixes:
                new_local_sd[key] = fit_tensor_to_local_shape(global_tensor, local_tensor)
            else:
                raise ValueError(f"Unrecognized LoRA suffix: {lora_type}")
        return new_local_sd
