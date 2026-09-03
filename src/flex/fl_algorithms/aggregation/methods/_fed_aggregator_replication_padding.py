from __future__ import annotations

from collections import OrderedDict

import torch

from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs
from ....ml_utils import console


class FedAggregator_ReplicationPadding(AbstractFedAggregator):
    """Replication Padding for heterogeneous-rank LoRA aggregation.

    For every LoRA factor, clients with the maximum rank are first averaged to
    form a donor factor.  Missing rows of ``lora_A`` and missing columns of
    ``lora_B`` are copied from that donor before the regular weighted average.
    The server therefore retains the high-rank tail instead of diluting it with
    padded zeros.  Non-LoRA floating-point tensors use the regular weighted
    average, while non-floating tensors are copied from the first client.

    The implementation supports the repository's Linear, Embedding, and
    Conv2d LoRA layouts: the rank axis is dimension 0 for ``lora_A`` and
    dimension 1 for ``lora_B``.
    """

    _LORA_A = "lora_A"
    _LORA_B = "lora_B"
    _WEIGHTING_MODES = {"data_volume", "client_count"}

    def __init__(self, args: FedAggregatorArgs | None = None):
        super().__init__(args)
        self._aggregation_method = "replication_padding"
        self._weighting_mode = str(
            self.args.get("weighting_mode", "data_volume")
        ).lower()
        if self._weighting_mode not in self._WEIGHTING_MODES:
            raise ValueError(
                "[ReplicationPadding] weighting_mode must be one of "
                f"{sorted(self._WEIGHTING_MODES)}, got "
                f"'{self._weighting_mode}'."
            )

    def _before_aggregation(self) -> None:
        return

    def _do_aggregation(self) -> None:
        pairs = self._aggregation_data_dict
        if not isinstance(pairs, list) or not pairs:
            raise ValueError(
                "[ReplicationPadding] Expected a non-empty list of "
                "(state_dict, data_volume) pairs."
            )

        state_dicts = [state_dict for state_dict, _ in pairs]
        if self._weighting_mode == "client_count":
            weights = [1.0] * len(pairs)
        else:
            weights = [float(volume) for _, volume in pairs]

        console.debug(
            f"\n[ReplicationPadding] Aggregating {len(state_dicts)} clients "
            f"with weighting_mode={self._weighting_mode}..."
        )

        state_dicts_on_device = [
            {key: value.to(self._device) for key, value in state_dict.items()}
            for state_dict in state_dicts
        ]
        aggregated = self.aggregate_state_dicts(state_dicts_on_device, weights)
        first_keys = list(state_dicts[0].keys())
        self._aggregated_weight = OrderedDict(
            (key, aggregated[key]) for key in first_keys
        )

    def _after_aggregation(self) -> None:
        console.debug("[ReplicationPadding] Aggregation completed.")

    @classmethod
    def _lora_type(cls, key: str) -> str | None:
        parts = key.split(".")
        if cls._LORA_A in parts:
            return cls._LORA_A
        if cls._LORA_B in parts:
            return cls._LORA_B
        return None

    @classmethod
    def _paired_lora_key(cls, key: str) -> str | None:
        parts = key.split(".")
        if cls._LORA_A not in parts:
            return None
        parts[parts.index(cls._LORA_A)] = cls._LORA_B
        return ".".join(parts)

    @staticmethod
    def _normalise_weights(weights: list[float]) -> list[float]:
        if not weights:
            raise ValueError("[ReplicationPadding] Cannot normalise empty weights.")
        if any(weight < 0 for weight in weights):
            raise ValueError("[ReplicationPadding] Client weights must be non-negative.")
        total = float(sum(weights))
        if total == 0:
            return [1.0 / len(weights)] * len(weights)
        return [float(weight) / total for weight in weights]

    @classmethod
    def _validate_state_dicts(cls, state_dicts: list[dict]) -> None:
        reference_keys = list(state_dicts[0].keys())
        reference_key_set = set(reference_keys)
        for index, state_dict in enumerate(state_dicts[1:], start=1):
            if set(state_dict.keys()) != reference_key_set:
                raise ValueError(
                    "[ReplicationPadding] All clients must provide identical "
                    f"state-dict keys; client {index} differs from client 0."
                )

        for a_key in reference_keys:
            b_key = cls._paired_lora_key(a_key)
            if b_key is None or b_key not in reference_key_set:
                continue
            for index, state_dict in enumerate(state_dicts):
                a_value = state_dict[a_key]
                b_value = state_dict[b_key]
                if a_value.ndim != 2 or b_value.ndim != 2:
                    raise ValueError(
                        "[ReplicationPadding] LoRA factors must be 2-D; "
                        f"client {index} has {a_key}{tuple(a_value.shape)} and "
                        f"{b_key}{tuple(b_value.shape)}."
                    )
                if a_value.shape[0] != b_value.shape[1]:
                    raise ValueError(
                        "[ReplicationPadding] Paired LoRA rank dimensions do "
                        f"not match for client {index}: {a_key}{tuple(a_value.shape)} "
                        f"vs {b_key}{tuple(b_value.shape)}."
                    )

    @classmethod
    def replicate_and_average_lora_tensors(
        cls,
        tensors: list[torch.Tensor],
        weights: list[float],
        *,
        rank_axis: int,
    ) -> torch.Tensor:
        """Replicate the maximum-rank donor tail, then take a weighted mean."""
        if not tensors:
            raise ValueError("[ReplicationPadding] Empty LoRA tensor list.")
        if len(tensors) != len(weights):
            raise ValueError(
                "[ReplicationPadding] Tensor and weight counts must match."
            )
        if rank_axis not in (0, 1):
            raise ValueError(
                f"[ReplicationPadding] rank_axis must be 0 or 1, got {rank_axis}."
            )
        if any(tensor.ndim != 2 for tensor in tensors):
            shapes = [tuple(tensor.shape) for tensor in tensors]
            raise ValueError(
                f"[ReplicationPadding] LoRA factors must be 2-D, got {shapes}."
            )

        non_rank_axis = 1 - rank_axis
        non_rank_size = tensors[0].shape[non_rank_axis]
        if any(
            tensor.shape[non_rank_axis] != non_rank_size for tensor in tensors
        ):
            shapes = [tuple(tensor.shape) for tensor in tensors]
            raise ValueError(
                "[ReplicationPadding] Non-rank LoRA dimensions must match, "
                f"got {shapes}."
            )

        normalised_weights = cls._normalise_weights(weights)
        max_rank = max(tensor.shape[rank_axis] for tensor in tensors)
        donor_indices = [
            index
            for index, tensor in enumerate(tensors)
            if tensor.shape[rank_axis] == max_rank
        ]
        donor_weights = cls._normalise_weights(
            [normalised_weights[index] for index in donor_indices]
        )

        donor = torch.zeros_like(tensors[donor_indices[0]])
        for donor_weight, donor_index in zip(donor_weights, donor_indices):
            donor.add_(tensors[donor_index], alpha=donor_weight)

        replicated: list[torch.Tensor] = []
        for tensor in tensors:
            local_rank = tensor.shape[rank_axis]
            if local_rank == max_rank:
                replicated.append(tensor)
                continue
            padded = donor.clone()
            if rank_axis == 0:
                padded[:local_rank, :] = tensor
            else:
                padded[:, :local_rank] = tensor
            replicated.append(padded)

        stacked = torch.stack(replicated, dim=0)
        weight_shape = (len(normalised_weights),) + (1,) * (stacked.ndim - 1)
        weight_tensor = torch.as_tensor(
            normalised_weights,
            dtype=stacked.dtype,
            device=stacked.device,
        ).view(*weight_shape)
        return (stacked * weight_tensor).sum(dim=0)

    @classmethod
    def aggregate_state_dicts(
        cls,
        state_dicts: list[dict],
        weights: list[float] | None = None,
    ) -> dict:
        if not state_dicts:
            raise ValueError("[ReplicationPadding] Empty state-dict list.")
        if weights is None:
            weights = [1.0] * len(state_dicts)
        if len(weights) != len(state_dicts):
            raise ValueError(
                "[ReplicationPadding] State-dict and weight counts must match."
            )

        cls._validate_state_dicts(state_dicts)
        normalised_weights = cls._normalise_weights(weights)
        aggregated: dict[str, torch.Tensor] = {}

        for key in state_dicts[0].keys():
            values = [state_dict[key] for state_dict in state_dicts]
            lora_type = cls._lora_type(key)
            if lora_type is not None:
                aggregated[key] = cls.replicate_and_average_lora_tensors(
                    values,
                    normalised_weights,
                    rank_axis=0 if lora_type == cls._LORA_A else 1,
                )
            elif not torch.is_floating_point(values[0]):
                aggregated[key] = values[0].clone()
            else:
                stacked = torch.stack(values, dim=0)
                weight_shape = (len(normalised_weights),) + (1,) * (
                    stacked.ndim - 1
                )
                weight_tensor = torch.as_tensor(
                    normalised_weights,
                    dtype=stacked.dtype,
                    device=stacked.device,
                ).view(*weight_shape)
                aggregated[key] = (stacked * weight_tensor).sum(dim=0)

        return aggregated
