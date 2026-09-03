from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class LoRALayerScaleProfile:
    """LoRA metadata required to preserve an effective ``scaling * B @ A``."""

    prefix: str
    key_A: str
    key_B: str
    rank: int
    alpha: float
    scaling: float


LoRAScaleProfile = dict[str, LoRALayerScaleProfile]


def _parameter_key(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _as_positive_finite_float(value, *, name: str, prefix: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError(
                f"LoRA layer '{prefix}' has non-scalar {name}: shape={tuple(value.shape)}"
            )
        value = value.detach().item()
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"LoRA layer '{prefix}' has unsupported {name}={value!r}; "
            "rank-scale alignment requires a scalar value"
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(
            f"LoRA layer '{prefix}' must have a positive finite {name}, got {result}"
        )
    return result


def build_lora_scale_profile(model: nn.Module) -> LoRAScaleProfile:
    """Build an exact per-layer LoRA scale profile from a model.

    The current FLEX LoRA modules expose ``lora_A``, ``lora_B``, ``r``,
    ``lora_alpha`` and ``scaling`` directly.  Reading those attributes is
    important for convolutional LoRA, whose tensor dimensions contain kernel
    factors and therefore must not be treated as the logical rank.
    """

    state_keys = set(model.state_dict().keys())
    profile: LoRAScaleProfile = {}

    for prefix, module in model.named_modules():
        has_A = hasattr(module, "lora_A")
        has_B = hasattr(module, "lora_B")
        if not has_A and not has_B:
            continue
        if has_A != has_B:
            raise ValueError(
                f"LoRA layer '{prefix}' must expose both lora_A and lora_B"
            )

        lora_A = getattr(module, "lora_A")
        lora_B = getattr(module, "lora_B")
        if not isinstance(lora_A, torch.Tensor) or not isinstance(lora_B, torch.Tensor):
            raise TypeError(
                f"LoRA layer '{prefix}' uses container-valued factors; "
                "rbla_plus_rank_scale currently requires tensor-valued lora_A/lora_B"
            )

        key_A = _parameter_key(prefix, "lora_A")
        key_B = _parameter_key(prefix, "lora_B")
        missing = [key for key in (key_A, key_B) if key not in state_keys]
        if missing:
            raise KeyError(
                f"LoRA layer '{prefix}' factors are missing from state_dict: {missing}"
            )

        rank_value = getattr(module, "r", None)
        if rank_value is None:
            raise AttributeError(f"LoRA layer '{prefix}' does not expose logical rank 'r'")
        rank = int(rank_value)
        if rank <= 0:
            raise ValueError(f"LoRA layer '{prefix}' must have rank > 0, got {rank}")

        scaling = _as_positive_finite_float(
            getattr(module, "scaling", None),
            name="scaling",
            prefix=prefix,
        )
        alpha_value = getattr(module, "lora_alpha", scaling * rank)
        alpha = _as_positive_finite_float(
            alpha_value,
            name="lora_alpha",
            prefix=prefix,
        )

        profile[prefix] = LoRALayerScaleProfile(
            prefix=prefix,
            key_A=key_A,
            key_B=key_B,
            rank=rank,
            alpha=alpha,
            scaling=scaling,
        )

    if not profile:
        raise ValueError("No tensor-valued LoRA layers were found in the model")
    return profile


def validate_compatible_scale_profiles(
    source: Mapping[str, LoRALayerScaleProfile],
    target: Mapping[str, LoRALayerScaleProfile],
) -> None:
    """Require source and target to describe the same named LoRA layers."""

    source_prefixes = set(source)
    target_prefixes = set(target)
    if source_prefixes != target_prefixes:
        missing_from_target = sorted(source_prefixes - target_prefixes)
        missing_from_source = sorted(target_prefixes - source_prefixes)
        raise ValueError(
            "Incompatible LoRA scale profiles: "
            f"missing_from_target={missing_from_target}, "
            f"missing_from_source={missing_from_source}"
        )

    for prefix in sorted(source_prefixes):
        source_layer = source[prefix]
        target_layer = target[prefix]
        if source_layer.key_A != target_layer.key_A or source_layer.key_B != target_layer.key_B:
            raise ValueError(
                f"LoRA layer '{prefix}' uses incompatible state-dict keys: "
                f"source=({source_layer.key_A}, {source_layer.key_B}), "
                f"target=({target_layer.key_A}, {target_layer.key_B})"
            )


def align_lora_state_dict_scale(
    state_dict: Mapping[str, torch.Tensor],
    source_profile: Mapping[str, LoRALayerScaleProfile],
    target_profile: Mapping[str, LoRALayerScaleProfile],
) -> OrderedDict[str, torch.Tensor]:
    """Express LoRA factors from ``source`` in ``target`` scale coordinates.

    For each layer, both factors are multiplied by
    ``sqrt(source.scaling / target.scaling)``.  Consequently,

    ``target.scaling * B_aligned @ A_aligned``
    equals ``source.scaling * B_source @ A_source``.

    The input mapping and its tensors are never modified in-place. Non-LoRA
    values are intentionally shared because the aggregation and broadcast
    consumers treat them as read-only.
    """

    validate_compatible_scale_profiles(source_profile, target_profile)
    aligned = OrderedDict(state_dict.items())

    for prefix, source_layer in source_profile.items():
        target_layer = target_profile[prefix]
        factor = math.sqrt(source_layer.scaling / target_layer.scaling)
        for key in (source_layer.key_A, source_layer.key_B):
            if key not in state_dict:
                raise KeyError(f"LoRA tensor '{key}' is missing from the state dict")
            tensor = state_dict[key]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"LoRA state '{key}' must be a torch.Tensor")
            aligned[key] = tensor * tensor.new_tensor(factor)

    return aligned
