"""Load complete aggregated LoRA factors into an independent evaluation model.

Partial participation can make the aggregate narrower than the largest client
in the population. Resizing the evaluator's factors preserves the aggregate's
entire B @ A without padding, truncation, or another decomposition. The original
server scaling remains fixed: changing factor width is not a scale conversion.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .impl.lora_ms import MSLoRAConv2d, MSLoRALinear, MSEmbedding


def resize_lora_evaluation_factors(
    model: nn.Module, weight: Mapping[str, torch.Tensor]
) -> None:
    """Resize supported A/B Parameters before a strict full-state load.

    ``model`` must be an independent evaluation copy. Input tensors are neither
    modified nor attached to the model, so evaluating cannot mutate the
    aggregate used for client broadcasts. Non-LoRA shapes and all state keys
    are checked before changing any Parameters.
    """
    current = model.state_dict()
    if current.keys() != weight.keys():
        missing = sorted(current.keys() - weight.keys())
        unexpected = sorted(weight.keys() - current.keys())
        raise ValueError(
            f"Aggregate evaluation state keys differ: missing={missing}, "
            f"unexpected={unexpected}"
        )

    changes = []
    factor_keys = set()
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") and not hasattr(module, "lora_B"):
            continue
        if not isinstance(module, (MSLoRAConv2d, MSLoRALinear, MSEmbedding)):
            raise TypeError(
                f"Aggregate-rank evaluation does not support LoRA module "
                f"{name!r} ({type(module).__name__})"
            )
        prefix = f"{name}." if name else ""
        key_a, key_b = prefix + "lora_A", prefix + "lora_B"
        a, b = weight[key_a], weight[key_b]
        if (
            a.ndim != 2 or b.ndim != 2 or a.shape[0] <= 0
            or a.shape[0] != b.shape[1]
            or a.shape[1] != module.lora_A.shape[1]
            or b.shape[0] != module.lora_B.shape[0]
        ):
            raise ValueError(f"Incompatible aggregate LoRA shapes for {name!r}")

        # This CNN implementation stores width r * kernel_size, while its
        # logical r and scaling refer to r. Never replace scaling with alpha/r.
        width_per_rank = module.kernel_size[0] if isinstance(module, MSLoRAConv2d) else 1
        if a.shape[0] % width_per_rank:
            raise ValueError(f"Aggregate width is not a valid CNN rank for {name!r}")
        changes.append((module, a.shape, b.shape, a.shape[0] // width_per_rank))
        factor_keys.update((key_a, key_b))

    for key, tensor in current.items():
        if key not in factor_keys and tensor.shape != weight[key].shape:
            raise ValueError(f"Non-LoRA aggregate shape differs for {key!r}")

    for module, shape_a, shape_b, logical_rank in changes:
        for attr, shape in (("lora_A", shape_a), ("lora_B", shape_b)):
            old = getattr(module, attr)
            if old.shape != shape:
                setattr(module, attr, nn.Parameter(
                    old.new_empty(shape), requires_grad=old.requires_grad
                ))
        module.r = logical_rank
        # The following strict load replaces the base weight with the
        # aggregate's unmerged base, even if an earlier evaluation merged it.
        module.merged = False
