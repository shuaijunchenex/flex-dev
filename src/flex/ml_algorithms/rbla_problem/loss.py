"""Strong-A proximal anchoring used only by the RBLA problem strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass(frozen=True)
class StrongAConfig:
    """Configuration for normalized, layer-balanced A-side anchoring."""

    lambda_a: float = 0.1
    eps: float = 1e-8

    @classmethod
    def from_dict(cls, config: dict | None) -> "StrongAConfig":
        data = config or {}
        return cls(
            lambda_a=float(data.get("lambda_a", cls.lambda_a)),
            eps=float(data.get("eps", cls.eps)),
        )


class StrongAProximalLoss(nn.Module):
    r"""Normalized proximal loss averaged first over slots, then layers.

    .. math::

        L_A = \frac{1}{L}\sum_l\frac{1}{r_l}\sum_s
              \frac{\|a_{i,l,s}-a_{g,l,s}\|_2^2}
                   {\|a_{g,l,s}\|_2^2+\epsilon}.

    Layer averaging keeps ``lambda_a`` comparable between small MNIST models
    and deeper models.  This module deliberately does not reuse SARA so the
    existing SARA implementation and experiments remain unchanged.
    """

    def __init__(self, config: StrongAConfig | None = None):
        super().__init__()
        self.config = config or StrongAConfig()

    @staticmethod
    def is_lora_a_key(key: str) -> bool:
        return "lora_A" in key.split(".")

    def forward(
        self,
        named_parameters: Dict[str, nn.Parameter],
        anchors: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        layer_losses = []
        for key, parameter in named_parameters.items():
            if not self.is_lora_a_key(key) or key not in anchors:
                continue

            anchor = anchors[key].to(device=parameter.device, dtype=parameter.dtype)
            rows = min(int(parameter.shape[0]), int(anchor.shape[0]))
            if rows <= 0 or parameter.dim() != 2 or anchor.dim() != 2:
                continue

            local_rows = parameter[:rows]
            anchor_rows = anchor[:rows]
            numerator = (local_rows - anchor_rows).pow(2).sum(dim=1)
            denominator = anchor_rows.pow(2).sum(dim=1) + self.config.eps
            layer_losses.append((numerator / denominator).mean())

        if not layer_losses:
            device = next(iter(named_parameters.values())).device if named_parameters else torch.device("cpu")
            return torch.zeros((), device=device)
        return torch.stack(layer_losses).mean()
