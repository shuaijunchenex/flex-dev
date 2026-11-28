"""Payloads for FedEx-LoRA aggregation and broadcast."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class FedExLayerResult:
    mean_a: Tensor
    mean_b: Tensor
    residual: Tensor


@dataclass(frozen=True)
class FedExBroadcast:
    version: int
    state: Mapping[str, Tensor]
