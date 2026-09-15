from __future__ import annotations

from typing import Any

import torch.nn as nn
import torch.nn.functional as F

from .. import AbstractNNModel, NNModelArgs, NNModel


class NNModel_SflAlignedClient(NNModel):
    """
    Client-side (front) segment of the aligned SFL model.

    Mirrors the first layer of NNModel_MnistNNBrenden exactly:
        Flatten → fc1(784→200) → ReLU
    Cut-layer output (smashed data) has dimension 200.
    """

    def __init__(self) -> None:
        super().__init__()
        self._flatten = nn.Flatten()
        self._fc1: nn.Linear

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        self._fc1 = nn.Linear(784, 200)
        return self

    def forward(self, x) -> Any:
        x = self._flatten(x)
        return F.relu(self._fc1(x))
