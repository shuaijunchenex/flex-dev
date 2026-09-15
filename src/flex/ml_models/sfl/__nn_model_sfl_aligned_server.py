from __future__ import annotations

from typing import Any

import torch.nn as nn
import torch.nn.functional as F

from .. import AbstractNNModel, NNModelArgs, NNModel


class NNModel_SflAlignedServer(NNModel):
    """
    Server-side (rear) segment of the aligned SFL model.

    Mirrors the second and third layers of NNModel_MnistNNBrenden exactly:
        fc2(200→200) → ReLU → fc3(200→10) → Softmax
    Input dimension must match client cut-layer output (200).
    """

    def __init__(self) -> None:
        super().__init__()
        self._fc2: nn.Linear
        self._fc3: nn.Linear

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        self._fc2 = nn.Linear(200, 200)
        self._fc3 = nn.Linear(200, 10)
        return self

    def forward(self, x) -> Any:
        x = F.relu(self._fc2(x))
        x = self._fc3(x)
        return x  # raw logits, CrossEntropyLoss handles softmax
