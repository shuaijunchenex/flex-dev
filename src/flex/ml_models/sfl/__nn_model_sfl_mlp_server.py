from __future__ import annotations

from typing import Any

import torch.nn as nn
import torch.nn.functional as F

from .. import AbstractNNModel, NNModelArgs, NNModel


class NNModel_SflMlpServer(NNModel):
    def __init__(self) -> None:
        super().__init__()
        self._activation = nn.ReLU()
        self._fc2: nn.Linear
        self._fc3: nn.Linear
        self._softmax_dim = 1

    # override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        self._fc2 = nn.Linear(200, 200)
        self._fc3 = nn.Linear(200, 10)
        return self

    # override
    def forward(self, x) -> Any:
        x = self._activation(self._fc2(x))
        x = self._fc3(x)
        return x  # raw logits, CrossEntropyLoss handles softmax
