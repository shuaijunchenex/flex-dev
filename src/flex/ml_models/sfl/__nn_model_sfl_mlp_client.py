from __future__ import annotations

from typing import Any

import torch.nn as nn

from .. import AbstractNNModel, NNModelArgs, NNModel


class NNModel_SflMlpClient(NNModel):
    def __init__(self) -> None:
        super().__init__()
        self._flatten = nn.Flatten()
        self._activation = nn.ReLU()
        self._fc1: nn.Linear

    # override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        self._fc1 = nn.Linear(784, 200)
        return self

    # override
    def forward(self, x) -> Any:
        x = self._flatten(x)
        x = self._fc1(x)
        return self._activation(x)
