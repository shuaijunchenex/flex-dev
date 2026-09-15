from ._nn_model_cnn_brenden import NNModel_CNNBrenden, NNModel_MnistCNNBrenden
from typing import Any
import torch.nn as nn
import torch.nn.functional as F

from .. import AbstractNNModel, NNModelArgs, NNModel


class NNModel_MnistCNNBrendenReLU(NNModel):
    """
    CNN for MNIST with ReLU activations (1×28×28 → 10).

    Architecture:
        Conv2d( 1, 32, 5) -> ReLU -> MaxPool2d(2)
        Conv2d(32, 64, 5) -> ReLU -> MaxPool2d(2)
        Flatten
        Linear(64*7*7, 512) -> ReLU
        Linear(512, 10)     -> Softmax
    """

    def __init__(self):
        super().__init__()
        self._conv1: nn.Conv2d
        self._conv2: nn.Conv2d
        self._pool: nn.MaxPool2d
        self._flatten: nn.Flatten
        self._fc1: nn.Linear
        self._fc2: nn.Linear

    # override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        num_classes: int = getattr(args, "num_classes", 10) or 10

        # 28 -> 14 -> 7, 64 channels: 64 * 7 * 7 = 3136
        self._conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self._conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self._pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self._flatten = nn.Flatten()
        self._fc1 = nn.Linear(64 * 7 * 7, 512)
        self._fc2 = nn.Linear(512, num_classes)
        return self

    # override
    def forward(self, x) -> Any:
        x = self._pool(F.relu(self._conv1(x)))
        x = self._pool(F.relu(self._conv2(x)))
        x = self._flatten(x)
        x = F.relu(self._fc1(x))
        x = self._fc2(x)  # raw logits, CrossEntropyLoss handles softmax
        return x


__all__ = ["NNModel_CNNBrenden", "NNModel_MnistCNNBrenden", "NNModel_MnistCNNBrendenReLU"]
