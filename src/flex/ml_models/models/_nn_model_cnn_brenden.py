from typing import Any
import torch.nn as nn
import torch.nn.functional as F

from .. import AbstractNNModel, NNModelArgs, NNModel


class NNModel_CNNBrenden(NNModel):
    """
    CNN model adapted from the Brendan McMahan benchmark.

    Architecture (PyTorch equivalent of the Keras version):
        Conv2d(C, 32, 5, padding='same') -> Sigmoid -> MaxPool2d(2)
        Conv2d(32, 64, 5, padding='same') -> ReLU    -> MaxPool2d(2)
        Flatten
        Linear(64 * (H/4) * (W/4), 512) -> ReLU
        Linear(512, num_classes)         -> Softmax

    Supports arbitrary input spatial size (e.g. 28x28 MNIST or 32x32 CIFAR).
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
        in_channels: int = args.in_channels if args.in_channels > 0 else 1
        input_size: int = getattr(args, "input_size", 28) or 28  # spatial H=W

        # After two MaxPool2d(2): size = input_size // 4
        pooled_size = input_size // 4
        fc1_in = 64 * pooled_size * pooled_size

        self._conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32,
                                kernel_size=5, stride=1, padding=2)  # 'same' padding
        self._conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                                kernel_size=5, stride=1, padding=2)  # 'same' padding
        self._pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self._flatten = nn.Flatten()
        self._fc1 = nn.Linear(fc1_in, 512)
        self._fc2 = nn.Linear(512, num_classes)
        return self  # Note: return self

    # override
    def forward(self, x) -> Any:
        x = self._pool(F.sigmoid(self._conv1(x)))
        x = self._pool(F.relu(self._conv2(x)))
        x = self._flatten(x)
        x = F.relu(self._fc1(x))
        x = self._fc2(x)  # raw logits, CrossEntropyLoss handles softmax
        return x


# Backward-compatible alias
NNModel_MnistCNNBrenden = NNModel_CNNBrenden
