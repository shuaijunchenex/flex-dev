from typing import Any
import torch.nn as nn
import torch.nn.functional as F

from .. import AbstractNNModel, NNModelArgs, NNModel


class NNModel_Cifar10CNNBrenden(NNModel):
    """
    CIFAR-10 CNN from the FedAvg paper (McMahan et al., 2017).

    Input: 3 × 24 × 24 (after RandomCrop)

    Architecture:
        Conv2d( 3, 64, 5) -> ReLU -> MaxPool2d(3, stride=2) -> LRN
        Conv2d(64, 64, 5) -> ReLU -> LRN -> MaxPool2d(3, stride=2)
        Flatten
        Linear(1600, 384)  -> ReLU
        Linear( 384, 192)  -> ReLU
        Linear( 192,  10)  -> logits (no softmax)
    """

    def __init__(self):
        super().__init__()
        self._conv1: nn.Conv2d
        self._conv2: nn.Conv2d
        self._lrn1: nn.LocalResponseNorm
        self._lrn2: nn.LocalResponseNorm
        self._pool: nn.MaxPool2d
        self._flatten: nn.Flatten
        self._fc1: nn.Linear
        self._fc2: nn.Linear
        self._fc3: nn.Linear

    # override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        num_classes: int = getattr(args, "num_classes", 10) or 10

        # Conv layers with "same" padding (kernel=5 → padding=2)
        self._conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self._conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)

        # Local Response Normalization (FedAvg paper uses LRN)
        self._lrn1 = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0)
        self._lrn2 = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0)

        # MaxPool: 3×3, stride 2
        # 24→12→5  (floor((N-3)/2)+1)
        self._pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self._flatten = nn.Flatten()

        # 64 channels × 5 × 5 = 1600
        self._fc1 = nn.Linear(1600, 384)
        self._fc2 = nn.Linear(384, 192)
        self._fc3 = nn.Linear(192, num_classes)
        return self

    # override
    def forward(self, x) -> Any:
        x = self._lrn1(self._pool(F.relu(self._conv1(x))))
        x = self._pool(self._lrn2(F.relu(self._conv2(x))))
        x = self._flatten(x)
        x = F.relu(self._fc1(x))
        x = F.relu(self._fc2(x))
        x = self._fc3(x)  # raw logits (no softmax)
        return x
