from typing import Any
import torch.nn as nn
import torch.nn.functional as F

from .. import AbstractNNModel, NNModelArgs, NNModel


class _MnistNNBrendenFront(nn.Module):
    """
    " Front part of the split: fc1 -> ReLU
    """

    def __init__(self, fc1: nn.Linear):
        super().__init__()
        self._fc1 = fc1

    def forward(self, x) -> Any:
        return F.relu(self._fc1(x.view(x.size(0), -1)))


class _MnistNNBrendenRear(nn.Module):
    """
    " Rear part of the split: fc2 -> ReLU -> fc3
    """

    def __init__(self, fc2: nn.Linear, fc3: nn.Linear):
        super().__init__()
        self._fc2 = fc2
        self._fc3 = fc3

    def forward(self, x) -> Any:
        return self._fc3(F.relu(self._fc2(x)))


class NNModel_MnistNNBrenden(NNModel):

    """
    " Private class for Mnist2NNBrenden model implementation
    """

    def __init__(self):
        super().__init__()

    #override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        
        self._relu = nn.ReLU()
        self._fc1 = nn.Linear(784, 200)
        self._fc2 = nn.Linear(200, 200)
        self._fc3 = nn.Linear(200, 10)
        return self         #Note: return self

    #override
    def forward(self, x)-> Any:
        x = x.view(x.size(0), -1)  # flatten (B, C, H, W) → (B, C*H*W)
        x = F.relu(self._fc1(x))
        x = F.relu(self._fc2(x))
        x = self._fc3(x)  # raw logits, CrossEntropyLoss handles softmax
        return x

    #override
    def split(self, cut: str) -> tuple[nn.Module, nn.Module]:
        if cut != "fc1":
            raise ValueError(f"Unknown cut '{cut}', expected 'fc1'.")
        return _MnistNNBrendenFront(self._fc1), _MnistNNBrendenRear(self._fc2, self._fc3)
