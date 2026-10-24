from typing import Any
import torch.nn as nn

from .. import AbstractNNModel, NNModelArgs, NNModel

class NNModel_CapstoneMLP(NNModel):

    """
    " Private class for CapstoneMLP model implementation
    """

    def __init__(self):
        super().__init__()
        

    #override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        
        self._flatten = nn.Flatten()
        self._fc1 = nn.Linear(784, 128)
        self._relu1 = nn.ReLU()
        self._fc2 = nn.Linear(128, 64)
        self._relu2 = nn.ReLU()
        self._fc3 = nn.Linear(64, 10)
        return self         #Note: return self

    #override
    def forward(self, x):
        x = self._flatten(x)
        x = self._relu1(self._fc1(x))
        x = self._relu2(self._fc2(x))
        x = self._fc3(x)
        return x
