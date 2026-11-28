from __future__ import annotations

import torch.nn as nn

from .nn_model_abc import NNModelArgs, AbstractNNModel

"""
NN Model virtual class
"""

class NNModel(AbstractNNModel):
    def __init__(self):
        super().__init__()

    # override
    def create_args(self) -> NNModelArgs:
        """
        " Create model args
        """
        return NNModelArgs()

    # override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        return super().create_model(args)

    # override
    def forward(self, x):
       return

    def split(self, cut: str) -> tuple[nn.Module, nn.Module]:
        """
        " Split the model into (front, rear) at cut, for split learning.
        " Both parts share this model's sub-modules, so their state_dict keys together
        " equal this model's keys. Override in models that support split learning.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support split learning.")
