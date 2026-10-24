from __future__ import annotations

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
