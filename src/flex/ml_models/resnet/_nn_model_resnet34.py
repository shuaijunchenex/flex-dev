import torch.nn as nn
from torchvision import models
from ..nn_model import NNModel
from ..nn_model_abc import AbstractNNModel
from ..nn_model_args import NNModelArgs


class NNModel_ResNet34(NNModel):
    """
    ResNet34 from torchvision
    """

    def __init__(self):
        super().__init__()
        self.model = None
        return

    # override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        
        weights = None
        if getattr(args, "pretrained", False):
            weights = models.ResNet34_Weights.DEFAULT
            
        self.model = models.resnet34(weights=weights)
        
        # Modify the last layer for specific num_classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, args.num_classes)
        
        return self

    # override
    def forward(self, x):
        return self.model(x)
