import torch.nn as nn
from torchvision import models
from ..nn_model import NNModel
from ..nn_model_abc import AbstractNNModel
from ..nn_model_args import NNModelArgs


class NNModel_MobileNetV2(NNModel):
    """
    MobileNetV2 from torchvision
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
            weights = models.MobileNet_V2_Weights.DEFAULT
            
        self.model = models.mobilenet_v2(weights=weights)
        
        # Modify the last layer for specific num_classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, args.num_classes)
        
        return self

    # override
    def forward(self, x):
        return self.model(x)
