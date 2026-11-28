from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import AbstractNNModel, NNModelArgs, NNModel

class NNModel_CifarConvnet(NNModel):

    """
    " Private class for CifarConvnet model implementation
    """

    def __init__(self):
        super().__init__()
        

    #override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        
        # Defining layers with same shapes but without the same padding attribute
        self._conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self._bn1 = nn.BatchNorm2d(32)
        self._conv2 = nn.Conv2d(32, 32, 3)
        self._pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._dropout1 = nn.Dropout(0.25)

        self._conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self._bn2 = nn.BatchNorm2d(64)
        self._conv4 = nn.Conv2d(64, 64, 3)
        self._pool2 = nn.MaxPool2d(2, 2)
        self._dropout2 = nn.Dropout(0.25)

        self._fc1 = nn.Linear(64 * 6 * 6, 512) # Final shape calculated
        self._dropout3 = nn.Dropout(0.25)
        self._fc2 = nn.Linear(512, 512)
        self._dropout4 = nn.Dropout(0.25)
        self._fc3 = nn.Linear(512, 10)
        return self         #Note: return self

    #override
    def forward(self, x) -> Any:
        # Forward pass through layers with calculated shapes and activations
        x = F.relu(self._conv1(x))
        x = self._bn1(x)
        x = F.relu(self._conv2(x))
        x = self._pool1(x)
        x = self._dropout1(x)

        x = F.relu(self._conv3(x))
        x = self._bn2(x)
        x = F.relu(self._conv4(x))
        x = self._pool2(x)
        x = self._dropout2(x)

        x = torch.flatten(x, 1)  # Flattening for fully connected layers
        x = F.relu(self._fc1(x))
        x = self._dropout3(x)
        x = F.relu(self._fc2(x))
        x = self._dropout4(x)
        x = self._fc3(x)
        return x  # raw logits, CrossEntropyLoss handles softmax
