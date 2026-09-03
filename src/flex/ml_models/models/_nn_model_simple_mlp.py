import torch.nn as nn

from .. import AbstractNNModel, NNModelArgs, NNModel

class NNModel_SimpleMLP(NNModel):

    def __init__(self):
        super().__init__()
        

    #override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        
        self._layer_input = nn.Linear(args.input_dim, args.hidden_dim)
        self._relu = nn.ReLU()
        self._dropout = nn.Dropout()
        self._layer_hidden = nn.Linear(args.hidden_dim, args.output_dim)
        self._softmax = nn.Softmax(args.softmax_dim)
        return self         #Note: return self

    #override
    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self._layer_input(x)
        x = self._dropout(x)
        x = self._relu(x)
        x = self._layer_hidden(x)
        return self._softmax(x)
