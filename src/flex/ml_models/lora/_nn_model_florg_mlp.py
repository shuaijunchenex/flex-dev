"""Linear-adapter model for the FLoRG training strategy."""

import torch
from torch import nn

from ...ml_algorithms.lora.impl.gram_lora import GramLoRALinear
from ..nn_model import NNModel, NNModelArgs


class NNModel_FlorgMLP(NNModel):
    def create_model(self, args: NNModelArgs) -> "NNModel_FlorgMLP":
        super().create_model(args)
        input_features = int(args.get("input_features", 784))
        hidden_features = int(args.get("hidden_features", 200))
        classes = int(args.get("num_classes", 10))
        rank = int(args.get("lora_rank", 4))
        bias = bool(args.get("use_bias", True))
        init_std = float(args.get("factor_init_std", 0.02))
        rank_ratio = float(args.get("rank_ratio", 1.0))
        if rank_ratio != 1.0:
            raise ValueError("florg_mlp requires the same configured rank for the server and clients")
        if min(input_features, hidden_features, classes) <= 0:
            raise ValueError("FLoRG model feature widths and class count must be positive")
        generator = torch.Generator().manual_seed(int(args.get("basis_seed", 0)))
        self.layers = nn.Sequential(
            nn.Flatten(),
            GramLoRALinear(
                nn.Linear(input_features, hidden_features, bias=bias), rank,
                generator=generator, factor_init_std=init_std,
            ),
            nn.ReLU(),
            GramLoRALinear(
                nn.Linear(hidden_features, hidden_features, bias=bias), rank,
                generator=generator, factor_init_std=init_std,
            ),
            nn.ReLU(),
            GramLoRALinear(
                nn.Linear(hidden_features, classes, bias=bias), rank,
                generator=generator, factor_init_std=init_std,
            ),
        )
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)
