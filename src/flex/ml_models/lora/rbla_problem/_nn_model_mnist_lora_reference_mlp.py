"""MNIST LoRA MLP with a rank-independent functional scaling."""
from __future__ import annotations

import torch.nn as nn

from flex.ml_algorithms.lora import MSLoRALinear
from flex.ml_models import AbstractNNModel, NNModel, NNModelArgs


class NNModel_MNISTLoRAReferenceMLP(NNModel):
    """Separate model used only by the RBLA reference-frame experiments.

    Each layer sets ``lora_alpha = local_rank * lora_scale``.  Therefore
    ``lora_alpha / local_rank`` is constant across heterogeneous clients,
    removing rank-dependent LoRA scaling as an experimental confounder.
    """

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        ratio = float(args.rank_ratio)
        hidden_rank = max(1, int(round(float(args.get("base_rank_hidden", 160)) * ratio)))
        output_rank = max(1, int(round(float(args.get("base_rank_output", 100)) * ratio)))
        lora_scale = float(args.get("lora_scale", 1.0))
        use_bias = bool(args.get("use_bias", True))

        hidden_alpha = hidden_rank * lora_scale
        output_alpha = output_rank * lora_scale

        self._flatten = nn.Flatten()
        self._fc1 = MSLoRALinear(
            784, 200, r=hidden_rank, lora_alpha=hidden_alpha,
            lora_dropout=0.0, fan_in_fan_out=False,
            merge_weights=False, bias=use_bias,
        )
        self._relu1 = nn.ReLU()
        self._fc2 = MSLoRALinear(
            200, 200, r=hidden_rank, lora_alpha=hidden_alpha,
            lora_dropout=0.0, fan_in_fan_out=False,
            merge_weights=False, bias=use_bias,
        )
        self._relu2 = nn.ReLU()
        self._fc3 = MSLoRALinear(
            200, 10, r=output_rank, lora_alpha=output_alpha,
            lora_dropout=0.0, fan_in_fan_out=False,
            merge_weights=False, bias=use_bias,
        )
        self._lora_config = {
            "suffix_A": "lora_A",
            "suffix_B": "lora_B",
            "sp_suffix": "sp_aggregated",
            "has_non_lora_params": False,
            "rank_independent_scaling": True,
            "lora_scale": lora_scale,
        }
        return self

    def forward(self, inputs):
        hidden = self._flatten(inputs)
        hidden = self._relu1(self._fc1(hidden))
        hidden = self._relu2(self._fc2(hidden))
        return self._fc3(hidden)
