from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import AbstractNNModel, NNModelArgs, NNModel
from ...ml_algorithms.lora import MSLoRALinear
from ...ml_utils import console

class NNModel_SimpleLoRAMLP(NNModel):
    """
    Simple MLP with LoRA-enabled Linear layers (using Microsoft MSLoRALinear).
    """

    def __init__(self):
        super().__init__()
        self.lora_mode = "standard" 

    # override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        rank = getattr(args, "lora_rank", 4)
        scaling = getattr(args, "lora_scaling", 0.5)
        use_bias = getattr(args, "use_bias", True)
        rank_ratio = getattr(args, "rank_ratio", 1)
        cap_rank_to_matrix_dim = bool(
            args.get("cap_lora_rank_to_matrix_dim", False)
        )

        def effective_rank(base_rank: int, in_features: int, out_features: int) -> int:
            local_rank = int(base_rank * rank_ratio)
            if cap_rank_to_matrix_dim:
                # Compact canonicalization requires a thin factorization with
                # r <= min(d_in, d_out). This is opt-in so existing experiments
                # keep their historical, potentially overcomplete LoRA ranks.
                local_rank = min(local_rank, in_features, out_features)
            return local_rank

        hidden_rank = effective_rank(160, 784, 200)
        output_rank = effective_rank(100, 200, 10)
        # Base ranks are hardcoded; warn if any is 1 while rank_ratio is enabled
        _base_ranks = [160, 160, 100]
        if any(br == 1 for br in _base_ranks) and rank_ratio != 1.0:
            console.warn(
                f"[Simple LoRA MLP] rank_ratio={rank_ratio} is configured but one of the "
                f"hardcoded base ranks is 1, effective rank will always be 1 regardless of rank_ratio."
            )

        self._flatten = nn.Flatten()
        self._fc1 = MSLoRALinear(784, 200, r=hidden_rank, lora_alpha=int(rank * scaling),
                                 lora_dropout=0.0, fan_in_fan_out=False,
                                 merge_weights=False, bias=use_bias)
        self._relu1 = nn.ReLU()
        self._fc2 = MSLoRALinear(200, 200, r=hidden_rank, lora_alpha=int(rank * scaling),
                                 lora_dropout=0.0, fan_in_fan_out=False,
                                 merge_weights=False, bias=use_bias)
        self._relu2 = nn.ReLU()
        self._fc3 = MSLoRALinear(200, 10, r=output_rank, lora_alpha=int(rank * scaling),
                                 lora_dropout=0.0, fan_in_fan_out=False,
                                 merge_weights=False, bias=use_bias)
        
        # Declare LoRA configuration for aggregation system
        self._lora_config = {
            "suffix_A": "lora_A",
            "suffix_B": "lora_B",
            "sp_suffix": "sp_aggregated",
            "has_non_lora_params": False,  # All weighted layers are LoRA-wrapped
        }

        return self  

    # override
    def forward(self, x):
        x = self._flatten(x)
        x = self._relu1(self._fc1(x))
        x = self._relu2(self._fc2(x))
        x = self._fc3(x)

        return x

    def set_lora_mode(self, mode: str):
        if mode not in ["standard", "lora_only", "lora_disabled", "scaling"]:
            raise ValueError(f"Unsupported lora_mode: {mode}")
        self.lora_mode = mode
        for layer in [self._fc1, self._fc2, self._fc3]:
            layer.lora_mode = mode
