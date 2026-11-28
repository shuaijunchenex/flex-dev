import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import AbstractNNModel, NNModel, NNModelArgs
from ...ml_algorithms.lora import LoRALinear
from ...ml_utils import console


class NNModel_SimpleLoRACNN(NNModel):
    def __init__(self):
        super().__init__()
        self.fc1 = None
        self.bn2 = None
        self.conv2 = None
        self.bn1 = None
        self.conv1 = None

    # override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        
        rank_ratio = float(getattr(args, "rank_ratio", 1))
        base_lora_rank = 5
        lora_rank = int(max(1, round(base_lora_rank * rank_ratio)))
        if base_lora_rank == 1 and rank_ratio != 1.0:
            console.warn(
                f"[Simple LoRA CNN] rank_ratio={rank_ratio} is configured but base lora_rank=1, "
                f"effective rank will always be 1 regardless of rank_ratio. "
                f"Consider increasing the base lora_rank in the model definition."
            )
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = LoRALinear(32 * 5 * 5, 10, rank=lora_rank)

        # Declare LoRA configuration for aggregation system
        self._lora_config = {
            "suffix_A": "lora_A",
            "suffix_B": "lora_B",
            "sp_suffix": "sp_aggregated",
            "has_non_lora_params": True,   # Conv2d + BatchNorm2d are non-LoRA
        }

        return self         # Note: return self

    # override
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
