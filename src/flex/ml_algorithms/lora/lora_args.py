from dataclasses import dataclass
from typing import Optional

import torch

from ...ml_utils import KeyValueArgs

@dataclass
class LoRAArgs(KeyValueArgs):
    name: str = "lora_linear"

    features_in: int = 0            # in_features (int): Number of input features.
    features_out: int = 0           # out_features (int): Number of output features.
    rank: int = 4                   # rank (int): Rank of the low-rank decomposition.
    scaling: float = 0.5            # alpha (float): Scaling factor for the low-rank update.
    lora_mode: str = "standard"     # lora_mode (str): Determines the LoRA inference mode. Options: "standard", "alternate", etc.
    device: str = "cpu"

    use_bias = True                 # use_bias (bool): If True, includes a bias parameter.
    pretrained_weight: Optional[torch.Tensor] = None      # pretrained_weight (Optional[torch.Tensor]): If provided, initializes self.weight with this tensor.

    alpha: float = 0

    def __init__(self, config_dict: dict, is_clone_dict = False):
        """
        Linear layer with Low-Rank Adaptation (LoRA).
        """
        super().__init__(config_dict, is_clone_dict)
        if config_dict != None and "lora" in config_dict:
            self.set_args(config_dict["lora"])

        self.name = self.get("name", "lora_linear")

        self.features_in: int = self.get("features_in", 0)
        self.features_out: int = self.get("features_out", 0)
        self.rank: int = self.get("rank", 0)
        self.lora_mode: str = self.get("lora_mode", "standard")

        self.device: str = self.get("device", "cpu")
        self.use_bias: bool = self.get("use_bias", True)
        return

    def with_pretrained_weight(self, weight: torch.Tensor):
        self.pretrained_weight = weight
        return self
