from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from ..ml_utils.key_value_args import KeyValueArgs


@dataclass
class NNModelArgs(KeyValueArgs):
    """
    NN Model args
    """
    # Common
    model_type: str = ""
    share_model: bool = True

    # Specified
    input_dim: int = 0
    hidden_dim: int = 0
    output_dim: int = 0
    softmax_dim: int = 1
    num_classes: int = 10
    num_heads: int = 8
    feature_dim: int = 0
    dropout: float = 0.0

    rank_ratio: int = 1
    lora_mode: str = "standard"
    in_channels: int = 1
    input_size: int = 28
    

    def __init__(self, config_dict: dict[str, Any]|None = None, is_clone_dict=False):
        """
        Model type enum
        """
        super().__init__(config_dict, is_clone_dict)
        if config_dict is not None and "nn_model" in config_dict:
            self.set_args(config_dict["nn_model"])

        self.model_type = self.get("name", "")
        self.input_dim = self.get("input_dim", 0)
        self.hidden_dim = self.get("hidden_dim", 0)
        self.output_dim = self.get("output_dim", 0)
        self.rank_ratio = self.get("rank_ratio", 1)
        self.lora_mode = self.get("lora_mode", "")
        self.softmax_dim = self.get("softmax_dim", 1)
        self.num_classes = self.get("num_classes", 10)
        self.num_heads = self.get("num_heads", 8)
        self.feature_dim = self.get("feature_dim", 0)
        self.dropout = self.get("dropout", 0.0)

        self.share_model = self.get("share_model", True)
        self.in_channels = self.get("in_channels", 1)
        self.input_size = self.get("input_size", 28)
        return
