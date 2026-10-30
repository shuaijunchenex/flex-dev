from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from ...ml_utils.key_value_args import KeyValueArgs


@dataclass
class DataHandlerArgs(KeyValueArgs):
    """
    Data handler args
    """
    distribution: str = "mnist_lt"
    data_volum_list = None

    verify_allocate: bool = True
    batch_size: int = 64
    shuffle: bool = False
    num_workers: int = 4

    def __init__(self, config_dict: Optional[dict] = None, is_clone_dict = False):
        super().__init__(config_dict, is_clone_dict)

        if config_dict is not None and "data_handler" in config_dict:
            self.set_args(config_dict["data_handler"], is_clone_dict)

        self.distribution = self.get("distribution", "mnist_lt")
        self.data_volum_list = self.get("data_volum_list", None)
        self.verify_allocate = self.get("verify_allocate", True)
        self.batch_size = self.get("batch_size", 64)
        self.shuffle = self.get("shuffle", False)
        return    
