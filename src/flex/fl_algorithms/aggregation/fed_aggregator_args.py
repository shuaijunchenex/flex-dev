from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from ...ml_utils.key_value_args import KeyValueArgs


@dataclass
class FedAggregatorArgs(KeyValueArgs):
    """
    " Dataset loader arguments
    """
    method: str = ""
    is_wbab: bool = False   # for RBLA use
    device: str = 'cpu'

    def __init__(self, config_dict: dict[str, Any]|None = None, is_clone_dict = False):
        """
        Args for aggregation methods
        """

        super().__init__(config_dict, is_clone_dict)
        if config_dict is not None and "aggregation" in config_dict:
            self.set_args(config_dict["aggregation"], is_clone_dict)

        self.method: str = self.get("method", "fedavg")
        self.device: str = self.get("device", "cpu")
        return
