from __future__ import annotations
from dataclasses import dataclass

from ...ml_utils import KeyValueArgs


@dataclass
class FedClientSelectorArgs(KeyValueArgs):
    """
    Fed selector args
    """
    select_method: str = "random"    # Select method, can be 'all', 'high_loss' or 'random'
    select_round: int = 1
    select_number: int = 2
    client_selection_round: int = 1
    random_seed: int = 42       # if -1 means random seed generate by time milliseconds
    
    def __init__(self, config_dict: dict|None = None, is_clone_dict = False):
        super().__init__(config_dict, is_clone_dict)

        if config_dict is not None and "client_selection" in config_dict:
             self.set_args(config_dict["client_selection"], is_clone_dict)

        self.select_method = self.get("method", "random")
        self.select_round = self.get("round", 1)
        self.select_number = self.get("number", 2)
        self.client_selection_round = self.get("client_selection_round", 1)
        self.random_seed = self.get("random_seed", 42)
        return
