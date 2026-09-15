from __future__ import annotations
import random

from ..fed_client_selector_args import FedClientSelectorArgs
from ..fed_client_selector_abc import FedClientSelector


class FedClientSelector_Random(FedClientSelector):
    """
    Random clients select class
    """

    def __init__(self, args: FedClientSelectorArgs|None = None):
        super().__init__(args)
        self._args.select_method = "random"
        return

    def select(self, client_list: list, select_number: int = -1):
        """
        Select clients from client list,
        Args:
            select_number: if selection_number <= 0, select number from args.select_number,
                           else use selection_number
        """        
        if select_number <= 0:
            select_number = self._args.select_number

        # If available clients are less than or equal to requested number, select all
        if len(client_list) <= select_number:
            return client_list

        return random.sample(client_list, select_number)
