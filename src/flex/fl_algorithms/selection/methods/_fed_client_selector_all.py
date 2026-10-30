from __future__ import annotations

from ..fed_client_selector_abc import FedClientSelector, FedClientSelectorArgs

"""
Select all clients
"""

class FedClientSelector_All(FedClientSelector):
    def __init__(self, args: FedClientSelectorArgs|None = None):
        super().__init__(args)
        self._args.select_method = "all"      # Select method     
        return

    #Override parent class virtual method
    def select(self, client_list: list, select_number: int = -1) -> list:
        """
        Select clients from client list
        """        
        return client_list