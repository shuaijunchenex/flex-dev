from __future__ import annotations

from ..fed_client_selector_args import FedClientSelectorArgs
from ..fed_client_selector_abc import FedClientSelector


class FedClientSelector_HighLoss(FedClientSelector):
    """
    High loss client selection class
    """

    def __init__(self, args: FedClientSelectorArgs|None = None):
        super().__init__(args)
        self._args.select_method = "high_loss"      # Select method     
        return

    #Override parent class virtual method
    def select(self, client_list: list, select_number: int = -1):
        """
        Select clients from client list
        """
        if select_number <= 0:
            select_number = self._args.select_number        

        def _get_loss(data):
            """Extract sqrt_train_loss_power_two_sum from either nested or flat format."""
            if not isinstance(data, dict):
                return 0.0
            # Nested path: data["train_record"]["sqrt_train_loss_power_two_sum"]
            inner = data.get("train_record", {})
            if isinstance(inner, dict) and "sqrt_train_loss_power_two_sum" in inner:
                return float(inner["sqrt_train_loss_power_two_sum"])
            # Flat path: data["sqrt_train_loss_power_two_sum"] (fedgra server format)
            return float(data.get("sqrt_train_loss_power_two_sum", 0.0))

        # Convert to list of (client_id, data_pack) and sort by loss descending
        sorted_clients = sorted(self._clients_data_dict.items(),
                              key = lambda item: _get_loss(item[1]),
                              reverse = True)

        # Take top-k
        self.__top_k = [client_id for client_id, _ in sorted_clients[:select_number]]
        return [client for client in client_list if client.node_id in self.__top_k]
