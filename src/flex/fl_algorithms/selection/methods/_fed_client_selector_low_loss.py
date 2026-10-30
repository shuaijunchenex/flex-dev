from __future__ import annotations

from ..fed_client_selector_args import FedClientSelectorArgs
from ..fed_client_selector_abc import FedClientSelector


class FedClientSelector_LowLoss(FedClientSelector):
    """
    Low loss client selection class
    """

    def __init__(self, args: FedClientSelectorArgs|None = None):
        super().__init__(args)
        self._args.select_method = "low_loss"      # Select method     
        return

    #Override parent class virtual method
    def select(self, client_list: list, select_number: int = -1):
        """
        Select clients from client list
        """
        if select_number <= 0:
            select_number = self._args.select_number        
            
        # Convert to list of (client_id, data_pack) and sort by avg_loss ascending
        sorted_clients = sorted(self._clients_data_dict.items(),
                              key = lambda item: item[1]["train_record"]["sqrt_train_loss_power_two_sum"],
                              reverse = False)

        # Take top-k
        self.__top_k = [client_id for client_id, _ in sorted_clients[:select_number]]
        return [client for client in client_list if client.node_id in self.__top_k]
