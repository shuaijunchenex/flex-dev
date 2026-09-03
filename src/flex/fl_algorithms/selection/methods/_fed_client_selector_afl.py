from __future__ import annotations
import pandas as pd
import numpy as np
import random
from typing import List

from ..fed_client_selector_args import FedClientSelectorArgs
from ..fed_client_selector_abc import FedClientSelector

class FedClientSelector_AFL(FedClientSelector):
    """
    AFL (Adaptive Federated Learning) client selection algorithm.
    """

    def __init__(self, args: FedClientSelectorArgs|None = None):
        super().__init__(args)
        self._args.select_method = "afl"
        return

    def select(self, client_list: list, select_number: int = -1) -> list:
        """
        Select clients using AFL algorithm.
        
        Args:
            client_list: List of available client objects
            select_number: Number of clients to select
            
        Returns:
            List of selected client objects
        """
        if select_number <= 0:
            select_number = self._args.select_number
            
        # 1. Prepare metric_matrix and name_dict
        name_dict = {client.node_id: client for client in client_list}
        
        # Extract loss from _clients_data_dict
        metric_matrix = {}
        for cid, data in self._clients_data_dict.items():
            # Try to find 'loss' metric
            # Adapting to structure seen in HighLoss selector or generic
            loss = 0.0
            if isinstance(data, dict):
                 if "train_record" in data:
                     rec = data["train_record"]
                     # Prioritize the metric used in HighLoss as it seems standard for this codebase
                     if "sqrt_train_loss_power_two_sum" in rec:
                         loss = rec["sqrt_train_loss_power_two_sum"]
                     elif "train_loss" in rec:
                         loss = rec["train_loss"]
                     elif "loss" in rec:
                         loss = rec["loss"]
            
            # Only add if we found a non-zero loss or if we want to track 0 loss?
            # Assuming 0 loss might mean no data, but let's include it.
            # But the logic uses exp(loss), so 0 is fine.
            metric_matrix[cid] = loss

        # Filter metric_matrix to only include available clients
        available_ids = set(name_dict.keys())
        metric_matrix = {k: v for k, v in metric_matrix.items() if k in available_ids}

        # If no metrics available (e.g. first round), fallback to random
        if not metric_matrix:
            if len(client_list) <= select_number:
                return client_list
            return random.sample(client_list, select_number)

        # 2. Logic adapted from user provided AFLSelector
        alpha1, alpha2, alpha3 = 0.8, 0.01, 0.1
        number = select_number
        
        # Create DataFrame from loss metrics
        df = pd.DataFrame.from_dict(
            metric_matrix, orient='index', 
            columns=['loss']
        )
        
        # Calculate valuation based on loss
        df['valuation'] = np.exp(alpha2 * df['loss'])
        
        # Sort clients based on valuation
        sorted_clients = df.sort_values(by='valuation', ascending=False)
        
        # Select top (1-alpha1)% of clients based on valuation
        num_top_clients = int((1 - alpha1) * len(sorted_clients))
        if num_top_clients == 0:
            num_top_clients = 1
        top_clients = sorted_clients.head(num_top_clients)
        # console.info(f"top_clients: {top_clients.index.tolist()}")
        
        # Select (1-alpha3) * number of clients based on adjusted probabilities
        # Normalize probabilities
        probabilities = top_clients['valuation'] / top_clients['valuation'].sum()
        # console.info(f"prob: {probabilities}")
        
        num_primary = int((1 - alpha3) * number)
        selected_clients_primary = pd.DataFrame()

        if num_primary > 0 and not top_clients.empty:
             # Ensure weights sum to 1 (fillNa handled implicitly by division if no NaNs)
             # Handle potential NaNs just in case
             probabilities = probabilities.fillna(0)
             if probabilities.sum() == 0:
                 # Uniform if all 0
                 selected_clients_primary = top_clients.sample(n=min(num_primary, len(top_clients)))
             else:
                 selected_clients_primary = top_clients.sample(n=min(num_primary, len(top_clients)), weights=probabilities)
        
        # Select alpha3 * number of clients uniformly from remaining
        remaining_clients = sorted_clients.drop(selected_clients_primary.index)
        # console.info(f"remaining_clients: {len(remaining_clients)}")
        
        num_secondary = int(alpha3 * number)
        if num_secondary > 0 and len(remaining_clients) > 0:
            selected_clients_secondary = remaining_clients.sample(n=min(num_secondary, len(remaining_clients)))
        else:
            selected_clients_secondary = pd.DataFrame()
        
        # Combine the two groups
        selected_clients = pd.concat([selected_clients_primary, selected_clients_secondary])
        
        # Get client IDs
        selected_client_keys = selected_clients.index.tolist()
        
        # Map back to client objects
        final_list = [name_dict[key] for key in selected_client_keys if key in name_dict]
        
        # Just in case we selected fewer than requested due to small pools and strict logic, 
        # the user code doesn't backfill. We return what we have.
        
        return final_list
