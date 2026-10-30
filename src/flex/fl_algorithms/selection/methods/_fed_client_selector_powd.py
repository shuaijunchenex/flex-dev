from __future__ import annotations
import pandas as pd
import random
from typing import List

from ..fed_client_selector_args import FedClientSelectorArgs
from ..fed_client_selector_abc import FedClientSelector


class FedClientSelector_Powd(FedClientSelector):
    """
    Pow-d client selection algorithm.
    """

    def __init__(self, args: FedClientSelectorArgs|None = None):
        super().__init__(args)
        self._args.select_method = "powd"
        return

    def powd_sample_clients_under_probability(self, info_df, d):
        """Sample clients under probability distribution based on data volume."""
        
        probability_list = []
        sample_client_df = {}
        
        # Check if data_volume is appropriate for this sampling method
        # The method creates a list with 'data_volume' entries for each client.
        # If data_volume is large (e.g. 5000 images), this list will be huge.
        # We might need to scale it or use `random.choices` with weights directly.
        # Implementing as requested, but adding a specialized handling for efficiency if needed.

        # User's logic:
        # for index, row in info_df.iterrows():
        #     for i in range(int(row['data_volume'])):
        #         probability_list.append(index)
        
        # Optimization: using random.choices with weights is much faster and cleaner than expanding the list
        # But to stick close to the user's logic which does "Remove the client from list" (sampling without replacement logic on the pool),
        # actually, `probability_list` being filtered `filter(lambda x: x != sample_client, probability_list)` implies
        # once a client is picked, it cannot be picked again.
        
        # Re-implementing user logic structure:
        indices = info_df.index.tolist()
        weights = info_df['data_volume'].tolist()
        
        # If any weight is <= 0 or NaN, handle it
        cleaned_weights = []
        cleaned_indices = []
        for idx, w in zip(indices, weights):
            if w > 0:
                cleaned_indices.append(idx)
                cleaned_weights.append(int(w))
                
        if not cleaned_indices:
            return pd.DataFrame()

        # To simulate the user's probability list approach efficiently:
        # We want to sample 'd' unique clients, where probability of picking client i is proportional to its weight.
        # And we remove picked clients.
        # This is equivalent to weighted random sampling without replacement.
        
        # However, standard weighted sampling without replacement is complex.
        # The user's code:
        # 1. Puts N copies of client into a list.
        # 2. Picks one item randomly.
        # 3. Removes ALL copies of that client from the list.
        # 4. Repeats.
        
        # This effectively means: Pick a client with prob P(i) = w_i / sum(w).
        # Then remove i.
        # Then Pick client j with prob P(j) = w_j / (sum(w) - w_i).
        
        selected_indices = []
        
        # Work with a mutable copy
        current_indices = list(cleaned_indices)
        current_weights = list(cleaned_weights)
        
        loop_count = min(d, len(current_indices))
        
        for _ in range(loop_count):
            total_weight = sum(current_weights)
            if total_weight == 0:
                break
                
            # Select one
            # random.choices returns a list [element]
            chosen_idx = random.choices(current_indices, weights=current_weights, k=1)[0]
            
            # Add to selected
            selected_indices.append(chosen_idx)
            
            # Remove from pool
            # Find index in current lists to pop
            list_pos = current_indices.index(chosen_idx)
            current_indices.pop(list_pos)
            current_weights.pop(list_pos)

        # Construct result DataFrame
        for idx in selected_indices:
            # User logic: sample_client_df[sample_client[0]] = [float(info_df['loss'][sample_client])]
            # Using .loc for safer access
            val = info_df.loc[idx, 'loss']
            # Handle if it returns series (duplicate index?) - assuming unique index
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            sample_client_df[idx] = [float(val)]

        # print(sample_client_df)
        return pd.DataFrame.from_dict(sample_client_df, orient='index', columns=['loss'])

    def select(self, client_list: list, select_number: int = -1) -> list:

        """
        Select clients using Pow-d algorithm.
        
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
        
        # Extract metrics
        metric_matrix = {}
        for cid, data in self._clients_data_dict.items():
            loss = 0.0
            data_volume = 1.0 # Default to 1 if not found to avoid errors
            
            if isinstance(data, dict):
                 # Loss
                 if "train_record" in data:
                     rec = data["train_record"]
                     if "sqrt_train_loss_power_two_sum" in rec:
                         loss = rec["sqrt_train_loss_power_two_sum"]
                     elif "train_loss" in rec:
                         loss = rec["train_loss"]
                     elif "loss" in rec:
                         loss = rec["loss"]
                 
                 # Data Volume
                 # Assuming data_volume might be in 'data_sample_num' or similar in client config
                 # In simulation, we often pass it in 'clients_data_dict' if it was gathered.
                 # If not present, we check if the client object has it.
                 if "data_sample_num" in data:
                     data_volume = data["data_sample_num"]
                 elif cid in name_dict:
                      client = name_dict[cid]
                      if hasattr(client, 'data_sample_num'):
                          data_volume = client.data_sample_num
                      elif hasattr(client, 'node_var') and client.node_var and hasattr(client.node_var, 'data_sample_num'):
                          data_volume = client.node_var.data_sample_num

            metric_matrix[cid] = {
                'loss': loss,
                'data_volume': data_volume
            }

        # Filter to available clients
        available_ids = set(name_dict.keys())
        metric_matrix = {k: v for k, v in metric_matrix.items() if k in available_ids}

        if not metric_matrix:
            if len(client_list) <= select_number:
                return client_list
            return random.sample(client_list, select_number)

        # 2. Logic adapted from user provided PowdSelector
        number = select_number
        
        # Create dataframe from metrics
        df = pd.DataFrame.from_dict(metric_matrix, orient='index')
        # Ensure columns exist
        if 'loss' not in df.columns: df['loss'] = 0.0
        if 'data_volume' not in df.columns: df['data_volume'] = 1.0

        # Sample by data volume (d = 2 * number)
        d = max(2 * number, number) # Ensure we sample at least 'number' if possible, though user logic implies 2*number candidate pool
        
        # If total clients < d, we can't sample d distinct ones.
        # But powd_sample_clients_under_probability handles loop_count = min(d, len)
        df_sample_clients = self.powd_sample_clients_under_probability(df, d)

        # Rank by loss (descending)
        if df_sample_clients.empty:
             return random.sample(client_list, min(len(client_list), number))

        client_rank = df_sample_clients.sort_values(ascending=False, by='loss')

        # Select top clients
        selected_client_keys = client_rank[:number].index.tolist()

        # Print selected clients (optional logging via console)
        # keys = [name_dict[key].node_id for key in selected_client_keys if key in name_dict] # Just IDs
        # console.info(f"Powd Selected: {keys}")
        
        # Return client objects
        final_list = [name_dict[key] for key in selected_client_keys if key in name_dict]
        
        return final_list
