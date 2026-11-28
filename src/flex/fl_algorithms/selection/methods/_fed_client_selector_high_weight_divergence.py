from __future__ import annotations

from ..fed_client_selector_args import FedClientSelectorArgs
from ..fed_client_selector_abc import FedClientSelector


class FedClientSelector_HighWeightDivergence(FedClientSelector):
    """
    High weight-divergence client selection.

    Ranks clients by the L2 distance between their local model weights
    *after* training and the global model weights *before* training
    (i.e. ``train_record["weight_l2_delta_keras"]``), then selects the top-k
    clients with the largest divergence.

    This metric captures how far each client's local update has moved
    away from the global model, which is a proxy for the informativeness
    of that client's gradient — clients with larger divergence have seen
    data that is more different from the current global model.

    The divergence is computed as the **sum of per-layer L2 norms**
    (matching the original Keras implementation), i.e.
    ``Σ_l ‖w_local_l − w_global_l‖₂``.
    """

    def __init__(self, args: FedClientSelectorArgs | None = None):
        super().__init__(args)
        self._args.select_method = "high_weight_divergence"

    # Override parent class virtual method
    def select(self, client_list: list, select_number: int = -1) -> list:
        """
        Select top-k clients ranked by weight L2 divergence (descending).

        ``_clients_data_dict`` is keyed by node_id and each value is expected
        to contain::

            {
                "train_record": {
                    "weight_l2_delta_keras": <float>,   # Σ_l ‖w_local_l − w_global_l‖₂
                    ...
                },
                ...
            }
        """
        if select_number <= 0:
            select_number = self._args.select_number

        # Sort clients by weight_l2_delta_keras descending (highest divergence first)
        sorted_clients = sorted(
            self._clients_data_dict.items(),
            key=lambda item: item[1].get("train_record", {}).get("weight_l2_delta_keras", 0.0),
            reverse=True,
        )

        top_k_ids = {str(client_id) for client_id, _ in sorted_clients[:select_number]}
        return [client for client in client_list if str(client.node_id) in top_k_ids]
