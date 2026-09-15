from __future__ import annotations

import random
from typing import Optional

from ..fed_client_selector_args import FedClientSelectorArgs
from ..fed_client_selector_abc import FedClientSelector


class FedClientSelector_MidLoss(FedClientSelector):
    """
    Mid loss client selection class: selects clients whose loss is closest to the mean loss.
    Falls back to random selection when no client metrics are available yet.
    """

    # Keys searched in order when extracting loss from a train_record dict.
    _LOSS_KEYS = (
        "sqrt_train_loss_power_two_sum",
        "avg_loss",
        "train_loss_sum",
        "train_loss",
        "loss",
    )

    def __init__(self, args: FedClientSelectorArgs | None = None):
        super().__init__(args)
        self._args.select_method = "mid_loss"
        return

    def _extract_loss(self, client_id) -> Optional[float]:
        """Try to extract a scalar loss value from _clients_data_dict for the given client."""
        entry = self._clients_data_dict.get(client_id) or self._clients_data_dict.get(str(client_id))
        if not isinstance(entry, dict):
            return None
        # The value may be stored directly under a loss key, or nested under "train_record".
        record = entry.get("train_record", entry)
        if isinstance(record, dict):
            for key in self._LOSS_KEYS:
                val = record.get(key)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return None

    # Override parent class virtual method
    def select(self, client_list: list, select_number: int = -1):
        """
        Select clients from client list whose loss is closest to the mean.
        Falls back to random selection when no metrics are available.
        """
        if select_number <= 0:
            select_number = self._args.select_number

        # Build (client, loss) pairs for clients that have metrics.
        scored: list[tuple] = []
        for client in client_list:
            loss = self._extract_loss(client.node_id)
            if loss is not None:
                scored.append((client, loss))

        # Fallback: no data yet — pick randomly so training can start.
        if not scored:
            k = min(select_number, len(client_list))
            return random.sample(client_list, k)

        mean_loss = sum(loss for _, loss in scored) / len(scored)

        scored.sort(key=lambda pair: abs(pair[1] - mean_loss))
        return [client for client, _ in scored[:select_number]]
