from __future__ import annotations

from ._rbla_client import RblaClientTrainingStrategy
from ...fl_algorithms.aggregation.methods._fed_aggregator_sp_plus import (
    FedAggregator_SPPlus,
)


class SpPlusClientTrainingStrategy(RblaClientTrainingStrategy):
    """RBLA local training with SP+ canonical-prefix broadcast reception."""

    def __init__(self, args, client_node):
        super().__init__(args, client_node)
        self._strategy_type = "sp_plus"

    def _create_inner(self, args, client_node) -> None:
        self._args = args
        self._strategy_type = "sp_plus"
        self._obj = client_node
        return self

    def set_local_weight(self) -> dict:
        node_var = self._obj.node_var
        node_var.model_weight = FedAggregator_SPPlus.broadcast_lora_state_dict(
            node_var.cache_weight,
            node_var.model_weight,
        )
        return

