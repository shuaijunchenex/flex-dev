from __future__ import annotations

from ._rbla_server import RblaServerStrategy


class SpPlusServerStrategy(RblaServerStrategy):
    """Server strategy for SP+ aggregation and rank-prefix broadcasting."""

    def __init__(self, args, server_node) -> None:
        super().__init__(args, server_node)
        self._strategy_type = "sp_plus"

    def _create_inner(self, args, server_node) -> None:
        self._args = args
        self._strategy_type = "sp_plus"
        self._obj = server_node
        return self

    def broadcast(self) -> None:
        for client in self._obj.client_nodes:
            client.receive_weight(self._obj.node_var.model_weight)
            client.set_local_weight()
        return

