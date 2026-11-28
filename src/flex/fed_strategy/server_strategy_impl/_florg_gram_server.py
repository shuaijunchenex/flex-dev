"""Stateful FLoRG server with separate initialization and factor broadcasts."""

from ...fl_algorithms.aggregation.methods._fed_aggregator_florg_gram import (
    FedAggregator_FLoRGGram,
    validate_factor_payload,
    validated_sample_count,
)
from ..client_strategy_impl._florg_gram_client import (
    FlorgGramClientTrainingStrategy,
    gram_factor_keys,
)
from ._rbla_server import RblaServerStrategy


def _payload_bytes(payload):
    return sum(value.numel() * value.element_size() for value in payload.values())


class FlorgGramServerStrategy(RblaServerStrategy):
    def __init__(self, args, server_node):
        super().__init__(args, server_node)
        self._strategy_type = "florg_gram"
        self._version = -1
        self._broadcast_version = -1
        self._pending = None
        self._payload = {}
        self._initial_state = {}
        self._clients_by_id = {}
        self.history = []
        self.initialization_stats = {}
        server_node.node_var.florg_gram_history = self.history
        server_node.node_var.florg_gram_initialization_stats = self.initialization_stats

    def _create_inner(self, args, server_node):
        self.__init__(args, server_node)
        return self

    def prepare(self, logger_header, client_nodes_in) -> None:
        node_var = self._obj.node_var
        if not isinstance(node_var.aggregation_method, FedAggregator_FLoRGGram):
            raise ValueError("The FLoRG server strategy requires the florg_gram aggregator")
        factor_keys = gram_factor_keys(node_var.model)
        initial = {key: value.detach().to(device="cpu", copy=True) for key, value in node_var.model_weight.items()}
        factors = {key: initial[key] for key in factor_keys}
        validate_factor_payload(factors, factors)
        if any(not value.count_nonzero().item() for value in factors.values()):
            raise ValueError("FLoRG requires nonzero initial factors")
        clients = {}
        for client in client_nodes_in:
            client_id = str(client.node_id)
            if client_id in clients:
                raise ValueError(f"Duplicate FLoRG client {client_id!r}")
            if not isinstance(client.strategy, FlorgGramClientTrainingStrategy):
                raise ValueError("The FLoRG server strategy requires florg_gram client strategies")
            client.strategy.validate_initial_state(initial)
            clients[client_id] = client
        super().prepare(logger_header, client_nodes_in)
        self._initial_state = initial
        self._payload = factors
        self._clients_by_id = clients

    def aggregation(self) -> None:
        if self._version < 0 or self._pending is not None or self._version != self._broadcast_version:
            raise RuntimeError("FLoRG aggregation requires a completed broadcast and no pending result")
        node_var = self._obj.node_var
        updates = [
            update for update in (node_var.client_updates or ())
            if validated_sample_count(update["train_record"].get("data_sample_num")) > 0
        ]
        if not updates:
            return
        seen = set()
        for update in updates:
            record = update["train_record"]
            client_id = str(update.get("node_id", record.get("node_id")))
            if client_id not in self._clients_by_id or client_id in seen:
                raise ValueError(f"Unknown or duplicate FLoRG client {client_id!r}")
            if update.get("base_version", record.get("base_version")) != self._version:
                raise ValueError(f"FLoRG client {client_id!r} trained on a stale broadcast")
            seen.add(client_id)
        aggregator = node_var.aggregation_method
        aggregator.set_reference(self._payload)
        result = aggregator.aggregate(updates)
        validate_factor_payload(result, self._payload)
        node_var.aggregated_weight = {
            key: value.detach().to(self._payload[key], copy=True) for key, value in result.items()
        }
        self._pending = {
            "round": self._version + 1,
            "client_ids": sorted(seen),
            "actual_upload_bytes": sum(_payload_bytes(update["updated_weights"]) for update in updates),
            "actual_download_bytes": 0,
            "aggregation_stats": aggregator.last_stats,
        }

    def apply_weight(self) -> None:
        if self._pending is None:
            return
        node_var = self._obj.node_var
        payload = node_var.aggregated_weight
        evaluated = {**self._initial_state, **payload}
        node_var.model_evaluator.update_model(evaluated)
        node_var.model_weight = evaluated
        self._payload = payload
        self._version += 1
        self.history.append(self._pending)
        self._pending = None

    def broadcast(self) -> None:
        if not self._initial_state:
            raise RuntimeError("Prepare the FLoRG server before broadcasting")
        if self._pending is not None:
            raise RuntimeError("Apply the FLoRG aggregate before broadcasting")
        clients = self._clients_by_id.values()
        if self._version < 0:
            for client in clients:
                client.strategy.initialize_from_server(self._initial_state)
            size = _payload_bytes(self._initial_state)
            self.initialization_stats.update({
                "initial_state_bytes_per_client": size,
                "actual_download_bytes": size * len(clients),
                "broadcast_recipients": len(clients),
            })
            self._version = 0
            self._broadcast_version = self._version
            return

        if self._broadcast_version == self._version:
            return
        message = {"version": self._version, "factors": self._payload}
        for client in clients:
            client.receive_weight(message)
            client.set_local_weight()
        self._broadcast_version = self._version
        if self.history:
            size = _payload_bytes(self._payload)
            self.history[-1].update({
                "actual_download_bytes": size * len(clients),
                "global_payload_bytes": size,
                "broadcast_recipients": len(clients),
            })
