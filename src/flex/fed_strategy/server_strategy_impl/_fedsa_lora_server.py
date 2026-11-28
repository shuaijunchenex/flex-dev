"""FedSA-LoRA server that coordinates shared A and receives scalar evaluations."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping

from ...fl_algorithms.aggregation.methods._fed_aggregator_fedsa_lora import (
    FedAggregator_FedSAShareA,
    validate_sample_count,
    validate_shared_a,
)
from ..client_strategy_impl._fedsa_lora_client import FedSALoRAClientTrainingStrategy, linear_lora_profile
from ._rbla_server import RblaServerStrategy


_IDLE = object()


def _payload_bytes(state) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in state.values())


class FedSALoRAServerStrategy(RblaServerStrategy):
    def __init__(self, args, server_node):
        super().__init__(args, server_node)
        self._strategy_type = "fedsa_lora"
        self._clients = {}
        self._initial_state = None
        self._shared_template = None
        self.version = 0
        self._broadcast_version = None
        self._pending_aggregate = _IDLE
        self._history = []
        self._obj.node_var.fedsa_history = self._history

    def _create_inner(self, args, server_node):
        self.__init__(args, server_node)
        return self

    def prepare(self, logger_header, client_nodes_in) -> None:
        if self._shared_template is not None:
            raise RuntimeError("FedSA-LoRA server is already prepared")
        node_var = self._obj.node_var
        if not isinstance(node_var.aggregation_method, FedAggregator_FedSAShareA):
            raise ValueError("FedSA-LoRA server requires the fedsa_lora aggregator")
        layers = linear_lora_profile(node_var.model)
        shared_keys = tuple(layer.key_A for layer in layers.values())
        private_keys = {layer.key_B for layer in layers.values()}
        initial = OrderedDict((key, value.detach().cpu().clone()) for key, value in node_var.model_weight.items()
                              if key not in private_keys)
        template = OrderedDict((key, initial[key]) for key in shared_keys)
        validate_shared_a(template, template)
        clients = {}
        for client in client_nodes_in:
            client_id = client.node_id
            if not isinstance(client_id, str) or not client_id or client_id in clients:
                raise ValueError(f"Missing or duplicate FedSA-LoRA client id '{client_id}'")
            if not isinstance(client.strategy, FedSALoRAClientTrainingStrategy):
                raise TypeError("FedSA-LoRA server requires fedsa_lora client strategies")
            client.strategy.configure_model()
            client.strategy.configure_evaluation_loader(getattr(node_var, "test_data_loader", None))
            client_layers = client.strategy.layers
            if client_layers.keys() != layers.keys() or any(
                (layer.rank, layer.scaling) != (client_layers[name].rank, client_layers[name].scaling)
                for name, layer in layers.items()
            ):
                raise ValueError("FedSA-LoRA clients and server require common adapter layers, ranks and scales")
            client.strategy.validate_initial_state(initial)
            clients[client_id] = client
        if not clients:
            raise ValueError("FedSA-LoRA requires at least one client")

        super().prepare(logger_header, client_nodes_in)
        self._clients = clients
        self._initial_state = initial
        self._shared_template = template
        node_var.model_weight = OrderedDict((key, value.clone()) for key, value in template.items())

    def broadcast(self) -> None:
        if self._shared_template is None:
            raise RuntimeError("FedSA-LoRA must be prepared before broadcasting")
        if self._broadcast_version == self.version:
            return
        if self._broadcast_version is None:
            for client in self._clients.values():
                if client.strategy.version is None:
                    client.strategy.initialize_from_server(self._initial_state)
            self._obj.node_var.fedsa_initialization_stats = {
                "initial_download_bytes": _payload_bytes(self._initial_state) * len(self._clients),
                "broadcast_recipients": len(self._clients),
            }
            self._initial_state = None
        else:
            shared = self._obj.node_var.model_weight
            validate_shared_a(shared, self._shared_template)
            message = {"version": self.version, "shared_a": shared}
            for client in self._clients.values():
                if client.strategy.version != self.version:
                    client.receive_weight(message)
                    client.set_local_weight()
            self._history[-1]["actual_download_bytes"] = _payload_bytes(shared) * len(self._clients)
        self._broadcast_version = self.version

    def receive_client_updates(self, client_updates) -> None:
        if self._broadcast_version != self.version or self._pending_aggregate is not _IDLE:
            raise RuntimeError("FedSA-LoRA requires a completed broadcast before accepting client updates")
        if not isinstance(client_updates, (list, tuple)):
            raise ValueError("FedSA-LoRA updates must be a list or tuple")
        accepted = []
        seen = set()
        for update in client_updates:
            if not isinstance(update, Mapping) or update.keys() != {"updated_weights", "train_record"}:
                raise ValueError("Invalid FedSA-LoRA update envelope")
            metadata = update["train_record"]
            if not isinstance(metadata, Mapping) or metadata.keys() != {"node_id", "data_sample_num", "fedsa_version"}:
                raise ValueError("FedSA-LoRA accepts only id, sample count and version as update metadata")
            client_id = metadata["node_id"]
            if not isinstance(client_id, str) or client_id not in self._clients or client_id in seen:
                raise ValueError(f"Unknown or duplicate FedSA-LoRA client id {client_id!r}")
            version = metadata["fedsa_version"]
            if type(version) is not int or version != self.version:
                raise ValueError(f"FedSA-LoRA update from '{client_id}' targets version {version!r}, expected {self.version}")
            validate_sample_count(metadata["data_sample_num"])
            shared = update["updated_weights"]
            validate_shared_a(shared, self._shared_template)
            accepted.append({"updated_weights": OrderedDict((key, value.detach().clone()) for key, value in shared.items()),
                             "train_record": dict(metadata)})
            seen.add(client_id)
        self._obj.node_var.client_updates = accepted

    def aggregation(self) -> None:
        if self._broadcast_version != self.version or self._pending_aggregate is not _IDLE:
            raise RuntimeError("FedSA-LoRA requires a completed broadcast before aggregation")
        node_var = self._obj.node_var
        if not any(update["train_record"]["data_sample_num"] > 0 for update in node_var.client_updates):
            self._pending_aggregate = None
            return
        pending = node_var.aggregation_method.aggregate(node_var.client_updates)
        validate_shared_a(pending, self._shared_template)
        node_var.aggregated_weight = pending
        self._pending_aggregate = pending

    def apply_weight(self) -> None:
        aggregate = self._pending_aggregate
        if aggregate is _IDLE:
            raise RuntimeError("FedSA-LoRA has no pending shared A aggregate")
        node_var = self._obj.node_var
        if aggregate is None:
            node_var.client_updates = []
            self._pending_aggregate = _IDLE
            return
        validate_shared_a(aggregate, self._shared_template)
        committed = OrderedDict((key, tensor.detach().to(self._shared_template[key]).clone())
                                for key, tensor in aggregate.items())
        validate_shared_a(committed, self._shared_template)
        node_var.model_weight = committed
        self.version += 1
        self._pending_aggregate = _IDLE
        self._history.append({
            "version": self.version,
            "upload_clients": len(node_var.client_updates),
            "actual_upload_bytes": sum(_payload_bytes(update["updated_weights"]) for update in node_var.client_updates),
        })
        node_var.client_updates = []

    def evaluate(self) -> None:
        if self._broadcast_version != self.version:
            raise RuntimeError("FedSA-LoRA must finish broadcasting before evaluation")
        metrics_by_client = [client.strategy.evaluate_local() for client in self._clients.values()]
        counts = [result["total_test_samples"] for result in metrics_by_client]
        total = sum(counts)
        metrics = {}
        for key in ("accuracy", "average_loss", "precision", "recall", "f1_score", "matthews_correlation",
                    "mcc", "primary_score"):
            if all(key in result for result in metrics_by_client):
                metrics[key] = math.fsum(count * result[key] for count, result in zip(counts, metrics_by_client)) / total
        primary = {result.get("primary_metric") for result in metrics_by_client}
        if len(primary) == 1 and None not in primary:
            metrics["primary_metric"] = primary.pop()
        metrics["perplexity"] = math.exp(min(metrics["average_loss"], 20.0))
        metrics["total_test_samples"] = total
        metrics["invalid_label_samples"] = sum(result.get("invalid_label_samples", 0) for result in metrics_by_client)
        metrics["fedsa_evaluation_clients"] = len(metrics_by_client)
        metrics["fedsa_evaluation_scope"] = "sample-weighted mean of personalized client metrics"
        self._obj.eval_results = metrics
