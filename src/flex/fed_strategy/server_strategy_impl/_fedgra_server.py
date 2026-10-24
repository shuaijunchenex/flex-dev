from __future__ import annotations
from typing import Dict, List, Any, Optional

import torch
from flex.fed_strategy.server_strategy import ServerStrategy
from flex.fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from flex.fl_algorithms.selection.fed_client_selector_factory import FedClientSelectorFactory
from flex.model_trainer.model_evaluator import ModelEvaluator
from flex.ml_utils import console


class FedgraServerStrategy(ServerStrategy):
    """Server wiring for FedGRA: forwards client feedback into selector."""

    def __init__(self, args, server_node) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "fedgra"
        self._obj = server_node

    def _create_inner(self, args, server_node) -> None:
        self._args = args
        self._strategy_type = "fedgra"
        self._obj = server_node
        return self

    def aggregation(self) -> dict:
        aggregator = self._obj.node_var.aggregation_method
        aggregated_weights = aggregator.aggregate(self._obj.node_var.client_updates)
        self._obj.node_var.aggregated_weight = aggregated_weights
        return

    def select_clients(self, available_clients) -> list:
        selector = self._obj.node_var.client_selection
        selected_clients = selector.select(available_clients, self._obj.node_var.config_dict["client_selection"]["number"])
        return selected_clients

    def record_evaluation(self) -> None:
        self._obj.node_var.training_logger.record(self._obj.eval_results)
        return

    def receive_client_updates(self, client_updates: List[Dict[str, Any]]) -> None:
        """Capture latest client feedback so the FedGRA selector can rank clients."""
        self._obj.node_var.client_updates = client_updates

        selector = self._obj.node_var.client_selection
        if selector and hasattr(selector, "with_clients_data"):
            feedback_dict: Dict[str, Any] = {}
            for update in client_updates:
                record = update.get("train_record", {}) or {}
                if not isinstance(record, dict):
                    continue
                node_id = record.get("node_id") or update.get("node_id")
                if node_id is None:
                    continue
                # Unpack: trainer stats are nested inside record["train_record"]
                inner = record.get("train_record", {}) or {}
                feedback = dict(inner) if isinstance(inner, dict) else {}
                feedback["node_id"] = str(node_id)
                feedback["data_sample_num"] = record.get("data_sample_num", inner.get("data_sample_num"))
                feedback["latency"] = update.get("latency", 1.0)
                feedback_dict[str(node_id)] = feedback
            selector.with_clients_data(feedback_dict)

    def _flatten_weight_delta(self, local_weights: dict, global_weights: dict) -> Optional[torch.Tensor]:
        """Flatten (local - global) weight delta into a single 1D float tensor."""
        parts = []
        for key in sorted(set(local_weights.keys()) & set(global_weights.keys())):
            lw = local_weights[key]
            gw = global_weights[key]
            if isinstance(lw, torch.Tensor) and isinstance(gw, torch.Tensor):
                try:
                    parts.append((lw.detach().cpu().float() - gw.detach().cpu().float()).flatten())
                except Exception:
                    continue
        return torch.cat(parts) if parts else None

    def receive_client_updates_for_selection(self, client_updates) -> None:
        """Feed real-training metrics from ALL clients to selector (no aggregation).

        Extracts train_record, unwraps nested stats, and calls
        selector.with_clients_data() so the FedGRA selector has fresh
        per-client metrics before select_clients().

        Also computes weight_cosine_distance for each client: the cosine distance
        between that client's weight update direction and the mean update direction
        across all clients.  Unlike weight_l2_delta (magnitude), this direction-
        based metric retains high cross-client variance even when all clients train
        with similar budgets, making it a far more effective WD signal for EWM/GRA.
        """
        selector = getattr(self._obj.node_var, "client_selection", None)
        if selector is None or not hasattr(selector, "with_clients_data"):
            return

        feedback_dict: Dict[str, Any] = {}
        delta_vectors: Dict[str, torch.Tensor] = {}
        global_weights = getattr(self._obj.node_var, "model_weight", None)

        for item in client_updates:
            if not isinstance(item, dict):
                continue
            record = item.get("train_record", {}) or {}
            if not isinstance(record, dict):
                continue
            node_id = record.get("node_id") or item.get("node_id")
            if node_id is None:
                continue
            # Unpack: trainer stats are in record["train_record"]
            inner = record.get("train_record", {}) or {}
            feedback = dict(inner) if isinstance(inner, dict) else {}
            feedback["node_id"] = str(node_id)
            feedback["data_sample_num"] = record.get("data_sample_num", inner.get("data_sample_num"))
            feedback["latency"] = item.get("latency", 1.0)
            feedback_dict[str(node_id)] = feedback

            # Collect weight-update vector for cosine-distance computation
            updated_weights = item.get("updated_weights")
            if global_weights is not None and isinstance(updated_weights, dict):
                delta = self._flatten_weight_delta(updated_weights, global_weights)
                if delta is not None:
                    delta_vectors[str(node_id)] = delta

        # Compute cosine distance of each client's update from the mean update.
        # High value → client is pulling the model in a unique direction → high diversity.
        if len(delta_vectors) >= 2:
            stacked = torch.stack(list(delta_vectors.values()))   # (N, D)
            mean_delta = stacked.mean(dim=0)                      # (D,)
            mean_norm = torch.norm(mean_delta)
            for cid, delta in delta_vectors.items():
                delta_norm = torch.norm(delta)
                if delta_norm > 1e-12 and mean_norm > 1e-12:
                    cos_sim = float(torch.dot(delta, mean_delta) / (delta_norm * mean_norm))
                    cosine_dist = 1.0 - cos_sim
                else:
                    cosine_dist = 0.0
                if cid in feedback_dict:
                    feedback_dict[cid]["weight_cosine_distance"] = cosine_dist

        if feedback_dict:
            selector.with_clients_data(feedback_dict)

    def broadcast(self) -> None:
        for client in self._obj.client_nodes:
            client.receive_weight(self._obj.node_var.model_weight)
            client.set_local_weight()
        return

    def run(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> None:
        self._obj.eval_results = self._obj.node_var.model_evaluator.evaluate()
        self._obj.node_var.model_evaluator.print_results()
        console.info("Server Evaluation Completed.\n")
        return

    def prepare(self, logger_header, client_nodes_in) -> None:
        self._obj.node_var.training_logger.begin(logger_header)
        self._obj.set_client_nodes(client_nodes_in)
        return

    def run(self) -> Dict[str, Any]:
        raise NotImplementedError
