from __future__ import annotations
from typing import Dict, List, Any

from flex.fed_strategy.server_strategy import ServerStrategy
from flex.ml_utils import console


class RepuFLServerStrategy(ServerStrategy):
    """Server-side strategy to wire RepuFL selector feedback each round."""

    def __init__(self, args, server_node) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "repufl"
        self._obj = server_node

    def _create_inner(self, args, server_node) -> None:
        self._args = args
        self._strategy_type = "repufl"
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
        """Pass training feedback to RepuFL selector via with_clients_data."""
        self._obj.node_var.client_updates = client_updates

        selector = self._obj.node_var.client_selection
        if selector and hasattr(selector, "with_clients_data"):
            feedback_dict: Dict[str, Any] = {}
            for update in client_updates:
                record = update.get("train_record", {}) or {}
                node_id = record.get("node_id") or update.get("node_id")
                if not node_id:
                    continue

                trainer_stats = record.get("train_record", {}) if isinstance(record.get("train_record", {}), dict) else {}
                feedback = {**trainer_stats}
                feedback["latency"] = update.get("latency", record.get("latency", 1.0))
                feedback["data_sample_num"] = record.get("data_sample_num", trainer_stats.get("data_sample_num"))
                feedback["node_id"] = node_id
                feedback_dict[str(node_id)] = feedback

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
