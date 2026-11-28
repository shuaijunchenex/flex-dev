from __future__ import annotations
from typing import Dict, List, Any

import torch
from flex.fed_strategy.server_strategy import ServerStrategy
from flex.fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from flex.fl_algorithms.selection.fed_client_selector_factory import FedClientSelectorFactory
from flex.model_trainer.model_evaluator import ModelEvaluator
from flex.ml_utils import console


class PyramidFLServerStrategy(ServerStrategy):
    """Server-side PyramidFL strategy; mirrors Oort server wiring but targets PyramidFL selector."""

    def __init__(self, args, server_node) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "pyramidfl"
        self._obj = server_node

    def _create_inner(self, args, server_node) -> None:
        self._args = args
        self._strategy_type = "pyramidfl"
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
        """Pass client feedback to the PyramidFL selector."""
        self._obj.node_var.client_updates = client_updates

        selector = self._obj.node_var.client_selection
        if selector and hasattr(selector, "with_clients_data"):
            feedback_dict = {}
            for update in client_updates:
                # Structure: update = {"updated_weights": ..., "train_record": {...}, "latency": ...}
                # train_record = {"node_id": ..., "train_record": {...trainer stats...}, "data_sample_num": ...}
                record = update.get("train_record", {})
                node_id = record.get("node_id")

                if node_id is not None:
                    # Extract the actual trainer statistics from the nested train_record
                    trainer_stats = record.get("train_record", {})

                    # Combine trainer stats with other metadata.
                    # shard_keep_ratio is reported back by the client so the server can
                    # use it to compute the global utility in the next round.
                    feedback_dict[str(node_id)] = {
                        **trainer_stats,  # epoch_loss, avg_loss, weight_l2_delta, etc.
                        "latency": update.get("latency", 1.0),
                        "data_sample_num": record.get("data_sample_num", trainer_stats.get("data_sample_num")),
                        # Client reports the shard_keep_ratio it actually used this round
                        "shard_keep_ratio": trainer_stats.get("shard_keep_ratio", record.get("shard_keep_ratio", 1.0)),
                        "node_id": node_id,
                    }
            selector.with_clients_data(feedback_dict)

    def broadcast(self) -> None:
        selector = self._obj.node_var.client_selection
        # Retrieve per-client local parameters computed by the selector after selection
        client_local_params: dict = {}
        if selector and hasattr(selector, "_client_local_params"):
            client_local_params = selector._client_local_params or {}

        for client in self._obj.client_nodes:
            cid = str(client.node_id)
            local_params = client_local_params.get(cid, {})
            client.receive_weight(self._obj.node_var.model_weight)
            client.set_local_weight()
            # Deliver PyramidFL per-client optimisation parameters (shard_keep_ratio, adaptive_iter)
            # to the client strategy directly (FedNode is an abstract node; strategy holds the logic)
            if local_params and hasattr(client.strategy, "receive_pyramidfl_params"):
                client.strategy.receive_pyramidfl_params(local_params)
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
