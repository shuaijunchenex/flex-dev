from __future__ import annotations
from typing import Dict, List, Any, Optional
from collections import defaultdict

import torch
from flex.fed_strategy.server_strategy import ServerStrategy
from flex.fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from flex.fl_algorithms.selection.fed_client_selector_factory import FedClientSelectorFactory
from flex.model_trainer.model_evaluator import ModelEvaluator
from flex.ml_utils import console

class OortServerStrategy(ServerStrategy):

    def __init__(self, args, server_node) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "oort"
        self._obj = server_node

    def _create_inner(self, args, server_node) -> None:
        self._args = args
        self._strategy_type = "oort"
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

    def record_evaluation(self)-> None:
        self._obj.node_var.training_logger.record(self._obj.eval_results)
        return

    def receive_client_updates(self, client_updates: List[Dict[str, Any]]) -> None:
        """
        Receive updates from clients and pass feedback to the Oort selector.
        """
        self._obj.node_var.client_updates = client_updates
        
        # Oort Feedback: Pass performance metrics to the selector
        selector = self._obj.node_var.client_selection
        if selector and hasattr(selector, "with_clients_data"):
            feedback_dict = {}
            for update in client_updates:
                record = update.get("train_record", {})
                node_id = record.get("node_id")
                if node_id:
                    # Oort needs loss/latency to calculate utility
                    feedback_dict[str(node_id)] = {
                        **record, 
                        "latency": update.get("latency", 1.0)
                    }
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