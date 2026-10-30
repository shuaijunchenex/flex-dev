from __future__ import annotations
from typing import Dict, List, Any, Optional
from collections import defaultdict

import torch
from flex.fed_strategy.server_strategy import ServerStrategy
from flex.fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from flex.fl_algorithms.selection.fed_client_selector_factory import FedClientSelectorFactory
from flex.model_trainer.model_evaluator import ModelEvaluator
from flex.ml_utils import console

class FedAvgServerStrategy(ServerStrategy):

    def __init__(self, args, server_node) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "fedavg"
        self._obj = server_node

    def _create_inner(self, args, server_node) -> None:
        self._args = args
        self._strategy_type = "fedavg"
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

    def receive_client_updates(self, client_updates) -> None:
        self._obj.node_var.client_updates = client_updates #{client1: {weight:"", data_vol:""}, client2: {weight:"", data_vol:""}}

    def receive_client_updates_for_selection(self, client_updates) -> None:
        """Feed client update metrics to the selector without touching client_updates.

        Extracts train_record from each update item and calls
        selector.with_clients_data() so that selectors that depend on
        per-client statistics (e.g. MidLoss, HighLoss) have fresh data
        available before the next select_clients() call.
        """
        selector = getattr(self._obj.node_var, "client_selection", None)
        if selector is None or not hasattr(selector, "with_clients_data"):
            return

        feedback_dict: dict = {}
        for item in client_updates:
            if not isinstance(item, dict):
                continue
            train_rec = item.get("train_record", {})
            # observation_step returns (weights, record) tuple — unwrap if needed.
            if isinstance(train_rec, tuple) and len(train_rec) == 2:
                train_rec = train_rec[1]
            if not isinstance(train_rec, dict):
                continue
            node_id = train_rec.get("node_id") or item.get("node_id")
            if node_id is None:
                continue
            feedback_dict[str(node_id)] = {
                "train_record": train_rec,
                "latency": item.get("latency", 1.0),
                "data_sample_num": train_rec.get("data_sample_num"),
            }

        if feedback_dict:
            selector.with_clients_data(feedback_dict)

    def broadcast(self) -> None:
        for client in self._obj.client_nodes:
            client.receive_weight(self._obj.node_var.model_weight)
            client.set_local_weight()
        return

    def run(self) -> None:
        raise NotImplementedError

    def evaluate(self, pipeline_test: bool | None = None) -> None:
        """
        Evaluate the global model on the server's validation dataset.

        :param pipeline_test: When True, runs evaluation on a single sample only
                              (pipeline verification mode).  When None (default),
                              auto-detects based on trainer type (glue_test /
                              pipeline_test → single sample).
        """
        # ── Auto-detect pipeline_test from trainer type ──────────────────
        if pipeline_test is None:
            trainer = getattr(self._obj.node_var, "trainer", None)
            trainer_type = ""
            if trainer is not None:
                trainer_args = getattr(trainer, "trainer_args", None)
                if trainer_args is not None:
                    trainer_type = getattr(trainer_args, "trainer_type", "") or ""
                if not trainer_type:
                    trainer_type = getattr(trainer, "trainer_type", "") or ""
            pipeline_test = trainer_type in ("glue_test", "pipeline_test")

        self._obj.eval_results = self._obj.node_var.model_evaluator.evaluate(
            pipeline_test=pipeline_test
        )
        self._obj.node_var.model_evaluator.print_results()
        console.info("Server Evaluation Completed.\n")

        return

    def prepare(self, logger_header, client_nodes_in) -> None:
        self._obj.node_var.training_logger.begin(logger_header)
        self._obj.set_client_nodes(client_nodes_in)
        return

    def run(self) -> Dict[str, Any]:
        raise NotImplementedError