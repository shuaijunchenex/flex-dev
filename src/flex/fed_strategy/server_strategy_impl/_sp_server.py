from __future__ import annotations
from typing import Dict, List, Any, Optional
from collections import defaultdict

import torch
from flex.fed_strategy.server_strategy import ServerStrategy
from flex.fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from flex.fl_algorithms.selection.fed_client_selector_factory import FedClientSelectorFactory
from flex.model_trainer.model_evaluator import ModelEvaluator
from flex.ml_utils import console
from flex.ml_algorithms.lora.lora_utils import LoRAUtils

class SpServerStrategy(ServerStrategy):

    def __init__(self, args, server_node) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "sp"
        self._obj = server_node        

    def _create_inner(self, args, server_node) -> None:
        self._args = args
        self._strategy_type = "sp"
        self._obj = server_node
        return self

    def aggregation(self) -> dict:
        aggregator = self._obj.node_var.aggregation_method
        aggregated_weights = aggregator.aggregate(self._obj.node_var.client_updates) #TODO: check
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
    
    def apply_weight(self, mode = "regular"):
        """
        Apply aggregated weights to the server model evaluator.

        Parameters
        ----------
        mode : str
            ``"regular"`` (default) — one-time SVD factorization + cache, broadcast
            the compact factors, and materialize the server-eval weights by slicing
            (see :meth:`_apply_factored_weight`).
            ``"replace_w"`` — use ``convert_lora_for_sp_inference`` to substitute
            base weights with their SP-aggregated counterparts and zero-init LoRA
            A/B tensors (no SVD decomposition).
        """
        if mode == "replace_w":
            self._obj.node_var.model_weight = self._obj.node_var.aggregated_weight
            inference_weight = LoRAUtils.convert_lora_for_sp_inference(
                self._obj.node_var.aggregated_weight,
                self._obj.node_var.model_weight,
            )
            inference_weight = LoRAUtils.sort_state_dict_by_suffix(inference_weight)
            self._obj.node_var.model_evaluator.update_model(inference_weight)
        else:
            self._apply_factored_weight()
        return

    def _apply_factored_weight(self) -> None:
        """SP "regular" weight application via one-time SVD factorization.

        Steps:
        1. Factorize every ``{prefix}.sp_aggregated`` (full ΔW) **once** to the
           server rank ``r_cap`` and cache the sliceable factors on ``node_var``.
        2. Set ``model_weight`` to the compact factors so ``broadcast()`` ships
           factors (not the full ΔW) to clients.
        3. Materialize the server evaluation weights by slicing the cached factors
           to the server's own rank (no extra SVD).

        Falls back to the generic base pipeline when there are no ``sp_aggregated``
        keys (e.g. plain FedAvg) or no model is available for rank inference.
        """
        node_var = self._obj.node_var
        aggregated = getattr(node_var, "aggregated_weight", None)
        if aggregated is None:
            return

        inference_model = getattr(node_var, "inference_model", None) or getattr(node_var, "model", None)
        lora_cfg = getattr(inference_model, "lora_config", None) or {}
        sp_suffix = lora_cfg.get("sp_suffix", "sp_aggregated")
        suffix_A = lora_cfg.get("suffix_A", "lora_A")
        suffix_B = lora_cfg.get("suffix_B", "lora_B")

        server_ranks = (
            LoRAUtils.get_lora_ranks(inference_model, suffix_A, suffix_B)
            if inference_model is not None else {}
        )
        has_sp = any(k.endswith(f".{sp_suffix}") for k in aggregated)

        if not (has_sp and server_ranks):
            # No sp_aggregated layers (or no model) — keep the generic pipeline.
            super().apply_weight()
            return

        # r_cap = server rank (>= every client rank), so slicing is lossless.
        r_cap = max(server_ranks.values())
        factored = LoRAUtils.cache_svd_factored_matrix(
            node_var, aggregated, r_cap, sp_suffix=sp_suffix,
        )
        # Broadcast payload = compact factors (broadcast() sends model_weight).
        node_var.model_weight = factored
        # Server evaluation model: slice the cached factors to its own rank.
        prepared = LoRAUtils.materialize_lora_from_factors(
            factored, server_ranks,
            lora_suffix_A=suffix_A, lora_suffix_B=suffix_B,
        )
        node_var.model_evaluator.update_model(prepared)
        return

    def broadcast(self) -> None:
        for client in self._obj.client_nodes:
            client.receive_weight(self._obj.node_var.model_weight)
            client.set_local_weight()
            #client.node_var.model_weight = self._obj.node_var.model_weight
        return

    def run(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> None:
        self._obj.eval_results =  self._obj.node_var.model_evaluator.evaluate()
        self._obj.node_var.model_evaluator.print_results()
        console.info("Server Evaluation Completed.\n")

        return

    def prepare(self, logger_header, client_nodes_in) -> None:
        self._obj.node_var.training_logger.begin(logger_header)
        self._obj.set_client_nodes(client_nodes_in)
        return

    def run(self) -> Dict[str, Any]:
        raise NotImplementedError