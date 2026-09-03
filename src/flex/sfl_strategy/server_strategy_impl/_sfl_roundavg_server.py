from __future__ import annotations

import copy
from typing import Any

import torch

from flex.fed_strategy.strategy_args import StrategyArgs
from flex.ml_utils.model_utils import ModelUtils
from flex.sfl_strategy.server_strategy_impl._sfl_server_example import SflServerStrategy, _SflFullMlp


class SflRoundAvgServerStrategy(SflServerStrategy):
    def __init__(self, args: StrategyArgs, server_node: Any) -> None:
        super().__init__(args, server_node)
        self._strategy_type = "sfl_roundavg"
        self._round_server_weight_snapshot = None
        self._round_optimizer_state_snapshot = None
        self._initial_weights_aligned = False

    def initialize_shared_roundavg_state(self, client_nodes) -> None:
        if self._initial_weights_aligned:
            return

        node_vars = self._obj.node_var
        with torch.random.fork_rng():
            torch.manual_seed(torch.initial_seed())
            full_model = _SflFullMlp()
        full_state = copy.deepcopy(full_model.state_dict())

        server_state_template = ModelUtils.unwrap_model(node_vars.model).state_dict()
        server_state = {
            key: full_state[key].detach().clone()
            for key in server_state_template.keys()
            if key in full_state
        }
        node_vars.model.load_state_dict(server_state, strict=True)
        node_vars.model_weight = copy.deepcopy(server_state)
        node_vars.aggregated_server_weight = copy.deepcopy(server_state)

        optimizer = getattr(node_vars, "optimizer", None)
        if optimizer is not None:
            ModelUtils.reset_optimizer_state(optimizer)

        front_state = None
        for client in client_nodes or []:
            client_model = getattr(client.node_var, "model", None)
            if client_model is None:
                continue

            client_state_template = client_model.state_dict()
            client_state = {
                key: full_state[key].detach().clone()
                for key in client_state_template.keys()
                if key in full_state
            }
            client_model.load_state_dict(client_state, strict=True)
            client.node_var.model_weight = copy.deepcopy(client_state)
            client.node_var.cache_weight = copy.deepcopy(client_state)
            if front_state is None:
                front_state = copy.deepcopy(client_state)

        if front_state is not None:
            node_vars.client_front_weights = [copy.deepcopy(front_state)]
            node_vars.aggregated_front_weight = copy.deepcopy(front_state)

        evaluator = getattr(node_vars, "model_evaluator", None)
        if evaluator is not None:
            evaluator.change_model(full_model, full_state)

        self._initial_weights_aligned = True

    def begin_round_training(self) -> None:
        '''
        Takes a snapshot of the server model weights and optimizer state at the beginning of each round.
        This snapshot will be used to initialize client models and optimizers before local training.'''
        node_vars = self._obj.node_var
        self._round_server_weight_snapshot = copy.deepcopy(ModelUtils.unwrap_model(node_vars.model).state_dict())
        optimizer = getattr(node_vars, "optimizer", None)
        self._round_optimizer_state_snapshot = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None

    def prepare_client_training(self) -> None:
        node_vars = self._obj.node_var
        if self._round_server_weight_snapshot is None:
            self.begin_round_training()

        node_vars.model.load_state_dict(self._round_server_weight_snapshot, strict=True)
        node_vars.model_weight = copy.deepcopy(self._round_server_weight_snapshot)

        optimizer = getattr(node_vars, "optimizer", None)
        if optimizer is not None and self._round_optimizer_state_snapshot is not None:
            optimizer.load_state_dict(copy.deepcopy(self._round_optimizer_state_snapshot))

    def finish_client_training(self):
        return copy.deepcopy(ModelUtils.unwrap_model(self._obj.node_var.model).state_dict())

    def aggregation(self) -> None:
        node_var = self._obj.node_var
        updates = getattr(node_var, "client_updates", []) or []

        front_weights = []
        server_weights = []
        sample_counts = []

        for update in updates:
            if not isinstance(update, dict):
                continue

            front_weight = update.get("front_weight")
            server_weight = update.get("server_weight")
            if front_weight is None or server_weight is None:
                continue

            front_weights.append(front_weight)
            server_weights.append(server_weight)
            sample_counts.append(float(update.get("data_sample_num", 1)))

        if not front_weights:
            raise ValueError("SFL round-average aggregation requires client front weights.")
        if not server_weights:
            raise ValueError("SFL round-average aggregation requires server-side weights.")

        total_samples = sum(sample_counts) if sample_counts else float(len(front_weights))
        node_var.client_front_weights = [self._weighted_average(front_weights, sample_counts, total_samples)]
        node_var.aggregated_front_weight = node_var.client_front_weights[0]
        node_var.aggregated_server_weight = self._weighted_average(server_weights, sample_counts, total_samples)

    @staticmethod
    def _weighted_average(weight_list, sample_counts, total_samples):
        agg_state = None
        effective_counts = sample_counts or [1.0] * len(weight_list)

        for weight, count in zip(weight_list, effective_counts):
            scale = count / total_samples
            if agg_state is None:
                agg_state = {k: v.detach().clone() * scale for k, v in weight.items()}
            else:
                for key, value in weight.items():
                    agg_state[key] += value.detach().clone() * scale

        return agg_state

    def broadcast(self) -> None:
        weight = getattr(self._obj.node_var, "aggregated_front_weight", None)
        if weight is None:
            return
        for client in getattr(self._obj, "client_nodes", []):
            client.receive_weight(weight)
            client.set_local_weight()

    def apply_weight(self) -> None:
        node_vars = self._obj.node_var
        aggregated_server_weight = getattr(node_vars, "aggregated_server_weight", None)
        if aggregated_server_weight is None:
            return

        node_vars.model.load_state_dict(aggregated_server_weight, strict=True)
        node_vars.model_weight = copy.deepcopy(aggregated_server_weight)

        optimizer = getattr(node_vars, "optimizer", None)
        if optimizer is not None:
            ModelUtils.reset_optimizer_state(optimizer)

        evaluator = getattr(node_vars, "model_evaluator", None)
        if evaluator is not None:
            full_model, full_state = self._compose_full_model_for_eval(node_vars)
            evaluator.change_model(full_model, full_state)
