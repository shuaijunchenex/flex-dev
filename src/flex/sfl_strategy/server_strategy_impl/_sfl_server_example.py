from __future__ import annotations

from typing import Any, Iterable

import torch.nn as nn
import torch.nn.functional as F

from flex.fed_strategy.server_strategy import ServerStrategy
from flex.fed_strategy.strategy_args import StrategyArgs
from flex.ml_utils import console
from flex.ml_utils.model_utils import ModelUtils


class _SflFullMlp(nn.Module):
    """Simple end-to-end MLP to stitch client and server parts for evaluation."""

    def __init__(self) -> None:
        super().__init__()
        self._flatten = nn.Flatten()
        self._activation = nn.ReLU()
        self._fc1 = nn.Linear(784, 200)
        self._fc2 = nn.Linear(200, 200)
        self._fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = self._flatten(x)
        x = self._activation(self._fc1(x))
        x = self._activation(self._fc2(x))
        x = self._fc3(x)
        return F.softmax(x, dim=1)


class SflServerStrategy(ServerStrategy):
    def __init__(self, args: StrategyArgs, server_node: Any) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "sfl"
        self._obj = server_node

    def _create_inner(self, args: StrategyArgs, server_node: Any) -> None:
        self._args = args
        self._obj = server_node
        return

    def aggregation(self) -> None:
        node_var = self._obj.node_var
        updates = getattr(node_var, "client_updates", []) or []

        front_weights = []
        sample_counts = []
        for u in updates:
            if not isinstance(u, dict):
                continue
            fw = u.get("front_weight")
            if fw is None:
                continue
            front_weights.append(fw)
            sample_counts.append(float(u.get("data_sample_num", 1)))

        if not front_weights:
            raise ValueError("SFL aggregation requires client front weights.")

        total_samples = sum(sample_counts) if sample_counts else len(front_weights)
        agg_state = None
        for fw, count in zip(front_weights, sample_counts or [1.0] * len(front_weights)):
            if agg_state is None:
                agg_state = {k: v.detach().clone() * (count / total_samples) for k, v in fw.items()}
            else:
                for k, v in fw.items():
                    agg_state[k] += v.detach().clone() * (count / total_samples)

        node_var.client_front_weights = [agg_state]
        node_var.aggregated_weight = agg_state

    def broadcast(self) -> None:
        weight = getattr(self._obj.node_var, "aggregated_weight", None)
        if weight is None:
            return
        for client in getattr(self._obj, "client_nodes", []):
            client.receive_weight(weight)
            client.set_local_weight()

    def run(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> None:
        evaluator = getattr(self._obj.node_var, "model_evaluator", None)
        if evaluator is None:
            return
        self._obj.eval_results = evaluator.evaluate()
        evaluator.print_results()
        console.info("Server Evaluation Completed.\n")

    def select_clients(self, available_clients) -> list:
        selector = getattr(self._obj.node_var, "client_selection", None)
        if selector is None:
            return list(available_clients)
        config = self._obj.node_var.config_dict.get("client_selection", {})
        number = config.get("number", len(available_clients))
        return selector.select(available_clients, number)

    def record_evaluation(self) -> None:
        logger = getattr(self._obj.node_var, "training_logger", None)
        if logger is not None and hasattr(self._obj, "eval_results"):
            logger.record(self._obj.eval_results)

    def receive_client_updates(self, client_updates) -> None:
        self._obj.node_var.client_updates = client_updates

    def prepare(self, logger_header, client_nodes_in) -> None:
        logger = getattr(self._obj.node_var, "training_logger", None)
        if logger is not None:
            logger.begin(logger_header)
        self._obj.set_client_nodes(client_nodes_in)

    def apply_weight(self) -> None:
        node_vars = self._obj.node_var
        node_vars.model_weight = node_vars.aggregated_weight
        evaluator = getattr(node_vars, "model_evaluator", None)
        if evaluator is not None:
            full_model, full_state = self._compose_full_model_for_eval(node_vars)
            evaluator.change_model(full_model, full_state)

    def train_on_smashed(self, smashed_data, labels, training: bool = True):
        node_vars = self._obj.node_var
        trainer = getattr(node_vars, "trainer", None)
        if trainer is None:
            raise RuntimeError("Server trainer is not prepared for SFL.")

        # Ensure trainer uses current server model/optimizer/loss
        trainer.set_model(node_vars.model)
        trainer.set_optimizer(node_vars.optimizer)
        trainer.trainer_args.loss_func = node_vars.loss_func
        trainer.trainer_args.device = getattr(node_vars, "device", "cpu")

        activation_grad, loss_value, metrics = trainer.train_step_from_remote(smashed_data, labels)
        node_vars.model_weight = ModelUtils.unwrap_model(node_vars.model).state_dict()
        return activation_grad, loss_value, metrics

    # New: explicit server forward pass on smashed data
    def run_server_forward(self, smashed_data, labels, training: bool = True):
        node_vars = self._obj.node_var
        trainer = getattr(node_vars, "trainer", None)
        if trainer is None:
            raise RuntimeError("Server trainer is not prepared for SFL.")

        trainer.set_model(node_vars.model)
        trainer.set_optimizer(node_vars.optimizer)
        trainer.trainer_args.loss_func = node_vars.loss_func
        trainer.trainer_args.device = getattr(node_vars, "device", "cpu")

        server_input, loss, metrics = trainer.forward_only(smashed_data, labels, training=training)
        return server_input, loss, metrics

    # New: explicit server backward to produce activation gradients
    def run_server_backward(self, server_input, loss, training: bool = True):
        node_vars = self._obj.node_var
        trainer = getattr(node_vars, "trainer", None)
        if trainer is None:
            raise RuntimeError("Server trainer is not prepared for SFL.")

        trainer.set_model(node_vars.model)
        trainer.set_optimizer(node_vars.optimizer)
        trainer.trainer_args.loss_func = node_vars.loss_func
        trainer.trainer_args.device = getattr(node_vars, "device", "cpu")

        activation_grad, loss_value = trainer.backward_only(server_input, loss, training=training)
        latest_state = ModelUtils.unwrap_model(node_vars.model).state_dict()
        node_vars.model_weight = latest_state
        return activation_grad, loss_value

    def _compose_full_model_for_eval(self, node_vars):
        client_front_list = getattr(node_vars, "client_front_weights", []) or []
        client_weight = client_front_list[0] if client_front_list else {}
        server_model = getattr(node_vars, "model", None)
        server_state = server_model.state_dict() if server_model is not None else {}

        full_model = _SflFullMlp()
        full_state = full_model.state_dict()

        # Inject client-side layer weights if present
        for key in ("_fc1.weight", "_fc1.bias"):
            if key in client_weight:
                full_state[key] = client_weight[key].detach().clone()
            elif key in server_state:
                full_state[key] = server_state[key].detach().clone()

        # Keep server-side layers from the server model
        for key in ("_fc2.weight", "_fc2.bias", "_fc3.weight", "_fc3.bias"):
            if key in server_state:
                full_state[key] = server_state[key].detach().clone()

        full_model.load_state_dict(full_state, strict=True)
        return full_model, full_state

