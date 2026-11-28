from __future__ import annotations

import torch

from flex.fed_strategy.server_strategy import ServerStrategy
from flex.fed_strategy.strategy_args import StrategyArgs
from flex.ml_utils import console
from flex.ml_utils.model_utils import tensor_bytes
from flex.sfl.front_sync import FrontSynchronizer
from flex.sfl.rear import RearStateStore


class SplitServerStrategy(ServerStrategy):
    """
    Split learning server, owns the rear model and manages front broadcasts independently
    from the rear training state.

    forward(), backward() and step() run the rear on the activation of one batch.
    """

    def __init__(self, args: StrategyArgs, server_node) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "split"
        self._obj = server_node
        self.front_sync: FrontSynchronizer | None = None
        self.rear_state: RearStateStore | None = None

    def _create_inner(self, args: StrategyArgs, server_node) -> None:
        self._args = args
        self._obj = server_node

    def initialize(self, clients, protocol: dict) -> None:
        node = self._obj.node_var
        node.trainer.set_model(node.model.to(node.device))
        node.model_weight = node.model.state_dict()
        if node.optimizer is not None:
            node.optimizer = node.optimizer_builder.rebuild(node.model.parameters())
        node.trainer.set_optimizer(node.optimizer)
        self.front_sync = FrontSynchronizer(clients, protocol["front_sync"])
        self.rear_state = RearStateStore(node.model, node.optimizer,
                                         mode=protocol["rear_mode"],
                                         history=protocol.get("rear_optimizer_history", "shared"))
        full_model = getattr(node, "full_model", None)
        if full_model is not None and self.front_sync.method == "fedavg":
            # Like FL's first broadcast: the server's own model is the global start.
            state = full_model.state_dict()
            self.front_sync.factors = {name: state[name].detach().clone() for name in self.front_sync.factors}

    def forward(self, activation: torch.Tensor, labels: torch.Tensor, *, need_cut_grad: bool):
        return self._obj.node_var.trainer.forward(activation, labels, need_cut_grad)

    def backward(self, cut: torch.Tensor, loss: torch.Tensor):
        return self._obj.node_var.trainer.backward(cut, loss)

    def step(self, activation: torch.Tensor, labels: torch.Tensor, *, need_cut_grad: bool):
        return self._obj.node_var.trainer.step(activation, labels, need_cut_grad)

    def finish_round(self, updates: list) -> int:
        self.rear_state.commit_round()
        return self.front_sync.aggregate(updates)

    def broadcast_front(self, clients) -> dict:
        sent = {}
        for client in clients:
            payload = self.front_sync.payload_for(client.node_id)
            if payload:
                client.strategy.install_front(payload, self.front_sync.version)
                sent[client.node_id] = tensor_bytes(payload)
        return sent

    def aggregation(self):
        raise NotImplementedError("Use finish_round.")

    def broadcast(self):
        raise NotImplementedError("Use broadcast_front.")

    def run(self):
        raise NotImplementedError

    def evaluate(self) -> None:
        """
        Evaluate the global model: the aggregated front plus the current rear, which the
        server's full model shares.
        """
        node = self._obj.node_var
        incompatible = node.full_model.load_state_dict(self.front_sync.factors, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(f"The aggregated front does not fit the full model: {incompatible.unexpected_keys}")
        self._obj.eval_results = node.model_evaluator.evaluate()
        node.model_evaluator.print_results()
        console.info("Server Evaluation Completed.\n")

    def select_clients(self, available_clients) -> list:
        node = self._obj.node_var
        selector = getattr(node, "client_selection", None)
        if selector is None:
            return list(available_clients)
        return selector.select(available_clients, node.config_dict["client_selection"]["number"])

    def record_evaluation(self) -> None:
        self._obj.node_var.training_logger.record(self._obj.eval_results)

    def receive_client_updates(self, client_updates):
        raise NotImplementedError("Use finish_round.")

    def prepare(self, logger_header, client_nodes_in) -> None:
        self._obj.node_var.training_logger.begin(logger_header)
        self._obj.set_client_nodes(client_nodes_in)
