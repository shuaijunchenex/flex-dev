from __future__ import annotations

import torch

from flex.fed_strategy.client_strategy import ClientStrategy
from flex.fed_strategy.strategy_args import StrategyArgs
from flex.ml_utils.model_utils import ModelUtils
from flex.sfl.client_update import make_client_update


class SplitClientStrategy(ClientStrategy):
    """
    Split learning client, owns the front model and parks its model and optimizer state between visits.

    A visit is driven by the runner: begin_visit(), then forward() and backward() for every
    batch, then end_visit() which returns the update for the front synchronization.
    """

    def __init__(self, args: StrategyArgs, client_node) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "split"
        self._obj = client_node
        self.front_version = 0
        self._train_front = any(p.requires_grad for p in client_node.node_var.model.parameters())

    def _create_inner(self, args: StrategyArgs, client_node) -> None:
        self._args = args
        self._obj = client_node

    @property
    def needs_cut_grad(self) -> bool:
        """
        A front without trainable parameters is frozen and needs no cut gradient.
        """
        return self._train_front

    @property
    def preserve_optimizer_state(self) -> bool:
        """
        Keep the optimizer state across front broadcasts, like FedAvg's preserve_optimizer_state.
        """
        return bool(getattr(self._args, "preserve_optimizer_state", False))

    def begin_visit(self) -> None:
        node = self._obj.node_var
        node.trainer.set_model(node.model.to(node.device))
        if self.needs_cut_grad:
            node.optimizer = node.optimizer_builder.rebuild(node.model.parameters())
            if node.persistent_optimizer_state is not None:
                node.optimizer.load_state_dict(node.persistent_optimizer_state)
        node.trainer.set_optimizer(node.optimizer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._obj.node_var.trainer.forward(inputs, need_cut_grad=self.needs_cut_grad)

    def backward(self, cut_grad: torch.Tensor) -> None:
        self._obj.node_var.trainer.backward(cut_grad)

    def export_front(self) -> dict:
        return {name: p.detach().to("cpu", copy=True)
                for name, p in self._obj.node_var.model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def install_front(self, factors: dict, version: int) -> None:
        node = self._obj.node_var
        params = {name: p for name, p in node.model.named_parameters() if p.requires_grad}
        if params.keys() != factors.keys():
            raise ValueError("The broadcast must contain exactly this front's adapter parameters")
        for name, value in factors.items():
            params[name].copy_(value)
        node.model_weight = node.model.state_dict()
        if not self.preserve_optimizer_state:
            node.persistent_optimizer_state = None
        self.front_version = version

    def end_visit(self, train_record: dict) -> dict:
        node = self._obj.node_var
        if node.optimizer is not None:
            node.persistent_optimizer_state = ModelUtils.snapshot_optimizer_state(node.optimizer)
            node.optimizer = None
            node.trainer.set_optimizer(None)
        node.model.to("cpu")
        node.model_weight = node.model.state_dict()
        return make_client_update(self._obj.node_id, self.export_front(), train_record, node.data_sample_num,
                                  base_version=self.front_version)

    def run_observation(self):
        raise NotImplementedError("The split client has no local observation.")

    def observation_step(self):
        raise NotImplementedError("The split client has no local observation.")

    def run_local_training(self):
        raise NotImplementedError("Use begin_visit, forward, backward and end_visit.")

    def local_training_step(self):
        raise NotImplementedError("Use begin_visit, forward, backward and end_visit.")
