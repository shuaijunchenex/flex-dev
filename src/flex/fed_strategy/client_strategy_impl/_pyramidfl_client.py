import copy
import torch
from typing import Any, Tuple
import torch.nn as nn

from flex.fed_strategy.strategy_args import StrategyArgs
from ..client_strategy import ClientStrategy
from ...ml_utils.model_utils import ModelUtils
from ...model_trainer import model_trainer_factory
from ...model_trainer.model_trainer_args import ModelTrainerArgs
from ...model_trainer.model_trainer_factory import ModelTrainerFactory
from ...ml_algorithms.optimizer_builder import OptimizerBuilder
from ...ml_data_loader import DatasetLoaderFactory
from ...ml_algorithms.loss_function_builder import LossFunctionBuilder
from ...ml_utils import console
from ...fed_node.fed_node_vars import FedNodeVars


class PyramidFLClientTrainingStrategy(ClientStrategy):
    """Client training wrapper for PyramidFL (mirrors Oort client flow)."""

    def __init__(self, args, client_node):
        super().__init__()
        self._args = args
        self._strategy_type = "pyramidfl"
        self._obj = client_node
        # Per-client parameters sent by the server after selection (Algorithm 1 Lines 14-17)
        self._pyramidfl_params: dict = {}

    def _create_inner(self, args, client_node) -> None:
        self._args = args
        self._strategy_type = "pyramidfl"
        self._obj = client_node
        return

    def receive_pyramidfl_params(self, params: dict) -> None:
        """
        Receive per-client optimisation parameters from the PyramidFL server.

        Called by the server during broadcast after selection. The selector computes:
            - shard_keep_ratio (P_i ∈ [a, b]): fraction of local data shards to use.
            - adaptive_iter (I_i): number of local training iterations for this round.
        """
        self._pyramidfl_params = params or {}

    def run_observation(self) -> dict:
        print(f"\n Observation Client [{self._obj.node_id}] ...\n")
        _, train_record = self.observation_step()
        return {
            "node_id": self._obj.node_id,
            "train_record": train_record,
            "data_sample_num": self._obj.node_var.data_sample_num,
        }

    def observation_step(self) -> Tuple[dict, Any]:
        node_vars: FedNodeVars = self._obj.node_var
        cfg: dict = self._obj.node_var.config_dict
        device = node_vars.device if hasattr(node_vars, "device") and node_vars.device else "cpu"

        observe_model: nn.Module = copy.deepcopy(node_vars.model).to(device)
        observe_model.load_state_dict(node_vars.model_weight, strict=True)
        optimizer = self._obj.node_var.optimizer_builder.rebuild(observe_model.parameters())

        ModelUtils.clear_all(observe_model, optimizer)

        tr = self._obj.node_var.trainer
        orig_model = tr.trainer_args.model
        orig_optimizer = tr.trainer_args.optimizer
        orig_device = getattr(tr.trainer_args, "device", None)

        try:
            tr.set_model(observe_model)
            tr.set_optimizer(optimizer)
            tr.trainer_args.device = device

            local_epochs = int(cfg.get("training", {}).get("local_epochs", 1))
            updated_weights, train_record = self.train_and_offload(tr, local_epochs)
        finally:
            if orig_model is not None:
                tr.set_model(orig_model)
            if orig_optimizer is not None:
                tr.set_optimizer(orig_optimizer)
            if orig_device is not None:
                tr.trainer_args.device = orig_device
            self.cleanup_training_resources(
                model=observe_model,
                optimizer=optimizer,
            )

        return updated_weights, train_record

    def run_local_training(self) -> dict:
        updated_weights, train_record = self.local_training_step()
        # Report back the shard_keep_ratio used so the server can update the selector state
        shard_keep_ratio = self._pyramidfl_params.get("shard_keep_ratio", 1.0)
        return updated_weights, {
            "node_id": self._obj.node_id,
            "updated_weights": updated_weights,
            "train_record": {
                **train_record,
                "shard_keep_ratio": shard_keep_ratio,
            },
            "data_sample_num": self._obj.node_var.data_sample_num,
        }

    def local_training_step(self) -> Tuple[dict, Any]:
        node_vars: FedNodeVars = self._obj.node_var
        cfg: dict = node_vars.config_dict
        device = getattr(node_vars, "device", None) or "cpu"

        training_model: nn.Module = copy.deepcopy(node_vars.model).to(device)
        training_model.load_state_dict(node_vars.model_weight, strict=True)

        optimizer = node_vars.optimizer_builder.rebuild(training_model.parameters())

        ModelUtils.clear_all(training_model, optimizer)

        tr = node_vars.trainer
        tr.set_model(training_model)
        tr.set_optimizer(optimizer)
        tr.trainer_args.device = device

        # --- Epoch count resolution ---
        # 1. Start from config value.
        config_epochs = int(cfg.get("training", {}).get("epochs", 1))

        # 2. Read strategy-level control params from config_dict["strategy"].
        strategy_cfg: dict = cfg.get("strategy", {})
        disable_adaptive_epochs: bool = bool(strategy_cfg.get("disable_adaptive_epochs", False))
        max_epoch_ratio = strategy_cfg.get("max_epoch_ratio", None)  # None means no cap

        # 3. Apply adaptive_iter from PyramidFL server (Algorithm 1 Line 16-17)
        #    unless the feature is explicitly disabled.
        adaptive_iter = self._pyramidfl_params.get("adaptive_iter", None)
        if (not disable_adaptive_epochs) and (adaptive_iter is not None) and (adaptive_iter > 0):
            local_epochs = int(adaptive_iter)
        else:
            local_epochs = config_epochs

        # 4. Apply max_epoch_ratio cap if set:
        #    local_epochs ≤ floor(config_epochs * max_epoch_ratio)
        if max_epoch_ratio is not None and float(max_epoch_ratio) > 0:
            max_epochs = int(float(max_epoch_ratio) * config_epochs)
            local_epochs = min(local_epochs, max(1, max_epochs))

        try:
            updated_weights, train_record = self.train_and_offload(tr, local_epochs)
        finally:
            self.cleanup_training_resources(
                model=training_model,
                optimizer=optimizer,
                trainer=tr,
            )

        node_vars.model_weight = updated_weights
        return updated_weights, train_record

    def receive_weight(self, global_weight) -> dict:
        self._obj.node_var.cache_weight = global_weight
        return

    def set_local_weight(self) -> dict:
        self._obj.node_var.model_weight = self._obj.node_var.cache_weight
        return
