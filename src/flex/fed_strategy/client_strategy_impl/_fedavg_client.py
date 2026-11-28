import copy
import torch
import torch.nn as nn
from typing import Any, Tuple

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

class FedAvgClientTrainingStrategy(ClientStrategy):
    def __init__(self, args, client_node):
        """
        client: a FedNodeClient (or FedNode) that owns a FedNodeVars in `client.node_var`
        config: high-level strategy/trainer config; falls back to `client.node_var.config_dict` when needed
        """
        super().__init__()
        self._args = args
        self._strategy_type = "fedavg"
        self._obj = client_node

    def _create_inner(self, args, client_node) -> None:
        self._args = args
        self._strategy_type = "fedavg"
        self._obj = client_node
        return

    # ------------------- Public: Observation wrapper -------------------
    def run_observation(self) -> dict:
        print(f"\n Observation Client [{self._obj.node_id}] ...\n")
        _, train_record = self.observation_step()
        return {
            "node_id": self._obj.node_id,
            "train_record": train_record,
            "data_sample_num": self._obj.node_var.data_sample_num,
        }

    # ------------------- Observation (no state write-back) -------------------
    def observation_step(self) -> Tuple[dict, Any]:

        node_vars: FedNodeVars = self._obj.node_var
        cfg: dict = self._obj.node_var.config_dict
        device = node_vars.device if hasattr(node_vars, "device") and node_vars.device else "cpu"

        observe_model: nn.Module = copy.deepcopy(node_vars.model).to(device)
        observe_model.load_state_dict(node_vars.model_weight, strict=True)
        optimizer = self._obj.node_var.optimizer_builder.rebuild(observe_model.parameters())

        ModelUtils.clear_all(observe_model, optimizer)

        # Bind trainer to the observation model/optimizer so weight updates are applied to the copy.
        tr = self._obj.node_var.trainer

        # Preserve original trainer bindings to avoid side effects on subsequent training.
        orig_model = tr.trainer_args.model
        orig_optimizer = tr.trainer_args.optimizer
        orig_device = getattr(tr.trainer_args, "device", None)

        try:
            tr.set_model(observe_model)
            tr.set_optimizer(optimizer)
            tr.trainer_args.device = device

            local_epochs = 1  # observation: quick 1-epoch probe for selector metrics
            updated_weights, train_record = self.train_and_offload(tr, local_epochs)
        finally:
            # Restore original trainer bindings.
            if orig_model is not None:
                tr.set_model(orig_model)
            if orig_optimizer is not None:
                tr.set_optimizer(orig_optimizer)
            if orig_device is not None:
                tr.trainer_args.device = orig_device
            # ── Release observation model GPU memory ──────────────────
            self.cleanup_training_resources(
                model=observe_model,
                optimizer=optimizer,
            )

        return updated_weights, train_record

    # ------------------- Public: Local training wrapper -------------------
    def run_local_training(self) -> dict:
        updated_weights, train_record = self.local_training_step()
        self._log_training_complete(train_record)
        return updated_weights, {
            "node_id": self._obj.node_id,
            "updated_weights": updated_weights,
            "train_record": train_record,
            "data_sample_num": self._obj.node_var.data_sample_num}

    # ------------------- Full local training (write-back to node_var) -------------------
    def local_training_step(self) -> Tuple[dict, Any]:
        node_vars: FedNodeVars = self._obj.node_var
        cfg: dict = node_vars.config_dict
        device = getattr(node_vars, "device", None) or "cpu"

        training_model: nn.Module = copy.deepcopy(node_vars.model).to(device)
        training_model.load_state_dict(node_vars.model_weight, strict=True)

        # Build a fresh optimizer bound to the new model parameters.
        optimizer = node_vars.optimizer_builder.rebuild(training_model.parameters())

        # Optionally restore accumulated optimizer state (momentum buffers etc.)
        # from the previous round. Controlled by strategy arg: preserve_optimizer_state.
        # On the very first round persistent_optimizer_state is None → cold start.
        preserve_opt_state = getattr(self._args, "preserve_optimizer_state", False)
        if preserve_opt_state:
            ModelUtils.restore_optimizer_state(
                optimizer, node_vars.persistent_optimizer_state, device
            )
            # Only clear gradients; do NOT reset optimizer state so momentum persists.
            ModelUtils.clear_model_grads(training_model)
        else:
            # Default: full reset each round (standard FedAvg behaviour)
            ModelUtils.clear_all(training_model, optimizer)
        ModelUtils.clear_cuda_cache()

        tr = node_vars.trainer
        tr.set_model(training_model)
        tr.set_optimizer(optimizer)
        tr.trainer_args.device = device

        local_epochs = int(cfg.get("training", {}).get("epochs", 1))
        try:
            updated_weights, train_record = self.train_and_offload(tr, local_epochs)

            # Snapshot optimizer state (CPU-resident) before releasing the optimizer
            if preserve_opt_state:
                node_vars.persistent_optimizer_state = ModelUtils.snapshot_optimizer_state(optimizer)
            else:
                node_vars.persistent_optimizer_state = None
        finally:
            # ── Release GPU memory via centralized cleaner ───────────
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