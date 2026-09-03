"""
SARA (Semantic-Anchored Rank Alignment) client training strategy.

Receives global LoRA prefix from server, trains locally with slot-level
and subspace-level alignment regularization, and uploads (A_i, B_i, n_i, r_i).
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from ..client_strategy import ClientStrategy
from ...fed_node.fed_node_vars import FedNodeVars
from ...fl_algorithms.aggregation.methods._fed_aggregator_rbla import FedAggregator_RBLA
from ...ml_algorithms.sara import SARAAlignmentLoss, SARAConfig
from ...ml_utils.model_utils import ModelUtils
from ...ml_utils import console


class SaraClientTrainingStrategy(ClientStrategy):
    """
    Client strategy for SARA federated LoRA training.

    Differs from plain RBLA by:
    1. Caching the global (A_g, B_g) anchor tensors received from the server.
    2. Injecting :class:`SARAAlignmentLoss` into the trainer before each round
       so that local training is regularised by slot/subspace alignment.
    3. Uploading only its local LoRA factors + metadata (same format as RBLA).
    """

    def __init__(self, args, client_node):
        super().__init__()
        self._args = args
        self._strategy_type = "sara"
        self._obj = client_node

        # ── SARA alignment module (lazy-init from config) ──
        self._sara_alignment: SARAAlignmentLoss | None = None
        self._sara_config: SARAConfig | None = None

        # ── Global anchors cached after receive_weight ──
        self._global_anchors: Dict[str, torch.Tensor] | None = None

    def _create_inner(self, args, client_node) -> None:
        self._args = args
        self._strategy_type = "sara"
        self._obj = client_node

    # ------------------------------------------------------------------
    # Lazy SARA config from node config dict
    # ------------------------------------------------------------------
    def _ensure_sara(self) -> SARAAlignmentLoss:
        if self._sara_alignment is None:
            cfg_dict = dict(self._obj.node_var.config_dict.get("sara", {}))
            if "rank_ratio_list" not in cfg_dict:
                server_node = getattr(self._obj, "server_node", None)
                server_cfg = getattr(getattr(server_node, "node_var", None), "config_dict", {}) or {}
                rank_ratio_list = server_cfg.get("rank_distribution", {}).get("rank_ratio_list", [])
                if rank_ratio_list:
                    cfg_dict["rank_ratio_list"] = list(rank_ratio_list)
            self._sara_config = SARAConfig.from_dict(cfg_dict)
            self._sara_alignment = SARAAlignmentLoss(self._sara_config)
        return self._sara_alignment

    # ------------------------------------------------------------------
    # Receive global weight (cached as anchors)
    # ------------------------------------------------------------------
    def receive_weight(self, global_weight: dict) -> None:
        self._obj.node_var.cache_weight = global_weight
        # Store as global anchors for alignment (detached copy)
        self._global_anchors = {k: v.detach().clone() for k, v in global_weight.items()}

    # ------------------------------------------------------------------
    # Broadcast: slice global LoRA to local rank
    # ------------------------------------------------------------------
    def set_local_weight(self) -> None:
        self._obj.node_var.model_weight = FedAggregator_RBLA.broadcast_lora_state_dict(
            self._obj.node_var.cache_weight,
            self._obj.node_var.model_weight,
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
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
        device = getattr(node_vars, "device", None) or "cpu"

        observe_model: nn.Module = copy.deepcopy(node_vars.model).to(device)
        observe_model.load_state_dict(node_vars.model_weight, strict=True)
        optimizer = node_vars.optimizer_builder.rebuild(observe_model.parameters())
        ModelUtils.clear_all(observe_model, optimizer)

        tr = node_vars.trainer
        orig_model = tr.trainer_args.model
        orig_optimizer = tr.trainer_args.optimizer
        orig_device = getattr(tr.trainer_args, "device", None)
        try:
            tr.set_model(observe_model)
            tr.set_optimizer(optimizer)
            tr.trainer_args.device = device
            local_epochs = int(node_vars.config_dict.get("training", {}).get("local_epochs", 1))
            updated_weights, train_record = self.train_and_offload(tr, local_epochs)
        finally:
            if orig_model is not None:
                tr.set_model(orig_model)
            if orig_optimizer is not None:
                tr.set_optimizer(orig_optimizer)
            if orig_device is not None:
                tr.trainer_args.device = orig_device
            self.cleanup_training_resources(model=observe_model, optimizer=optimizer)

        return updated_weights, train_record

    # ------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------
    def run_local_training(self) -> dict:
        updated_weights, train_record = self.local_training_step()
        self._log_training_complete(train_record)
        return updated_weights, {
            "node_id": self._obj.node_id,
            "updated_weights": updated_weights,
            "train_record": train_record,
            "data_sample_num": self._obj.node_var.data_sample_num,
        }

    def local_training_step(self) -> Tuple[dict, Any]:
        """Full local training with SARA alignment."""
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

        # ── Inject SARA context ──
        self._inject_sara_context(tr)

        local_epochs = int(cfg.get("training", {}).get("epochs", 1))
        try:
            updated_weights, train_record = self.train_and_offload(tr, local_epochs)
        finally:
            self.cleanup_training_resources(model=training_model, optimizer=optimizer, trainer=tr)

        node_vars.model_weight = updated_weights
        return updated_weights, train_record

    # ------------------------------------------------------------------
    # SARA context injection
    # ------------------------------------------------------------------
    def _inject_sara_context(self, trainer) -> None:
        """Set global anchors, round index, and alignment module on the trainer."""
        if not hasattr(trainer, "set_sara_context"):
            return  # trainer doesn't support SARA — skip alignment

        sara = self._ensure_sara()

        # Determine local rank r_i from the model's lora_A shape
        r_i = self._infer_client_rank()

        # Retrieve round index from node config (set by runner)
        cfg = self._obj.node_var.config_dict
        round_idx = int(cfg.get("_round_idx", 0))

        # Global anchors: server's full-rank A_g, B_g
        global_sd = self._global_anchors or self._obj.node_var.cache_weight or {}

        trainer.set_sara_context(
            global_anchors=global_sd,
            round_idx=round_idx,
            r_i=r_i,
            sara_alignment=sara,
        )

    def _infer_client_rank(self) -> int:
        """Infer local rank r_i from the first lora_A tensor in model_weight."""
        mw = self._obj.node_var.model_weight
        for key, tensor in mw.items():
            if "lora_A" in key:
                return int(tensor.shape[0])
        # Fallback: use rank_ratio from config
        return int(self._obj.node_var.config_dict.get("nn_model", {}).get("rank_ratio", 1))
