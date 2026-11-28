"""New clients for diagnostic, hard-reference, and Strong-A RBLA variants."""
from __future__ import annotations

import copy
from typing import Any, Tuple

import torch
import torch.nn as nn

from .._rbla_client import RblaClientTrainingStrategy
from ....fed_node.fed_node_vars import FedNodeVars
from ....ml_utils.model_utils import ModelUtils


def _is_lora_a_key(key: str) -> bool:
    return "lora_A" in key.split(".")


class RblaRefDiagClientTrainingStrategy(RblaClientTrainingStrategy):
    """Vanilla RBLA local training under a distinct registered name."""

    def __init__(self, args, client_node):
        super().__init__(args, client_node)
        self._strategy_type = "rbla_refdiag"

    def _create_inner(self, args, client_node) -> None:
        super()._create_inner(args, client_node)
        self._strategy_type = "rbla_refdiag"


class RblaFreezeAClientTrainingStrategy(RblaClientTrainingStrategy):
    """Hard gauge fixing: receive the RBLA prefix, freeze A, and train B."""

    def __init__(self, args, client_node):
        super().__init__(args, client_node)
        self._strategy_type = "rbla_freeze_a"

    def _create_inner(self, args, client_node) -> None:
        super()._create_inner(args, client_node)
        self._strategy_type = "rbla_freeze_a"

    def local_training_step(self) -> Tuple[dict, Any]:
        node_vars: FedNodeVars = self._obj.node_var
        cfg: dict = node_vars.config_dict
        device = getattr(node_vars, "device", None) or "cpu"

        training_model: nn.Module = copy.deepcopy(node_vars.model).to(device)
        training_model.load_state_dict(node_vars.model_weight, strict=True)
        a_anchors = {
            key: value.detach().cpu().clone()
            for key, value in training_model.state_dict().items()
            if _is_lora_a_key(key)
        }
        for key, parameter in training_model.named_parameters():
            if _is_lora_a_key(key):
                parameter.requires_grad_(False)

        # Reference/support RBLA variants also rebuild and clear optimizer state
        # each round, so canonical prefixes never reuse old slot momentum.
        optimizer = node_vars.optimizer_builder.rebuild(training_model.parameters())
        ModelUtils.clear_all(training_model, optimizer)

        trainer = node_vars.trainer
        trainer.set_model(training_model)
        trainer.set_optimizer(optimizer)
        trainer.trainer_args.device = device

        local_epochs = int(cfg.get("training", {}).get("epochs", 1))
        try:
            updated_weights, train_record = self.train_and_offload(trainer, local_epochs)
            max_change = max(
                (
                    updated_weights[key].detach().cpu() - anchor
                ).abs().max().item()
                for key, anchor in a_anchors.items()
            ) if a_anchors else 0.0
            if max_change != 0.0:
                raise RuntimeError(f"Freeze-A invariant violated: max |A_after-A_anchor|={max_change}")
            train_record["freeze_a_max_abs_change"] = float(max_change)
        finally:
            self.cleanup_training_resources(
                model=training_model,
                optimizer=optimizer,
                trainer=trainer,
            )

        node_vars.model_weight = updated_weights
        return updated_weights, train_record


class RblaStrongAClientTrainingStrategy(RblaClientTrainingStrategy):
    """RBLA local training with a trainable normalized A-side proximal anchor."""

    def __init__(self, args, client_node):
        super().__init__(args, client_node)
        self._strategy_type = "rbla_strong_a"

    def _create_inner(self, args, client_node) -> None:
        super()._create_inner(args, client_node)
        self._strategy_type = "rbla_strong_a"

    def local_training_step(self) -> Tuple[dict, Any]:
        node_vars: FedNodeVars = self._obj.node_var
        cfg: dict = node_vars.config_dict
        device = getattr(node_vars, "device", None) or "cpu"

        training_model: nn.Module = copy.deepcopy(node_vars.model).to(device)
        training_model.load_state_dict(node_vars.model_weight, strict=True)
        anchors = {
            key: value.detach().clone()
            for key, value in node_vars.model_weight.items()
            if _is_lora_a_key(key)
        }
        optimizer = node_vars.optimizer_builder.rebuild(training_model.parameters())
        ModelUtils.clear_all(training_model, optimizer)

        trainer = node_vars.trainer
        if not hasattr(trainer, "set_strong_a_context"):
            raise TypeError(
                "rbla_strong_a requires trainer_type=rbla_strong_a; "
                f"got {type(trainer).__name__}"
            )
        trainer.set_model(training_model)
        trainer.set_optimizer(optimizer)
        trainer.trainer_args.device = device
        trainer.set_strong_a_context(
            anchors=anchors,
            config=cfg.get("reference_alignment", {}),
        )

        local_epochs = int(cfg.get("training", {}).get("epochs", 1))
        try:
            updated_weights, train_record = self.train_and_offload(trainer, local_epochs)
        finally:
            trainer.clear_strong_a_context()
            self.cleanup_training_resources(
                model=training_model,
                optimizer=optimizer,
                trainer=trainer,
            )

        node_vars.model_weight = updated_weights
        return updated_weights, train_record


class RblaRefDiagAnalysisClientTrainingStrategy(RblaRefDiagClientTrainingStrategy):
    def __init__(self, args, client_node):
        super().__init__(args, client_node)
        self._strategy_type = "rbla_refdiag_analysis"


class RblaFreezeAAnalysisClientTrainingStrategy(RblaFreezeAClientTrainingStrategy):
    def __init__(self, args, client_node):
        super().__init__(args, client_node)
        self._strategy_type = "rbla_freeze_a_analysis"


class RblaStrongAAnalysisClientTrainingStrategy(RblaStrongAClientTrainingStrategy):
    def __init__(self, args, client_node):
        super().__init__(args, client_node)
        self._strategy_type = "rbla_strong_a_analysis"


class RblaFreezeASupportGammaClientTrainingStrategy(RblaFreezeAClientTrainingStrategy):
    def __init__(self, args, client_node):
        super().__init__(args, client_node)
        self._strategy_type = "rbla_freeze_a_support_gamma"


class RblaP8RefDiagSupportScalingClientTrainingStrategy(RblaRefDiagClientTrainingStrategy):
    def __init__(self, args, client_node):
        super().__init__(args, client_node)
        self._strategy_type = "rbla_p8_refdiag_support_scaling"


class RblaP8StrongASupportScalingClientTrainingStrategy(RblaStrongAClientTrainingStrategy):
    def __init__(self, args, client_node):
        super().__init__(args, client_node)
        self._strategy_type = "rbla_p8_strong_a_support_scaling"


class RblaP10FreezeASupportScalingClientTrainingStrategy(RblaFreezeAClientTrainingStrategy):
    def __init__(self, args, client_node):
        super().__init__(args, client_node)
        self._strategy_type = "rbla_p10_freeze_a_support_scaling"


class RblaP9FreezeASupportScalingClientTrainingStrategy(RblaFreezeAClientTrainingStrategy):
    def __init__(self, args, client_node):
        super().__init__(args, client_node)
        self._strategy_type = "rbla_p9_freeze_a_support_scaling"
