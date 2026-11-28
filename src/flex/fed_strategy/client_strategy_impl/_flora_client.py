from __future__ import annotations

import copy
import math
from collections import OrderedDict
from typing import Any, Mapping, Tuple

import torch
import torch.nn as nn

from flex.fed_strategy.strategy_args import StrategyArgs
from ..client_strategy import ClientStrategy
from ...ml_utils.model_utils import ModelUtils
from ...ml_utils import console
from ...fed_node.fed_node_vars import FedNodeVars


class FloraClientTrainingStrategy(ClientStrategy):
    """FLoRA (Wang et al., NeurIPS 2024) — client-side training strategy.

    **Every round:**

    1. Receive the server‑broadcast full *W* (already merged with the
       previous round's :math:`\\Delta W`).
    2. Load *W* as the **frozen backbone**; freshly initialise LoRA A/B so
       :math:`BA=0` without setting both factors to zero.
    3. Train only LoRA A/B for the configured number of local epochs.
    4. Upload the trained A/B (plus any non‑LoRA params) to the server.

    **Freeze semantics:**  Only parameters whose name ends with ``lora_A`` or
    ``lora_B`` have ``requires_grad=True``; all others are frozen.
    """

    def __init__(self, args: StrategyArgs, client_node) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "flora"
        self._obj = client_node

    def _create_inner(self, args, client_node) -> None:
        self._args = args
        self._strategy_type = "flora"
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
        cfg: dict = node_vars.config_dict
        device = getattr(node_vars, "device", None) or "cpu"

        observe_model: nn.Module = copy.deepcopy(node_vars.model).to(device)
        observe_model.load_state_dict(node_vars.model_weight, strict=True)

        # Freeze backbone, only A/B trainable
        self._freeze_backbone_train_lora(observe_model)
        batchnorm_hook = self._register_frozen_batchnorm_hook(observe_model)

        optimizer = node_vars.optimizer_builder.rebuild(
            observe_model.parameters()
        )
        ModelUtils.clear_all(observe_model, optimizer)

        tr = node_vars.trainer
        orig_model = tr.trainer_args.model
        orig_optimizer = tr.trainer_args.optimizer
        orig_device = getattr(tr.trainer_args, "device", None)

        try:
            tr.set_model(observe_model)
            tr.set_optimizer(optimizer)
            tr.trainer_args.device = device
            local_epochs = int(cfg.get("training", {}).get("epochs", 1))
            updated_weights, train_record = self.train_and_offload(tr, local_epochs)
        finally:
            if batchnorm_hook is not None:
                batchnorm_hook.remove()
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

    # ------------------- Public: Local training wrapper -------------------
    def run_local_training(self) -> dict:
        updated_weights, train_record = self.local_training_step()
        self._log_training_complete(train_record)
        return updated_weights, {
            "node_id": self._obj.node_id,
            "updated_weights": updated_weights,
            "train_record": train_record,
            "data_sample_num": self._obj.node_var.data_sample_num,
        }

    # ------------------- Full local training (write-back to node_var) -------------------
    def local_training_step(self) -> Tuple[dict, Any]:
        """Train only LoRA A/B with a frozen backbone.

        The backbone *W* is loaded from ``node_var.model_weight`` (set by
        :meth:`set_local_weight` from the server broadcast).  We freeze
        every parameter except those ending with ``lora_A`` / ``lora_B``,
        train, and return the full state dict.
        """
        node_vars: FedNodeVars = self._obj.node_var
        cfg: dict = node_vars.config_dict
        device = getattr(node_vars, "device", None) or "cpu"

        training_model: nn.Module = copy.deepcopy(node_vars.model).to(device)
        training_model.load_state_dict(node_vars.model_weight, strict=True)

        # ---- Freeze backbone, only A/B trainable ----
        self._freeze_backbone_train_lora(training_model)
        batchnorm_hook = self._register_frozen_batchnorm_hook(training_model)

        optimizer = node_vars.optimizer_builder.rebuild(
            training_model.parameters()
        )
        ModelUtils.clear_all(training_model, optimizer)

        tr = node_vars.trainer
        tr.set_model(training_model)
        tr.set_optimizer(optimizer)
        tr.trainer_args.device = device

        local_epochs = int(cfg.get("training", {}).get("epochs", 1))
        try:
            updated_weights, train_record = self.train_and_offload(tr, local_epochs)
        finally:
            if batchnorm_hook is not None:
                batchnorm_hook.remove()
            self.cleanup_training_resources(
                model=training_model,
                optimizer=optimizer,
                trainer=tr,
            )

        lora_cfg = getattr(node_vars.model, "lora_config", {}) or {}
        updated_weights = self._restore_frozen_state(
            updated_weights,
            node_vars.model_weight,
            suffix_A=lora_cfg.get("suffix_A", "lora_A"),
            suffix_B=lora_cfg.get("suffix_B", "lora_B"),
        )
        node_vars.model_weight = updated_weights
        return updated_weights, train_record

    # ------------------- Weight reception & local preparation -------------------
    def receive_weight(self, global_weight: dict) -> None:
        """Cache the server‑broadcast full *W*."""
        self._obj.node_var.cache_weight = global_weight

    def set_local_weight(self) -> None:
        """Load the merged backbone *W* and freshly initialise LoRA A/B.

        The server broadcasts the full merged *W* (not LoRA factors).
        We copy every backbone key from the cache and initialise local-rank
        LoRA factors with a zero product and a viable first gradient.
        """
        cache = self._obj.node_var.cache_weight
        if cache is None:
            return

        model_state = self._obj.node_var.model_weight
        lora_cfg = getattr(self._obj.node_var.model, "lora_config", {}) or {}
        suffix_A = lora_cfg.get("suffix_A", "lora_A")
        suffix_B = lora_cfg.get("suffix_B", "lora_B")

        local_weight = OrderedDict()
        for key in model_state.keys():
            if self._is_lora_key(key, suffix_A, suffix_B):
                # Preserve the client's tensor shape; server factors may use
                # a different rank and must never be copied into this client.
                local_weight[key] = model_state[key].clone().detach()
            elif key in cache:
                # Copy backbone parameter from server broadcast
                if tuple(cache[key].shape) != tuple(model_state[key].shape):
                    raise ValueError(
                        "FLoRA backbone shape mismatch for "
                        f"'{key}': server {tuple(cache[key].shape)}, "
                        f"client {tuple(model_state[key].shape)}"
                    )
                local_weight[key] = cache[key].clone().detach()
            else:
                # Keep local fallback (e.g. classifier head not in cache)
                local_weight[key] = model_state[key].clone().detach()

        self._obj.node_var.model_weight = self._reset_lora_state(
            self._obj.node_var.model,
            local_weight,
            suffix_A=suffix_A,
            suffix_B=suffix_B,
        )

    # ------------------- Private helpers -------------------
    @staticmethod
    def _is_lora_key(
        key: str,
        suffix_A: str = "lora_A",
        suffix_B: str = "lora_B",
    ) -> bool:
        leaf = key.rsplit(".", 1)[-1]
        return leaf == suffix_A or leaf == suffix_B

    @classmethod
    def _reset_lora_state(
        cls,
        model: nn.Module,
        state: Mapping[str, torch.Tensor],
        *,
        suffix_A: str = "lora_A",
        suffix_B: str = "lora_B",
    ) -> OrderedDict:
        """Reset only LoRA factors without touching the frozen backbone.

        Linear, merged-linear, and convolution adapters use the standard LoRA
        initialisation (A random, B zero). Embedding adapters in this codebase
        use the transposed convention (A zero, B random). Both conventions
        start from an exactly zero adapter update without blocking gradients.
        """
        modules = dict(model.named_modules())
        reset_state = OrderedDict()

        for key, value in state.items():
            leaf = key.rsplit(".", 1)[-1]
            if leaf != suffix_A and leaf != suffix_B:
                reset_state[key] = value.clone().detach()
                continue

            prefix = key[: -(len(leaf) + 1)] if "." in key else ""
            owner = modules.get(prefix, model if not prefix else None)
            fresh = torch.empty_like(value)
            is_embedding = isinstance(owner, nn.Embedding)

            if (leaf == suffix_A and is_embedding) or (
                leaf == suffix_B and not is_embedding
            ):
                nn.init.zeros_(fresh)
            elif leaf == suffix_A:
                nn.init.kaiming_uniform_(fresh, a=math.sqrt(5))
            else:
                nn.init.normal_(fresh)

            reset_state[key] = fresh

        return reset_state

    @classmethod
    def _restore_frozen_state(
        cls,
        updated_state: Mapping[str, torch.Tensor],
        backbone_state: Mapping[str, torch.Tensor],
        suffix_A: str = "lora_A",
        suffix_B: str = "lora_B",
    ) -> OrderedDict:
        """Remove non-LoRA parameter and buffer drift from a client upload."""
        restored = OrderedDict()
        for key, value in updated_state.items():
            if cls._is_lora_key(key, suffix_A, suffix_B) or key not in backbone_state:
                restored[key] = value.clone().detach()
            else:
                restored[key] = backbone_state[key].clone().detach()
        return restored

    @staticmethod
    def _register_frozen_batchnorm_hook(model: nn.Module):
        """Keep frozen BatchNorm modules in eval mode during LoRA training."""
        batchnorm_modules = tuple(
            module
            for module in model.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        )
        if not batchnorm_modules:
            return None

        def _set_batchnorm_eval(_module, _inputs) -> None:
            for batchnorm in batchnorm_modules:
                batchnorm.eval()

        return model.register_forward_pre_hook(_set_batchnorm_eval)

    @staticmethod
    def _freeze_backbone_train_lora(
        model: nn.Module,
        suffix_A: str = "lora_A",
        suffix_B: str = "lora_B",
    ) -> None:
        """Freeze all parameters except those whose name ends with *suffix_A* or *suffix_B*."""
        for name, param in model.named_parameters():
            if name.endswith(f".{suffix_A}") or name.endswith(f".{suffix_B}"):
                param.requires_grad = True
            else:
                param.requires_grad = False
