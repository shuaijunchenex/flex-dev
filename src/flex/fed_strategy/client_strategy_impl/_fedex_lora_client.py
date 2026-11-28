"""FedEx-LoRA clients load the server base and retain its adapter factors."""

from __future__ import annotations

from collections import OrderedDict

from ...ml_algorithms.lora.fedex_lora import FedExBroadcast
from ...ml_utils.model_utils import ModelUtils
from ._flora_client import FloraClientTrainingStrategy


class FedExLoRAClientTrainingStrategy(FloraClientTrainingStrategy):
    def __init__(self, args, client_node) -> None:
        super().__init__(args, client_node)
        self._strategy_type = "fedex_lora"
        self._version: int | None = None

    def _create_inner(self, args, client_node) -> None:
        super()._create_inner(args, client_node)
        self._strategy_type = "fedex_lora"
        self._version = None

    def receive_weight(self, global_weight: FedExBroadcast) -> None:
        if not isinstance(global_weight, FedExBroadcast):
            raise TypeError("FedEx-LoRA requires a versioned full-state broadcast")
        node_vars = self._obj.node_var
        pending = getattr(node_vars, "cache_weight", None)
        latest = pending.version if pending is not None else self._version
        if latest is not None and global_weight.version < latest:
            raise ValueError("FedEx-LoRA cannot install an older base version")
        if global_weight.version == self._version:
            return
        node_vars.cache_weight = global_weight

    def set_local_weight(self) -> None:
        node_vars = self._obj.node_var
        payload = getattr(node_vars, "cache_weight", None)
        if payload is None:
            return

        template = ModelUtils.unwrap_model(node_vars.model).state_dict()
        ModelUtils.validate_state_dict(payload.state, template, context="FedEx-LoRA")
        node_vars.model_weight = OrderedDict(
            (key, value.detach().clone()) for key, value in payload.state.items()
        )
        self._version = payload.version
        node_vars.cache_weight = None

    def run_local_training(self) -> tuple[dict, dict]:
        if self._version is None:
            raise RuntimeError("FedEx-LoRA client must install the server state before training")
        updated_weights, train_record = super().run_local_training()
        train_record["base_version"] = self._version
        train_record.pop("updated_weights")
        adapter_weights = {
            key: value for key, value in updated_weights.items() if self._is_lora_key(key)
        }
        return adapter_weights, train_record
