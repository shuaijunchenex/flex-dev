"""Client training and factor-only exchange for FLoRG."""

import torch

from ...fl_algorithms.aggregation.methods._fed_aggregator_florg_gram import (
    GRAM_FACTOR_NAME,
    validate_factor_payload,
)
from ...ml_algorithms.lora.impl.gram_lora import GramLoRALinear
from ...ml_utils.model_utils import ModelUtils
from ._rbla_client import RblaClientTrainingStrategy


def gram_factor_keys(model) -> tuple[str, ...]:
    model = ModelUtils.unwrap_model(model)
    keys = tuple(
        f"{name}.{GRAM_FACTOR_NAME}" if name else GRAM_FACTOR_NAME
        for name, module in model.named_modules()
        if isinstance(module, GramLoRALinear)
    )
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    if not keys or trainable != set(keys):
        raise ValueError("FLoRG requires GramLoRALinear factors as the only trainable parameters")
    return keys


class FlorgGramClientTrainingStrategy(RblaClientTrainingStrategy):
    def __init__(self, args, client_node):
        super().__init__(args, client_node)
        self._strategy_type = "florg_gram"
        self._version = -1
        self._factor_keys = gram_factor_keys(client_node.node_var.model)

    def _create_inner(self, args, client_node):
        self.__init__(args, client_node)
        return self

    def validate_initial_state(self, shared_state) -> None:
        local = self._obj.node_var.model_weight
        if local.keys() != shared_state.keys() or any(
            shared_state[key].shape != tensor.shape or shared_state[key].dtype != tensor.dtype
            for key, tensor in local.items()
        ):
            raise ValueError("FLoRG requires matching server and client model state keys, shapes, and dtypes")

    def initialize_from_server(self, shared_state) -> None:
        self.validate_initial_state(shared_state)
        if self._version > 0:
            raise RuntimeError("FLoRG cannot overwrite an initialized client's training with round zero")
        self._obj.node_var.model_weight = {key: tensor.detach().clone() for key, tensor in shared_state.items()}
        self._version = 0

    def receive_weight(self, broadcast) -> None:
        version, factors = broadcast["version"], broadcast["factors"]
        if self._version < 0 or version not in (self._version, self._version + 1):
            raise ValueError("FLoRG broadcast does not follow the client's installed version")
        local = self._obj.node_var.model_weight
        validate_factor_payload(factors, {key: local[key] for key in self._factor_keys})
        self._obj.node_var.cache_weight = broadcast

    def set_local_weight(self) -> None:
        node_var = self._obj.node_var
        broadcast = node_var.cache_weight
        version = broadcast["version"]
        if version == self._version + 1:
            local = dict(node_var.model_weight)
            for key, tensor in broadcast["factors"].items():
                local[key] = tensor.detach().to(local[key], copy=True)
            node_var.model_weight = local
            self._version = version

    def run_local_training(self):
        payload, record = super().run_local_training()
        record["base_version"] = self._version
        return payload, record

    def local_training_step(self):
        if self._version < 0:
            raise RuntimeError("FLoRG requires round-zero initialization before training")
        node_var = self._obj.node_var
        previous = node_var.model_weight
        updated, record = super().local_training_step()
        if updated.keys() != previous.keys() or any(
            not torch.equal(value, updated[key].to(value.device))
            for key, value in previous.items() if key not in self._factor_keys
        ):
            node_var.model_weight = previous
            raise RuntimeError("FLoRG local training changed the frozen model state")
        payload = {key: updated[key].detach().to(device="cpu", copy=True) for key in self._factor_keys}
        try:
            validate_factor_payload(payload, {key: previous[key] for key in self._factor_keys})
        except ValueError:
            node_var.model_weight = previous
            raise
        return payload, record
