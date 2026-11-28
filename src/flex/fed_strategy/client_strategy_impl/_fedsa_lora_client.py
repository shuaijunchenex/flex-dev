"""FedSA-LoRA trains A and B locally and shares only A each round."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch

from ...fl_algorithms.aggregation.methods._fed_aggregator_fedsa_lora import validate_sample_count, validate_shared_a
from ...ml_algorithms.lora.impl.lora_ms import MSLoRALinear
from ...ml_algorithms.lora.rank_scale_alignment import build_lora_scale_profile
from ...model_trainer.model_evaluator import ModelEvaluator
from .._florist_batchnorm import freeze_batchnorm_forwards
from ._rbla_client import RblaClientTrainingStrategy


def linear_lora_profile(model):
    layers = build_lora_scale_profile(model)
    for name, layer in layers.items():
        module = model.get_submodule(name) if name else model
        if not isinstance(module, MSLoRALinear):
            raise TypeError(f"FedSA-LoRA requires MSLoRALinear adapters, got {type(module).__name__} at '{name}'")
        if module.fan_in_fan_out or module.merge_weights or module.merged:
            raise ValueError(f"FedSA-LoRA requires standard unmerged linear weights at '{name}'")
        if (module.lora_A.shape != (layer.rank, module.in_features)
                or module.lora_B.shape != (module.out_features, layer.rank)):
            raise ValueError(f"Invalid FedSA-LoRA factor shapes at '{name}'")
    return layers


class FedSALoRAClientTrainingStrategy(RblaClientTrainingStrategy):
    def __init__(self, args, client_node):
        super().__init__(args, client_node)
        self._strategy_type = "fedsa_lora"
        self.evaluation_loader = args.get("evaluation_loader", "shared_test")
        if self.evaluation_loader not in ("shared_test", "client_test"):
            raise ValueError("FedSA-LoRA evaluation_loader must be 'shared_test' or 'client_test'")
        self._evaluation_data = None
        self.layers = None
        self.shared_keys = ()
        self.private_keys = ()
        self.version = None
        self._pending_broadcast = None
        self._cached_upload = None

    def _create_inner(self, args, client_node):
        self.__init__(args, client_node)
        return self

    def configure_model(self) -> None:
        model = self._obj.node_var.model
        self.layers = linear_lora_profile(model)
        self.shared_keys = tuple(layer.key_A for layer in self.layers.values())
        self.private_keys = tuple(layer.key_B for layer in self.layers.values())
        trainable_keys = set(self.shared_keys) | set(self.private_keys)
        for key, parameter in model.named_parameters():
            parameter.requires_grad_(key in trainable_keys)

    def configure_evaluation_loader(self, shared_loader) -> None:
        loader = shared_loader if self.evaluation_loader == "shared_test" else getattr(self._obj.node_var, "test_data_loader", None)
        if loader is None:
            raise ValueError(f"FedSA-LoRA has no test data loader for '{self.evaluation_loader}'")
        self._evaluation_data = loader

    def validate_initial_state(self, shared_state) -> None:
        local = self._obj.node_var.model_weight
        if not isinstance(shared_state, Mapping) or shared_state.keys() != local.keys() - set(self.private_keys):
            raise ValueError("FedSA-LoRA initialization must contain the frozen base and shared A only")
        for key, value in shared_state.items():
            if not isinstance(value, torch.Tensor) or value.shape != local[key].shape or value.dtype != local[key].dtype:
                raise ValueError(f"FedSA-LoRA initial state shape or dtype differs at '{key}'")
        validate_shared_a({key: shared_state[key] for key in self.shared_keys},
                          {key: local[key] for key in self.shared_keys})

    def initialize_from_server(self, shared_state) -> None:
        if self.version is not None:
            raise RuntimeError("FedSA-LoRA client is already initialized")
        self.validate_initial_state(shared_state)
        local = self._obj.node_var.model_weight
        self._obj.node_var.model_weight = OrderedDict(
            (key, torch.zeros_like(value) if key in self.private_keys else shared_state[key].detach().to(value).clone())
            for key, value in local.items()
        )
        self.version = 0

    def train_and_offload(self, trainer, epochs):
        hook = freeze_batchnorm_forwards(trainer.trainer_args.model)
        try:
            return super().train_and_offload(trainer, epochs)
        finally:
            if hook is not None:
                hook.remove()

    def local_training_step(self):
        if self.version is None or self._pending_broadcast is not None:
            raise RuntimeError("FedSA-LoRA requires an applied shared A version before local training")
        node_var = self._obj.node_var
        previous = node_var.model_weight
        adapter_keys = set(self.shared_keys) | set(self.private_keys)
        try:
            updated, train_record = super().local_training_step()
            for key in previous.keys() - adapter_keys:
                if not torch.equal(updated[key].detach().to(previous[key]), previous[key]):
                    raise RuntimeError(f"FedSA-LoRA frozen state changed during training: '{key}'")
            payload = OrderedDict((key, updated[key].detach().clone()) for key in self.shared_keys)
            validate_shared_a(payload, {key: previous[key] for key in self.shared_keys})
            for key in self.private_keys:
                if not torch.isfinite(updated[key]).all().item():
                    raise ValueError(f"Nonfinite FedSA-LoRA private B factor '{key}'")
        except Exception:
            node_var.model_weight = previous
            raise
        return payload, train_record

    def run_local_training(self):
        if self.version is None or self._pending_broadcast is not None:
            raise RuntimeError("FedSA-LoRA requires an applied shared A version before local training")
        node_var = self._obj.node_var
        count = node_var.data_sample_num
        validate_sample_count(count)
        if count == 0:
            payload = OrderedDict((key, node_var.model_weight[key].detach().clone()) for key in self.shared_keys)
        else:
            if self._cached_upload is None:
                self._cached_upload, train_record = self.local_training_step()
                self._log_training_complete(train_record)
            payload = OrderedDict((key, value.clone()) for key, value in self._cached_upload.items())
        return payload, {
            "node_id": self._obj.node_id,
            "data_sample_num": count,
            "fedsa_version": self.version,
        }

    def receive_weight(self, message) -> None:
        if not isinstance(message, Mapping) or message.keys() != {"version", "shared_a"}:
            raise ValueError("FedSA-LoRA broadcast requires version and shared_a")
        version = message["version"]
        if type(version) is not int or self.version is None or version != self.version + 1:
            raise ValueError(f"Unexpected FedSA-LoRA shared A version {version!r} after {self.version!r}")
        local = self._obj.node_var.model_weight
        shared = message["shared_a"]
        template = {key: local[key] for key in self.shared_keys}
        validate_shared_a(shared, template)
        owned = OrderedDict((key, tensor.detach().to(template[key]).clone()) for key, tensor in shared.items())
        validate_shared_a(owned, template)
        self._pending_broadcast = (version, owned)

    def set_local_weight(self) -> None:
        if self._pending_broadcast is None:
            raise RuntimeError("FedSA-LoRA has no pending shared A broadcast")
        version, shared = self._pending_broadcast
        node_var = self._obj.node_var
        updated = OrderedDict(node_var.model_weight.items())
        updated.update(shared)
        node_var.model_weight = updated
        self.version = version
        self._pending_broadcast = None
        self._cached_upload = None

    def evaluate_local(self):
        if self.version is None or self._pending_broadcast is not None:
            raise RuntimeError("FedSA-LoRA requires an applied shared A version before evaluation")
        node_var = self._obj.node_var
        config = node_var.config_dict.get("data_loader", {})
        task = config.get("name") if config.get("task_type") == "nlp" else None
        if self._evaluation_data is None:
            raise RuntimeError("FedSA-LoRA evaluation loader has not been configured")
        evaluator = ModelEvaluator(node_var.model, self._evaluation_data, node_var.loss_func,
                                   node_var.device, task_name=task)
        evaluator.update_model(node_var.model_weight)
        try:
            result = evaluator.evaluate()
        finally:
            node_var.model.cpu()
        if result["total_test_samples"] <= 0:
            raise ValueError("FedSA-LoRA requires labeled samples in the configured client test loader")
        # Leave model tensors, predictions, and labels within the client strategy.
        fields = ("accuracy", "average_loss", "precision", "recall", "f1_score", "matthews_correlation",
                  "mcc", "primary_metric", "primary_score", "total_test_samples", "invalid_label_samples")
        return {key: result[key] for key in fields if key in result}
