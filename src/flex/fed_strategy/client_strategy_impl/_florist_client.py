"""FLoRIST clients: fixed local ranks and a permanently frozen backbone."""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
from torch import nn

from ...ml_algorithms.lora.impl.lora_ms import MSLoRAConv2d, MSLoRALinear
from ...ml_algorithms.lora.rank_scale_alignment import build_lora_scale_profile
from .._florist_batchnorm import batchnorm_mode, batchnorm_keys, freeze_batchnorm_forwards
from ._rbla_client import RblaClientTrainingStrategy


def _head_prefixes(args) -> tuple[str, ...]:
    values = args.get("trainable_head_prefixes", [])
    if not isinstance(values, (list, tuple)) or any(
        not isinstance(value, str) or not value or value.endswith(".")
        for value in values
    ):
        raise ValueError("trainable_head_prefixes must be a list of nonempty parameter/module prefixes")
    return tuple(values)


def _inspect_model(model: nn.Module, head_prefixes: tuple[str, ...]):
    """Validate supported factor layouts while retaining logical ranks and scales."""
    for prefix, module in model.named_modules():
        if not (hasattr(module, "lora_A") or hasattr(module, "lora_B")):
            continue
        if not isinstance(module, (MSLoRALinear, MSLoRAConv2d)):
            raise TypeError(f"FLoRIST supports MSLoRALinear and MSLoRAConv2d only, got {type(module).__name__} at '{prefix}'")
        if not prefix:
            raise ValueError("FLoRIST requires named LoRA submodules")
        if module.merge_weights or module.merged:
            raise ValueError(f"FLoRIST requires merge_weights=False at '{prefix}'")
        if not isinstance(getattr(module, "lora_A", None), torch.Tensor) or not isinstance(getattr(module, "lora_B", None), torch.Tensor):
            raise ValueError(f"FLoRIST requires tensor-valued A/B factors at '{prefix}'")
        if isinstance(module, MSLoRAConv2d):
            if len(module.kernel_size) != 2 or module.kernel_size[0] != module.kernel_size[1]:
                raise ValueError(f"FLoRIST requires square convolution kernels at '{prefix}'")
            if module.groups != 1 or module.padding_mode != "zeros":
                raise ValueError(f"FLoRIST requires groups=1 and padding_mode=zeros at '{prefix}'")
            kernel = module.kernel_size[0]
            shape_A = (module.r * kernel, module.in_channels * kernel)
            shape_B = (module.out_channels * kernel, module.r * kernel)
            shape_weight = (module.out_channels, module.in_channels, kernel, kernel)
        else:
            if module.fan_in_fan_out:
                raise ValueError(f"FLoRIST requires standard [out, in] weights at '{prefix}'")
            shape_A = (module.r, module.in_features)
            shape_B = (module.out_features, module.r)
            shape_weight = (module.out_features, module.in_features)
        if tuple(module.lora_A.shape) != shape_A or tuple(module.lora_B.shape) != shape_B or tuple(module.weight.shape) != shape_weight:
            raise ValueError(f"Invalid FLoRIST factor shapes at '{prefix}'")

    layers = build_lora_scale_profile(model)
    adapter_keys = {key for layer in layers.values() for key in (layer.key_A, layer.key_B)}
    parameters = dict(model.named_parameters())
    head_keys = set()
    for prefix in head_prefixes:
        matches = {key for key in parameters if key == prefix or key.startswith(prefix + ".")}
        if not matches:
            raise ValueError(f"FLoRIST trainable head prefix '{prefix}' matches no parameters")
        if any(key in adapter_keys or any(key.startswith(layer + ".") if layer else True for layer in layers) for key in matches):
            raise ValueError(f"FLoRIST head prefix '{prefix}' overlaps a LoRA layer")
        head_keys.update(matches)
    return layers, adapter_keys, head_keys


def _payload_bytes(payload) -> int:
    return sum(value.numel() * value.element_size() for value in payload.values())


def _validate_payload(payload, layers, extra_keys, template, ranks=None, counter_keys=()) -> None:
    expected = {key for layer in layers.values() for key in (layer.key_A, layer.key_B)} | set(extra_keys)
    if set(payload) != expected:
        raise ValueError(f"FLoRIST payload keys differ: missing={sorted(expected - set(payload))}, unexpected={sorted(set(payload) - expected)}")
    for key, value in payload.items():
        if key in counter_keys:
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.int64
                    or value.dtype != template[key].dtype or value.shape != template[key].shape
                    or value.ndim != 0 or value.item() < 0):
                raise ValueError(f"FLoRIST BatchNorm counter must be a nonnegative int64 scalar: {key}")
            continue
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise ValueError(f"FLoRIST payload '{key}' must be a floating tensor")
        # if not torch.isfinite(value).all():
        #     raise ValueError(f"FLoRIST payload '{key}' must be a finite floating tensor")
    for prefix, layer in layers.items():
        A, B = payload[layer.key_A], payload[layer.key_B]
        expected_A, expected_B = template[layer.key_A], template[layer.key_B]
        if A.ndim != 2 or B.ndim != 2 or A.shape[0] < 1 or A.shape[0] != B.shape[1] or A.shape[1] != expected_A.shape[1] or B.shape[0] != expected_B.shape[0]:
            raise ValueError(f"FLoRIST factor shape mismatch at '{prefix}'")
        if ranks is not None and A.shape[0] != ranks[prefix]:
            raise ValueError(f"FLoRIST upload rank mismatch at '{prefix}'")
    for key in extra_keys:
        if tuple(payload[key].shape) != tuple(template[key].shape):
            raise ValueError(f"FLoRIST auxiliary state shape mismatch for '{key}'")


class FloristClientTrainingStrategy(RblaClientTrainingStrategy):
    """Reuse the RBLA trainer/optimizer lifecycle, changing only adapter exchange."""

    def __init__(self, args, client_node):
        super().__init__(args, client_node)
        self._strategy_type = "florist"
        self.head_prefixes = _head_prefixes(args)
        self.batchnorm_mode = batchnorm_mode(args)
        self.layers = None
        self.factor_ranks = {}
        self.adapter_keys = set()
        self.head_keys = set()
        self.batchnorm_keys = set()
        self.bn_counter_keys = set()
        self.extra_keys = set()
        self._frozen_weight = None

    def _create_inner(self, args, client_node):
        self.__init__(args, client_node)
        return self

    def configure_model(self) -> None:
        if self.layers is None:
            layers, adapter_keys, head_keys = _inspect_model(
                self._obj.node_var.model, self.head_prefixes
            )
            bn_keys, counter_keys = batchnorm_keys(
                self._obj.node_var.model, self.batchnorm_mode, head_keys
            )
            self.layers, self.adapter_keys, self.head_keys = layers, adapter_keys, head_keys
            self.batchnorm_keys, self.bn_counter_keys = bn_keys, counter_keys
            self.extra_keys = self.head_keys | self.batchnorm_keys
            state = self._obj.node_var.model.state_dict()
            # Conv2d stores r*k factor columns, but its actual scaling remains alpha/r.
            self.factor_ranks = {prefix: int(state[layer.key_A].shape[0]) for prefix, layer in self.layers.items()}
        for key, parameter in self._obj.node_var.model.named_parameters():
            parameter.requires_grad_(key in self.adapter_keys or key in self.extra_keys)

    def train_and_offload(self, trainer, epochs):
        # RBLA has already created and loaded the actual training model copy.
        model = trainer.trainer_args.model
        hook = freeze_batchnorm_forwards(model) if self.batchnorm_mode == "frozen" else None
        try:
            return super().train_and_offload(trainer, epochs)
        finally:
            if hook is not None:
                hook.remove()

    def initialize_from_server(self, backbone) -> None:
        """Synchronize W0 once; initialize every client's full local rank A/random B/0."""
        if self._frozen_weight is not None:
            raise RuntimeError("FLoRIST client was already initialized")
        self.configure_model()
        local = self._obj.node_var.model_weight
        if set(backbone) != set(local) - self.adapter_keys:
            raise ValueError("FLoRIST initial backbone keys differ between server and client")
        initialized = OrderedDict()
        for key, value in local.items():
            if key in self.adapter_keys:
                initialized[key] = torch.empty_like(value)
            else:
                if tuple(backbone[key].shape) != tuple(value.shape):
                    raise ValueError(f"FLoRIST initial backbone shape mismatch for '{key}'")
                initialized[key] = backbone[key].detach().to(value).clone()
        for layer in self.layers.values():
            nn.init.kaiming_uniform_(initialized[layer.key_A], a=math.sqrt(5))
            nn.init.zeros_(initialized[layer.key_B])
        # References are read-only; all training occurs in RBLA's separate model copy.
        self._frozen_weight = {key: value for key, value in backbone.items() if key not in self.extra_keys}
        self._obj.node_var.model_weight = initialized

    def receive_weight(self, global_weight) -> None:
        # All clients receive the same compact payload. Never change its tensors.
        self._obj.node_var.cache_weight = global_weight

    def set_local_weight(self) -> None:
        if self._frozen_weight is None:
            raise RuntimeError("FLoRIST client must receive the initial shared backbone first")
        cache = self._obj.node_var.cache_weight
        local = self._obj.node_var.model_weight
        _validate_payload(cache, self.layers, self.extra_keys, local, counter_keys=self.bn_counter_keys)
        fitted = OrderedDict(local.items())
        for prefix, layer in self.layers.items():
            A = torch.zeros_like(local[layer.key_A])
            B = torch.zeros_like(local[layer.key_B])
            keep = min(self.factor_ranks[prefix], cache[layer.key_A].shape[0])
            A[:keep].copy_(cache[layer.key_A][:keep].to(A))
            # Preserve the official A=Vt / B=U Sigma coordinates: scale only B.
            B[:, :keep].copy_((cache[layer.key_B][:, :keep] / layer.scaling).to(B))
            fitted[layer.key_A], fitted[layer.key_B] = A, B
        for key in self.extra_keys:
            fitted[key] = cache[key].detach().to(local[key]).clone()
        self._obj.node_var.model_weight = fitted

    def local_training_step(self):
        if self._frozen_weight is None:
            raise RuntimeError("FLoRIST client must be initialized before training")
        self.configure_model()
        updated, record = super().local_training_step()
        self.assert_frozen_backbone(updated)
        payload = OrderedDict()
        for layer in self.layers.values():
            payload[layer.key_A] = updated[layer.key_A].detach().clone()
            payload[layer.key_B] = updated[layer.key_B].detach() * layer.scaling
        for key in sorted(self.extra_keys):
            payload[key] = updated[key].detach().clone()
        _validate_payload(payload, self.layers, self.extra_keys, updated, self.factor_ranks, self.bn_counter_keys)
        # super() retains the full local state for the next round; only compact data leaves.
        return payload, record

    def assert_frozen_backbone(self, state=None) -> None:
        if self._frozen_weight is None:
            raise RuntimeError("FLoRIST client is not initialized")
        state = self._obj.node_var.model_weight if state is None else state
        for key, reference in self._frozen_weight.items():
            if not torch.equal(state[key].detach().cpu(), reference.detach().cpu()):
                raise RuntimeError(f"FLoRIST frozen backbone changed during local training: '{key}'")
