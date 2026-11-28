"""Commit compensated base weights and retained adapters once per round."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from ...fl_algorithms.aggregation.methods._fed_aggregator_fedex_lora import (
    FedAggregator_FedExLoRA,
)
from ...ml_algorithms.lora.fedex_lora import FedExBroadcast, FedExLayerResult
from ...ml_algorithms.lora.impl.lora_ms import MSLoRALinear
from ...ml_algorithms.lora.rank_scale_alignment import LoRAScaleProfile, build_lora_scale_profile
from ...ml_utils.model_utils import ModelUtils
from ._fedavg_server import FedAvgServerStrategy


class FedExLoRAServerStrategy(FedAvgServerStrategy):
    def __init__(self, args, server_node) -> None:
        super().__init__(args, server_node)
        self._strategy_type = "fedex_lora"
        self._version = 0
        self._pending: dict[str, FedExLayerResult] | None = None

    def _create_inner(self, args, server_node) -> None:
        super()._create_inner(args, server_node)
        self._strategy_type = "fedex_lora"
        self._version = 0
        self._pending = None
        return self

    def prepare(self, logger_header, client_nodes_in) -> None:
        node_vars = self._obj.node_var
        model = ModelUtils.unwrap_model(node_vars.model)
        self._layers = self._build_lora_profile(model)
        model_template = model.state_dict()
        ModelUtils.validate_state_dict(node_vars.model_weight, model_template, context="FedEx-LoRA")

        evaluator_model = ModelUtils.unwrap_model(node_vars.model_evaluator.model)
        self._validate_destination(evaluator_model, model_template, "evaluator")
        for client in client_nodes_in:
            client_model = ModelUtils.unwrap_model(client.node_var.model)
            self._validate_destination(client_model, model_template, str(client.node_id))

        node_vars.aggregation_method.configure_layers(self._layers)
        initial = OrderedDict(
            (key, value.detach().clone()) for key, value in node_vars.model_weight.items()
        )
        node_vars.model_evaluator.update_model(initial)
        node_vars.model_weight = initial
        super().prepare(logger_header, client_nodes_in)

    @staticmethod
    def _build_lora_profile(model: nn.Module) -> LoRAScaleProfile:
        """Inspect named, unmerged Linear layers with a standard [out, in] base."""
        for name, module in model.named_modules():
            if not (hasattr(module, "lora_A") or hasattr(module, "lora_B")):
                continue
            if not isinstance(module, MSLoRALinear):
                raise TypeError(
                    f"FedEx-LoRA supports MSLoRALinear only, got {type(module).__name__} at '{name}'"
                )
            if not name or module.fan_in_fan_out or module.merge_weights or module.merged:
                raise ValueError(f"FedEx-LoRA requires named, unmerged [out, in] Linear layers at '{name}'")
            expected_a = (module.r, module.in_features)
            expected_b = (module.out_features, module.r)
            expected_weight = (module.out_features, module.in_features)
            if (
                tuple(module.lora_A.shape) != expected_a
                or tuple(module.lora_B.shape) != expected_b
                or tuple(module.weight.shape) != expected_weight
            ):
                raise ValueError(f"FedEx-LoRA factor or backbone shape mismatch at '{name}'")
        return build_lora_scale_profile(model)

    def _validate_destination(self, model, template, destination: str) -> None:
        profile = self._build_lora_profile(model)
        if profile.keys() != self._layers.keys():
            raise ValueError(f"FedEx-LoRA adapter layers differ for {destination}")
        for prefix, server_layer in self._layers.items():
            layer = profile[prefix]
            if (
                layer.rank != server_layer.rank
                or layer.scaling != server_layer.scaling
                or layer.key_A != server_layer.key_A
                or layer.key_B != server_layer.key_B
            ):
                raise ValueError(
                    f"FedEx-LoRA requires a common rank and effective scaling at '{prefix}' for {destination}"
                )
        ModelUtils.validate_state_dict(model.state_dict(), template, context="FedEx-LoRA")

    def receive_client_updates(self, client_updates) -> None:
        if self._pending is not None:
            raise RuntimeError("FedEx-LoRA cannot replace a pending aggregate")
        known_ids = {str(client.node_id) for client in self._obj.client_nodes}
        received_ids: set[str] = set()
        active = []
        for update in client_updates:
            record = update["train_record"]
            node_id = str(record["node_id"])
            if node_id not in known_ids or node_id in received_ids:
                raise ValueError(f"Unknown or duplicate FedEx-LoRA client '{node_id}'")
            if record.get("base_version") != self._version:
                raise ValueError(f"Stale or missing FedEx-LoRA base version for '{node_id}'")
            received_ids.add(node_id)
            count = record["data_sample_num"]
            FedAggregator_FedExLoRA.validate_sample_count(count)
            if count > 0:
                active.append(update)
        super().receive_client_updates(active)

    def aggregation(self) -> None:
        node_vars = self._obj.node_var
        if self._pending is None and node_vars.client_updates:
            self._pending = node_vars.aggregation_method.aggregate(node_vars.client_updates)

    def apply_weight(self) -> None:
        if self._pending is None:
            return
        node_vars = self._obj.node_var
        previous = node_vars.model_weight
        candidate = OrderedDict(previous)
        for prefix, layer in self._layers.items():
            result = self._pending[prefix]
            weight_key = f"{prefix}.weight"
            base = previous[weight_key]
            if result.residual.shape != base.shape:
                raise ValueError(f"FedEx-LoRA residual shape does not match '{weight_key}'")
            candidate[weight_key] = (
                base.to(dtype=result.residual.dtype) + result.residual.to(device=base.device)
            ).to(dtype=base.dtype)
            candidate[layer.key_A] = result.mean_a.to(previous[layer.key_A])
            candidate[layer.key_B] = result.mean_b.to(previous[layer.key_B])
            for key in (weight_key, layer.key_A, layer.key_B):
                if not torch.isfinite(candidate[key]).all():
                    raise ValueError(f"FedEx-LoRA aggregation produced a nonfinite tensor for '{key}'")
        ModelUtils.validate_state_dict(candidate, previous, context="FedEx-LoRA")
        node_vars.model_evaluator.update_model(candidate)

        node_vars.model_weight = candidate
        node_vars.aggregated_weight = candidate
        node_vars.client_updates = []
        self._pending = None
        self._version += 1

    def broadcast(self) -> None:
        payload = FedExBroadcast(self._version, self._obj.node_var.model_weight)
        for client in self._obj.client_nodes:
            client.receive_weight(payload)
            client.set_local_weight()
