"""FLoRIST server: compact global factors and full-rank temporary evaluation."""

from __future__ import annotations

import copy
from collections import OrderedDict

import torch

from ...ml_algorithms.lora.lora_utils import LoRAUtils
from .._florist_batchnorm import batchnorm_mode, batchnorm_keys
from ..client_strategy_impl._florist_client import (
    FloristClientTrainingStrategy,
    _head_prefixes,
    _inspect_model,
    _payload_bytes,
    _validate_payload,
)
from ._rbla_server import RblaServerStrategy


class FloristServerStrategy(RblaServerStrategy):
    def __init__(self, args, server_node):
        super().__init__(args, server_node)
        self._strategy_type = "florist"
        self.head_prefixes = _head_prefixes(args)
        self.batchnorm_mode = batchnorm_mode(args)
        self._initial_weight = None
        self._initial_broadcast_done = False
        self.history = []
        self.initialization_stats = {}
        self._obj.node_var.florist_history = self.history

    def _create_inner(self, args, server_node):
        self.__init__(args, server_node)
        return self

    def _initialize_context(self) -> None:
        if self._initial_weight is not None:
            return
        node_var = self._obj.node_var
        self.layers, self.adapter_keys, self.head_keys = _inspect_model(node_var.model, self.head_prefixes)
        self.batchnorm_keys, self.bn_counter_keys = batchnorm_keys(node_var.model, self.batchnorm_mode, self.head_keys)
        self.extra_keys = self.head_keys | self.batchnorm_keys
        self._clients_by_id = {}
        for client in self._obj.client_nodes:
            client_id = str(client.node_id)
            if client_id in self._clients_by_id:
                raise ValueError(f"Duplicate FLoRIST client id '{client_id}'")
            if not isinstance(client.strategy, FloristClientTrainingStrategy):
                raise TypeError("FLoRIST requires the florist client strategy")
            client.strategy.configure_model()
            if set(client.strategy.layers) != set(self.layers) or client.strategy.head_keys != self.head_keys:
                raise ValueError("FLoRIST client/server adapter layers or trainable head keys differ")
            if (client.strategy.batchnorm_mode != self.batchnorm_mode
                    or client.strategy.batchnorm_keys != self.batchnorm_keys
                    or client.strategy.bn_counter_keys != self.bn_counter_keys):
                raise ValueError("FLoRIST client/server BatchNorm policy or state keys differ")
            self._clients_by_id[client_id] = client
        # Commit initialization only after every client has passed policy checks.
        self._initial_weight = OrderedDict((key, value.detach().cpu().clone()) for key, value in node_var.model_weight.items())
        self._backbone = OrderedDict((key, value) for key, value in self._initial_weight.items() if key not in self.adapter_keys)

    def prepare(self, logger_header, client_nodes_in) -> None:
        super().prepare(logger_header, client_nodes_in)
        self._initialize_context()

    def _validate_state(self, payload, ranks=None):
        _validate_payload(
            payload, self.layers, self.extra_keys, self._initial_weight, ranks, self.bn_counter_keys)
        # if any(not torch.isfinite(value).all().item() for value in payload.values()):
        #     raise ValueError("Nonfinite FLoRIST server payload")

    def broadcast(self) -> None:
        self._initialize_context()
        if not self._initial_broadcast_done:
            for client in self._obj.client_nodes:
                client.strategy.initialize_from_server(self._backbone)
            self.initialization_stats = {
                "backbone_bytes_per_client": _payload_bytes(self._backbone),
                "actual_download_bytes": _payload_bytes(self._backbone) * len(self._obj.client_nodes),
                "broadcast_recipients": len(self._obj.client_nodes),
                "scope": "one-time shared backbone and head initialization; no adapter factors sent",
            }
            self._obj.node_var.florist_initialization_stats = self.initialization_stats
            self._initial_broadcast_done = True
            return

        payload = self._obj.node_var.model_weight
        self._validate_state(payload)
        for client in self._obj.client_nodes:
            client.receive_weight(payload)
            client.set_local_weight()
        if self.history:
            self.history[-1].update({
                "actual_download_bytes": _payload_bytes(payload) * len(self._obj.client_nodes),
                "broadcast_recipients": len(self._obj.client_nodes),
                "global_payload_bytes": _payload_bytes(payload),
                "batchnorm_download_bytes": _payload_bytes({key: payload[key] for key in self.batchnorm_keys}) * len(self._obj.client_nodes),
            })

    @staticmethod
    def _update_client_id(update) -> str:
        node_id = update.get("node_id")
        if node_id is None:
            node_id = update.get("train_record", {}).get("node_id")
        if node_id is None:
            raise ValueError("FLoRIST client update is missing node_id")
        return str(node_id)

    def aggregation(self) -> None:
        if not self._initial_broadcast_done:
            raise RuntimeError("FLoRIST initial broadcast must precede aggregation")
        updates = self._obj.node_var.client_updates
        if not updates:
            raise ValueError("FLoRIST requires at least one client update")
        seen = set()
        for update in updates:
            client_id = self._update_client_id(update)
            if client_id not in self._clients_by_id or client_id in seen:
                raise ValueError(f"Unknown or duplicate FLoRIST client '{client_id}'")
            seen.add(client_id)
            client = self._clients_by_id[client_id]
            client.strategy.assert_frozen_backbone()
            self._validate_state(update["updated_weights"], client.strategy.factor_ranks)
        aggregator = self._obj.node_var.aggregation_method
        # Preserve the SVD aggregator exactly. It already averages auxiliary floats.
        # Integer BN counters are explicit strategy state, never SVD/FedAvg inputs.
        float_updates = [dict(update, updated_weights={
            key: value for key, value in update["updated_weights"].items() if key not in self.bn_counter_keys
        }) for update in updates] if self.bn_counter_keys else updates
        aggregated = OrderedDict(aggregator.aggregate(float_updates))
        for key in sorted(self.bn_counter_keys):
            active = [update["updated_weights"][key] for update in updates if update["train_record"]["data_sample_num"] > 0]
            aggregated[key] = torch.stack([value.detach().cpu() for value in active]).max()
        self._validate_state(aggregated)
        self._obj.node_var.aggregated_weight = OrderedDict((key, value.detach().clone()) for key, value in aggregated.items())
        self.history.append({
            "round": len(self.history) + 1,
            "num_clients": len(updates),
            "client_ids": sorted(seen),
            "actual_upload_bytes": sum(_payload_bytes(update["updated_weights"]) for update in updates),
            "actual_download_bytes": 0,
            "global_rank_by_layer": {prefix: int(aggregated[layer.key_A].shape[0]) for prefix, layer in self.layers.items()},
            "local_rank_by_client": {client_id: {prefix: layer.rank for prefix, layer in client.strategy.layers.items()} for client_id, client in self._clients_by_id.items()},
            "local_factor_rank_by_client": {client_id: dict(client.strategy.factor_ranks) for client_id, client in self._clients_by_id.items()},
            "aggregation_stats": copy.deepcopy(getattr(aggregator, "last_stats", {})),
            "batchnorm_mode": self.batchnorm_mode,
            "batchnorm_upload_bytes": sum(_payload_bytes({key: update["updated_weights"][key] for key in self.batchnorm_keys}) for update in updates),
            "batchnorm_counter_policy": "max over positive-sample participants; fixed BN momentum" if self.bn_counter_keys else "frozen",
            "aggregation_stats_scope": "LoRA and auxiliary floating tensors; integer BN counters excluded",
            "frozen_backbone_unchanged": True,
            "communication_scope": "tensor payload bytes in the simulator; excludes serialization/network overhead; initialization recorded separately",
        })

    def apply_weight(self) -> None:
        """Evaluate W0 + Bg Ag in a separate state; never truncate to the evaluator rank."""
        payload = self._obj.node_var.aggregated_weight
        self._validate_state(payload)
        self._obj.node_var.model_weight = payload
        evaluated = OrderedDict((key, value.clone()) for key, value in self._initial_weight.items())
        for prefix, layer in self.layers.items():
            A, B = payload[layer.key_A], payload[layer.key_B]
            work_dtype = torch.float64 if A.dtype == torch.float64 or B.dtype == torch.float64 else torch.float32
            delta = B.to(work_dtype) @ A.to(work_dtype)
            weight_key = f"{prefix}.weight" if prefix else "weight"
            base = self._initial_weight[weight_key]
            delta = LoRAUtils._align_flora_delta_to_backbone(delta, base, weight_key=weight_key)
            evaluated[weight_key] = (base.to(device=delta.device, dtype=work_dtype) + delta).to(base)
            evaluated[layer.key_A].zero_()
            evaluated[layer.key_B].zero_()
        for key in self.extra_keys:
            evaluated[key] = payload[key].detach().to(evaluated[key]).clone()
        self._obj.node_var.model_evaluator.update_model(evaluated)
