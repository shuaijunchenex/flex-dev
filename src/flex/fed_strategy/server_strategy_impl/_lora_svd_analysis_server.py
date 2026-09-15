from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from flex.fed_strategy.server_strategy import ServerStrategy
from flex.ml_algorithms.lora.lora_utils import LoRAUtils
from flex.ml_utils import console
from flex.ml_utils.model_utils import ModelUtils


class LoraSvdAnalysisServerStrategy(ServerStrategy):
    """
    Server strategy for the LoRA SVD analysis experiment.

    Uses SP-style aggregation and broadcast logic:
      - apply_weight: SVD-splits the aggregated full-weight matrix for the evaluator.
      - broadcast:    Each client receives the full aggregated weight; set_local_weight
                      (on the SP client) will SVD-split it back into LoRA A/B components.
    The heavy SVD analysis is done in the runner strategy.
    """

    def __init__(self, args, server_node) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "lora_svd"
        self._obj = server_node

    def _create_inner(self, args=None, server_node=None) -> None:
        return self

    # ------------------------------------------------------------------
    # Lifecycle – SP-style
    # ------------------------------------------------------------------
    def aggregation(self) -> None:
        aggregator = self._obj.node_var.aggregation_method
        aggregated_weights = aggregator.aggregate(self._obj.node_var.client_updates)
        self._obj.node_var.aggregated_weight = aggregated_weights

    def select_clients(self, available_clients) -> list:
        selector = self._obj.node_var.client_selection
        n = self._obj.node_var.config_dict["client_selection"]["number"]
        return selector.select(available_clients, n)

    def record_evaluation(self) -> None:
        self._obj.node_var.training_logger.record(self._obj.eval_results)

    def receive_client_updates(self, client_updates) -> None:
        self._obj.node_var.client_updates = client_updates

    def broadcast(self) -> None:
        """SP-style: send the full aggregated weight; each SP client will SVD-split it locally."""
        for client in self._obj.client_nodes:
            client.receive_weight(self._obj.node_var.model_weight)
            client.set_local_weight()

    def evaluate(self) -> None:
        self._obj.eval_results = self._obj.node_var.model_evaluator.evaluate()
        self._obj.node_var.model_evaluator.print_results()
        console.info("Server Evaluation Completed.\n")

    def prepare(self, logger_header, client_nodes_in) -> None:
        self._obj.node_var.training_logger.begin(logger_header)
        self._obj.set_client_nodes(client_nodes_in)

    # ------------------------------------------------------------------
    # Probe training — train only W (freeze lora_A / lora_B) on a COPY of
    # the global model.  Returns mean gradients of W; no weight is updated.
    # ------------------------------------------------------------------
    def probe_train_global_model(
        self,
        probe_epochs: int = 3,
    ) -> dict[str, np.ndarray]:
        """
        Deep-copy the current global model, **freeze** every ``lora_A`` /
        ``lora_B`` parameter and keep only the base weight ``W`` trainable.
        Run an inline training loop for *probe_epochs* epochs, accumulate the
        real gradient of each ``*.weight`` parameter, and return the mean
        gradient dict without writing anything back to ``node_var``.

        Returns
        -------
        dict[str, np.ndarray]
            ``{ param_name -> mean_grad }`` for every trainable ``*.weight``
            that received a gradient.  Empty dict on error.
        """
        nv = self._obj.node_var

        if nv.model is None or nv.data_loader is None or nv.loss_func is None:
            console.warn(
                "[LoRA SVD probe] Skipping: server node_var missing model / "
                "data_loader / loss_func."
            )
            return {}

        device = getattr(nv, "device", "cpu") or "cpu"

        # 1. Deep-copy and load current global weights.
        probe_model: nn.Module = copy.deepcopy(ModelUtils.unwrap_model(nv.model)).to(device)
        try:
            probe_model.load_state_dict(nv.model_weight, strict=True)
        except RuntimeError:
            try:
                svd_w = LoRAUtils.svd_split_global_weight(
                    nv.model_weight, LoRAUtils.get_lora_ranks(probe_model)
                )
                probe_model.load_state_dict(svd_w, strict=True)
            except Exception as exc:
                console.warn(f"[LoRA SVD probe] Cannot load global weights: {exc}")
                return {}

        # 2. Freeze lora_A / lora_B; keep only *.weight trainable.
        weight_params: list[tuple[str, nn.Parameter]] = []
        for name, param in probe_model.named_parameters():
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                param.requires_grad_(False)
            elif name.endswith(".weight"):
                param.requires_grad_(True)
                weight_params.append((name, param))
            else:
                param.requires_grad_(False)

        if not weight_params:
            console.warn("[LoRA SVD probe] No trainable *.weight params found.")
            return {}

        # 3. Build optimizer over W params only.
        probe_optimizer = nv.optimizer_builder.rebuild([p for _, p in weight_params])
        ModelUtils.clear_all(probe_model, probe_optimizer)

        # 4. Inline training loop — accumulate real W gradients.
        probe_model.train()
        train_dl = nv.data_loader.data_loader
        grad_accum: dict[str, torch.Tensor | None] = {name: None for name, _ in weight_params}
        backward_count = 0

        console.info(
            f"[LoRA SVD probe] Training W only for {probe_epochs} epoch(s) "
            f"({len(weight_params)} weight param(s))..."
        )
        try:
            for _ in range(probe_epochs):
                for inputs, labels in train_dl:
                    inputs = inputs.to(device)
                    labels = labels.to(device).long()

                    probe_optimizer.zero_grad()
                    outputs = probe_model(inputs)
                    loss = nv.loss_func(outputs, labels)
                    loss.backward()

                    # Accumulate gradient BEFORE step.
                    for name, param in weight_params:
                        if param.grad is not None:
                            g = param.grad.detach().float().cpu()
                            grad_accum[name] = (
                                g if grad_accum[name] is None else grad_accum[name] + g
                            )

                    backward_count += 1
                    probe_optimizer.step()
        except Exception as exc:
            console.warn(f"[LoRA SVD probe] Probe training failed: {exc}")
            return {}

        if backward_count == 0:
            return {}

        console.info(
            f"[LoRA SVD probe] Done: {probe_epochs} epoch(s), "
            f"{backward_count} backward passes."
        )
        return {
            name: (grad_accum[name] / backward_count).numpy()
            for name, _ in weight_params
            if grad_accum[name] is not None
        }

    def run(self) -> Dict[str, Any]:
        raise NotImplementedError
