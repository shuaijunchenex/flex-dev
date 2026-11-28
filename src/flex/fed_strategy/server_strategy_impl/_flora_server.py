from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any, Dict

from flex.fed_strategy.server_strategy import ServerStrategy
from flex.ml_algorithms.lora.lora_utils import LoRAUtils
from flex.ml_utils import console


class FloraServerStrategy(ServerStrategy):
    """FLoRA (Wang et al., NeurIPS 2024) — server-side merge strategy.

    Core flow per round:

    1. **Aggregation** — FLoRA stacking :math:`\\Delta W = \\sum_i p_i \\, B_i A_i`
       (performed by ``FedAggregator_Flora``, outputs ``{prefix}.sp_aggregated``).
    2. **Merge** — :math:`W \\leftarrow W + \\Delta W` for every LoRA layer.
       The updated *W* becomes the new frozen backbone for the next round.
    3. **Broadcast** — the full merged *W* (not LoRA factors) is sent to all
       clients, who freeze it and freshly initialise local-rank A/B with a
       zero product.
    4. **Evaluation** — the evaluator model is loaded with the merged *W* and
       **zero‑init** LoRA A/B (because :math:`\\Delta W` is already absorbed).

    Key difference from the SP‑style pathway:
    SP keeps the backbone permanently frozen and broadcasts SVD‑decomposed
    LoRA factors.  FLoRA **updates** the backbone each round by adding the
    stacked :math:`\\Delta W`, so the broadcast payload is the full weight
    matrix and A/B are freshly re-initialised every round.

    Non‑LoRA parameters and buffers remain at the synchronized frozen
    backbone state; only LoRA A/B contribute a trainable update.
    """

    def __init__(self, args, server_node) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "flora"
        self._obj = server_node
        # Per‑round accumulating backbone (initialised on first apply_weight).
        self._backbone_weight: Dict[str, Any] | None = None

    def _create_inner(self, args, server_node) -> None:
        self._args = args
        self._strategy_type = "flora"
        self._obj = server_node
        return self

    # ------------------------------------------------------------------
    # Core strategy methods
    # ------------------------------------------------------------------

    def aggregation(self) -> None:
        """Run FLoRA stacking aggregation → ΔW (stored in aggregated_weight)."""
        aggregator = self._obj.node_var.aggregation_method
        aggregated_weights = aggregator.aggregate(self._obj.node_var.client_updates)
        self._obj.node_var.aggregated_weight = aggregated_weights

    def select_clients(self, available_clients) -> list:
        selector = self._obj.node_var.client_selection
        return selector.select(
            available_clients,
            self._obj.node_var.config_dict["client_selection"]["number"],
        )

    def receive_client_updates(self, client_updates) -> None:
        self._obj.node_var.client_updates = client_updates

    def broadcast(self) -> None:
        """Broadcast the full merged *W* matrix to every client.

        Clients receive the complete backbone weight (not LoRA factors) and
        freeze it while freshly initialising their own-rank LoRA factors.
        """
        if self._backbone_weight is None:
            # Snapshot the same initial W that the runner broadcasts before
            # round 0.  The first aggregate is therefore merged into the
            # backbone from which every client actually trained.
            self._backbone_weight = copy.deepcopy(self._obj.node_var.model_weight)
        for client in self._obj.client_nodes:
            client.receive_weight(self._obj.node_var.model_weight)
            client.set_local_weight()

    # ------------------------------------------------------------------
    # Weight application — the FLoRA merge step
    # ------------------------------------------------------------------

    def apply_weight(self) -> None:
        """Merge ΔW into the frozen backbone and update the evaluator.

        Step‑by‑step:

        1. Snapshot the initial backbone on first call.
        2. For every ``{prefix}.sp_aggregated`` key in the aggregated output,
           compute ``W_new = W_old + ΔW``.
        3. Store *W_new* as ``model_weight`` for broadcast.
        4. Build an evaluation‑compatible state dict (W_new + zero A/B) and
           update the evaluator model.
        """
        aggregated = self._obj.node_var.aggregated_weight
        node_var = self._obj.node_var

        # ---- 1. Lazy‑init the backbone snapshot ----
        if self._backbone_weight is None:
            self._backbone_weight = copy.deepcopy(node_var.model_weight)

        # ---- 2. Merge ΔW into backbone ----
        merged_weight = LoRAUtils.merge_flora_delta_to_backbone(
            aggregated,
            self._backbone_weight,
            sp_suffix="sp_aggregated",
        )

        # ---- 3. Cache for next round + broadcast ----
        self._backbone_weight = merged_weight
        node_var.model_weight = merged_weight

        # ---- 4. Build evaluator weight (W_new + zero A/B) ----
        eval_model = (
            getattr(node_var, "inference_model", None)
            or getattr(node_var, "model", None)
        )
        eval_weight = LoRAUtils.build_eval_weight_with_merged_backbone(
            merged_weight,
            eval_model,
        )
        node_var.model_evaluator.update_model(eval_weight)

    # ------------------------------------------------------------------
    # Evaluation & logging
    # ------------------------------------------------------------------

    def evaluate(self) -> None:
        self._obj.eval_results = self._obj.node_var.model_evaluator.evaluate()
        self._obj.node_var.model_evaluator.print_results()
        console.info("Server Evaluation Completed.\n")

    def record_evaluation(self) -> None:
        self._obj.node_var.training_logger.record(self._obj.eval_results)

    def prepare(self, logger_header, client_nodes_in) -> None:
        self._obj.node_var.training_logger.begin(logger_header)
        self._obj.set_client_nodes(client_nodes_in)

    def run(self) -> Dict[str, Any]:
        raise NotImplementedError
