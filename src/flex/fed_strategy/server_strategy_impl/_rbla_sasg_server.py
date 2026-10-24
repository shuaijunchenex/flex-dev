"""
RBLA-SASG server strategy.

Broadcasts per-client semantic slots instead of full weight dicts.
Aggregates per semantic slot.
"""

from __future__ import annotations

from typing import Any, Dict, List

from flex.fed_strategy.server_strategy import ServerStrategy
from flex.ml_algorithms.rblasa.semantic_grid import (
    LinearSemanticGrid,
    AdaptiveSemanticGrid,
    slice_global_for_client,
)
from flex.ml_utils import console


class RblaSasgServerStrategy(ServerStrategy):
    """
    Server strategy for RBLA-SASG.

    - Broadcast: sends only assigned semantic slots to each client.
    - Aggregation: delegates to the SASG aggregator.
    - Maintains global LoRA slots A_g / B_g across rounds.
    - Supports ``LinearSemanticGrid`` (default) and ``AdaptiveSemanticGrid``
      via the ``rbla_sasg.adaptive_grid`` config key.
    """

    def __init__(self, args, server_node) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "rbla_sasg"
        self._obj = server_node
        # Per-prefix global slot tensors (updated each round)
        self._global_A: Dict[str, Any] = {}
        self._global_B: Dict[str, Any] = {}
        self._max_rank: int = 0

        # Semantic grid (initialized lazily in _ensure_grid)
        self._grid: LinearSemanticGrid | AdaptiveSemanticGrid | None = None
        self._grid_initialized: bool = False

    def _create_inner(self, args, server_node) -> None:
        self._args = args
        self._strategy_type = "rbla_sasg"
        self._obj = server_node
        return self

    # ------------------------------------------------------------------
    # Grid helper
    # ------------------------------------------------------------------

    def _ensure_grid(self, max_rank: int) -> None:
        """Create the semantic grid instance if not yet initialized,
        or if ``max_rank`` has changed."""
        if self._grid is not None and self._grid.max_rank == max_rank:
            return

        cfg = self._obj.node_var.config_dict.get("rbla_sasg", {})
        adaptive = bool(cfg.get("adaptive_grid", False))

        if adaptive:
            self._grid = AdaptiveSemanticGrid(
                max_rank=max_rank,
                warmup_rounds=int(cfg.get("warmup_rounds", 3)),
                rho=float(cfg.get("grid_rho", 0.9)),
            )
        else:
            self._grid = LinearSemanticGrid(max_rank=max_rank)

    # ------------------------------------------------------------------
    # Core strategy methods
    # ------------------------------------------------------------------

    def aggregation(self) -> None:
        aggregator = self._obj.node_var.aggregation_method
        aggregated_weights = aggregator.aggregate(self._obj.node_var.client_updates)
        self._obj.node_var.aggregated_weight = aggregated_weights

        # Cache per-slot A/B from aggregator for broadcast
        self._global_A = getattr(aggregator, "slot_A", {})
        self._global_B = getattr(aggregator, "slot_B", {})
        self._max_rank = getattr(aggregator, "max_rank", 0)

        # ── Update adaptive grid EMA from global factors ──
        if isinstance(self._grid, AdaptiveSemanticGrid):
            for prefix in self._global_A:
                if prefix not in self._global_B:
                    continue
                A_g = self._global_A[prefix]
                B_g = self._global_B[prefix]
                R_p = A_g.shape[0]
                self._ensure_grid(R_p)
                self._grid.update_strengths(A_g, B_g)

    def select_clients(self, available_clients) -> list:
        selector = self._obj.node_var.client_selection
        n = self._obj.node_var.config_dict["client_selection"]["number"]
        return selector.select(available_clients, n)

    def record_evaluation(self) -> None:
        self._obj.node_var.training_logger.record(self._obj.eval_results)

    def receive_client_updates(self, client_updates) -> None:
        """Store enriched client_updates (include r_i, Phi_i)."""
        self._obj.node_var.client_updates = client_updates

    # ------------------------------------------------------------------
    # Broadcast — per-client semantic slot slicing
    # ------------------------------------------------------------------
    def broadcast(self) -> None:
        """Send each client only its assigned semantic slots."""
        for client in self._obj.client_nodes:
            self._broadcast_to_client(client)

    def _broadcast_to_client(self, client) -> None:
        node_var = client.node_var
        if node_var is None or node_var.model is None:
            return

        # Determine client local rank PER PREFIX from its model
        model = node_var.model
        from flex.ml_algorithms.lora.lora_utils import LoRAUtils
        rank_dict = LoRAUtils.get_lora_ranks(model)
        r_i = max(rank_dict.values()) if rank_dict else 0
        if r_i <= 0:
            r_i = 1

        # Determine max_rank and global A/B source
        if self._global_A and self._global_B:
            # After aggregation: use cached global slots
            max_rank = getattr(self, "_max_rank", 0) or 0
            if max_rank <= 0:
                for A in self._global_A.values():
                    max_rank = max(max_rank, A.shape[0])
            A_src = self._global_A
            B_src = self._global_B
        else:
            # Before first aggregation: extract from server model_weight
            full_weight = getattr(self._obj.node_var, "model_weight", None) or {}
            A_src, B_src = {}, {}
            for key, tensor in full_weight.items():
                if not hasattr(tensor, "dim"):
                    continue
                suf = key.rsplit(".", 1)[-1]
                prefix = key.rsplit(".", 1)[0]
                if suf == "lora_A":
                    A_src[prefix] = tensor
                elif suf == "lora_B":
                    B_src[prefix] = tensor
            max_rank = 0
            for A in A_src.values():
                max_rank = max(max_rank, A.shape[0])
            for B in B_src.values():
                max_rank = max(max_rank, B.shape[1])
            if max_rank <= 0:
                max_rank = r_i

        r_i = min(r_i, max_rank)
        self._ensure_grid(max_rank)
        Phi_i = self._grid.get_slot_mapping(r_i)

        # Build per-prefix sliced state_dict
        sliced_state: Dict[str, Any] = {}

        # Copy non-LoRA keys from server model_weight
        full_weight = getattr(self._obj.node_var, "model_weight", None) or {}
        for key, val in full_weight.items():
            if not isinstance(val, (int, float)) and hasattr(val, "dim"):
                suf = key.rsplit(".", 1)[-1]
                if suf in ("lora_A", "lora_B"):
                    continue
            sliced_state[key] = val.clone() if hasattr(val, "clone") else val

        # ── Slice LoRA factors per prefix using client's actual per-prefix rank ──
        for prefix in A_src:
            if prefix not in B_src:
                continue
            A_g = A_src[prefix]   # [R_prefix, d_in]
            B_g = B_src[prefix]   # [d_out, R_prefix]
            R_prefix = A_g.shape[0]

            # Client's actual rank for this prefix
            r_i_prefix = rank_dict.get(prefix, r_i)
            r_i_prefix = min(r_i_prefix, R_prefix)
            self._ensure_grid(R_prefix)
            Phi_prefix = self._grid.get_slot_mapping(r_i_prefix)

            A_i, B_i = slice_global_for_client(A_g, B_g, Phi_prefix)
            sliced_state[f"{prefix}.lora_A"] = A_i
            sliced_state[f"{prefix}.lora_B"] = B_i

        if hasattr(client.strategy, "receive_sasg_slots"):
            # Build per-prefix metadata from the slicing loop above
            rank_by_prefix: Dict[str, int] = {}
            Phi_by_prefix: Dict[str, List[int]] = {}
            for prefix in A_src:
                if prefix not in B_src:
                    continue
                R_prefix = A_src[prefix].shape[0]
                r_i_p = rank_dict.get(prefix, r_i)
                r_i_p = min(r_i_p, R_prefix)
                rank_by_prefix[prefix] = r_i_p
                self._ensure_grid(R_prefix)
                Phi_by_prefix[prefix] = self._grid.get_slot_mapping(r_i_p)

            client.strategy.receive_sasg_slots(
                sliced_state, r_i, Phi_i,
                rank_by_prefix=rank_by_prefix,
                Phi_by_prefix=Phi_by_prefix,
            )
        else:
            client.receive_weight(sliced_state)
            client.set_local_weight()

    # ------------------------------------------------------------------
    # Prepare / Evaluate
    # ------------------------------------------------------------------
    def evaluate(self) -> None:
        self._obj.eval_results = self._obj.node_var.model_evaluator.evaluate()
        self._obj.node_var.model_evaluator.print_results()
        console.info("[RBLA-SASG] Server Evaluation Completed.\n")

    def prepare(self, logger_header, client_nodes_in) -> None:
        self._obj.node_var.training_logger.begin(logger_header)
        self._obj.set_client_nodes(client_nodes_in)

    def run(self) -> None:
        raise NotImplementedError

    def server_update(self, weight) -> None:
        """Optional: apply an external weight update (e.g. from runner simulation)."""
        pass
