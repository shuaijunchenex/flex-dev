from __future__ import annotations

from typing import Any, Dict

from flex.fed_strategy.server_strategy_impl._sp_server import SpServerStrategy
from flex.ml_algorithms.lora.lora_utils import LoRAUtils


class CloseOptimalServerStrategy(SpServerStrategy):
    """Close-optimal server pipeline with optional LoRA scale preservation."""

    def __init__(self, args, server_node) -> None:
        super().__init__(args, server_node)
        self._strategy_type = "close_optimal"
        self._maintain_lora_scale_ratio = bool(
            args.get("maintain_lora_scale_ratio", False)
        )
        self._client_profiles: Dict[str, Dict[str, Any]] = {}

    def _create_inner(self, args, server_node) -> None:
        self._args = args
        self._strategy_type = "close_optimal"
        self._obj = server_node
        self._maintain_lora_scale_ratio = bool(
            args.get("maintain_lora_scale_ratio", False)
        )
        return self

    @staticmethod
    def _lora_scalings(model) -> Dict[str, float]:
        return {
            name: float(getattr(module, "scaling", 1.0))
            for name, module in model.named_modules()
            if hasattr(module, "lora_A") and hasattr(module, "lora_B")
        }

    @staticmethod
    def _profile(model) -> Dict[str, Any]:
        lora_cfg = getattr(model, "lora_config", None) or {}
        suffix_A = lora_cfg.get("suffix_A", "lora_A")
        suffix_B = lora_cfg.get("suffix_B", "lora_B")
        return {
            "ranks": LoRAUtils.get_lora_ranks(model, suffix_A, suffix_B),
            "scalings": CloseOptimalServerStrategy._lora_scalings(model),
            "suffix_A": suffix_A,
            "suffix_B": suffix_B,
        }

    def _materialize_for_profile(self, factored, profile):
        weight = LoRAUtils.materialize_lora_from_factors(
            factored,
            profile["ranks"],
            lora_suffix_A=profile["suffix_A"],
            lora_suffix_B=profile["suffix_B"],
        )
        if self._maintain_lora_scale_ratio:
            for prefix, scaling in profile["scalings"].items():
                scale_root = scaling ** 0.5
                key_A = f"{prefix}.{profile['suffix_A']}"
                key_B = f"{prefix}.{profile['suffix_B']}"
                weight[key_A] = weight[key_A] / scale_root
                weight[key_B] = weight[key_B] / scale_root
        return weight

    def _refresh_client_profiles(self) -> None:
        self._client_profiles = {
            str(client.node_id): self._profile(client.node_var.model)
            for client in self._obj.client_nodes
        }

    def prepare(self, logger_header, client_nodes_in) -> None:
        super().prepare(logger_header, client_nodes_in)
        self._refresh_client_profiles()

    def _effective_client_updates(self, client_updates):
        scaled_updates = []
        for update in client_updates:
            client_id = str(update["train_record"]["node_id"])
            profile = self._client_profiles[client_id]
            scaled_state = update["updated_weights"].copy()
            suffix_B = profile["suffix_B"]
            for prefix, scaling in profile["scalings"].items():
                key_B = f"{prefix}.{suffix_B}"
                if key_B in scaled_state:
                    scaled_state[key_B] = scaled_state[key_B] * scaling

            scaled_update = update.copy()
            scaled_update["updated_weights"] = scaled_state
            scaled_updates.append(scaled_update)
        return scaled_updates

    def aggregation(self) -> None:
        self._refresh_client_profiles()
        client_updates = self._obj.node_var.client_updates
        if self._maintain_lora_scale_ratio:
            client_updates = self._effective_client_updates(client_updates)

        aggregator = self._obj.node_var.aggregation_method
        self._obj.node_var.aggregated_weight = aggregator.aggregate(client_updates)

    def _rank_cap(self, server_ranks: Dict[str, int]) -> Dict[str, int]:
        rank_cap = dict(server_ranks)
        for profile in self._client_profiles.values():
            for prefix, rank in profile["ranks"].items():
                rank_cap[prefix] = max(rank_cap.get(prefix, 0), rank)
        return rank_cap

    def apply_weight(self) -> None:
        node_var = self._obj.node_var
        inference_model = getattr(node_var, "inference_model", None) or node_var.model
        server_profile = self._profile(inference_model)
        lora_cfg = getattr(inference_model, "lora_config", None) or {}

        factored = LoRAUtils.cache_svd_factored_matrix(
            node_var,
            node_var.aggregated_weight,
            self._rank_cap(server_profile["ranks"]),
            sp_suffix=lora_cfg.get("sp_suffix", "sp_aggregated"),
        )
        node_var.model_weight = factored

        prepared = self._materialize_for_profile(factored, server_profile)
        node_var.model_evaluator.update_model(prepared)

    def broadcast(self) -> None:
        factored = getattr(self._obj.node_var, "sp_factored_weight", None)
        if factored is None or not LoRAUtils.has_factored_keys(factored):
            super().broadcast()
            return

        self._refresh_client_profiles()
        for client in self._obj.client_nodes:
            profile = self._client_profiles[str(client.node_id)]
            local_weight = self._materialize_for_profile(factored, profile)
            client.receive_weight(local_weight)
            client.set_local_weight()
