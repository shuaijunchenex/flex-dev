from __future__ import annotations

from typing import Any

from flex.ml_algorithms.lora.rank_scale_alignment import (
    LoRAScaleProfile,
    build_lora_scale_profile,
    validate_compatible_scale_profiles,
)
from flex.ml_utils import console
from flex.ml_utils.model_utils import ModelUtils

from ._sp_plus_server import SpPlusServerStrategy


class RblaPlusRankScaleServerStrategy(SpPlusServerStrategy):
    """SP+/RBLA+ with bidirectional, per-layer LoRA scale alignment.

    Client factors are expressed in the server's ``lora_alpha / rank``
    coordinates before the existing SP+ aggregator runs.  The inverse mapping
    is applied per client before the existing canonical-prefix receive path.
    No existing RBLA or SP+ implementation is modified by this strategy.
    """

    def __init__(self, args, server_node) -> None:
        super().__init__(args, server_node)
        self._strategy_type = "rbla_plus_rank_scale"
        self._server_rank_scale_profile: LoRAScaleProfile | None = None
        self._clients_by_id: dict[str, Any] = {}

    def _create_inner(self, args, server_node) -> None:
        self._args = args
        self._strategy_type = "rbla_plus_rank_scale"
        self._obj = server_node
        self._server_rank_scale_profile = None
        self._clients_by_id = {}
        return self

    @property
    def server_rank_scale_profile(self) -> LoRAScaleProfile:
        if self._server_rank_scale_profile is None:
            self._prepare_rank_scale_context()
        return self._server_rank_scale_profile

    def _prepare_rank_scale_context(self) -> None:
        """Build the server profile and ask each client to cache its profile."""

        server_model = ModelUtils.unwrap_model(self._obj.node_var.model)
        self._server_rank_scale_profile = build_lora_scale_profile(server_model)
        self._clients_by_id = {}

        for client in self._obj.client_nodes:
            client_id = str(client.node_id)
            if client_id in self._clients_by_id:
                raise ValueError(f"Duplicate client id '{client_id}'")

            client_strategy = getattr(client, "strategy", None)
            if not hasattr(client_strategy, "refresh_rank_scale_profile"):
                raise TypeError(
                    f"Client '{client_id}' must use the rbla_plus_rank_scale "
                    "client strategy"
                )

            client_profile = client_strategy.refresh_rank_scale_profile()
            validate_compatible_scale_profiles(
                client_profile,
                self._server_rank_scale_profile,
            )
            self._clients_by_id[client_id] = client

    def _ensure_rank_scale_context(self) -> None:
        if self._server_rank_scale_profile is None or not self._clients_by_id:
            self._prepare_rank_scale_context()

    def _get_client(self, client_id: str):
        if client_id not in self._clients_by_id:
            raise KeyError(
                f"No rank-scale client is registered with id '{client_id}'"
            )
        return self._clients_by_id[client_id]

    @staticmethod
    def _client_id_from_update(update: dict[str, Any]) -> str:
        if "node_id" in update:
            return str(update["node_id"])
        train_record = update.get("train_record")
        if isinstance(train_record, dict) and "node_id" in train_record:
            return str(train_record["node_id"])
        raise KeyError(
            "A client update for rbla_plus_rank_scale must contain node_id "
            "either directly or in train_record"
        )

    def prepare(self, logger_header, client_nodes_in) -> None:
        super().prepare(logger_header, client_nodes_in)
        self._prepare_rank_scale_context()

    def _normalize_one_update(self, update: dict[str, Any]) -> dict[str, Any]:
        client_id = self._client_id_from_update(update)
        client = self._get_client(client_id)
        if "updated_weights" not in update:
            raise KeyError(
                f"Client '{client_id}' update does not contain updated_weights"
            )

        normalized_weight = client.strategy.normalize_upload_for_server(
            update["updated_weights"],
            self.server_rank_scale_profile,
        )
        normalized_update = dict(update)
        normalized_update["updated_weights"] = normalized_weight
        return normalized_update

    def aggregation(self) -> None:
        self._ensure_rank_scale_context()
        normalized_updates = [
            self._normalize_one_update(update)
            for update in self._obj.node_var.client_updates
        ]
        aggregator = self._obj.node_var.aggregation_method
        self._obj.node_var.aggregated_weight = aggregator.aggregate(normalized_updates)

    def broadcast(self) -> None:
        self._ensure_rank_scale_context()
        global_weight = self._obj.node_var.model_weight

        for client in self._obj.client_nodes:
            client_id = str(client.node_id)
            registered_client = self._get_client(client_id)
            client_weight = registered_client.strategy.prepare_broadcast_from_server(
                global_weight,
                self.server_rank_scale_profile,
            )
            registered_client.receive_weight(client_weight)
            registered_client.set_local_weight()

        console.debug(
            "[RBLA+ RankScale] Broadcast server-scale factors in each client's "
            "LoRA scale coordinates."
        )
