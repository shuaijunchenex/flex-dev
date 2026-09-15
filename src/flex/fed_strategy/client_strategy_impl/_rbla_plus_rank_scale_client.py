from __future__ import annotations

from typing import Mapping

import torch

from flex.ml_algorithms.lora.rank_scale_alignment import (
    LoRAScaleProfile,
    align_lora_state_dict_scale,
    build_lora_scale_profile,
)
from flex.ml_utils.model_utils import ModelUtils

from ._sp_plus_client import SpPlusClientTrainingStrategy


class RblaPlusRankScaleClientTrainingStrategy(SpPlusClientTrainingStrategy):
    """Unchanged SP+ local training for the scale-aware RBLA+ pipeline."""

    def __init__(self, args, client_node):
        super().__init__(args, client_node)
        self._strategy_type = "rbla_plus_rank_scale"
        self._rank_scale_profile: LoRAScaleProfile | None = None
        self._rank_scale_profile_model_id: int | None = None

    def _create_inner(self, args, client_node) -> None:
        self._args = args
        self._strategy_type = "rbla_plus_rank_scale"
        self._obj = client_node
        self._rank_scale_profile = None
        self._rank_scale_profile_model_id = None
        return self

    def refresh_rank_scale_profile(self) -> LoRAScaleProfile:
        """Read and cache this client's fixed LoRA rank/scale metadata."""

        model = ModelUtils.unwrap_model(self._obj.node_var.model)
        self._rank_scale_profile = build_lora_scale_profile(model)
        self._rank_scale_profile_model_id = id(model)
        return self._rank_scale_profile

    @property
    def rank_scale_profile(self) -> LoRAScaleProfile:
        """Return the cached profile, rebuilding it if the model was replaced."""

        model = ModelUtils.unwrap_model(self._obj.node_var.model)
        if (
            self._rank_scale_profile is None
            or self._rank_scale_profile_model_id != id(model)
        ):
            return self.refresh_rank_scale_profile()
        return self._rank_scale_profile

    def normalize_upload_for_server(
        self,
        uploaded_weight: Mapping[str, torch.Tensor],
        server_profile: LoRAScaleProfile,
    ) -> dict[str, torch.Tensor]:
        """Express this client's uploaded factors in server scale coordinates."""

        return align_lora_state_dict_scale(
            uploaded_weight,
            source_profile=self.rank_scale_profile,
            target_profile=server_profile,
        )

    def prepare_broadcast_from_server(
        self,
        global_weight: Mapping[str, torch.Tensor],
        server_profile: LoRAScaleProfile,
    ) -> dict[str, torch.Tensor]:
        """Express server factors in this client's local scale coordinates."""

        return align_lora_state_dict_scale(
            global_weight,
            source_profile=server_profile,
            target_profile=self.rank_scale_profile,
        )
