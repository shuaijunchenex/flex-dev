from __future__ import annotations

import copy

from ..fed_aggregator_args import FedAggregatorArgs
from ._fed_aggregator_rbla import FedAggregator_RBLA


class FedAggregator_SPPlus(FedAggregator_RBLA):
    """Standalone SP+ aggregation used by the former RBLA+ experiment path.

    SP+ intentionally preserves the RBLA+ computation: heterogeneous LoRA
    factors are aggregated with RBLA and then canonicalized before broadcast.
    Keeping it behind its own method name prevents the optional RBLA
    canonicalization switch from being used as the experiment identity.
    """

    def __init__(self, args: FedAggregatorArgs | None = None):
        config = copy.deepcopy(args.key_value_dict.data) if args is not None else {}
        canonicalization = dict(config.get("canonicalization") or {})
        # Canonical prefix ordering is the defining SP+ behavior.  Selecting
        # method=sp_plus therefore always enables it; scheduling and numerical
        # options remain configurable through the same mapping.
        canonicalization["enabled"] = True
        # Preserve the configured LoRA tensor shapes when a layer's requested
        # rank exceeds its intrinsic matrix rank. Users can explicitly switch
        # this back to "error" for strict validation.
        canonicalization.setdefault("overcomplete_policy", "zero_pad")
        config["canonicalization"] = canonicalization

        super().__init__(FedAggregatorArgs(config))
        self._aggregation_method = "sp_plus"

    @staticmethod
    def broadcast_lora_state_dict(
        global_sd: dict,
        local_sd: dict,
        lora_suffixes: set[str] = {"lora_A", "lora_B"},
    ) -> dict:
        """Broadcast the leading canonical factors for a client's local rank."""
        return FedAggregator_RBLA.broadcast_lora_state_dict(
            global_sd,
            local_sd,
            lora_suffixes,
        )
