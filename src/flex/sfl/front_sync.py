from flex.fl_algorithms.aggregation.fed_aggregator_args import FedAggregatorArgs
from flex.fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from flex.ml_algorithms.lora.rank_scale_alignment import build_lora_scale_profile, align_lora_state_dict_scale
from .sfl_type import EFrontSyncMethod


class FrontSynchronizer:
    """Aggregate only the uploaded front factors; the server model is the rear."""

    def __init__(self, clients, method):
        self.method = EFrontSyncMethod(method)
        self.version = 0
        self.factors = {}
        self.templates = {c.node_id: c.strategy.export_front() for c in clients}
        active = [c for c in clients if self.templates[c.node_id]]
        if not active or self.method == EFrontSyncMethod.local:
            return
        if self.method == EFrontSyncMethod.fedavg:
            # Dense front: average every uploaded parameter, no LoRA structure needed.
            self.factors = self.templates[active[0].node_id]
            self.aggregator = self._create_aggregator("fedavg")
            return
        self.profiles = {c.node_id: build_lora_scale_profile(c.node_var.model) for c in active}
        ranks = {cid: max(layer.rank for layer in p.values()) for cid, p in self.profiles.items()}
        if self.method == EFrontSyncMethod.factor_avg and len(set(ranks.values())) > 1:
            raise ValueError("Different front ranks require rbla_plus")
        self.reference_id = max(ranks, key=ranks.get)
        self.factors = self.templates[self.reference_id]
        if self.method == EFrontSyncMethod.factor_avg:
            self.aggregator = self._create_aggregator("fedavg")
        else:
            # "nan" padding: a rank missing on a client is ignored instead of averaged as zero.
            self.aggregator = self._create_aggregator("sp_plus", pad_mode="nan")

    @staticmethod
    def _create_aggregator(method, **options):
        return FedAggregatorFactory.create_aggregator(FedAggregatorArgs({"method": method, "device": "cpu", **options}))

    def aggregate(self, updates):
        active = [u for u in updates if u["updated_weights"] and u["train_record"]["steps"]]
        if self.method == EFrontSyncMethod.local or not active:
            return self.version
        if any(u["base_version"] != self.version for u in active):
            raise ValueError("Front update came from a different broadcast version")
        if self.method == EFrontSyncMethod.rbla_plus:
            ref = self.profiles[self.reference_id]
            active = [
                {**u, "updated_weights": align_lora_state_dict_scale(
                    u["updated_weights"], self.profiles[u["node_id"]], ref)}
                for u in active
            ]
        self.factors = self.aggregator.aggregate(active)
        self.version += 1
        return self.version

    def payload_for(self, client_id):
        local = self.templates[client_id]
        if not local or self.method == EFrontSyncMethod.local:
            return {}
        if self.method in (EFrontSyncMethod.factor_avg, EFrontSyncMethod.fedavg):
            return self.factors
        aligned = align_lora_state_dict_scale(self.factors, self.profiles[self.reference_id],
                                              self.profiles[client_id])
        return self.aggregator.broadcast_lora_state_dict(aligned, local)

