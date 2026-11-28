import copy

import torch

from flex.fl_algorithms.aggregation.fed_aggregator_args import FedAggregatorArgs
from flex.fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from .sfl_type import ERearHistory, ERearMode


def _snapshot(params):
    return {name: p.detach().to("cpu", copy=True) for name, p in params.items()}


class RearStateStore:
    """One active rear, with explicit ownership of its adapter and optimizer slots."""

    def __init__(self, model, optimizer, *, mode="shared", history="shared"):
        self.model = model
        self.optimizer = optimizer
        self.mode = ERearMode(mode)
        self.history = ERearHistory(history)
        self.trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if mode == "private_adapter":
            self.body = {name: self.trainable[name] for name in model.adapter_keys() if name in self.trainable}
        elif mode == "shared" and history != "shared":
            head = set(model.head.parameters())
            self.body = {name: p for name, p in self.trainable.items() if p not in head}
        else:
            self.body = {}
        self.initial = _snapshot(self.body) if mode == "private_adapter" else {}
        self.slot_params = self.trainable if mode == "round_average" else self.body
        # Some optimizers (Adagrad) allocate slots at construction; others do so on first step.
        self.cold_slots = self._save_slots()
        self.private = {}
        self.slots = {}
        self.last_client = None
        self.aggregator = FedAggregatorFactory.create_aggregator(FedAggregatorArgs({"method": "fedavg", "device": "cpu"}))

    @torch.no_grad()
    def _install(self, weights):
        for name, tensor in weights.items():
            self.trainable[name].copy_(tensor)

    def _slot_ids(self):
        names = {id(p): n for n, p in self.slot_params.items()}
        # PyTorch serializes parameter IDs in optimizer-group order.
        parameters = [p for g in self.optimizer.param_groups for p in g["params"]]
        return {i: names[id(p)] for i, p in enumerate(parameters) if id(p) in names}

    def _save_slots(self):
        if self.optimizer is None:
            return {}
        state = self.optimizer.state_dict()["state"]
        return {name: {k: v.detach().to("cpu", copy=True) if torch.is_tensor(v) else v
                       for k, v in state[i].items()}
                for i, name in self._slot_ids().items() if i in state}

    def _load_slots(self, slots=None):
        if self.optimizer is None:
            return
        if slots is None:
            slots = copy.deepcopy(self.cold_slots)
        packed = self.optimizer.state_dict()
        for i, name in self._slot_ids().items():
            packed["state"].pop(i, None)
            if name in slots:
                packed["state"][i] = slots[name]
        self.optimizer.load_state_dict(packed)

    def begin_round(self):
        if self.mode == "round_average":
            self.round_start = _snapshot(self.trainable)
            self.visits = []

    def activate(self, client_id):
        if self.mode == "round_average":
            self._install(self.round_start)
            self._load_slots(self.slots.pop(client_id, None))
        elif self.mode == "private_adapter":
            self._install(self.private.get(client_id, self.initial))
            self._load_slots(self.slots.pop(client_id, None))
        elif self.history == "per_client":
            self._load_slots(self.slots.pop(client_id, None))
        elif self.history == "reset_on_switch" and client_id != self.last_client:
            self._load_slots()
        self.last_client = client_id

    def finish_visit(self, client_id, shard_size):
        if self.mode == "round_average":
            # Same aggregator as FL, so a single client reproduces FedAvg bit-exactly.
            self.visits.append({"updated_weights": _snapshot(self.trainable),
                                "train_record": {"data_sample_num": shard_size}})
            if self.history == "per_client":
                self.slots[client_id] = self._save_slots()
        elif self.mode == "private_adapter":
            self.private[client_id] = _snapshot(self.body)
            self.slots[client_id] = self._save_slots()
        elif self.history == "per_client":
            self.slots[client_id] = self._save_slots()

    def commit_round(self):
        if self.mode == "round_average" and self.visits:
            self._install(self.aggregator.aggregate(self.visits))
            self._load_slots()

    def state_for_eval(self, client_id):
        state = dict(self.model.state_dict())
        if self.mode == "private_adapter":
            state.update(self.private.get(client_id, self.initial))
        return state

