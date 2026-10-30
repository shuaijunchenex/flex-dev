"""Small, offline FLoRIST lifecycle tests using real autograd and SGD."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import yaml

from flex.fed_strategy.client_strategy_impl._florist_client import (
    FloristClientTrainingStrategy,
    _payload_bytes,
)
from flex.fed_strategy.server_strategy_impl._florist_server import FloristServerStrategy
from flex.fed_strategy.runner_strategy_impl._rbla_runner_strategy import RblaRunnerStrategy
from flex.fed_strategy.strategy_args import StrategyArgs
from flex.fed_strategy.strategy_factory import StrategyFactory
from flex.fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from flex.fl_algorithms.aggregation.methods._fed_aggregator_florist import FedAggregator_Florist
from flex.ml_algorithms.lora.impl.lora_ms import MSLoRALinear, MSEmbedding


class _Model(nn.Module):
    def __init__(self, rank, alpha=8, merge=False):
        super().__init__()
        self.layer = MSLoRALinear(5, 4, r=rank, lora_alpha=alpha, merge_weights=merge)
        self.head = nn.Linear(4, 2)
        self.register_buffer("frozen_counter", torch.tensor(13))

    def forward(self, x):
        return self.head(torch.tanh(self.layer(x)))


class _Trainer:
    def __init__(self, count):
        generator = torch.Generator().manual_seed(count)
        self.x = torch.randn(count, 5, generator=generator)
        self.y = torch.arange(count) % 2
        self.trainer_args = SimpleNamespace(model=None, optimizer=None, device="cpu")
        self.inputs = []

    def set_model(self, model):
        self.trainer_args.model = model

    def set_optimizer(self, optimizer):
        self.trainer_args.optimizer = optimizer

    def train(self, epochs):
        model, optimizer = self.trainer_args.model, self.trainer_args.optimizer
        self.inputs.append({key: value.clone() for key, value in model.state_dict().items()})
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(self.x), self.y)
            loss.backward()
            optimizer.step()
        return ({key: value.detach().clone() for key, value in model.state_dict().items()},
                {"data_sample_num": len(self.y), "avg_loss": float(loss.detach())})


class _OptimizerBuilder:
    def __init__(self):
        self.created = []

    def rebuild(self, parameters):
        optimizer = torch.optim.SGD(parameters, lr=0.1, momentum=0.3)
        self.created.append(optimizer)
        return optimizer


class _Client:
    def __init__(self, node_id, rank, count=4, heads=("head",), model=None):
        self.node_id = node_id
        model = _Model(rank) if model is None else model
        self.node_var = SimpleNamespace(
            model=model, model_weight={key: value.clone() for key, value in model.state_dict().items()},
            cache_weight=None, device="cpu", config_dict={"training": {"epochs": 1}},
            data_sample_num=count, trainer=_Trainer(count), optimizer_builder=_OptimizerBuilder(),
        )
        self.strategy = FloristClientTrainingStrategy(
            StrategyArgs({"role": "client", "strategy_name": "florist", "trainable_head_prefixes": list(heads)}), self
        )

    def receive_weight(self, weight):
        self.strategy.receive_weight(weight)

    def set_local_weight(self):
        self.strategy.set_local_weight()

    def set_server_node(self, server):
        self.server_node = server


class _Evaluator:
    def __init__(self, model):
        self.model = copy.deepcopy(model)

    def update_model(self, weights):
        self.model.load_state_dict(weights, strict=True)


class _Server:
    def __init__(self, clients, rank=1, heads=("head",), threshold=0.9):
        self.client_nodes = clients
        model = _Model(rank)
        self.node_var = SimpleNamespace(
            model=model, model_weight={key: value.clone() for key, value in model.state_dict().items()},
            model_evaluator=_Evaluator(model),
            training_logger=SimpleNamespace(begin=lambda header: None),
            aggregation_method=FedAggregatorFactory.create_aggregator(
                FedAggregatorFactory.create_args({"method": "florist", "threshold": threshold})
            ),
        )
        self.strategy = FloristServerStrategy(
            StrategyArgs({"role": "server", "strategy_name": "florist", "trainable_head_prefixes": list(heads)}), self
        )

    def set_client_nodes(self, clients):
        self.client_nodes = clients


def _initialized(clients=None, threshold=0.9):
    torch.manual_seed(12)
    clients = clients or [_Client("small", 2, 4), _Client("large", 5, 7)]
    server = _Server(clients, threshold=threshold)
    server.strategy.prepare("test", clients)
    server.strategy.broadcast()
    return server, clients


def _global_payload(rank, heads=True):
    generator = torch.Generator().manual_seed(32)
    payload = {
        "layer.lora_A": torch.randn(rank, 5, generator=generator),
        "layer.lora_B": torch.randn(4, rank, generator=generator),
    }
    if heads:
        payload.update({"head.weight": torch.randn(2, 4, generator=generator), "head.bias": torch.randn(2, generator=generator)})
    return payload


def test_initial_sync_keeps_full_local_random_rank_and_freezes_backbone():
    server, clients = _initialized()
    for client in clients:
        state = client.node_var.model_weight
        assert torch.count_nonzero(state["layer.lora_B"]) == 0
        assert torch.all(torch.count_nonzero(state["layer.lora_A"], dim=1) > 0)
        assert state["layer.lora_A"].shape[0] == client.node_var.model.layer.r
        for key, reference in server.strategy._backbone.items():
            assert torch.equal(state[key], reference)
        for key, parameter in client.node_var.model.named_parameters():
            assert parameter.requires_grad == (key.startswith("head.") or "lora_" in key)
    assert server.strategy.initialization_stats["actual_download_bytes"] == 2 * _payload_bytes(server.strategy._backbone)
    with pytest.raises(RuntimeError, match="already initialized"):
        clients[0].strategy.initialize_from_server(server.strategy._backbone)


def test_same_global_payload_is_immutable_and_local_receive_uses_only_B_scale():
    server, clients = _initialized()
    payload = _global_payload(3)
    snapshot = {key: value.clone() for key, value in payload.items()}
    server.node_var.model_weight = payload
    server.strategy.broadcast()
    for client in clients:
        state, layer = client.node_var.model_weight, client.node_var.model.layer
        keep = min(layer.r, 3)
        assert client.node_var.cache_weight is payload
        torch.testing.assert_close(state["layer.lora_A"][:keep], payload["layer.lora_A"][:keep])
        torch.testing.assert_close(state["layer.lora_B"][:, :keep] * layer.scaling, payload["layer.lora_B"][:, :keep])
        assert torch.count_nonzero(state["layer.lora_A"][keep:]) == 0
        assert torch.count_nonzero(state["layer.lora_B"][:, keep:]) == 0
        expected = payload["layer.lora_B"][:, :keep] @ payload["layer.lora_A"][:keep]
        torch.testing.assert_close(layer.scaling * state["layer.lora_B"] @ state["layer.lora_A"], expected)
    assert all(torch.equal(payload[key], value) for key, value in snapshot.items())


def test_complete_global_rank_evaluation_never_accumulates_or_truncates():
    server, _ = _initialized()
    payload = _global_payload(3)  # Server evaluator rank is only one.
    base = server.strategy._initial_weight["layer.weight"].clone()
    server.node_var.aggregated_weight = payload
    server.strategy.apply_weight()
    evaluated = server.node_var.model_evaluator.model
    torch.testing.assert_close(evaluated.layer.weight, base + payload["layer.lora_B"] @ payload["layer.lora_A"])
    assert torch.count_nonzero(evaluated.layer.lora_A) == 0
    assert torch.count_nonzero(evaluated.layer.lora_B) == 0
    assert torch.equal(server.strategy._initial_weight["layer.weight"], base)
    x = torch.randn(3, 5)
    expected = nn.functional.linear(torch.tanh(nn.functional.linear(x, base + payload["layer.lora_B"] @ payload["layer.lora_A"], server.strategy._initial_weight["layer.bias"])), payload["head.weight"], payload["head.bias"])
    torch.testing.assert_close(evaluated(x), expected)
    server.strategy.apply_weight()
    torch.testing.assert_close(evaluated(x), expected)
    assert server.node_var.model_weight is payload


def test_two_real_sgd_rounds_preserve_base_and_continue_from_received_factors():
    server, clients = _initialized(threshold=0.9)
    initial = {key: value.clone() for key, value in server.strategy._initial_weight.items()}
    received = None
    for round_index in range(2):
        updates = []
        for client in clients:
            payload, record = client.strategy.run_local_training()
            updates.append({"updated_weights": payload, "train_record": record})
            assert set(payload) == {"layer.lora_A", "layer.lora_B", "head.weight", "head.bias"}
            local = client.node_var.model_weight
            assert set(local) == set(initial)
            torch.testing.assert_close(payload["layer.lora_A"], local["layer.lora_A"])
            torch.testing.assert_close(payload["layer.lora_B"], local["layer.lora_B"] * client.node_var.model.layer.scaling)
            if received is not None:
                for key in ("layer.lora_A", "layer.lora_B"):
                    assert torch.equal(client.node_var.trainer.inputs[-1][key], received[client.node_id][key])
            client.strategy.assert_frozen_backbone()
        server.strategy.receive_client_updates(updates)
        server.strategy.aggregation()
        server.strategy.apply_weight()
        payload_snapshot = {key: value.clone() for key, value in server.node_var.model_weight.items()}
        server.strategy.broadcast()
        assert all(torch.equal(value, server.node_var.model_weight[key]) for key, value in payload_snapshot.items())
        received = {client.node_id: {key: value.clone() for key, value in client.node_var.model_weight.items()} for client in clients}
        history = server.strategy.history[-1]
        assert history["round"] == round_index + 1
        assert history["actual_upload_bytes"] == sum(_payload_bytes(update["updated_weights"]) for update in updates)
        assert history["actual_download_bytes"] == len(clients) * _payload_bytes(server.node_var.model_weight)
        assert history["frozen_backbone_unchanged"]
        assert history["aggregation_stats"]["threshold"] == 0.9
        assert torch.count_nonzero(server.node_var.model_weight["layer.lora_B"]) > 0
    for client in clients:
        assert len(client.node_var.optimizer_builder.created) == 2
        assert client.node_var.optimizer_builder.created[0] is not client.node_var.optimizer_builder.created[1]
    assert server.node_var.florist_history is server.strategy.history
    assert all(torch.equal(value, initial[key]) for key, value in server.strategy._initial_weight.items())


def test_nonfinite_local_parameters_are_uploaded_with_original_scaling(monkeypatch):
    _, clients = _initialized()
    client = clients[0]
    trainer = client.node_var.trainer
    original_train = trainer.train

    def train(epochs):
        state, record = original_train(epochs)
        state["layer.lora_A"][0, 0] = float("nan")
        state["layer.lora_B"][0, 0] = float("inf")
        state["head.bias"][0] = -float("inf")
        return state, record

    monkeypatch.setattr(trainer, "train", train)
    payload, record = client.strategy.run_local_training()
    local = client.node_var.model_weight
    assert torch.isnan(payload["layer.lora_A"][0, 0])
    assert torch.isposinf(payload["layer.lora_B"][0, 0])
    assert torch.isneginf(payload["head.bias"][0])
    torch.testing.assert_close(payload["layer.lora_A"], local["layer.lora_A"], equal_nan=True)
    torch.testing.assert_close(payload["layer.lora_B"], local["layer.lora_B"] * client.node_var.model.layer.scaling)
    assert record["data_sample_num"] == client.node_var.data_sample_num
    assert trainer.trainer_args.model is None
    client.strategy.assert_frozen_backbone()


def test_nonfinite_global_parameters_are_applied_and_broadcast_unchanged():
    server, clients = _initialized()
    payload = _global_payload(2)
    payload["layer.lora_A"][0, 0] = float("nan")
    payload["layer.lora_B"][0, 0] = float("inf")
    payload["head.bias"][0] = float("nan")
    server.node_var.aggregated_weight = payload
    server.strategy.apply_weight()
    server.strategy.broadcast()
    assert server.node_var.model_weight is payload
    evaluated = server.node_var.model_evaluator.model
    assert not torch.isfinite(evaluated.layer.weight).all()
    assert torch.isnan(evaluated.head.bias[0])
    for client in clients:
        assert client.node_var.cache_weight is payload
        local = client.node_var.model_weight
        torch.testing.assert_close(local["layer.lora_A"][:2], payload["layer.lora_A"], equal_nan=True)
        torch.testing.assert_close(local["layer.lora_B"][:, :2] * client.node_var.model.layer.scaling,
                                   payload["layer.lora_B"])
        assert torch.isnan(local["head.bias"][0])
        client.strategy.assert_frozen_backbone()


def test_server_passes_all_nonfinite_updates_and_propagates_aggregation_error(monkeypatch):
    server, clients = _initialized()
    updates = [{"updated_weights": _global_payload(client.node_var.model.layer.r),
                "train_record": {"node_id": client.node_id,
                                 "data_sample_num": client.node_var.data_sample_num}}
               for client in clients]
    updates[0]["updated_weights"]["layer.lora_A"][0, 0] = float("nan")
    received = []
    error = RuntimeError("backend SVD failure")

    def aggregate(values):
        received.append(values)
        raise error

    monkeypatch.setattr(server.node_var.aggregation_method, "aggregate", aggregate)
    server.strategy.receive_client_updates(updates)
    with pytest.raises(RuntimeError) as caught:
        server.strategy.aggregation()
    assert caught.value is error
    assert received == [updates]
    assert received[0] is updates
    assert [item["train_record"]["data_sample_num"] for item in received[0]] == [4, 7]
    assert not hasattr(server.node_var, "aggregated_weight")
    assert server.strategy.history == []


def test_default_head_is_frozen_and_not_uploaded():
    client = _Client("client", 2, heads=())
    server = _Server([client], heads=())
    server.strategy.broadcast()
    payload, _ = client.strategy.local_training_step()
    assert set(payload) == {"layer.lora_A", "layer.lora_B"}
    assert not client.node_var.model.head.weight.requires_grad


def test_disabled_lora_scaling_uses_actual_module_scale_on_both_paths():
    model = _Model(2, alpha=8)
    model.layer = MSLoRALinear(5, 4, r=2, lora_alpha=8, merge_weights=False, use_lora_scaling=False)
    client = _Client("unscaled", 2, model=model)
    server, _ = _initialized([client])
    assert client.strategy.layers["layer"].scaling == 1.0
    payload = _global_payload(2)
    server.node_var.model_weight = payload
    server.strategy.broadcast()
    assert torch.equal(client.node_var.model_weight["layer.lora_B"], payload["layer.lora_B"])
    upload, _ = client.strategy.local_training_step()
    assert torch.equal(upload["layer.lora_B"], client.node_var.model_weight["layer.lora_B"])


@pytest.mark.parametrize("kind", ["merge", "embedding", "fan", "head_overlap", "head_typo"])
def test_unsupported_configurations_fail_before_training(kind):
    model = _Model(2, merge=kind == "merge")
    heads = ("head",)
    if kind == "embedding":
        model.layer = MSEmbedding(5, 4, r=2, merge_weights=False)
    if kind == "fan":
        model.layer.fan_in_fan_out = True
    if kind == "head_overlap":
        heads = ("layer",)
    if kind == "head_typo":
        heads = ("missing",)
    client = _Client("invalid", 2, heads=heads, model=model)
    with pytest.raises((TypeError, ValueError)):
        client.strategy.configure_model()


def test_frozen_drift_and_unexpected_payload_fail_closed():
    server, clients = _initialized()
    clients[0].node_var.model_weight["layer.weight"].add_(1)
    with pytest.raises(RuntimeError, match="frozen backbone changed"):
        clients[0].strategy.assert_frozen_backbone()
    payload = _global_payload(2)
    payload["layer.weight"] = torch.zeros(4, 5)
    clients[1].strategy.receive_weight(payload)
    with pytest.raises(ValueError, match="unexpected"):
        clients[1].strategy.set_local_weight()


def test_factories_and_yaml_fragments_select_florist_without_new_runner():
    yaml_root = Path(__file__).resolve().parents[1] / "yamls"
    fragments = {}
    for directory in ("client_strategy", "server_strategy", "runner_strategy"):
        fragments[directory] = yaml.safe_load((yaml_root / directory / "florist.yaml").read_text())
        assert fragments[directory]["strategy"]["strategy_name"] == "florist"
    aggregation = yaml.safe_load((yaml_root / "aggregation" / "aggregation_florist.yaml").read_text())
    assert isinstance(FedAggregatorFactory.create_aggregator(FedAggregatorFactory.create_args(aggregation)), FedAggregator_Florist)
    client = _Client("client", 2)
    assert isinstance(StrategyFactory.create_client_strategy(StrategyArgs(fragments["client_strategy"]["strategy"]), client), FloristClientTrainingStrategy)
    server = _Server([client])
    assert isinstance(StrategyFactory.create_server_strategy(StrategyArgs(fragments["server_strategy"]["strategy"]), server), FloristServerStrategy)
    runner = StrategyFactory.create_runner_strategy(StrategyArgs(fragments["runner_strategy"]["strategy"]), SimpleNamespace(), [client], server)
    assert isinstance(runner, RblaRunnerStrategy)
    client.strategy.create(StrategyArgs({"role": "client", "strategy_name": "florist", "trainable_head_prefixes": ["head"]}), client)
    server.strategy.create(StrategyArgs({"role": "server", "strategy_name": "florist", "trainable_head_prefixes": ["head"]}), server)
    assert client.strategy._strategy_type == server.strategy._strategy_type == "florist"
