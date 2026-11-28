"""Offline checks for the explicit CNN BatchNorm policies around FLoRIST."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from flex.fed_strategy.client_strategy_impl._florist_client import (
    FloristClientTrainingStrategy,
    _payload_bytes,
)
from flex.fed_strategy.server_strategy_impl._florist_server import FloristServerStrategy
from flex.fed_strategy.strategy_args import StrategyArgs
from flex.fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from flex.ml_algorithms.lora.impl.lora_ms import MSLoRAConv2d


BN_KEYS = {"bn.weight", "bn.bias", "bn.running_mean", "bn.running_var", "bn.num_batches_tracked"}
COUNTER = "bn.num_batches_tracked"


def _copy_state(state):
    return {key: value.detach().clone() for key, value in state.items()}


class _Model(nn.Module):
    def __init__(self, rank=1, *, momentum=0.1, track_running_stats=True):
        super().__init__()
        self.conv = MSLoRAConv2d(3, 4, 3, r=rank, lora_alpha=4, padding=1, merge_weights=False)
        self.bn = nn.BatchNorm2d(4, momentum=momentum, track_running_stats=track_running_stats)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(4, 2)
        self.register_buffer("frozen_counter", torch.tensor(7))
        self.observed_modes = []

    def forward(self, x):
        self.observed_modes.append((self.training, self.bn.training, self.dropout.training))
        features = self.dropout(torch.relu(self.bn(self.conv(x))))
        return self.head(features.mean(dim=(2, 3)))


class _Trainer:
    def __init__(self, count):
        generator = torch.Generator().manual_seed(700 + count)
        self.x = torch.randn(count, 3, 6, 6, generator=generator) + 0.7
        self.y = torch.arange(count) % 2
        self.trainer_args = SimpleNamespace(model=None, optimizer=None, device="cpu")
        self.models, self.inputs = [], []
        self.fail = False

    def set_model(self, model):
        self.trainer_args.model = model

    def set_optimizer(self, optimizer):
        self.trainer_args.optimizer = optimizer

    def train(self, epochs):
        model, optimizer = self.trainer_args.model, self.trainer_args.optimizer
        self.models.append(model)
        self.inputs.append(_copy_state(model.state_dict()))
        for _ in range(epochs):
            # The actual CV trainer re-enters training mode at each epoch.
            model.train()
            for start in range(0, len(self.y), 2):
                optimizer.zero_grad()
                logits = model(self.x[start:start + 2])
                if self.fail:
                    raise RuntimeError("injected training failure")
                loss = nn.functional.cross_entropy(logits, self.y[start:start + 2])
                loss.backward()
                optimizer.step()
        return _copy_state(model.state_dict()), {"data_sample_num": len(self.y), "avg_loss": float(loss.detach())}


def _args(role, mode, heads=("head",)):
    values = {"role": role, "strategy_name": "florist", "trainable_head_prefixes": list(heads)}
    if mode is not None:
        values["batchnorm_mode"] = mode
    return StrategyArgs(values)


class _Client:
    def __init__(self, node_id, rank, count, mode="frozen", *, model=None, heads=("head",)):
        self.node_id = node_id
        model = _Model(rank) if model is None else model
        self.node_var = SimpleNamespace(
            model=model, model_weight=_copy_state(model.state_dict()), cache_weight=None,
            device="cpu", config_dict={"training": {"epochs": 2}}, data_sample_num=count,
            trainer=_Trainer(count),
            optimizer_builder=SimpleNamespace(rebuild=lambda parameters: torch.optim.SGD(parameters, lr=0.04)),
        )
        self.strategy = FloristClientTrainingStrategy(_args("client", mode, heads), self)

    def receive_weight(self, weights):
        self.strategy.receive_weight(weights)

    def set_local_weight(self):
        self.strategy.set_local_weight()


class _Evaluator:
    def __init__(self, model):
        self.model = copy.deepcopy(model)

    def update_model(self, weights):
        self.model.load_state_dict(weights, strict=True)


class _Server:
    def __init__(self, clients, mode="frozen"):
        self.client_nodes = clients
        model = _Model()
        self.node_var = SimpleNamespace(
            model=model, model_weight=_copy_state(model.state_dict()),
            model_evaluator=_Evaluator(model),
            aggregation_method=FedAggregatorFactory.create_aggregator(
                FedAggregatorFactory.create_args({"method": "florist", "threshold": 0.9})
            ),
        )
        self.strategy = FloristServerStrategy(_args("server", mode), self)


def _initialized(mode="frozen", counts=(4, 6)):
    torch.manual_seed(456)
    clients = [_Client(str(index), index + 1, count, mode) for index, count in enumerate(counts)]
    server = _Server(clients, mode)
    server.strategy.broadcast()
    return server, clients


def _synthetic_payload(client):
    state = client.node_var.model_weight
    result = {key: state[key].detach().clone() for key in client.strategy.adapter_keys | client.strategy.extra_keys}
    result["conv.lora_B"].fill_(0.03 * client.node_var.model.conv.scaling)
    return result


def _update(client, payload, count=None):
    return {"updated_weights": payload, "train_record": {
        "node_id": client.node_id,
        "data_sample_num": client.node_var.data_sample_num if count is None else count,
    }}


@pytest.mark.parametrize("mode", ["frozen", "fedavg"])
def test_two_rounds_propagate_bn_state_and_preserve_other_backbone(mode):
    server, clients = _initialized(mode)
    initial = _copy_state(server.strategy._initial_weight)
    received = None
    for round_index in range(2):
        updates = []
        for client in clients:
            payload, record = client.strategy.run_local_training()
            updates.append({"updated_weights": payload, "train_record": record})
            trainer = client.node_var.trainer
            training_model = trainer.models[-1]
            assert training_model is not client.node_var.model
            assert training_model.observed_modes
            assert all(modes == (True, mode == "fedavg", True) for modes in training_model.observed_modes)
            assert not any(module._forward_pre_hooks for module in training_model.modules())
            for parameter in (client.node_var.model.bn.weight, client.node_var.model.bn.bias):
                assert parameter.requires_grad == (mode == "fedavg")
            client.strategy.assert_frozen_backbone()
            assert torch.count_nonzero(payload["conv.lora_B"]) > 0
            if received is not None:
                for key in BN_KEYS:
                    assert torch.equal(trainer.inputs[-1][key], received[key])
            if mode == "frozen":
                assert not BN_KEYS.intersection(payload)
                assert all(torch.equal(client.node_var.model_weight[key], initial[key]) for key in BN_KEYS)
            else:
                assert BN_KEYS.issubset(payload)
                expected_counter = (0 if received is None else int(received[COUNTER])) + client.node_var.data_sample_num
                assert int(payload[COUNTER]) == expected_counter  # Two epochs, batches of two.
                assert not torch.equal(payload["bn.running_mean"], trainer.inputs[-1]["bn.running_mean"])
                assert any(not torch.equal(payload[key], trainer.inputs[-1][key]) for key in ("bn.weight", "bn.bias"))
        server.strategy.receive_client_updates(updates)
        server.strategy.aggregation()
        aggregated = server.node_var.aggregated_weight
        if mode == "fedavg":
            for key in BN_KEYS - {COUNTER}:
                expected = sum(update["updated_weights"][key] * client.node_var.data_sample_num / 10 for update, client in zip(updates, clients))
                torch.testing.assert_close(aggregated[key], expected)
            assert aggregated[COUNTER].dtype == torch.int64
            assert int(aggregated[COUNTER]) == (round_index + 1) * 6
        server.strategy.apply_weight()
        before_broadcast = _copy_state(server.node_var.model_weight)
        server.strategy.broadcast()
        assert all(torch.equal(value, server.node_var.model_weight[key]) for key, value in before_broadcast.items())
        evaluator_state = server.node_var.model_evaluator.model.state_dict()
        received = {key: evaluator_state[key].clone() for key in BN_KEYS}
        for client in clients:
            assert all(torch.equal(client.node_var.model_weight[key], value) for key, value in received.items())
            client.strategy.assert_frozen_backbone()
        history = server.strategy.history[-1]
        assert history["actual_upload_bytes"] == sum(_payload_bytes(update["updated_weights"]) for update in updates)
        assert history["actual_download_bytes"] == len(clients) * _payload_bytes(before_broadcast)
        expected_bn_upload = sum(_payload_bytes({key: update["updated_weights"][key] for key in BN_KEYS}) for update in updates) if mode == "fedavg" else 0
        expected_bn_download = len(clients) * _payload_bytes({key: before_broadcast[key] for key in BN_KEYS}) if mode == "fedavg" else 0
        assert history["batchnorm_upload_bytes"] == expected_bn_upload
        assert history["batchnorm_download_bytes"] == expected_bn_download
        if mode == "fedavg":
            counter_bytes = sum(update["updated_weights"][COUNTER].element_size() for update in updates)
            assert history["actual_upload_bytes"] == history["aggregation_stats"]["upload_bytes"] + counter_bytes
            assert _payload_bytes(before_broadcast) == history["aggregation_stats"]["download_bytes"] + before_broadcast[COUNTER].element_size()
    assert all(torch.equal(value, server.strategy._initial_weight[key]) for key, value in initial.items())


def test_frozen_is_default_and_exception_removes_training_hook():
    server, clients = _initialized(mode=None)
    client = clients[0]
    client.node_var.trainer.fail = True
    with pytest.raises(RuntimeError, match="injected training failure"):
        client.strategy.local_training_step()
    model = client.node_var.trainer.models[-1]
    assert model.observed_modes == [(True, False, True)]
    assert not any(module._forward_pre_hooks for module in model.modules())
    assert client.node_var.trainer.trainer_args.model is None
    client.strategy.assert_frozen_backbone()


def test_zero_weight_client_cannot_raise_global_batch_counter():
    server, clients = _initialized("fedavg")
    updates = []
    for index, client in enumerate(clients):
        payload = _synthetic_payload(client)
        payload[COUNTER].fill_(3 if index == 0 else 1000)
        payload["bn.running_mean"].fill_(2 if index == 0 else 100)
        updates.append(_update(client, payload, count=4 if index == 0 else 0))
    server.strategy.receive_client_updates(updates)
    server.strategy.aggregation()
    assert int(server.node_var.aggregated_weight[COUNTER]) == 3
    torch.testing.assert_close(server.node_var.aggregated_weight["bn.running_mean"], torch.full((4,), 2.0))


@pytest.mark.parametrize("invalid", ["float_counter", "negative_counter", "counter_shape", "missing_key"])
def test_bad_bn_payload_rejected_on_upload_and_broadcast(invalid):
    server, clients = _initialized("fedavg")
    client = clients[0]
    payload = _synthetic_payload(client)
    if invalid == "float_counter":
        payload[COUNTER] = payload[COUNTER].float()
    elif invalid == "negative_counter":
        payload[COUNTER].fill_(-1)
    elif invalid == "counter_shape":
        payload[COUNTER] = torch.zeros(1, dtype=torch.int64)
    else:
        del payload["bn.bias"]
    client.strategy.receive_weight(payload)
    with pytest.raises(ValueError):
        client.strategy.set_local_weight()
    server.strategy.receive_client_updates([_update(client, payload)])
    with pytest.raises(ValueError):
        server.strategy.aggregation()


@pytest.mark.parametrize("key,value", [("bn.running_var", -0.01),
                                       ("bn.running_mean", float("nan")),
                                       ("bn.running_var", float("inf"))])
def test_bn_floating_values_are_aggregated_applied_and_broadcast_without_value_checks(key, value):
    server, clients = _initialized("fedavg")
    updates = []
    for client in clients:
        payload = _synthetic_payload(client)
        payload[key][0] = value
        client.strategy.receive_weight(payload)
        client.strategy.set_local_weight()
        torch.testing.assert_close(client.node_var.model_weight[key], payload[key], equal_nan=True)
        updates.append(_update(client, payload))
    server.strategy.receive_client_updates(updates)
    server.strategy.aggregation()
    expected = sum(update["updated_weights"][key] * client.node_var.data_sample_num / 10
                   for update, client in zip(updates, clients))
    torch.testing.assert_close(server.node_var.aggregated_weight[key], expected, equal_nan=True)
    assert server.strategy.history[-1]["num_clients"] == len(clients)
    server.strategy.apply_weight()
    server.strategy.broadcast()
    torch.testing.assert_close(server.node_var.model_evaluator.model.state_dict()[key], expected, equal_nan=True)
    for client in clients:
        torch.testing.assert_close(client.node_var.model_weight[key], expected, equal_nan=True)
        client.strategy.assert_frozen_backbone()


@pytest.mark.parametrize("case", ["mode_mismatch", "key_mismatch", "head_overlap", "cumulative_momentum", "missing_running_stats", "unknown_mode"])
def test_invalid_bn_configuration_fails_before_training(case):
    with pytest.raises((TypeError, ValueError)):
        if case == "unknown_mode":
            client = _Client("0", 1, 4, mode="automatic")
            client.strategy.configure_model()
        elif case == "head_overlap":
            client = _Client("0", 1, 4, mode="frozen", heads=("head", "bn"))
            client.strategy.configure_model()
        elif case == "cumulative_momentum":
            client = _Client("0", 1, 4, mode="fedavg", model=_Model(momentum=None))
            client.strategy.configure_model()
        elif case == "missing_running_stats":
            client = _Client("0", 1, 4, mode="frozen", model=_Model(track_running_stats=False))
            client.strategy.configure_model()
        elif case == "mode_mismatch":
            client = _Client("0", 1, 4, mode="fedavg")
            _Server([client], mode="frozen").strategy.broadcast()
        else:
            model = _Model()
            model.bn_extra = nn.BatchNorm2d(4)
            client = _Client("0", 1, 4, mode="fedavg", model=model)
            _Server([client], mode="fedavg").strategy.broadcast()


def test_fedavg_bn_exception_does_not_allow_other_backbone_drift():
    _, clients = _initialized("fedavg")
    client = clients[0]
    client.node_var.model_weight["bn.running_mean"].add_(1)
    client.strategy.assert_frozen_backbone()
    client.node_var.model_weight["conv.weight"].add_(1)
    with pytest.raises(RuntimeError, match="frozen backbone changed"):
        client.strategy.assert_frozen_backbone()
