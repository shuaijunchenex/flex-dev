"""Offline convolution layout and numerical checks for the unchanged FLoRIST core."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from flex.fed_strategy.client_strategy_impl._florist_client import FloristClientTrainingStrategy
from flex.fed_strategy.server_strategy_impl._florist_server import FloristServerStrategy
from flex.fed_strategy.strategy_args import StrategyArgs
from flex.fl_algorithms.aggregation.fed_aggregator_args import FedAggregatorArgs
from flex.fl_algorithms.aggregation.methods._fed_aggregator_florist import FedAggregator_Florist
from flex.ml_algorithms.lora.impl.lora_ms import MSLoRAConv2d, MSLoRALinear


class _ConvModel(nn.Module):
    def __init__(self, rank=1, alpha=2, scaling=True):
        super().__init__()
        self.conv = MSLoRAConv2d(3, 4, 3, r=rank, lora_alpha=alpha,
                                 merge_weights=False, use_lora_scaling=scaling, padding=1)
        self.linear = MSLoRALinear(4, 2, r=rank, lora_alpha=alpha,
                                   merge_weights=False, use_lora_scaling=scaling)

    def forward(self, x):
        return self.linear(torch.tanh(self.conv(x)).mean(dim=(2, 3)))


class _Trainer:
    def __init__(self):
        self.trainer_args = SimpleNamespace(model=None, optimizer=None, device="cpu")
        generator = torch.Generator().manual_seed(720)
        self.x = torch.randn(4, 3, 5, 5, generator=generator)
        self.y = torch.tensor([0, 1, 1, 0])

    def set_model(self, model):
        self.trainer_args.model = model

    def set_optimizer(self, optimizer):
        self.trainer_args.optimizer = optimizer

    def train(self, epochs):
        model, optimizer = self.trainer_args.model, self.trainer_args.optimizer
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(self.x), self.y)
            loss.backward()
            optimizer.step()
        return ({key: value.detach().clone() for key, value in model.state_dict().items()},
                {"data_sample_num": len(self.y), "avg_loss": loss.detach().item()})


class _Client:
    def __init__(self, node_id, rank, alpha=2, scaling=True, model=None):
        self.node_id = node_id
        model = model if model is not None else _ConvModel(rank, alpha, scaling)
        self.node_var = SimpleNamespace(
            model=model, model_weight=copy.deepcopy(model.state_dict()), cache_weight=None,
            device="cpu", config_dict={"training": {"epochs": 1}}, data_sample_num=4,
            trainer=_Trainer(), optimizer_builder=SimpleNamespace(
                rebuild=lambda parameters: torch.optim.SGD(parameters, lr=0.1)),
        )
        self.strategy = FloristClientTrainingStrategy(
            StrategyArgs({"role": "client", "strategy_name": "florist"}), self)

    def receive_weight(self, weights):
        self.strategy.receive_weight(weights)

    def set_local_weight(self):
        self.strategy.set_local_weight()


class _Evaluator:
    def __init__(self, model):
        self.model = copy.deepcopy(model)

    def update_model(self, weights):
        self.model.load_state_dict(weights, strict=True)


def _initialized(clients=None, threshold=1.0):
    torch.manual_seed(523)
    clients = clients if clients is not None else [_Client("small", 1, 0.5), _Client("large", 2, 8)]
    model = _ConvModel(1)
    server = SimpleNamespace(client_nodes=clients, node_var=SimpleNamespace(
        model=model, model_weight=copy.deepcopy(model.state_dict()), model_evaluator=_Evaluator(model),
        aggregation_method=FedAggregator_Florist(FedAggregatorArgs({"threshold": threshold})),
    ))
    server.strategy = FloristServerStrategy(
        StrategyArgs({"role": "server", "strategy_name": "florist"}), server)
    server.strategy.broadcast()
    return server, clients


def _global_payload(conv_rank, linear_rank=2):
    generator = torch.Generator().manual_seed(35)
    return {
        "conv.lora_A": torch.randn(conv_rank, 9, generator=generator),
        "conv.lora_B": torch.randn(12, conv_rank, generator=generator),
        "linear.lora_A": torch.randn(linear_rank, 4, generator=generator),
        "linear.lora_B": torch.randn(2, linear_rank, generator=generator),
    }


def convolution_alignment(dtype, threshold):
    """Return measured errors against a dense SVD oracle, also usable by validation reports."""
    generator = torch.Generator().manual_seed(439)
    ranks, scales, counts = (1, 2, 3), (0.5, 2.0, 4.0), (4, 6, 8)
    updates, dense = [], torch.zeros(12, 9, dtype=dtype)
    for rank, scale, count in zip(ranks, scales, counts):
        A = torch.randn(rank * 3, 9, generator=generator, dtype=dtype)
        B = torch.randn(12, rank * 3, generator=generator, dtype=dtype)
        updates.append({"updated_weights": {"conv.lora_A": A, "conv.lora_B": B * scale},
                        "train_record": {"data_sample_num": count}})
        dense += (count / sum(counts)) * (B * scale @ A)
    original = copy.deepcopy(updates)
    aggregator = FedAggregator_Florist(FedAggregatorArgs({"threshold": threshold}))
    actual = aggregator.aggregate(updates)
    u, singular, vh = torch.linalg.svd(dense, full_matrices=False)
    rank = len(singular) if threshold == 1 else int(torch.searchsorted(
        singular.square().cumsum(0) / singular.square().sum(), threshold).item()) + 1
    expected = (u[:, :rank] * singular[:rank]) @ vh[:rank]
    product = actual["conv.lora_B"] @ actual["conv.lora_A"]
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(product, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(product.reshape(4, 3, 3, 3), expected.reshape(4, 3, 3, 3),
                               atol=tolerance, rtol=tolerance)
    assert actual["conv.lora_A"].shape[0] == rank
    for old, current in zip(original, updates):
        assert all(torch.equal(value, current["updated_weights"][key])
                   for key, value in old["updated_weights"].items())
    return {"dtype": str(dtype), "threshold": threshold, "logical_ranks": list(ranks),
            "factor_ranks": [rank * 3 for rank in ranks], "scales": list(scales),
            "global_factor_rank": rank,
            "max_absolute_error": float((product - expected).abs().max()),
            "relative_frobenius_error": float(torch.linalg.norm(product - expected) / torch.linalg.norm(expected)),
            "relative_truncation_error": float(torch.linalg.norm(product - dense) / torch.linalg.norm(dense))}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("threshold", [0.9, 1.0])
def test_convolution_aggregation_matches_weighted_dense_svd(dtype, threshold):
    convolution_alignment(dtype, threshold)


def test_convolution_logical_rank_factor_width_and_actual_scale_are_distinct():
    server, clients = _initialized()
    for client in clients:
        strategy, model = client.strategy, client.node_var.model
        assert strategy.layers["conv"].rank == model.conv.r
        assert strategy.factor_ranks == {"conv": model.conv.r * 3, "linear": model.linear.r}
        assert strategy.layers["conv"].scaling == model.conv.lora_alpha / model.conv.r
        state = client.node_var.model_weight
        assert torch.all(torch.count_nonzero(state["conv.lora_A"], dim=1) > 0)
        assert torch.count_nonzero(state["conv.lora_B"]) == 0
        for key, value in server.strategy._backbone.items():
            assert torch.equal(state[key], value)


@pytest.mark.parametrize("global_rank", [2, 4, 8])
def test_convolution_broadcast_uses_factor_width_and_preserves_global_payload(global_rank):
    server, clients = _initialized()
    payload = _global_payload(global_rank)
    snapshot = copy.deepcopy(payload)
    server.node_var.model_weight = payload
    server.strategy.broadcast()
    for client in clients:
        state, conv = client.node_var.model_weight, client.node_var.model.conv
        keep = min(conv.r * 3, global_rank)
        expected = payload["conv.lora_B"][:, :keep] @ payload["conv.lora_A"][:keep]
        actual = conv.scaling * state["conv.lora_B"] @ state["conv.lora_A"]
        torch.testing.assert_close(actual, expected)
        assert torch.count_nonzero(state["conv.lora_A"][keep:]) == 0
        assert torch.count_nonzero(state["conv.lora_B"][:, keep:]) == 0
        client.strategy.assert_frozen_backbone()
    assert all(torch.equal(value, payload[key]) for key, value in snapshot.items())


def test_convolution_evaluation_uses_complete_global_rank_and_correct_kernel_order():
    server, _ = _initialized()
    payload = _global_payload(7)  # Not divisible by k=3; evaluator has only three factor columns.
    server.node_var.aggregated_weight = payload
    snapshot = copy.deepcopy(server.strategy._initial_weight)
    server.strategy.apply_weight()
    evaluated = server.node_var.model_evaluator.model
    conv_weight = snapshot["conv.weight"] + (payload["conv.lora_B"] @ payload["conv.lora_A"]).reshape(4, 3, 3, 3)
    linear_weight = snapshot["linear.weight"] + payload["linear.lora_B"] @ payload["linear.lora_A"]
    x = torch.randn(3, 3, 7, 7)
    features = torch.tanh(nn.functional.conv2d(x, conv_weight, snapshot["conv.bias"], padding=1)).mean((2, 3))
    expected = nn.functional.linear(features, linear_weight, snapshot["linear.bias"])
    torch.testing.assert_close(evaluated(x), expected)
    torch.testing.assert_close(evaluated.conv.weight, conv_weight)
    for key in server.strategy.adapter_keys:
        assert torch.count_nonzero(evaluated.state_dict()[key]) == 0
    server.strategy.apply_weight()
    torch.testing.assert_close(evaluated(x), expected)
    assert all(torch.equal(value, server.strategy._initial_weight[key]) for key, value in snapshot.items())
    assert server.node_var.model_weight is payload


@pytest.mark.parametrize("scaling", [True, False])
def test_real_convolution_training_upload_and_server_history(scaling):
    server, clients = _initialized([_Client("client", 2, alpha=8, scaling=scaling)])
    client = clients[0]
    upload, record = client.strategy.local_training_step()
    assert record["data_sample_num"] == 4
    assert set(upload) == client.strategy.adapter_keys
    assert torch.count_nonzero(upload["conv.lora_B"]) > 0
    expected_scale = 4 if scaling else 1
    torch.testing.assert_close(upload["conv.lora_B"], client.node_var.model_weight["conv.lora_B"] * expected_scale)
    client.strategy.assert_frozen_backbone()
    server.node_var.client_updates = [{"node_id": client.node_id, "updated_weights": upload, "train_record": record}]
    server.strategy.aggregation()
    history = server.strategy.history[-1]
    assert history["local_rank_by_client"] == {"client": {"conv": 2, "linear": 2}}
    assert history["local_factor_rank_by_client"] == {"client": {"conv": 6, "linear": 2}}
    assert history["aggregation_stats"]["layers"]["conv"]["input_stack_rank"] == 6


def test_server_rejects_upload_using_logical_rank_as_factor_width():
    server, clients = _initialized()
    client = clients[0]
    upload = _global_payload(1, 1)  # Client logical r=1, but conv factor width must be three.
    server.node_var.client_updates = [{"node_id": client.node_id, "updated_weights": upload,
                                      "train_record": {"data_sample_num": 4}}]
    with pytest.raises(ValueError, match="upload rank mismatch at 'conv'"):
        server.strategy.aggregation()


@pytest.mark.parametrize("kind", ["groups", "padding", "nonsquare", "merge", "merged", "A", "B", "weight"])
def test_unsupported_convolution_layouts_fail_before_training(kind):
    model = _ConvModel(2)
    if kind == "groups":
        model.conv = MSLoRAConv2d(4, 4, 3, r=2, groups=2, merge_weights=False)
    elif kind == "padding":
        model.conv.padding_mode = "reflect"
    elif kind == "nonsquare":
        model.conv.kernel_size = (3, 1)
    elif kind == "merge":
        model.conv.merge_weights = True
    elif kind == "merged":
        model.conv.merged = True
    elif kind == "A":
        model.conv.lora_A = nn.Parameter(torch.zeros(2, 9))
    elif kind == "B":
        model.conv.lora_B = nn.Parameter(torch.zeros(12, 2))
    else:
        model.conv.weight = nn.Parameter(torch.zeros(4, 3, 3, 2))
    client = _Client("invalid", 2, model=model)
    with pytest.raises(ValueError):
        client.strategy.configure_model()
