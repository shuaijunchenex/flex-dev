"""FedSA-LoRA regression tests for the shared/private round boundary."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flex.fed_node import FedNodeClient, FedNodeServer
from flex.fed_strategy.client_strategy_impl._fedsa_lora_client import FedSALoRAClientTrainingStrategy
from flex.fed_strategy.runner_strategy_impl._rbla_runner_strategy import RblaRunnerStrategy
from flex.fed_strategy.server_strategy_impl._fedsa_lora_server import FedSALoRAServerStrategy
from flex.fed_strategy.strategy_args import StrategyArgs
from flex.fl_algorithms.aggregation.fed_aggregator_args import FedAggregatorArgs
from flex.fl_algorithms.aggregation.methods._fed_aggregator_fedsa_lora import FedAggregator_FedSAShareA
from flex.ml_algorithms.lora.impl.lora_ms import MSLoRALinear


A_KEY = "adapter.lora_A"
B_KEY = "adapter.lora_B"


class _Model(nn.Module):
    def __init__(self, rank=2, alpha=4):
        super().__init__()
        self.adapter = MSLoRALinear(4, 4, r=rank, lora_alpha=alpha, merge_weights=False)
        self.head = nn.Linear(4, 2)

    def forward(self, inputs):
        return self.head(torch.tanh(self.adapter(inputs)))


class _Trainer:
    def __init__(self, count):
        generator = torch.Generator().manual_seed(count)
        self.inputs = torch.randn(count, 4, generator=generator)
        self.labels = torch.arange(count) % 2
        self.trainer_args = SimpleNamespace(model=None, optimizer=None, device="cpu")
        self.starting_weights = []

    def set_model(self, model):
        self.trainer_args.model = model

    def set_optimizer(self, optimizer):
        self.trainer_args.optimizer = optimizer

    def train(self, epochs):
        model, optimizer = self.trainer_args.model, self.trainer_args.optimizer
        self.starting_weights.append(copy.deepcopy(model.state_dict()))
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(self.inputs), self.labels)
            loss.backward()
            optimizer.step()
        state = copy.deepcopy(model.state_dict())
        # The runner's metadata must exclude even trainer-produced private tensors.
        return state, {"avg_loss": float(loss.detach()), "private_debug": state[B_KEY]}


def _client(name, count, rank=2, alpha=4, evaluation_loader="shared_test"):
    model = _Model(rank, alpha)
    client = FedNodeClient(name)
    client.node_var = SimpleNamespace(
        model=model, model_weight=copy.deepcopy(model.state_dict()), device="cpu",
        data_sample_num=count, config_dict={"training": {"epochs": 2}},
        loss_func=nn.CrossEntropyLoss(), trainer=_Trainer(count),
        optimizer_builder=SimpleNamespace(rebuild=lambda params: torch.optim.SGD(params, lr=0.2)),
    )
    args = StrategyArgs({"role": "client", "strategy_name": "fedsa_lora", "evaluation_loader": evaluation_loader})
    client.strategy = FedSALoRAClientTrainingStrategy(args, client)
    return client


def _server(clients):
    model = _Model()
    loader = DataLoader(TensorDataset(torch.eye(4), torch.tensor([0, 0, 1, 1])), batch_size=2)
    server = FedNodeServer("server")
    server.node_var = SimpleNamespace(
        model=model, model_weight=copy.deepcopy(model.state_dict()), test_data_loader=loader,
        training_logger=SimpleNamespace(begin=lambda _header: None), client_updates=[],
        aggregation_method=FedAggregator_FedSAShareA(FedAggregatorArgs({"method": "fedsa_lora"})),
    )
    server.strategy = FedSALoRAServerStrategy(StrategyArgs({"role": "server", "strategy_name": "fedsa_lora"}), server)
    runner = RblaRunnerStrategy(SimpleNamespace(), StrategyArgs({"role": "runner", "strategy_name": "rbla"}),
                                clients, server)
    return server, runner


def _apply_round(server, updates):
    server.receive_client_updates(updates)
    server.aggregation()
    server.apply_weight()
    server.broadcast()


@pytest.mark.parametrize("mode, expected", [("samples", 2.5), ("uniform", 2.0)])
def test_shared_a_average_supports_explicit_weights(mode, expected):
    aggregator = FedAggregator_FedSAShareA(FedAggregatorArgs({"weighting_mode": mode}))
    updates = [{"updated_weights": {A_KEY: torch.full((2, 4), value)}, "train_record": {"data_sample_num": count}}
               for value, count in [(1.0, 2), (3.0, 6), (500.0, 0)]]
    result = aggregator.aggregate(updates)
    assert result.keys() == {A_KEY}
    torch.testing.assert_close(result[A_KEY], torch.full((2, 4), expected))


def test_private_b_survives_two_rounds_and_a_skipped_client():
    torch.manual_seed(15)
    clients = [_client("c1", 4), _client("c2", 6), _client("c3", 8)]
    server, runner = _server(clients)
    server.prepare({}, clients)
    server.broadcast()
    initial_a = server.node_var.model_weight[A_KEY].clone()
    for client in clients:
        assert set(key for key, p in client.node_var.model.named_parameters() if p.requires_grad) == {A_KEY, B_KEY}
        torch.testing.assert_close(client.node_var.model_weight[A_KEY], initial_a)
        assert torch.count_nonzero(client.node_var.model_weight[B_KEY]) == 0

    first = list(runner.simulate_client_local_training_process(clients[:2]))
    private = [client.node_var.model_weight[B_KEY].clone() for client in clients]
    assert torch.count_nonzero(private[0]) and torch.count_nonzero(private[1])
    assert not torch.equal(private[0], private[1])
    for update in first:
        assert update["updated_weights"].keys() == {A_KEY}
        assert update["train_record"].keys() == {"node_id", "data_sample_num", "fedsa_version"}
    retry, _ = clients[0].run_local_training()
    torch.testing.assert_close(retry[A_KEY], first[0]["updated_weights"][A_KEY])
    assert len(clients[0].node_var.trainer.starting_weights) == 1

    expected = 0.4 * first[0]["updated_weights"][A_KEY] + 0.6 * first[1]["updated_weights"][A_KEY]
    _apply_round(server, first)
    assert not torch.equal(expected, initial_a)
    for client, previous_b in zip(clients, private):
        torch.testing.assert_close(client.node_var.model_weight[A_KEY], expected)
        torch.testing.assert_close(client.node_var.model_weight[B_KEY], previous_b)

    second = list(runner.simulate_client_local_training_process([clients[0], clients[2]]))
    for index in (0, 2):
        starting = clients[index].node_var.trainer.starting_weights[-1]
        torch.testing.assert_close(starting[A_KEY], expected)
        torch.testing.assert_close(starting[B_KEY], private[index])
    after_local = [client.node_var.model_weight[B_KEY].clone() for client in clients]
    _apply_round(server, second)
    server.broadcast()  # repeated dispatch is safe
    assert server.strategy.version == 2
    assert server.node_var.model_weight.keys() == {A_KEY}
    assert server.node_var.aggregated_weight.keys() == {A_KEY}
    for client, previous_b in zip(clients, after_local):
        assert client.strategy.version == 2
        torch.testing.assert_close(client.node_var.model_weight[A_KEY], server.node_var.model_weight[A_KEY])
        torch.testing.assert_close(client.node_var.model_weight[B_KEY], previous_b)

    server.evaluate()
    assert server.eval_results["total_test_samples"] == 3 * 4
    assert server.eval_results["fedsa_evaluation_clients"] == 3
    assert set(server.eval_results).isdisjoint({"predictions", "weights", A_KEY, B_KEY})


def test_invalid_update_is_rejected_before_server_records_it():
    client = _client("c1", 3)
    server, runner = _server([client])
    server.prepare({}, [client])
    server.broadcast()
    updates = list(runner.simulate_client_local_training_process([client]))
    private_tensor = client.node_var.model_weight[B_KEY]
    invalid = copy.deepcopy(updates)
    invalid[0]["updated_weights"][B_KEY] = private_tensor
    with pytest.raises(ValueError, match="exactly the shared A"):
        server.receive_client_updates(invalid)
    invalid = copy.deepcopy(updates)
    invalid[0]["train_record"]["data_sample_num"] = private_tensor
    with pytest.raises(ValueError, match="sample counts"):
        server.receive_client_updates(invalid)
    with pytest.raises(ValueError, match="duplicate"):
        server.receive_client_updates(updates * 2)
    assert not server.node_var.client_updates

    _apply_round(server, updates)
    with pytest.raises(ValueError, match="targets version"):
        server.receive_client_updates(updates)
    with pytest.raises(ValueError, match="Unexpected.*version"):
        client.receive_weight({"version": 1, "shared_a": server.node_var.model_weight})


@pytest.mark.parametrize("empty_selection", [False, True])
def test_round_without_effective_uploads_keeps_version_and_client_state(empty_selection):
    active = _client("active", 4)
    empty = _client("empty", 0)
    server, runner = _server([active, empty])
    server.prepare({}, [active, empty])
    server.broadcast()
    _apply_round(server, list(runner.simulate_client_local_training_process([active])))
    committed = server.node_var.model_weight
    aggregate = server.node_var.aggregated_weight
    private = active.node_var.model_weight[B_KEY].clone()

    participants = [] if empty_selection else [empty]
    _apply_round(server, list(runner.simulate_client_local_training_process(participants)))
    assert server.strategy.version == active.strategy.version == empty.strategy.version == 1
    assert server.node_var.model_weight is committed
    assert server.node_var.aggregated_weight is aggregate
    assert not empty.node_var.trainer.starting_weights
    torch.testing.assert_close(active.node_var.model_weight[B_KEY], private)
    assert not server.node_var.client_updates


@pytest.mark.parametrize("rank, alpha", [(1, 4), (2, 8)])
def test_setup_rejects_incompatible_shared_factor_coordinates(rank, alpha):
    client = _client("c1", 3, rank=rank, alpha=alpha)
    server, _ = _server([client])
    with pytest.raises(ValueError, match="common adapter layers, ranks and scales"):
        server.prepare({}, [client])
    assert client.strategy.version is None


def test_client_test_mode_requires_an_explicit_client_loader():
    client = _client("c1", 3, evaluation_loader="client_test")
    server, _ = _server([client])
    with pytest.raises(ValueError, match="no test data loader for 'client_test'"):
        server.prepare({}, [client])
