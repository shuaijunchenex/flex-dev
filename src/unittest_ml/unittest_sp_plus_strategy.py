from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import torch

from flex.fl_algorithms import FedAggregatorArgs, FedAggregatorFactory
from flex.fl_algorithms.aggregation.methods._fed_aggregator_rbla import (
    FedAggregator_RBLA,
)
from flex.fl_algorithms.aggregation.methods._fed_aggregator_sp import FedAggregator_SP
from flex.fl_algorithms.aggregation.methods._fed_aggregator_sp_plus import (
    FedAggregator_SPPlus,
)
from flex.fed_strategy.client_strategy_impl._sp_plus_client import (
    SpPlusClientTrainingStrategy,
)
from flex.fed_strategy.server_strategy_impl._sp_plus_server import (
    SpPlusServerStrategy,
)
from flex.fed_strategy.strategy_args import StrategyArgs
from flex.fed_strategy.strategy_factory import StrategyFactory


def _client_data(states: list[OrderedDict]) -> list[dict]:
    return [
        {
            "updated_weights": state,
            "train_record": {"data_sample_num": index + 1},
        }
        for index, state in enumerate(states)
    ]


def _heterogeneous_states() -> list[OrderedDict]:
    generator = torch.Generator().manual_seed(20260721)
    return [
        OrderedDict(
            [
                ("layer.lora_A", torch.randn(2, 5, generator=generator)),
                ("layer.lora_B", torch.randn(4, 2, generator=generator)),
                ("other", torch.tensor([1.0])),
            ]
        ),
        OrderedDict(
            [
                ("layer.lora_A", torch.randn(4, 5, generator=generator)),
                ("layer.lora_B", torch.randn(4, 4, generator=generator)),
                ("other", torch.tensor([3.0])),
            ]
        ),
    ]


def test_sp_plus_matches_the_previous_rbla_plus_computation() -> None:
    states = _heterogeneous_states()
    previous = FedAggregatorFactory.create_aggregator(
        FedAggregatorArgs(
            {
                "method": "rbla",
                "device": "cpu",
                "canonicalization": {"enabled": True, "log_diagnostics": False},
            }
        )
    ).aggregate(_client_data(states))

    sp_plus = FedAggregatorFactory.create_aggregator(
        FedAggregatorArgs(
            {
                "method": "sp_plus",
                "device": "cpu",
                "canonicalization": {"log_diagnostics": False},
            }
        )
    )
    actual = sp_plus.aggregate(_client_data(states))

    assert isinstance(sp_plus, FedAggregator_SPPlus)
    assert sp_plus.aggregated_method == "sp_plus"
    assert sp_plus.canonicalization_applied_last_round
    assert "broadcast_lora_state_dict" in FedAggregator_SPPlus.__dict__
    assert list(actual) == list(previous)
    for key in previous:
        assert torch.equal(actual[key], previous[key])


def test_sp_plus_broadcasts_the_canonical_rank_prefix() -> None:
    aggregator = FedAggregatorFactory.create_aggregator(
        FedAggregatorArgs(
            {
                "method": "sp_plus",
                "device": "cpu",
                "canonicalization": {"log_diagnostics": False},
            }
        )
    )
    global_state = aggregator.aggregate(_client_data(_heterogeneous_states()))
    local_state = OrderedDict(
        [
            ("layer.lora_A", torch.empty(2, 5)),
            ("layer.lora_B", torch.empty(4, 2)),
            ("other", torch.empty(1)),
        ]
    )
    node_var = SimpleNamespace(cache_weight=global_state, model_weight=local_state)
    client_node = SimpleNamespace(node_var=node_var)
    strategy = StrategyFactory.create_client_strategy(
        StrategyArgs({"role": "client", "strategy_name": "sp_plus"}),
        client_node,
    )
    strategy.set_local_weight()

    assert isinstance(strategy, SpPlusClientTrainingStrategy)
    assert torch.equal(node_var.model_weight["layer.lora_A"], global_state["layer.lora_A"][:2])
    assert torch.equal(node_var.model_weight["layer.lora_B"], global_state["layer.lora_B"][:, :2])
    assert torch.equal(node_var.model_weight["other"], global_state["other"])


def test_sp_plus_registration_does_not_replace_rbla_or_sp() -> None:
    rbla = FedAggregatorFactory.create_aggregator(
        FedAggregatorArgs({"method": "rbla", "device": "cpu"})
    )
    sp = FedAggregatorFactory.create_aggregator(
        FedAggregatorArgs({"method": "sp", "device": "cpu"})
    )
    assert type(rbla) is FedAggregator_RBLA
    assert type(sp) is FedAggregator_SP

    server = StrategyFactory.create_server_strategy(
        StrategyArgs({"role": "server", "strategy_name": "sp_plus"}),
        SimpleNamespace(),
    )
    assert isinstance(server, SpPlusServerStrategy)
    assert server._strategy_type == "sp_plus"
