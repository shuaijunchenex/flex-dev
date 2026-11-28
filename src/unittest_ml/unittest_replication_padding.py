from collections import OrderedDict

import pytest
import torch

from flex.fl_algorithms.aggregation.fed_aggregator_args import FedAggregatorArgs
from flex.fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from flex.fl_algorithms.aggregation.methods._fed_aggregator_rbla import (
    FedAggregator_RBLA,
)
from flex.fl_algorithms.aggregation.methods._fed_aggregator_replication_padding import (
    FedAggregator_ReplicationPadding,
)


def _state(a, b, *, dense=0.0, counter=0):
    return OrderedDict(
        [
            ("layer.weight", torch.tensor([dense], dtype=torch.float32)),
            ("layer.counter", torch.tensor(counter, dtype=torch.int64)),
            ("layer.lora_A", torch.as_tensor(a, dtype=torch.float32).clone()),
            ("layer.lora_B", torch.as_tensor(b, dtype=torch.float32).clone()),
        ]
    )


def test_replication_padding_matches_hand_computed_multirank_result():
    states = [
        _state([[1.0, 1.0]], [[10.0], [20.0]], dense=1.0, counter=7),
        _state(
            [[2.0, 2.0], [3.0, 3.0]],
            [[30.0, 40.0], [50.0, 60.0]],
            dense=2.0,
            counter=8,
        ),
        _state(
            [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
            [[70.0, 80.0, 90.0], [100.0, 110.0, 120.0]],
            dense=3.0,
            counter=9,
        ),
    ]

    result = FedAggregator_ReplicationPadding.aggregate_state_dicts(
        states, [1.0, 2.0, 3.0]
    )

    donor_a = states[2]["layer.lora_A"]
    donor_b = states[2]["layer.lora_B"]
    padded_a_0 = donor_a.clone()
    padded_a_0[:1] = states[0]["layer.lora_A"]
    padded_a_1 = donor_a.clone()
    padded_a_1[:2] = states[1]["layer.lora_A"]
    padded_b_0 = donor_b.clone()
    padded_b_0[:, :1] = states[0]["layer.lora_B"]
    padded_b_1 = donor_b.clone()
    padded_b_1[:, :2] = states[1]["layer.lora_B"]

    expected_a = (padded_a_0 + 2 * padded_a_1 + 3 * donor_a) / 6
    expected_b = (padded_b_0 + 2 * padded_b_1 + 3 * donor_b) / 6
    assert torch.allclose(result["layer.lora_A"], expected_a)
    assert torch.allclose(result["layer.lora_B"], expected_b)
    assert torch.allclose(result["layer.weight"], torch.tensor([14.0 / 6.0]))
    assert result["layer.counter"].item() == 7


def test_multiple_max_rank_clients_form_weighted_donor():
    states = [
        _state([[1.0], [10.0]], [[2.0, 20.0]]),
        _state([[3.0], [30.0]], [[4.0, 40.0]]),
        _state([[5.0]], [[6.0]]),
    ]
    result = FedAggregator_ReplicationPadding.aggregate_state_dicts(
        states, [1.0, 3.0, 4.0]
    )

    donor_a_tail = (1.0 * 10.0 + 3.0 * 30.0) / 4.0
    donor_b_tail = (1.0 * 20.0 + 3.0 * 40.0) / 4.0
    assert result["layer.lora_A"][1, 0].item() == pytest.approx(donor_a_tail)
    assert result["layer.lora_B"][0, 1].item() == pytest.approx(donor_b_tail)


def test_equal_rank_reduces_to_weighted_factor_average():
    states = [
        _state([[1.0], [2.0]], [[3.0, 4.0]]),
        _state([[5.0], [6.0]], [[7.0, 8.0]]),
    ]
    result = FedAggregator_ReplicationPadding.aggregate_state_dicts(
        states, [1.0, 3.0]
    )
    assert torch.allclose(
        result["layer.lora_A"],
        (states[0]["layer.lora_A"] + 3 * states[1]["layer.lora_A"]) / 4,
    )
    assert torch.allclose(
        result["layer.lora_B"],
        (states[0]["layer.lora_B"] + 3 * states[1]["layer.lora_B"]) / 4,
    )


def test_client_count_mode_ignores_data_volume_and_factory_builds_method():
    args = FedAggregatorArgs(
        {
            "aggregation": {
                "method": "replication_padding",
                "device": "cpu",
                "weighting_mode": "client_count",
            }
        }
    )
    aggregator = FedAggregatorFactory.create_aggregator(args)
    states = [
        _state([[1.0]], [[2.0]], dense=1.0),
        _state([[3.0]], [[4.0]], dense=3.0),
    ]
    result = aggregator.aggregate(
        [
            {"updated_weights": states[0], "train_record": {"data_sample_num": 1}},
            {"updated_weights": states[1], "train_record": {"data_sample_num": 99}},
        ]
    )
    assert isinstance(aggregator, FedAggregator_ReplicationPadding)
    assert result["layer.weight"].item() == pytest.approx(2.0)


def test_rbla_broadcast_slices_replication_result_to_local_rank():
    states = [
        _state([[1.0]], [[2.0]]),
        _state([[3.0], [4.0]], [[5.0, 6.0]]),
    ]
    global_state = FedAggregator_ReplicationPadding.aggregate_state_dicts(
        states, [1.0, 1.0]
    )
    local = FedAggregator_RBLA.broadcast_lora_state_dict(global_state, states[0])
    assert local["layer.lora_A"].shape == states[0]["layer.lora_A"].shape
    assert local["layer.lora_B"].shape == states[0]["layer.lora_B"].shape
    assert torch.equal(local["layer.lora_A"], global_state["layer.lora_A"][:1])
    assert torch.equal(local["layer.lora_B"], global_state["layer.lora_B"][:, :1])


def test_conv_style_expanded_rank_axes_are_replicated():
    low = _state(
        torch.arange(12, dtype=torch.float32).reshape(2, 6),
        torch.arange(18, dtype=torch.float32).reshape(9, 2),
    )
    high = _state(
        torch.arange(24, dtype=torch.float32).reshape(4, 6) + 100,
        torch.arange(36, dtype=torch.float32).reshape(9, 4) + 200,
    )
    result = FedAggregator_ReplicationPadding.aggregate_state_dicts(
        [low, high], [1.0, 1.0]
    )
    assert result["layer.lora_A"].shape == (4, 6)
    assert result["layer.lora_B"].shape == (9, 4)
    assert torch.equal(result["layer.lora_A"][2:], high["layer.lora_A"][2:])
    assert torch.equal(result["layer.lora_B"][:, 2:], high["layer.lora_B"][:, 2:])


def test_mismatched_paired_rank_is_rejected():
    state = _state([[1.0], [2.0]], [[3.0]])
    with pytest.raises(ValueError, match="rank dimensions do not match"):
        FedAggregator_ReplicationPadding.aggregate_state_dicts([state], [1.0])
