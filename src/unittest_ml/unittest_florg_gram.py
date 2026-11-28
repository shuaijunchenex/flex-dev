import pytest
import torch
from torch import nn
from torch.nn import functional as F

from flex.fl_algorithms.aggregation.fed_aggregator_args import FedAggregatorArgs
from flex.fl_algorithms.aggregation.methods._fed_aggregator_florg_gram import FedAggregator_FLoRGGram
from flex.ml_algorithms.lora.impl.gram_lora import GramLoRALinear


def test_gram_linear_matches_dense_rectangular_update_and_only_trains_factor():
    base = nn.Linear(5, 3, dtype=torch.float64)
    layer = GramLoRALinear(base, 2, generator=torch.Generator().manual_seed(45))
    inputs = torch.randn(4, 2, 5, dtype=torch.float64)
    eye = torch.eye(3, dtype=torch.float64)
    torch.testing.assert_close(layer.basis_left.T @ layer.basis_left, eye)
    torch.testing.assert_close(layer.basis_right @ layer.basis_right.T, eye)

    dense = layer.basis_left @ layer.gram_factor.T @ layer.gram_factor @ layer.basis_right
    torch.testing.assert_close(layer.effective_update(), dense)
    torch.testing.assert_close(layer(inputs), F.linear(inputs, base.weight + dense, base.bias))
    layer(inputs).square().sum().backward()
    assert torch.count_nonzero(layer.gram_factor.grad)
    assert {name for name, parameter in layer.named_parameters() if parameter.requires_grad} == {"gram_factor"}
    assert base.weight.grad is None and base.bias.grad is None


def test_rank_overflow_truncates_gram_before_alignment_and_is_gauge_invariant():
    left = torch.tensor([[3., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
    right = torch.tensor([[0., 2., 0.], [0., 0., 1.]], dtype=torch.float64)
    previous = torch.tensor([[0., 1., 0.], [1., 0., 0.]], dtype=torch.float64)
    fitted = FedAggregator_FLoRGGram.aggregate_gram_factors([left, right], [1, 1], previous)
    rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
    gauge = FedAggregator_FLoRGGram.aggregate_gram_factors([rotation @ left, -right], [1, 1], previous)
    expected = torch.diag(torch.tensor([4.5, 2.5, 0.], dtype=torch.float64))
    torch.testing.assert_close(fitted.factor.T @ fitted.factor, expected)
    torch.testing.assert_close(gauge.factor, fitted.factor)
    torch.testing.assert_close(fitted.eigenvalues, torch.tensor([4.5, 2.5, .5], dtype=torch.float64))
    assert fitted.gram_rank == 3
    assert fitted.gram_projection_residual == pytest.approx(.5)
    assert fitted.factor_drift_after < fitted.factor_drift_before


def test_rank_deficient_and_zero_grams_respect_shape_and_keep_exact_gram():
    previous = torch.tensor([[1., 0., 0.], [0., 2., 0.]])
    factor = torch.tensor([[1., 2., 1.], [0., 0., 0.]])
    result = FedAggregator_FLoRGGram.aggregate_gram_factors([factor], [1], previous, align=False)
    assert result.factor.shape == previous.shape
    assert result.gram_rank == 1
    assert result.factor_drift_before == result.factor_drift_after
    torch.testing.assert_close(result.factor.T @ result.factor, factor.T @ factor)

    zeros = torch.zeros_like(factor)
    zero_result = FedAggregator_FLoRGGram.aggregate_gram_factors([zeros], [1], previous)
    torch.testing.assert_close(zero_result.factor, zeros)
    assert zero_result.gram_rank == 0
    assert zero_result.gram_projection_residual == 0

    with pytest.raises(ValueError, match="positive total weight"):
        FedAggregator_FLoRGGram.aggregate_gram_factors([factor], [0], previous)
    with pytest.raises(ValueError, match="broadcast shape"):
        FedAggregator_FLoRGGram.aggregate_gram_factors([factor[:1]], [1], previous)


@pytest.mark.parametrize(("mode", "expected"), [("uniform", 5.0), ("samples", 7.0)])
def test_aggregator_respects_configured_weighting_and_requires_compact_payload(mode, expected):
    key = "layer.gram_factor"
    reference = {key: torch.tensor([[1., 0.]])}
    aggregator = FedAggregator_FLoRGGram(FedAggregatorArgs({"weighting_mode": mode, "align": True}))
    aggregator.set_reference(reference)
    updates = [
        {"updated_weights": {key: torch.tensor([[float(value), 0.]])},
         "train_record": {"data_sample_num": volume}}
        for value, volume in [(1, 1), (3, 3), (100, 0)]
    ]
    result = aggregator.aggregate(updates)[key]
    assert (result.T @ result)[0, 0].item() == pytest.approx(expected)
    assert aggregator.last_stats[key]["gram_rank"] == 1
    updates[2]["train_record"]["data_sample_num"] = float("nan")
    with pytest.raises(ValueError, match="finite and nonnegative"):
        aggregator.aggregate(updates)
    updates[2]["train_record"]["data_sample_num"] = 0
    updates[0]["updated_weights"]["backbone"] = torch.zeros(2)
    with pytest.raises(ValueError, match="exactly"):
        aggregator.aggregate(updates)



def test_live_round_keeps_nonzero_factor_and_empty_round_is_noop():
    from copy import deepcopy
    from types import SimpleNamespace

    from flex.fed_strategy.client_strategy_impl._florg_gram_client import FlorgGramClientTrainingStrategy
    from flex.fed_strategy.server_strategy_impl._florg_gram_server import FlorgGramServerStrategy

    model = GramLoRALinear(nn.Linear(4, 3), 2)
    client_model = deepcopy(model)
    client = SimpleNamespace(
        node_id="client",
        node_var=SimpleNamespace(model=client_model, model_weight=client_model.state_dict()),
    )
    client.strategy = FlorgGramClientTrainingStrategy(None, client)
    client.receive_weight = client.strategy.receive_weight
    client.set_local_weight = client.strategy.set_local_weight
    server = SimpleNamespace(node_var=SimpleNamespace(
        model=model,
        model_weight=model.state_dict(),
        model_evaluator=SimpleNamespace(update_model=model.load_state_dict),
        aggregation_method=FedAggregator_FLoRGGram(),
        training_logger=SimpleNamespace(begin=lambda header: None),
    ))
    server.set_client_nodes = lambda clients: setattr(server, "client_nodes", clients)
    strategy = FlorgGramServerStrategy(None, server)
    strategy.prepare({}, [client])
    strategy.broadcast()
    previous = client.node_var.model_weight["gram_factor"].clone()
    assert torch.count_nonzero(previous)

    zero_sample = {
        "updated_weights": {},
        "train_record": {"node_id": "client", "data_sample_num": 0, "base_version": 0},
    }
    for inactive_updates in ([], [zero_sample]):
        strategy.receive_client_updates(inactive_updates)
        strategy.aggregation()
        strategy.apply_weight()
        strategy.broadcast()
        assert client.strategy._version == 0
        assert not strategy.history

    upload = {"gram_factor": torch.zeros_like(previous)}
    update = {
        "updated_weights": upload,
        "train_record": {"node_id": "client", "data_sample_num": 5, "base_version": 0},
    }
    strategy.receive_client_updates([update])
    strategy.aggregation()
    strategy.apply_weight()
    strategy.broadcast()
    assert client.strategy._version == 1
    torch.testing.assert_close(client.node_var.model_weight["gram_factor"], previous)
    stats = strategy.history[0]["aggregation_stats"]["gram_factor"]
    assert stats["kept_previous_factor"]
    assert stats["broadcast_gram_residual"] > 0
    assert strategy.history[0]["actual_download_bytes"] == previous.numel() * previous.element_size()
    with pytest.raises(ValueError, match="stale"):
        strategy.aggregation()
