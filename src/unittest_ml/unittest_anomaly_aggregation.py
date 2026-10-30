"""Raw FLoRIST numeric behavior; independent recorder tests stay separate."""

from __future__ import annotations

import copy
from unittest.mock import Mock, patch

import pytest
import torch

from flex.ml_utils.anomaly_detector import AnomalyDetector
from unittest_ml.unittest_florist_aggregation import client, make_aggregator, random_clients
from unittest_ml.unittest_florist_strategy import _global_payload, _initialized


@pytest.fixture(autouse=True)
def preserve_rng():
    with torch.random.fork_rng(devices=[]):
        yield


def attach(aggregator, directory, enabled=True):
    detector = AnomalyDetector(directory, enabled=enabled)
    for name in ("observe", "remember", "record_exception"):
        setattr(detector, name, Mock(side_effect=AssertionError("Core must not call " + name)))
    aggregator.anomaly_detector = detector
    aggregator.anomaly_context = {"round": 93, "client_id": None}
    return detector


def attach_server(server, directory):
    detector = attach(server.node_var.aggregation_method, directory)
    server.node_var.anomaly_detector = detector
    server.node_var.anomaly_context = {"round": 93, "client_id": None}
    return detector


def assert_unused(detector):
    assert not detector.triggered
    detector.close()
    assert not detector.output_dir.exists()


def assert_same_state(left, right):
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_attributes_preserve_factors_rng_and_three_svds(tmp_path, enabled):
    updates = random_clients()
    original = copy.deepcopy(updates)
    expected = make_aggregator(0.85).aggregate(updates)
    aggregator = make_aggregator(0.85)
    detector = attach(aggregator, tmp_path / "unused", enabled)
    state = torch.get_rng_state().clone()
    with patch("torch.linalg.svd", wraps=torch.linalg.svd) as svd:
        actual = aggregator.aggregate(updates)
    assert svd.call_count == 3
    assert all(c.kwargs == {"full_matrices": False} for c in svd.call_args_list)
    assert torch.equal(torch.get_rng_state(), state)
    assert_same_state(actual, expected)
    for before, after in zip(original, updates):
        assert_same_state(before["updated_weights"], after["updated_weights"])
    assert_unused(detector)


@pytest.mark.parametrize("factor", ["A", "B"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "negative_inf"])
def test_nonfinite_factors_reach_svd_with_all_clients_and_original_weights(tmp_path, factor, bad):
    updates = random_clients()
    updates[1]["updated_weights"]["proj.lora_" + factor][0, 0] = bad
    original = copy.deepcopy(updates)
    aggregator = make_aggregator()
    detector = attach(aggregator, tmp_path / "raw_input")
    weights = [0.1, 0.3, 0.6]
    expected = [torch.cat([u["updated_weights"]["proj.lora_B"] for u in updates], dim=1),
                torch.cat([u["updated_weights"]["proj.lora_A"] * w for u, w in zip(updates, weights)], dim=0)]
    original_svd, calls = torch.linalg.svd, []
    error = RuntimeError("sentinel from nonfinite SVD input")

    def raw_svd(value, **kwargs):
        index = len(calls)
        calls.append(value.detach().clone())
        assert kwargs == {"full_matrices": False}
        torch.testing.assert_close(value, expected[index], rtol=0, atol=0, equal_nan=True)
        if not torch.isfinite(value).all():
            raise error
        return original_svd(value, **kwargs)

    # Former guards rejected before SVD; bad A with finite B now reaches SVD 2.
    with patch("torch.linalg.svd", side_effect=raw_svd), pytest.raises(RuntimeError) as caught:
        aggregator.aggregate(updates)
    assert caught.value is error
    assert len(calls) == (2 if factor == "A" else 1)
    assert aggregator._sample_weights == weights
    for before, after in zip(original, updates):
        assert_same_state(before["updated_weights"], after["updated_weights"])
    assert_unused(detector)


def test_finite_factor_overflow_reaches_original_core_svd(tmp_path):
    state = {"proj.lora_A": torch.tensor([[1e30]]), "proj.lora_B": torch.tensor([[1e30]])}
    aggregator = make_aggregator()
    detector = attach(aggregator, tmp_path / "raw_core")
    original_svd, calls = torch.linalg.svd, []
    error = RuntimeError("sentinel at core SVD")

    def raw_svd(value, **kwargs):
        calls.append(value.detach().clone())
        if len(calls) == 3:
            assert torch.isinf(value).all()
            raise error
        assert torch.isfinite(value).all()
        return original_svd(value, **kwargs)

    with patch("torch.linalg.svd", side_effect=raw_svd), pytest.raises(RuntimeError) as caught:
        aggregator.aggregate([client(state, 1)])
    assert caught.value is error
    assert len(calls) == 3
    assert_unused(detector)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "negative_inf"])
def test_nonfinite_head_is_averaged_without_filter_or_guard(tmp_path, bad):
    updates = random_clients()
    updates[1]["updated_weights"]["head.weight"][0, 0] = bad
    aggregator = make_aggregator()
    detector = attach(aggregator, tmp_path / "raw_head")
    expected = sum(u["updated_weights"]["head.weight"] * w for u, w in zip(updates, (0.1, 0.3, 0.6)))
    with patch("torch.linalg.svd", wraps=torch.linalg.svd) as svd:
        result = aggregator.aggregate(updates)
    assert svd.call_count == 3
    torch.testing.assert_close(result["head.weight"], expected, rtol=0, atol=1e-15, equal_nan=True)
    assert not torch.isfinite(result["head.weight"][0, 0])
    assert aggregator.last_stats["num_clients"] == 3
    assert_unused(detector)


@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_svd_exception_propagates_same_object_without_recording(tmp_path, failure_index):
    aggregator = make_aggregator()
    detector = attach(aggregator, tmp_path / "svd_exception")
    original_svd, calls = torch.linalg.svd, []
    error = RuntimeError("injected finite-input backend failure")

    def fail_svd(value, **kwargs):
        assert torch.isfinite(value).all()
        calls.append(value.shape)
        if len(calls) - 1 == failure_index:
            raise error
        return original_svd(value, **kwargs)

    with patch("torch.linalg.svd", side_effect=fail_svd), pytest.raises(RuntimeError) as caught:
        aggregator.aggregate(random_clients())
    assert caught.value is error
    assert len(calls) == failure_index + 1
    assert_unused(detector)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_nonfinite_backend_outputs_are_used_without_post_svd_guard(tmp_path, index):
    aggregator = make_aggregator()
    detector = attach(aggregator, tmp_path / "svd_output")
    original_svd, calls = torch.linalg.svd, []

    def corrupt_svd(value, **kwargs):
        result = original_svd(value, **kwargs)
        calls.append(value.shape)
        if len(calls) - 1 == index:
            # U_b / Vh_a / U_core affect final factors, not later SVD inputs.
            result[2 if index == 1 else 0][0, 0] = float("nan")
        return result

    with patch("torch.linalg.svd", side_effect=corrupt_svd):
        result = aggregator.aggregate(random_clients())
    assert len(calls) == 3
    assert any(not torch.isfinite(value).all() for value in result.values())
    assert_unused(detector)


def test_server_structure_validation_remains_strict(tmp_path):
    server, _ = _initialized()
    detector = attach_server(server, tmp_path / "structure")
    payload = _global_payload(2)
    payload["unrecognized.weight"] = torch.ones(2)
    server.node_var.aggregated_weight = payload
    with pytest.raises(ValueError, match="unexpected"):
        server.strategy.apply_weight()
    assert_unused(detector)


@pytest.mark.parametrize("stage", ["apply", "broadcast"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "negative_inf"])
def test_nonfinite_global_head_reaches_clients_or_evaluator(tmp_path, stage, bad):
    server, clients = _initialized()
    detector = attach_server(server, tmp_path / stage)
    payload = _global_payload(2)
    payload["head.weight"][0, 0] = bad
    server.node_var.aggregated_weight = payload
    server.node_var.model_weight = payload
    getattr(server.strategy, "apply_weight" if stage == "apply" else stage)()
    states = ([server.node_var.model_evaluator.model.state_dict()] if stage == "apply"
              else [node.node_var.model_weight for node in clients])
    for state in states:
        torch.testing.assert_close(state["head.weight"], payload["head.weight"], rtol=0, atol=0, equal_nan=True)
    assert_unused(detector)


@pytest.mark.parametrize("kind", ["delta", "evaluated_weight"])
def test_server_reconstruction_overflow_reaches_evaluator(tmp_path, kind):
    server, _ = _initialized()
    detector = attach_server(server, tmp_path / kind)
    payload = _global_payload(1)
    payload["layer.lora_A"].fill_(1e30 if kind == "delta" else 1e19)
    payload["layer.lora_B"].fill_(1e30 if kind == "delta" else 2e19)
    if kind == "evaluated_weight":
        server.strategy._initial_weight["layer.weight"].fill_(2e38)
    server.node_var.aggregated_weight = payload
    server.strategy.apply_weight()
    assert torch.isinf(server.node_var.model_evaluator.model.state_dict()["layer.weight"]).all()
    assert_unused(detector)


def test_server_healthy_evaluation_ignores_legacy_attributes_exactly(tmp_path):
    expected_server, _ = _initialized()
    expected_server.node_var.aggregated_weight = _global_payload(3)
    expected_server.strategy.apply_weight()
    server, _ = _initialized()
    detector = attach_server(server, tmp_path / "healthy_server")
    server.node_var.aggregated_weight = _global_payload(3)
    state = torch.get_rng_state().clone()
    server.strategy.apply_weight()
    assert torch.equal(state, torch.get_rng_state())
    assert_same_state(server.node_var.model_evaluator.model.state_dict(),
                      expected_server.node_var.model_evaluator.model.state_dict())
    assert_unused(detector)
