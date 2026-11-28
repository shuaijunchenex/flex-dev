"""Core training ignores legacy diagnostic settings and preserves raw failures."""

from __future__ import annotations

import copy
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flex.fed_strategy.runner_strategy_impl._rbla_runner_strategy import RblaRunnerStrategy
from flex.fed_strategy.strategy_args import StrategyArgs
from flex.ml_algorithms.lora.impl.lora_ms import MSLoRALinear
from flex.ml_utils.anomaly_detector import AnomalyDetector
from flex.model_trainer.model_trainer_args import ModelTrainerArgs
from flex.model_trainer.trainer._model_trainer_standard import ModelTrainer_Standard
from unittest_ml.unittest_florist_strategy import _Client, _Server


@pytest.fixture(autouse=True)
def restore_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.25)
        self.layer = MSLoRALinear(3, 2, r=2, lora_alpha=4, merge_weights=False)
        self.seen = []

    def forward(self, value):
        self.seen.append(value.detach().clone())
        return self.layer(self.dropout(value))


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, corrupt=False):
        super().__init__(parameters, lr=0.04, momentum=0.7, weight_decay=0.001)
        self.step_count = 0
        self.corrupt = corrupt

    def step(self, closure=None):
        result = super().step(closure)
        self.step_count += 1
        if self.corrupt:
            with torch.no_grad():
                next(parameter for group in self.param_groups for parameter in group["params"]
                     if parameter.requires_grad).fill_(float("inf"))
        return result


class _OptimizerBuilder:
    def __init__(self):
        self.created = []

    def rebuild(self, parameters):
        optimizer = _CountingSGD(parameters)
        self.created.append(optimizer)
        return optimizer


def _make_trainer(model, optimizer, input_dim=3, loss_func=None):
    values = torch.arange(27 if input_dim == 3 else 30, dtype=torch.float32).reshape(-1, input_dim) / 20
    labels = torch.arange(len(values)) % 2
    args = ModelTrainerArgs()
    args.set_trainer_args(model, optimizer, loss_func or nn.CrossEntropyLoss(),
                          SimpleNamespace(data_loader=DataLoader(TensorDataset(values, labels),
                                                                batch_size=3, shuffle=True)))
    args.device = "cpu"
    return ModelTrainer_Standard(args)


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_equal(a, b)
    else:
        assert left == right


def _unused_detector(directory, enabled=True):
    detector = AnomalyDetector(directory, enabled=enabled)
    for name in ("observe", "remember", "record_exception"):
        setattr(detector, name, Mock(side_effect=AssertionError("Core must not call " + name)))
    return detector


def _training_run(directory, enabled, fault=None):
    random.seed(21)
    np.random.seed(22)
    torch.manual_seed(23)
    model = _Model()
    optimizer = _CountingSGD(model.parameters(), corrupt=fault == "optimizer_step")
    observed_losses = []

    def loss_func(outputs, labels):
        loss = nn.functional.cross_entropy(outputs, labels)
        observed_losses.append(float(loss.detach()))
        return loss * float("nan") if fault == "forward" else loss

    if fault == "backward":
        model.layer.lora_A.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    trainer = _make_trainer(model, optimizer, loss_func=loss_func)
    detector = _unused_detector(directory, enabled)
    trainer.trainer_args.anomaly_detector = detector
    trainer.trainer_args.anomaly_context = {"client_id": "client.0", "round": 4}
    state, stats = trainer.train(2)
    result = {
        "state": copy.deepcopy(state), "optimizer": copy.deepcopy(optimizer.state_dict()),
        "rng": AnomalyDetector.capture_rng_state(), "step_count": optimizer.step_count,
        "seen": model.seen, "stats": stats, "observed_losses": observed_losses,
    }
    detector.close(status="completed")
    return result, detector


def test_legacy_diagnostic_attributes_do_not_change_parameters_optimizer_rng_or_batches(tmp_path):
    disabled, disabled_detector = _training_run(tmp_path / "disabled", False)
    enabled, enabled_detector = _training_run(tmp_path / "enabled", True)
    _assert_equal(disabled, enabled)
    assert enabled["step_count"] == 6
    assert enabled["stats"]["num_batches_sum"] == 6
    assert enabled["stats"]["num_samples_sum"] == 18
    assert not disabled_detector.triggered
    assert not enabled_detector.triggered
    assert not (tmp_path / "disabled").exists()
    assert not (tmp_path / "enabled").exists()


@pytest.mark.parametrize("stage", ["forward", "backward", "optimizer_step"])
def test_numeric_fault_keeps_all_optimizer_steps_without_automatic_recording(tmp_path, stage):
    result, detector = _training_run(tmp_path / stage, True, fault=stage)
    assert not detector.triggered
    assert result["step_count"] == 6
    assert result["stats"]["num_batches_sum"] == 6
    assert any(not torch.isfinite(value).all() for value in result["state"].values())
    assert np.isfinite(result["observed_losses"][0])
    assert not (tmp_path / stage).exists()


@pytest.mark.parametrize("stage", ["forward", "backward", "optimizer_step"])
def test_real_torch_exception_propagates_same_object_without_recording(tmp_path, stage):
    directory = tmp_path / stage
    model = _Model()
    optimizer = _CountingSGD(model.parameters())
    original = RuntimeError(f"{stage} implementation failed")

    def failing_loss(outputs, labels):
        raise original

    if stage == "backward":
        model.layer.lora_A.register_hook(lambda gradient: failing_loss(None, None))
    elif stage == "optimizer_step":
        optimizer.step = lambda: failing_loss(None, None)
    trainer = _make_trainer(model, optimizer, loss_func=failing_loss if stage == "forward" else None)
    detector = _unused_detector(directory)
    trainer.trainer_args.anomaly_detector = detector
    trainer.trainer_args.anomaly_context = {"client_id": "client.0", "round": 2}
    with pytest.raises(RuntimeError) as caught:
        trainer.train(1)
    assert caught.value is original
    assert optimizer.step_count == 0
    detector.close(status="failed")
    assert not detector.triggered
    assert not directory.exists()


def _two_clients(directory, enabled, inject_nan=True, initialize=True):
    torch.manual_seed(31)
    clients = [_Client("client.0", 2, 6), _Client("client.1", 2, 6)]
    detector = _unused_detector(directory, enabled)
    for client in clients:
        variables = client.node_var
        variables.optimizer_builder = _OptimizerBuilder()
        variables.trainer = _make_trainer(variables.model, _CountingSGD(variables.model.parameters()), input_dim=5)
        variables.config_dict["training"]["epochs"] = 2
        variables.anomaly_detector = detector
        variables.anomaly_context = {"client_id": client.node_id, "round": 1}
        variables.trainer.trainer_args.anomaly_detector = detector
        variables.trainer.trainer_args.anomaly_context = dict(variables.anomaly_context)
    server = _Server(clients, rank=2)
    if initialize:
        server.strategy.prepare("diagnostic test", clients)
        server.strategy.broadcast()
    server.node_var.anomaly_detector = detector
    server.node_var.anomaly_context = {"round": 1}
    server.node_var.aggregation_method.anomaly_detector = detector
    server.node_var.aggregation_method.anomaly_context = {"round": 1}
    # Forward hooks survive the client's model deepcopy; every bad-client batch
    # produces an actual NaN loss rather than injecting an invalid upload by hand.
    if inject_nan:
        clients[0].node_var.model.register_forward_hook(lambda module, args, output: output * float("nan"))
    runner = RblaRunnerStrategy(SimpleNamespace(), StrategyArgs({"role": "runner", "strategy_name": "rbla",
                                                               "training_rounds": 1}), clients, server)
    return runner, clients, server, detector


def test_healthy_florist_round_ignores_legacy_diagnostic_attributes(tmp_path):
    results = []
    for enabled in (False, True):
        directory = tmp_path / str(enabled)
        runner, clients, server, detector = _two_clients(directory, enabled, inject_nan=False)
        updates = list(runner.simulate_client_local_training_process(clients))
        server.strategy.receive_client_updates(updates)
        server.strategy.aggregation()
        server.strategy.apply_weight()
        results.append({
            "client_weights": [copy.deepcopy(client.node_var.model_weight) for client in clients],
            "global_factors": copy.deepcopy(server.node_var.model_weight),
            "evaluation_weights": copy.deepcopy(server.node_var.model_evaluator.model.state_dict()),
            "step_counts": [client.node_var.optimizer_builder.created[-1].step_count for client in clients],
            "rng": AnomalyDetector.capture_rng_state(),
        })
        assert results[-1]["step_counts"] == [4, 4]
        assert not detector.triggered
        detector.close()
        assert not directory.exists()
    _assert_equal(results[0], results[1])


@pytest.mark.parametrize("enabled", [False, True])
def test_nan_client_finishes_and_remaining_clients_reach_original_svd(tmp_path, monkeypatch, enabled):
    directory = tmp_path / "bad_round"
    runner, clients, server, detector = _two_clients(directory, enabled)
    updates = list(runner.simulate_client_local_training_process(clients))
    assert len(updates) == 2
    assert all(client.node_var.optimizer_builder.created[-1].step_count == 4 for client in clients)
    assert any(not torch.isfinite(value).all() for value in updates[0]["updated_weights"].values())
    assert all(torch.isfinite(value).all() for value in updates[1]["updated_weights"].values())
    server.strategy.receive_client_updates(updates)
    svd_calls = []

    original = RuntimeError("sentinel from original SVD")

    def raw_svd(value, **kwargs):
        svd_calls.append(value.detach().clone())
        assert kwargs == {"full_matrices": False}
        # The old diagnostic guard stopped here before SVD. Both client B blocks
        # must now reach the backend, including the original nonfinite block.
        expected = torch.cat([update["updated_weights"]["layer.lora_B"] for update in updates], dim=1)
        torch.testing.assert_close(value, expected, rtol=0, atol=0, equal_nan=True)
        assert not torch.isfinite(value).all()
        raise original

    monkeypatch.setattr(torch.linalg, "svd", raw_svd)
    with pytest.raises(RuntimeError) as caught:
        server.strategy.aggregation()
    assert caught.value is original
    detector.close(status="failed")
    assert len(svd_calls) == 1
    assert server.node_var.aggregation_method._sample_weights == [0.5, 0.5]
    assert not server.strategy.history
    assert not detector.triggered
    assert not directory.exists()


def _full_runner(directory, enabled, inject_nan=False):
    runner, clients, server, _ = _two_clients(directory, False, inject_nan=inject_nan, initialize=False)
    objects = [server.node_var, server.node_var.aggregation_method]
    for client in clients:
        objects.extend((client.node_var, client.node_var.trainer.trainer_args))
    # A stale YAML flag must no longer bind a detector to any core object.
    for obj in objects:
        for name in ("anomaly_detector", "anomaly_context", "anomaly_global_weight"):
            if hasattr(obj, name):
                delattr(obj, name)
    server.node_var.config_dict = {"general": {"anomaly_detection": {"enabled": enabled}},
                                   "experiment_name": "runner fixture"}
    server.node_var.training_logger.path = str(directory)
    server.prepare = server.strategy.prepare
    server.broadcast = server.strategy.broadcast
    server.select_clients = lambda available: list(available)
    server.receive_client_updates = server.strategy.receive_client_updates
    server.aggregation = server.strategy.aggregation
    server.apply_weight = server.strategy.apply_weight
    evaluations = []

    def evaluate():
        with torch.no_grad():
            outputs = server.node_var.model_evaluator.model(torch.zeros(2, 5))
        assert torch.isfinite(outputs).all()
        server.eval_results = {"accuracy": float((outputs.argmax(1) == 0).float().mean())}

    server.evaluate = evaluate
    server.record_evaluation = lambda: evaluations.append(dict(server.eval_results))
    return runner, clients, server, objects, evaluations


def _assert_unbound(runner, objects):
    assert not hasattr(runner, "anomaly_detector")
    assert not hasattr(runner, "anomaly_output_dir")
    for obj in objects:
        assert not hasattr(obj, "anomaly_detector")
        assert not hasattr(obj, "anomaly_context")
        assert not hasattr(obj, "anomaly_global_weight")


def test_real_runner_ignores_old_switch_and_never_constructs_detector(tmp_path, monkeypatch):
    results = []
    for enabled in (False, True):
        directory = tmp_path / str(enabled)
        runner, clients, server, objects, evaluations = _full_runner(directory, enabled)
        with monkeypatch.context() as local_patch:
            local_patch.setattr(AnomalyDetector, "__init__", Mock(side_effect=AssertionError("automatic detector")))
            runner.run()
        _assert_unbound(runner, objects)
        assert len(server.strategy.history) == 2  # Preserve the runner's existing inclusive rounds0..1.
        assert len(evaluations) == 2
        assert all(len(client.node_var.optimizer_builder.created) == 2 for client in clients)
        assert all(optimizer.step_count == 4 for client in clients
                   for optimizer in client.node_var.optimizer_builder.created)
        assert not directory.exists()
        results.append({"state": copy.deepcopy(server.node_var.model_weight),
                        "rng": AnomalyDetector.capture_rng_state(), "evaluations": evaluations})
    _assert_equal(results[0], results[1])


def test_full_runner_passes_raw_failure_after_all_clients_without_diagnostics(tmp_path, monkeypatch):
    directory = tmp_path / "failed_logs"
    runner, clients, server, objects, evaluations = _full_runner(directory, True, inject_nan=True)
    original = RuntimeError("backend failure")
    calls = []

    def raw_svd(value, **kwargs):
        calls.append(value)
        assert not torch.isfinite(value).all()
        raise original

    monkeypatch.setattr(torch.linalg, "svd", raw_svd)
    monkeypatch.setattr(AnomalyDetector, "__init__", Mock(side_effect=AssertionError("automatic detector")))
    with pytest.raises(RuntimeError) as caught:
        runner.run()
    assert caught.value is original
    _assert_unbound(runner, objects)
    assert len(calls) == 1
    assert all(client.node_var.optimizer_builder.created[-1].step_count == 4 for client in clients)
    assert not evaluations
    assert not directory.exists()
