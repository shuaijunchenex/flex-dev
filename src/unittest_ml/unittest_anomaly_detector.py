"""Small CPU-only checks for opt-in diagnostics; no training or downloads."""

from __future__ import annotations

import copy
import json
import math
import random
from collections.abc import Mapping

import numpy as np
import pytest
import torch

from flex.ml_utils.anomaly_detector import AnomalyDetector


@pytest.fixture(autouse=True)
def restore_rng():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)


def context(client="client.1", batch=0):
    return {"client_id": client, "round": 3, "epoch": 0, "batch": batch}


def strict_json(text):
    def reject_constant(value):
        raise AssertionError(f"Nonstandard JSON constant: {value}")
    return json.loads(text, parse_constant=reject_constant)


def events(directory):
    return [strict_json(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def snapshots(directory):
    return [torch.load(path, map_location="cpu", weights_only=False)
            for path in sorted(directory.glob("*.pt")) if path.name != "remembered.pt"]


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def test_disabled_does_not_scan_capture_or_create_files(tmp_path, monkeypatch):
    directory = tmp_path / "disabled"
    detector = AnomalyDetector(directory, enabled=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled diagnostics must not inspect or capture tensors")

    monkeypatch.setattr(torch, "isfinite", forbidden)
    assert detector.enabled is False
    assert detector.observe("loss", {"loss": torch.tensor(float("nan"))}, context(), snapshot=forbidden)
    detector.remember("state", {"tensor": object()})
    detector.record_exception("forward", ValueError("expected"), context(), snapshot=forbidden)
    detector.close()
    detector.close()
    assert detector.triggered is False
    assert not directory.exists()


def test_healthy_observation_preserves_autograd_tensor_gradient_and_rng(tmp_path):
    directory = tmp_path / "healthy"
    detector = AnomalyDetector(directory, metadata={"experiment": "small CPU check"})
    parameter = torch.tensor([1.0, -2.0], requires_grad=True)
    parameter.grad = torch.tensor([0.5, 0.25])
    intermediate = parameter.square()
    before = parameter.detach().clone(), parameter.grad.clone(), parameter._version
    rng_before = AnomalyDetector.capture_rng_state()
    snapshot_calls = []

    assert detector.observe("forward", {"output": intermediate, "unused_grad": None}, context(),
                            snapshot=lambda: snapshot_calls.append(True))
    assert detector.observe("gradient", {"weight": parameter, "grad": parameter.grad}, context())
    rng_after = AnomalyDetector.capture_rng_state()
    assert_state_equal(rng_before, rng_after)
    assert torch.equal(parameter.detach(), before[0])
    assert torch.equal(parameter.grad, before[1])
    assert parameter._version == before[2]
    assert snapshot_calls == []
    assert detector.triggered is False
    detector.close()
    assert not directory.exists()
    intermediate.sum().backward()
    assert torch.equal(parameter.grad, before[1] + 2 * before[0])


def test_capture_rng_state_is_detached_and_does_not_advance_rng():
    random.seed(102)
    np.random.seed(103)
    torch.manual_seed(104)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    captured = AnomalyDetector.capture_rng_state()
    assert_state_equal(before, (random.getstate(), np.random.get_state(), torch.get_rng_state()))
    frozen = copy.deepcopy(captured)
    random.random()
    np.random.random()
    torch.rand(3)
    assert_state_equal(captured, frozen)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_each_nonfinite_kind_triggers_diagnostics(tmp_path, invalid):
    directory = tmp_path / "invalid"
    detector = AnomalyDetector(directory)
    assert detector.observe("loss", {"loss": torch.tensor(invalid)}, context()) is False
    assert detector.triggered is True
    detector.close(status="failed")
    assert events(directory)
    assert strict_json((directory / "summary.json").read_text(encoding="utf-8"))


def test_statistics_use_fp64_norm_and_all_invalid_values_are_strict_json(tmp_path):
    directory = tmp_path / "statistics"
    detector = AnomalyDetector(directory)
    huge = torch.tensor([1e20, -1e20], dtype=torch.float32)
    invalid = torch.tensor([float("nan"), float("inf"), -float("inf")])
    assert not detector.observe("gradient", {"huge": huge, "nested": {"bad": invalid},
                                             "empty": torch.empty(0), "unused_grad": None}, context())
    detector.close(status="failed")
    event = next(event for event in events(directory) if "huge" in event.get("tensors", {}))
    huge_stats = event["tensors"]["huge"]
    assert math.isfinite(huge_stats["norm"])
    assert huge_stats["norm"] == pytest.approx(torch.linalg.vector_norm(huge.double()).item())
    invalid_stats = event["tensors"]["nested.bad"]
    assert invalid_stats["finite_count"] == 0
    assert invalid_stats["nan_count"] == 1
    assert invalid_stats["posinf_count"] == 1
    assert invalid_stats["neginf_count"] == 1
    assert invalid_stats["norm"] is None
    assert invalid_stats["min"] is None
    assert invalid_stats["max"] is None
    assert event["tensors"]["empty"]["norm"] is None
    strict_json((directory / "summary.json").read_text(encoding="utf-8"))


def test_snapshot_is_lazy_and_first_capture_is_independent_for_each_owner(tmp_path):
    directory = tmp_path / "owners"
    detector = AnomalyDetector(directory)
    calls = []

    def snapshot(owner):
        calls.append(owner)
        return {"owner_value": owner}

    for owner in ("client.1", "client.2"):
        assert detector.observe("before_forward", {"loss": torch.tensor(1.0)}, context(owner),
                                snapshot=lambda owner=owner: snapshot(owner))
    assert calls == []
    assert not directory.exists()
    for owner in ("client.1", "client.2"):
        assert not detector.observe("loss", {"loss": torch.tensor(float("nan"))}, context(owner, 1),
                                    snapshot=lambda owner=owner: snapshot(owner))
        assert not detector.observe("gradient", {"grad": torch.tensor(float("inf"))}, context(owner, 1),
                                    snapshot=lambda: pytest.fail("An owner must not be captured twice"))
    detector.close(status="failed")
    assert calls == ["client.1", "client.2"]
    assert sorted(item["snapshot"]["owner_value"] for item in snapshots(directory)) == calls


def test_remember_and_snapshot_are_independent_detached_cpu_copies(tmp_path, monkeypatch):
    directory = tmp_path / "copies"
    detector = AnomalyDetector(directory)
    remembered = torch.tensor([2.0, 3.0], requires_grad=True)
    detector.remember("pre_step", {"weights": remembered, "nested": [remembered.square()]})
    with torch.no_grad():
        remembered.add_(10)
    current = torch.tensor([5.0, 6.0], requires_grad=True)
    source = {"weights": current, "nested": [current.square()]}
    captured = []
    original_save = torch.save

    def capture_save(value, *args, **kwargs):
        if isinstance(value, dict) and "snapshot" in value:
            captured.append(value)
        return original_save(value, *args, **kwargs)

    monkeypatch.setattr(torch, "save", capture_save)
    assert not detector.observe("loss", {"loss": torch.tensor(float("nan"))}, context(),
                                snapshot=lambda: source)
    with torch.no_grad():
        current.fill_(99)
        source["nested"][0].fill_(99)
    detector.close(status="failed")
    assert len(captured) == 1
    for record in (captured[0], snapshots(directory)[0]):
        weight = record["snapshot"]["weights"]
        assert weight.device.type == "cpu"
        assert not weight.requires_grad
        assert weight.grad_fn is None
        assert torch.equal(weight, torch.tensor([5.0, 6.0]))
        assert torch.equal(record["snapshot"]["nested"][0], torch.tensor([25.0, 36.0]))
        assert torch.equal(record["remembered"]["pre_step"]["weights"], torch.tensor([2.0, 3.0]))
        assert torch.equal(record["remembered"]["pre_step"]["nested"][0], torch.tensor([4.0, 9.0]))


def test_record_exception_keeps_original_message_and_traceback(tmp_path):
    directory = tmp_path / "exception"
    detector = AnomalyDetector(directory)
    original = ValueError("original training failure sentinel")

    def train_operation():
        raise original

    with pytest.raises(ValueError) as raised:
        try:
            train_operation()
        except ValueError as error:
            detector.record_exception("forward", error, context())
            raise
    assert raised.value is original
    detector.close(status="failed")
    saved = json.dumps(events(directory), ensure_ascii=False)
    assert "original training failure sentinel" in saved
    assert "Traceback" in saved
    assert "train_operation" in saved


def test_io_error_does_not_replace_original_training_exception(tmp_path):
    blocked = tmp_path / "file_instead_of_directory"
    blocked.write_text("fixture", encoding="utf-8")
    detector = AnomalyDetector(blocked)
    original = RuntimeError("training failure must remain visible")
    with pytest.raises(RuntimeError) as raised:
        try:
            raise original
        except RuntimeError as error:
            detector.record_exception("training", error, context())
            detector.close(status="failed")
            raise
    assert raised.value is original
    assert detector.observe("gradient", {"grad": torch.tensor(float("nan"))}, context()) is True
    detector.close(status="failed")


def test_detectors_and_repeated_close_are_independent(tmp_path):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first = AnomalyDetector(first_dir)
    second = AnomalyDetector(second_dir)
    assert first.observe("loss", {"loss": torch.tensor(float("nan"))}, context()) is False
    assert second.observe("loss", {"loss": torch.tensor(1.0)}, context()) is True
    first.close(status="failed")
    first_events = (first_dir / "events.jsonl").read_bytes()
    first_summary = (first_dir / "summary.json").read_bytes()
    first.close(status="completed")
    assert (first_dir / "events.jsonl").read_bytes() == first_events
    assert (first_dir / "summary.json").read_bytes() == first_summary
    assert second.triggered is False
    second.close()
    assert not second_dir.exists()


@pytest.mark.parametrize("continues_nonfinite", [False, True])
def test_history_and_followup_are_bounded_by_batches(tmp_path, continues_nonfinite):
    directory = tmp_path / "bounded"
    detector = AnomalyDetector(directory)
    for batch in range(1, 21):
        detector.observe("loss", {"loss": torch.tensor(1.0)}, context(batch=batch))
    detector.observe("loss", {"loss": torch.tensor(float("nan"))}, context(batch=20))
    for batch in range(21, 61):
        value = float("nan") if continues_nonfinite else 1.0
        detector.observe("loss", {"loss": torch.tensor(value)}, context(batch=batch))
    detector.observe("epoch", {}, {**context(), "batch": None}, summary={"batches": 60})
    detector.close(status="failed")
    report = strict_json((directory / "summary.json").read_text(encoding="utf-8"))
    owner = report["owners"]["client_client.1"]
    assert [item["context"]["batch"] for item in owner["before_batches"]] == list(range(5, 21))
    assert [item["context"]["batch"] for item in owner["after_batches"]] == list(range(21, 29))
    assert owner["repeat_count"] == (40 if continues_nonfinite else 0)
    detail = [event for event in events(directory) if event["type"] == "observation"]
    assert len(detail) == 8
    assert any(event["type"] == "summary" and event["summary"]["batches"] == 60
               for event in events(directory))


def test_snapshot_budget_is_explicit_and_backbone_is_shared_once(tmp_path):
    directory = tmp_path / "shared"
    detector = AnomalyDetector(directory)
    detector.remember("backbone", {"weight": torch.ones(2)})
    for owner in ("a/b", "a_b"):
        detector.observe("loss", {"loss": torch.tensor(float("nan"))}, context(owner))
    detector.close(status="failed")
    saved = snapshots(directory)
    assert len(saved) == 2
    assert all("backbone" not in item["remembered"] for item in saved)
    assert all(item["shared_remembered_file"] == "remembered.pt" for item in saved)
    shared = torch.load(directory / "remembered.pt", map_location="cpu", weights_only=False)
    assert torch.equal(shared["backbone"]["weight"], torch.ones(2))

    limited = AnomalyDetector(tmp_path / "limited")
    limited.MAX_SNAPSHOT_BYTES = 1
    with pytest.warns(RuntimeWarning, match="budget"):
        limited.observe("loss", {"loss": torch.tensor(float("nan"))}, context(),
                        snapshot=lambda: {"weight": torch.ones(2)})
    limited.close(status="failed")
    assert not list(limited.output_dir.glob("*.pt"))
    report = strict_json((limited.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert report["snapshot_bytes"] == 0
    assert report["io_errors"]
