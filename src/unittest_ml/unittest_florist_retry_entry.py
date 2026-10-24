"""Check the failed-run retry entry without loading data or starting jobs."""

from __future__ import annotations

from collections import Counter
import copy
import importlib.util
from pathlib import Path, PureWindowsPath
import re
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


SRC = Path(__file__).resolve().parents[1]
LOCAL_ENTRY = SRC / "test" / "rbla_plus_experiment_local.py"
LORA_ENTRY = SRC / "entries" / "lora.py"
BATCH_SUMMARY = SRC / "test" / "batch_summary-20261017_202318-2c427f3e85583e73.log"


def _load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def local_entry(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    return _load_file(LOCAL_ENTRY, "florist_retry_entry_under_test")


def _historical_paths():
    result = {"successes": set(), "failures": set()}
    section = None
    for line in BATCH_SUMMARY.read_text(encoding="utf-8").splitlines():
        if line.strip() in ("successes:", "failures:"):
            section = line.strip().rstrip(":")
        match = re.match(r"^    - (.+\.yaml) \(", line)
        if match:
            parts = PureWindowsPath(match.group(1)).parts
            relative = "/".join(parts[parts.index("experiment") + 1:])
            result[section].add(relative)
    return result


def _fake_entries(monkeypatch, **children):
    package = ModuleType("entries")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "entries", package)
    for name, value in children.items():
        module = ModuleType(f"entries.{name}")
        for attribute, function in value.items():
            setattr(module, attribute, function)
        setattr(package, name, module)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return package


def test_retry_configs_match_all_19_failures_and_exclude_13_successes(local_entry):
    historical = _historical_paths()
    configs = local_entry.collect_configs()
    relative = {Path(config).relative_to(local_entry.EXPERIMENT_ROOT).as_posix() for config in configs}

    assert len(historical["failures"]) == 19
    assert len(historical["successes"]) == 13
    assert len(configs) == len(relative) == 19
    assert relative == historical["failures"]
    assert relative.isdisjoint(historical["successes"])
    assert all(Path(config).is_file() for config in configs)
    assert Counter(Path(config).relative_to(local_entry.EXPERIMENT_ROOT).parts[0]
                   for config in configs) == {
        "rank_match": 7, "rank_medium_confusion": 4, "rank_mismatch": 8,
    }


def test_dry_run_imports_no_training_entry_data_or_models(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"entries", "flex", "datasets", "torch", "torchvision", "transformers"}:
            raise AssertionError(f"Dry run tried to import training dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    entry = _load_file(LOCAL_ENTRY, "florist_retry_dry_run_under_test")
    entry.main(["--dry-run"])


def test_main_dispatches_original_lora_entry_without_starting_subprocesses(local_entry, monkeypatch):
    dispatch = Mock()
    _fake_entries(monkeypatch, run_all={"run_all": dispatch})
    configs = local_entry.collect_configs()

    local_entry.main([])

    dispatch.assert_called_once_with(configs, entry_module="entries.lora")
    assert not hasattr(local_entry, "run_experiment")
    assert not hasattr(local_entry, "ANOMALY_DATASETS")


@pytest.mark.parametrize("initial_enabled", [False, True])
def test_lora_preserves_loaded_configuration_and_execution_order(monkeypatch, initial_enabled):
    calls = []
    original = {
        "server_yaml": {
            "general": {"device": "cuda", "training_rounds": 100,
                        "anomaly_detection": {"enabled": initial_enabled, "output_dir": "existing-output"}},
            "aggregation": {"method": "florist", "threshold": 0.9},
        },
        "client_yaml": {"training": {"epochs": 5}, "optimizer": {"lr": 0.01}},
    }

    class FakeLoRAAppEntry:
        def __init__(self):
            self.objects = copy.deepcopy(original)

        def load_app_config(self, config):
            calls.append(("load", config))

        def get_app_object(self, name):
            calls.append(("get", name))
            return self.objects[name]

        def run(self, device):
            calls.append(("run", device))

    package = _fake_entries(monkeypatch)
    app = ModuleType("entries.app")
    app.__path__ = []
    lora_app = ModuleType("entries.app.lora_entry")
    lora_app.LoRAAppEntry = FakeLoRAAppEntry
    app.lora_entry = lora_app
    package.app = app
    monkeypatch.setitem(sys.modules, "entries.app", app)
    monkeypatch.setitem(sys.modules, "entries.app.lora_entry", lora_app)

    utils = ModuleType("flex.ml_utils")
    utils.__path__ = []
    utils.console = SimpleNamespace()
    model_utils = ModuleType("flex.ml_utils.model_utils")
    training_utils = ModuleType("flex.ml_utils.training_utils")

    def get_device():
        calls.append(("device",))
        return "test-device"

    model_utils.ModelUtils = SimpleNamespace(accelerator_device=get_device)
    training_utils.TrainingUtils = SimpleNamespace(
        set_seed_all=lambda seed: calls.append(("seed", seed)))
    monkeypatch.setitem(sys.modules, "flex.ml_utils", utils)
    monkeypatch.setitem(sys.modules, "flex.ml_utils.model_utils", model_utils)
    monkeypatch.setitem(sys.modules, "flex.ml_utils.training_utils", training_utils)
    monkeypatch.setattr(sys, "path", list(sys.path))
    # The real entry changes CWD at import; this isolated import must not do so.
    import os
    monkeypatch.setattr(os, "chdir", lambda _path: None)

    entry = _load_file(LORA_ENTRY, "florist_retry_lora_under_test")
    entry.main("fixture-config.yaml")

    assert entry.g_app.objects == original
    assert calls == [
        ("seed", 42), ("load", "fixture-config.yaml"), ("device",), ("run", "test-device"),
    ]
