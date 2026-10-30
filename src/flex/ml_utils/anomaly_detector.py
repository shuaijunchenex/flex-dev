"""Run-local numerical diagnostics; observations never change training tensors."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import copy
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import random
import re
import traceback
import warnings

import numpy as np
import torch


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return "NaN" if math.isnan(value) else ("+Inf" if value > 0 else "-Inf")
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _named_tensors(value, prefix="tensor"):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, (np.ndarray, np.generic, float, int)) and not isinstance(value, bool):
        yield prefix, torch.as_tensor(value)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key) if prefix == "tensor" else f"{prefix}.{key}"
            yield from _named_tensors(item, name)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _named_tensors(item, f"{prefix}.{index}")


def _tensor_stats(value):
    value = value.detach()
    # LoRA tensors are real. Counting components also makes complex diagnostics safe.
    measured = torch.view_as_real(value) if value.is_complex() else value
    finite = torch.isfinite(measured)
    good = measured[finite]
    finite_count = int(finite.sum().item())
    norm_values = good.cpu() if good.device.type == "mps" else good
    norm = float(norm_values.to(torch.float64).norm().item()) if finite_count else None
    return {
        "dtype": str(value.dtype), "device": str(value.device),
        "shape": list(value.shape), "numel": value.numel(),
        "nbytes": value.numel() * value.element_size(),
        "finite_count": finite_count,
        "nan_count": int(torch.isnan(measured).sum().item()),
        "posinf_count": int(torch.isposinf(measured).sum().item()),
        "neginf_count": int(torch.isneginf(measured).sum().item()),
        "finite_absmax": float(good.abs().max().item()) if finite_count else None,
        "finite_l2": norm, "norm": norm,
        "min": float(good.min().item()) if finite_count else None,
        "max": float(good.max().item()) if finite_count else None,
    }


class AnomalyDetector:
    """Bounded, opt-in diagnostics shared by one run's clients and aggregator.

    ``observe`` returns False for NaN/Inf but never raises for a numerical
    anomaly. Callers retain responsibility for algorithm boundaries such as
    refusing to pass nonfinite matrices to SVD. None gradients are ignored.
    """

    HISTORY_BATCHES = 16
    FOLLOWUP_BATCHES = 8
    MAX_OWNERS = 64
    MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024

    def __init__(self, output_dir, enabled=True, metadata=None):
        self.output_dir = Path(output_dir)
        self.enabled = bool(enabled)
        self.triggered = False
        self.metadata = _json_safe(metadata) if self.enabled and metadata is not None else {}
        self._owners = OrderedDict()
        self._remembered = {}
        self._first_event = None
        self._closed = False
        self._status = "running"
        self._snapshot_bytes = 0
        self._shared_file = None
        self.io_errors = []

    @staticmethod
    def capture_rng_state():
        numpy_state = np.random.get_state()
        return {
            "python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized() else None,
        }

    def remember(self, name, value):
        if not self.enabled or self._closed:
            return
        try:
            self._remembered[str(name)] = _cpu_copy(value)
        except Exception as error:
            self._warn(f"remember({name!r}) failed: {error}")

    def _warn(self, message):
        if message in self.io_errors:
            return
        if len(self.io_errors) < 32:
            self.io_errors.append(message)
        try:
            warnings.warn(f"AnomalyDetector: {message}", RuntimeWarning, stacklevel=3)
        except Warning:
            # Warnings configured as errors must not replace a training error.
            pass

    @staticmethod
    def _owner(context):
        client = context.get("client_id")
        if client is not None:
            return f"client_{client}"
        return str(context.get("owner") or "aggregation")

    @staticmethod
    def _batch_key(context):
        def get(*keys):
            return next((context[key] for key in keys if key in context), None)
        batch = get("batch", "batch_idx", "batch_index")
        if batch is None:
            return None
        return json.dumps([get("round", "round_idx", "round_index"),
                           get("epoch", "epoch_idx", "local_epoch"), batch], sort_keys=True)

    def _state(self, owner):
        if owner not in self._owners:
            if len(self._owners) >= self.MAX_OWNERS:
                self._warn(f"Owner limit ({self.MAX_OWNERS}) reached; additional owners are not retained")
                return None
            self._owners[owner] = {
                "before": OrderedDict(), "after": OrderedDict(), "latest_context": None,
                "triggered": False, "trigger_batch": None, "seen_stages": set(),
                "repeat_count": 0, "snapshot_attempted": False, "snapshot_file": None,
            }
        return self._owners[owner]

    @staticmethod
    def _cache_batch(cache, key, event, limit):
        if key not in cache:
            cache[key] = {"context": event["context"], "stages": {}}
        cache[key]["stages"][event["stage"]] = event
        while len(cache) > limit:
            cache.popitem(last=False)

    def _event(self, stage, context, **fields):
        return {"time": datetime.now(timezone.utc).isoformat(), "stage": str(stage),
                "owner": self._owner(context), "context": context, **fields}

    def _write_event(self, event):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(_json_safe(event), allow_nan=False) + "\n")
        except Exception as error:
            self._warn(f"Writing events.jsonl failed: {error}")

    def _save_tensor_file(self, name, value):
        try:
            buffer = io.BytesIO()
            torch.save(value, buffer)
            size = buffer.tell()
            if self._snapshot_bytes + size > self.MAX_SNAPSHOT_BYTES:
                self._warn(f"Snapshot budget ({self.MAX_SNAPSHOT_BYTES} bytes) exceeded; {name} ({size} bytes) not saved")
                return None
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / name).write_bytes(buffer.getvalue())
            self._snapshot_bytes += size
            return name
        except Exception as error:
            self._warn(f"Saving {name} failed: {error}")
            return None

    def _snapshot(self, owner, state, event, snapshot):
        if state["snapshot_attempted"]:
            return
        state["snapshot_attempted"] = True
        try:
            value = snapshot() if callable(snapshot) else snapshot
            payload = {
                "snapshot": _cpu_copy(value),
                "remembered": _cpu_copy({key: item for key, item in self._remembered.items() if key != "backbone"}),
                "shared_remembered_file": self._shared_file,
                "context": event["context"], "stage": event["stage"],
            }
            safe_owner = re.sub(r"[^A-Za-z0-9_.-]", "_", owner)[:100] or "owner"
            # A numeric prefix prevents distinct unsafe IDs from sharing a file.
            filename = f"{safe_owner}.pt"
            if any(item["snapshot_file"] == filename for item in self._owners.values()):
                filename = f"{safe_owner}_{list(self._owners).index(owner)}.pt"
            state["snapshot_file"] = self._save_tensor_file(filename, payload)
        except Exception as error:
            self._warn(f"Capturing {owner} snapshot failed: {error}")

    def _anomaly(self, event, state, snapshot):
        first = not self.triggered
        if first:
            self.triggered = True
            self._first_event = copy.deepcopy(event)
            self._shared_file = self._save_tensor_file("remembered.pt", self._remembered)
        if not state["triggered"]:
            state["triggered"] = True
            state["trigger_batch"] = self._batch_key(event["context"])
            event["history"] = {name: list(item["before"].values()) for name, item in self._owners.items()}
            event["latest_context"] = {name: item["latest_context"] for name, item in self._owners.items()}
        stage_key = (event["type"], event["stage"])
        if stage_key not in state["seen_stages"]:
            state["seen_stages"].add(stage_key)
            if first and event["type"] != "exception":
                event["type"] = "trigger"
            self._snapshot(event["owner"], state, event, snapshot)
            self._write_event(event)
            self._write_summary()
            return True
        else:
            state["repeat_count"] += 1
            return False

    def _followup(self, state, key, event, write=True):
        if key is None or key == state["trigger_batch"]:
            return
        if key in state["after"] or len(state["after"]) < self.FOLLOWUP_BATCHES:
            repeated = key in state["after"] and event["stage"] in state["after"][key]["stages"]
            self._cache_batch(state["after"], key, event, self.FOLLOWUP_BATCHES)
            if write and not repeated:
                detail = dict(event, type="observation")
                self._write_event(detail)

    def observe(self, stage, tensors, context, snapshot=None, summary=None):
        if not self.enabled or self._closed:
            return True
        context = _json_safe(context or {})
        try:
            named = list(_named_tensors(tensors))
            finite = all(torch.isfinite(value.detach()).all().item() for _, value in named)
        except Exception as error:
            self.record_exception(f"{stage}.finite_check", error, context, snapshot=snapshot)
            return False
        stats = {}
        for name, value in named:
            try:
                stats[name] = _tensor_stats(value)
            except Exception as error:
                self._warn(f"Statistics for {stage}/{name} failed: {error}")
                stats[name] = {"statistics_error": str(error), "shape": list(value.shape),
                               "dtype": str(value.dtype), "device": str(value.device)}
        event = self._event(stage, context, type="observation" if finite else "anomaly",
                            finite=finite, tensors=stats, summary=_json_safe(summary))
        state = self._state(event["owner"])
        if state is None:
            return finite
        key = self._batch_key(context)
        if not finite:
            written = self._anomaly(event, state, snapshot)
            self._followup(state, key, event, write=not written)
        elif not state["triggered"]:
            if key is None:
                state["latest_context"] = event
            else:
                self._cache_batch(state["before"], key, event, self.HISTORY_BATCHES)
        else:
            self._followup(state, key, event)
        if key is None:
            state["latest_context"] = event
            if self.triggered and finite and summary is not None:
                self._write_event(dict(event, type="summary"))
        return finite

    def record_exception(self, stage, error, context, snapshot=None):
        if not self.enabled or self._closed:
            return
        context = _json_safe(context or {})
        event = self._event(stage, context, type="exception", error_type=type(error).__name__,
                            message=str(error), traceback="".join(traceback.format_exception(type(error), error, error.__traceback__)))
        state = self._state(event["owner"])
        if state is not None:
            self._anomaly(event, state, snapshot)

    def _write_summary(self):
        report = {
            "metadata": self.metadata, "status": self._status, "triggered": self.triggered,
            "first_event": self._first_event, "shared_remembered_file": self._shared_file,
            "snapshot_bytes": self._snapshot_bytes, "io_errors": self.io_errors,
            "owners": {name: {
                "before_batches": list(state["before"].values()),
                "after_batches": list(state["after"].values()),
                "latest_context": state["latest_context"], "repeat_count": state["repeat_count"],
                "snapshot_file": state["snapshot_file"],
            } for name, state in self._owners.items()},
        }
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "summary.json").write_text(
                json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        except Exception as error:
            self._warn(f"Writing summary.json failed: {error}")

    def close(self, status="completed"):
        if self._closed:
            return
        self._status = str(status)
        if self.enabled and self.triggered:
            self._write_summary()
        self._closed = True
        self._owners.clear()
        self._remembered.clear()
        self._first_event = None
        self.metadata = {}
