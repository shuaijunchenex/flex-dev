"""Isolated MNIST experiment harness for RBLA compact broadcasting.

This entry intentionally leaves the normal runner and the SP aggregation path
unchanged.  It only supplies per-round calibration activations to the existing
RBLA aggregator and records read-only prefix diagnostics without materializing
``B @ A`` as a dense weight matrix.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from entries.app.lora_entry import LoRAAppEntry
from flex.fed_node import FedNodeVars
from flex.fed_runner import FedRunner
from flex.fl_algorithms.aggregation.methods._fed_aggregator_rbla import FedAggregator_RBLA
from flex.fl_algorithms.noniid.noniid_data_generator import NoniidDataGenerator
from flex.ml_data_loader.dataset_loader_args import DatasetLoaderArgs
from flex.ml_data_loader.dataset_loader_factory import DatasetLoaderFactory
from flex.ml_models import NNModelFactory
from flex.ml_utils.model_utils import ModelUtils
from flex.ml_utils.training_utils import TrainingUtils


PREFIX_RANKS = (1, 2, 4, 8)
CALIBRATION_SEED = 20260718
EPS = 1e-12


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _balanced_indices(dataset: Any, per_class: int, seed: int) -> list[int]:
    targets = torch.as_tensor(dataset.targets)
    generator = torch.Generator().manual_seed(seed)
    selected: list[int] = []
    for label in range(10):
        candidates = torch.nonzero(targets == label, as_tuple=False).flatten()
        if candidates.numel() < per_class:
            raise ValueError(
                f"class {label} has {candidates.numel()} samples, expected at least {per_class}"
            )
        order = torch.randperm(candidates.numel(), generator=generator)[:per_class]
        selected.extend(int(index) for index in candidates[order])
    return sorted(selected)


def _clone_state_dict(state: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((key, value.detach().clone()) for key, value in state.items())


@torch.no_grad()
def _collect_activations(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Collect temporary per-LoRA-layer inputs keyed by exact lora_A key."""

    buffers: dict[str, list[torch.Tensor]] = {}
    handles = []
    for module_name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        key = f"{module_name}.lora_A" if module_name else "lora_A"
        buffers[key] = []

        def capture(_module: nn.Module, inputs: tuple[Any, ...], *, state_key: str = key) -> None:
            activation = inputs[0]
            if not isinstance(activation, torch.Tensor):
                raise TypeError(f"activation for '{state_key}' is not a tensor")
            buffers[state_key].append(
                activation.detach().reshape(-1, activation.shape[-1]).to("cpu")
            )

        handles.append(module.register_forward_pre_hook(capture))

    if not handles:
        raise RuntimeError("no LoRA linear layers were found for activation collection")

    original_device = next(model.parameters()).device
    model.eval().to(device)
    try:
        for inputs, _labels in loader:
            model(inputs.to(device))
    finally:
        for handle in handles:
            handle.remove()
        model.to(original_device)

    result: dict[str, torch.Tensor] = {}
    for key, chunks in buffers.items():
        if not chunks:
            raise RuntimeError(f"forward hook for '{key}' collected no activations")
        result[key] = torch.cat(chunks, dim=0)
    return result


def _pair_keys(state: dict[str, torch.Tensor]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for key in state:
        parts = key.split(".")
        positions = [i for i, part in enumerate(parts) if part == "lora_A"]
        if len(positions) != 1:
            continue
        other = list(parts)
        other[positions[0]] = "lora_B"
        b_key = ".".join(other)
        if b_key not in state:
            raise KeyError(f"'{key}' is missing exact counterpart '{b_key}'")
        pairs.append((key, b_key))
    return pairs


@torch.no_grad()
def _component_response(
    activation: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Return component output energies without creating the dense update."""

    a = lora_A.detach().to(dtype=torch.float64, device="cpu")
    b = lora_B.detach().to(dtype=torch.float64, device="cpu")
    column_norms = b.square().sum(dim=0)
    result = torch.zeros(a.shape[0], dtype=torch.float64)
    flat = activation.reshape(-1, activation.shape[-1])
    for start in range(0, flat.shape[0], chunk_size):
        projected = flat[start : start + chunk_size].to(torch.float64) @ a.T
        result.add_(projected.square().sum(dim=0) * column_norms)
    return result


@torch.no_grad()
def _general_prefix_energies(
    activation: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    ranks: tuple[int, ...],
    chunk_size: int = 4096,
) -> tuple[float, dict[int, float]]:
    """Exact low-rank output energies, valid for the non-canonical baseline."""

    a = lora_A.detach().to(dtype=torch.float64, device="cpu")
    b = lora_B.detach().to(dtype=torch.float64, device="cpu")
    flat = activation.reshape(-1, activation.shape[-1])
    total = 0.0
    omitted = {rank: 0.0 for rank in ranks}
    for start in range(0, flat.shape[0], chunk_size):
        z = flat[start : start + chunk_size].to(torch.float64) @ a.T
        full_output = z @ b.T
        total += float(full_output.square().sum().item())
        for rank in ranks:
            cutoff = min(rank, a.shape[0])
            residual = z[:, cutoff:] @ b[:, cutoff:].T
            omitted[rank] += float(residual.square().sum().item())
    return total, omitted


def _weight_norm_squared(lora_A: torch.Tensor, lora_B: torch.Tensor) -> float:
    """Compute ||B A||_F^2 from rank-sized Gram matrices."""

    a = lora_A.detach().to(dtype=torch.float64, device="cpu")
    b = lora_B.detach().to(dtype=torch.float64, device="cpu")
    return float(((b.T @ b) * (a @ a.T)).sum().item())


@torch.no_grad()
def _factor_diagnostics(
    state: dict[str, torch.Tensor],
    calibration: dict[str, torch.Tensor],
    heldout: dict[str, torch.Tensor],
    aggregator: FedAggregator_RBLA,
    method: str,
    round_index: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    canonical = bool(aggregator.canonicalization_applied_last_round)
    aggregate: dict[str, float] = {}
    details: list[dict[str, Any]] = []
    cal_totals = {rank: [0.0, 0.0] for rank in PREFIX_RANKS}
    held_totals = {rank: [0.0, 0.0] for rank in PREFIX_RANKS}
    weight_totals = {rank: [0.0, 0.0] for rank in PREFIX_RANKS}
    layer_errors = {rank: {"cal": [], "held": [], "weight": [], "overlap": []} for rank in PREFIX_RANKS}

    for a_key, b_key in _pair_keys(state):
        a, b = state[a_key], state[b_key]
        rank_count = int(a.shape[0])
        diag = copy.deepcopy(aggregator.canonicalization_diagnostics.get(a_key, {}))
        order = [int(item) for item in diag.get("ordering_indices", range(rank_count))]
        if len(order) != rank_count:
            order = list(range(rank_count))

        if canonical:
            singular = aggregator.canonicalization_singular_values[a_key].to(torch.float64)
            cal_component = _component_response(calibration[a_key], a, b)
            held_component = _component_response(heldout[a_key], a, b)
            weight_component = singular.square()
            cal_total = float(cal_component.sum().item())
            held_total = float(held_component.sum().item())
            weight_total = float(weight_component.sum().item())
            cal_omitted = {}
            held_omitted = {}
            weight_omitted = {}
            for rank in PREFIX_RANKS:
                cutoff = min(rank, rank_count)
                cal_omitted[rank] = float(cal_component[cutoff:].sum().item())
                held_omitted[rank] = float(held_component[cutoff:].sum().item())
                weight_omitted[rank] = float(weight_component[cutoff:].sum().item())
        else:
            singular = torch.empty(0, dtype=torch.float64)
            cal_total, cal_omitted = _general_prefix_energies(calibration[a_key], a, b, PREFIX_RANKS)
            held_total, held_omitted = _general_prefix_energies(heldout[a_key], a, b, PREFIX_RANKS)
            weight_total = _weight_norm_squared(a, b)
            weight_omitted = {}
            for rank in PREFIX_RANKS:
                cutoff = min(rank, rank_count)
                weight_omitted[rank] = _weight_norm_squared(a[cutoff:, :], b[:, cutoff:])

        per_rank: dict[str, Any] = {}
        for rank in PREFIX_RANKS:
            cal_error = cal_omitted[rank] / max(cal_total, EPS)
            held_error = held_omitted[rank] / max(held_total, EPS)
            weight_error = weight_omitted[rank] / max(weight_total, EPS)
            overlap = (
                len(set(order[: min(rank, rank_count)]) & set(range(min(rank, rank_count))))
                / max(min(rank, rank_count), 1)
                if canonical
                else math.nan
            )
            per_rank[str(rank)] = {
                "functional_error_calibration": cal_error,
                "functional_error_heldout": held_error,
                "weight_error": weight_error,
                "overlap": overlap,
                "reorder_rate": 1.0 - overlap if math.isfinite(overlap) else math.nan,
            }
            cal_totals[rank][0] += cal_omitted[rank]
            cal_totals[rank][1] += cal_total
            held_totals[rank][0] += held_omitted[rank]
            held_totals[rank][1] += held_total
            weight_totals[rank][0] += weight_omitted[rank]
            weight_totals[rank][1] += weight_total
            layer_errors[rank]["cal"].append(cal_error)
            layer_errors[rank]["held"].append(held_error)
            layer_errors[rank]["weight"].append(weight_error)
            if math.isfinite(overlap):
                layer_errors[rank]["overlap"].append(overlap)

        details.append(
            {
                "round": round_index + 1,
                "method": method,
                "layer": a_key,
                "singular_values": singular.tolist(),
                "ordering_indices": order if canonical else [],
                "diagnostics": diag,
                "per_rank": per_rank,
                "calibration_total_response_energy": cal_total,
                "heldout_total_response_energy": held_total,
                "weight_total_energy": weight_total,
            }
        )

    for rank in PREFIX_RANKS:
        values = layer_errors[rank]
        aggregate[f"mean_functional_error_calibration_r{rank}"] = sum(values["cal"]) / len(values["cal"])
        aggregate[f"weighted_functional_error_calibration_r{rank}"] = cal_totals[rank][0] / max(cal_totals[rank][1], EPS)
        aggregate[f"mean_functional_error_heldout_r{rank}"] = sum(values["held"]) / len(values["held"])
        aggregate[f"weighted_functional_error_heldout_r{rank}"] = held_totals[rank][0] / max(held_totals[rank][1], EPS)
        aggregate[f"mean_weight_error_r{rank}"] = sum(values["weight"]) / len(values["weight"])
        aggregate[f"weighted_weight_error_r{rank}"] = weight_totals[rank][0] / max(weight_totals[rank][1], EPS)
        overlaps = values["overlap"]
        aggregate[f"mean_overlap_r{rank}"] = sum(overlaps) / len(overlaps) if overlaps else math.nan
        aggregate[f"mean_reorder_rate_r{rank}"] = 1.0 - aggregate[f"mean_overlap_r{rank}"] if overlaps else math.nan
    return aggregate, details


@torch.no_grad()
def _evaluate_model(model: nn.Module, loader: Iterable, device: torch.device) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval().to(device)
    correct = total = 0
    loss_sum = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device).long()
        output = model(inputs)
        loss_sum += float(criterion(output, labels).item())
        correct += int((output.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())
    model.to("cpu")
    return correct / max(total, 1), loss_sum / max(total, 1)


@torch.no_grad()
def _prefix_accuracies(
    global_state: dict[str, torch.Tensor],
    model_config: dict[str, Any],
    test_loader: Iterable,
    device: torch.device,
) -> dict[str, float]:
    results: dict[str, float] = {}
    for rank in PREFIX_RANKS:
        config = copy.deepcopy(model_config)
        config["rank_ratio"] = rank / 100.0
        model = NNModelFactory.create(NNModelFactory.create_args(config))
        local_state = FedAggregator_RBLA.broadcast_lora_state_dict(
            global_state, model.state_dict()
        )
        model.load_state_dict(local_state, strict=True)
        accuracy, loss = _evaluate_model(model, test_loader, device)
        results[f"prefix_test_accuracy_r{rank}"] = accuracy
        results[f"prefix_test_loss_r{rank}"] = loss
        del model
    return results


def _train_loss(client_updates: list[dict[str, Any]]) -> float:
    weighted = total = 0.0
    for update in client_updates:
        wrapper = update["train_record"]
        stats = wrapper.get("train_record", wrapper)
        volume = float(wrapper.get("data_sample_num", stats.get("data_sample_num", 1)))
        weighted += volume * float(stats["avg_loss"])
        total += volume
    return weighted / max(total, 1.0)


def _install_aggregation_timing(aggregator: FedAggregator_RBLA):
    from flex.ml_algorithms.lora import canonicalization as canonicalization_module

    original_do = aggregator._do_aggregation
    original_after = aggregator._after_aggregation
    original_ordering = canonicalization_module._apply_activation_aware_ordering

    def timed_do() -> None:
        start = time.perf_counter()
        original_do()
        aggregator._aasb_core_seconds = time.perf_counter() - start

    def timed_ordering(*args: Any, **kwargs: Any):
        start = time.perf_counter()
        result = original_ordering(*args, **kwargs)
        aggregator._aasb_score_seconds += time.perf_counter() - start
        return result

    def timed_after() -> None:
        pre_state = _clone_state_dict(aggregator._aggregated_weight)
        aggregator._aasb_score_seconds = 0.0
        before = time.perf_counter()
        original_after()
        aggregator._aasb_canonical_seconds = time.perf_counter() - before
        relative_errors: list[float] = []
        probes = getattr(aggregator, "_aasb_probe_activations", {})
        if aggregator.canonicalization_applied_last_round:
            for a_key, b_key in _pair_keys(aggregator._aggregated_weight):
                activation = probes.get(a_key)
                if activation is None:
                    continue
                x = activation.reshape(-1, activation.shape[-1]).to(torch.float64)
                old_a = pre_state[a_key].to(dtype=torch.float64, device="cpu")
                old_b = pre_state[b_key].to(dtype=torch.float64, device="cpu")
                new_a = aggregator._aggregated_weight[a_key].to(dtype=torch.float64, device="cpu")
                new_b = aggregator._aggregated_weight[b_key].to(dtype=torch.float64, device="cpu")
                old_output = (x @ old_a.T) @ old_b.T
                new_output = (x @ new_a.T) @ new_b.T
                error = torch.linalg.vector_norm(old_output - new_output)
                scale = torch.linalg.vector_norm(old_output).clamp_min(EPS)
                relative_errors.append(float((error / scale).item()))
        aggregator._aasb_full_rank_probe_relative_error = max(relative_errors, default=0.0)

    aggregator._do_aggregation = timed_do  # type: ignore[method-assign]
    aggregator._after_aggregation = timed_after  # type: ignore[method-assign]
    canonicalization_module._apply_activation_aware_ordering = timed_ordering

    def restore() -> None:
        canonicalization_module._apply_activation_aware_ordering = original_ordering

    return restore


def _prepare_experiment(
    app: LoRAAppEntry,
    device: torch.device,
    training_seed: int,
    output_dir: Path,
) -> tuple[FedRunner, Any, list[Any], DataLoader, DataLoader, list[int], list[int]]:
    runner_yaml = app.get_app_object("runner")
    client_yaml = app.get_app_object("client_yaml")
    server_yaml = app.get_app_object("server_yaml")
    output_dir.mkdir(parents=True, exist_ok=True)
    for config in (runner_yaml, client_yaml, server_yaml):
        config["training_logger"]["path"] = str(output_dir)
        config["training_logger"]["prefix"] = f"framework_seed{training_seed}_"

    TrainingUtils.apply_train_optimization(server_yaml)
    runner = FedRunner()
    runner.training_rounds = int(runner_yaml["general"]["training_rounds"])
    runner.with_yaml(runner_yaml)
    runner.create_nodes()
    runner.create_run_strategy()

    server_var = FedNodeVars(server_yaml)
    server_var.set_device(device)
    server_var.prepare()
    app._LoRAAppEntry__attach_event_handler(server_var)
    server_var.owner_nodes = runner.server_node
    runner.server_node.node_var = server_var
    runner.server_node.prepare_strategy()

    train_dataset = server_var.data_loader.data_set
    test_dataset = server_var.data_loader._test_dataset
    calibration_indices = _balanced_indices(train_dataset, 20, CALIBRATION_SEED)
    functional_indices = _balanced_indices(test_dataset, 100, CALIBRATION_SEED + 1)
    calibration_set = set(calibration_indices)
    remaining_indices = [index for index in range(len(train_dataset)) if index not in calibration_set]
    loader_config = server_yaml["data_loader"]
    remaining_loader = DataLoader(
        Subset(train_dataset, remaining_indices),
        batch_size=int(loader_config.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(loader_config.get("num_workers", 0)),
    )
    allocated = NoniidDataGenerator(remaining_loader).generate_noniid_data(
        distribution_config=server_yaml["data_distribution"]
    )
    if len(allocated) != len(runner.client_node_list):
        raise RuntimeError(f"data allocation produced {len(allocated)} clients, expected {len(runner.client_node_list)}")

    clients = []
    for index, node in enumerate(runner.client_node_list):
        client_loader_cfg = client_yaml.get("data_loader", {})
        custom_args = DatasetLoaderArgs(
            {
                "name": "custom",
                "root": ".dataset",
                "split": "",
                "batch_size": client_loader_cfg.get("batch_size", 64),
                "shuffle": client_loader_cfg.get("shuffle", True),
                "num_workers": client_loader_cfg.get("num_workers", 0),
                "is_download": False,
                "is_load_train_set": True,
                "is_load_test_set": False,
                "task_type": "cv",
                "generator_seed": training_seed + index + 1,
                "dataset": allocated[index],
            }
        )
        client_loader = DatasetLoaderFactory().create(custom_args)
        client_var = FedNodeVars(client_yaml, is_clone_dict=True)
        client_var.set_device(device)
        client_var.config_dict["nn_model"]["rank_ratio"] = server_yaml["rank_distribution"]["rank_ratio_list"][index]
        client_var.config_dict["nn_model"]["share_model"] = False
        client_var.prepare()
        client_var.data_loader = client_loader
        client_var.data_sample_num = client_loader.data_sample_num
        client_var.trainer.set_train_loader(client_loader)
        app._LoRAAppEntry__attach_event_handler(client_var)
        client_var.owner_nodes = node
        node.node_var = client_var
        node.prepare_strategy()
        clients.append(client_var)

    # The server's prepared inference model has the maximum configured rank.
    # Make it the explicit initial global state before the first prefix broadcast.
    server_var.model_weight = _clone_state_dict(server_var.model_evaluator.model.state_dict())
    calibration_loader = DataLoader(
        Subset(train_dataset, calibration_indices), batch_size=64, shuffle=False
    )
    functional_loader = DataLoader(
        Subset(test_dataset, functional_indices), batch_size=64, shuffle=False
    )
    return (
        runner,
        server_var,
        clients,
        calibration_loader,
        functional_loader,
        calibration_indices,
        functional_indices,
    )


def run(config_path: str) -> Path:
    seed = int(os.environ.get("AASB_SEED", "42"))
    run_name = os.environ.get("AASB_RUN_NAME")
    config_file = Path(config_path).resolve()
    if run_name is None:
        run_name = f"{config_file.stem}_seed{seed}"
    output_root = Path(os.environ.get("AASB_OUTPUT_ROOT", "src/test/experiment_results/aasb_mnist"))
    output_dir = (output_root / run_name).resolve()

    app = LoRAAppEntry()
    app.load_app_config(str(config_file))
    server_yaml = app.get_app_object("server_yaml")
    client_yaml = app.get_app_object("client_yaml")
    runner_yaml = app.get_app_object("runner")
    ordering = server_yaml.get("canonicalization", {}).get("ordering", "none")
    enabled = bool(server_yaml.get("canonicalization", {}).get("enabled", False))
    method = "rbla" if not enabled else ("rbla_cc_activation" if ordering == "activation_aware" else "rbla_cc_svd")
    device = ModelUtils.accelerator_device()

    TrainingUtils.set_seed_all(seed)
    FedNodeVars.share_model = None
    original_set_seed = TrainingUtils.set_seed

    def experiment_set_seed(seed_input: int = 42) -> None:
        original_set_seed(seed if int(seed_input) == 42 else int(seed_input))

    TrainingUtils.set_seed = staticmethod(experiment_set_seed)
    try:
        (
            runner,
            server_var,
            _clients,
            calibration_loader,
            functional_loader,
            calibration_indices,
            functional_indices,
        ) = _prepare_experiment(app, device, seed, output_dir)

        shutil.copy2(config_file, output_dir / "source_config.yaml")
        (output_dir / "effective_config.json").write_text(
            json.dumps(
                _jsonable({"runner": runner_yaml, "client": client_yaml, "server": server_yaml}),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        (output_dir / "sample_indices.json").write_text(
            json.dumps(
                {
                    "calibration_seed": CALIBRATION_SEED,
                    "calibration_train_indices": calibration_indices,
                    "functional_test_indices": functional_indices,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        aggregator: FedAggregator_RBLA = server_var.aggregation_method
        restore_timing = _install_aggregation_timing(aggregator)
        strategy = runner.runner_strategy
        header = TrainingUtils.build_training_header(runner.server_node)
        runner.server_node.prepare(header, runner.client_node_list)
        runner.server_node.broadcast()
        test_loader = server_var.data_loader.test_data_loader

        round_csv = output_dir / "round_metrics.csv"
        diagnostics_jsonl = output_dir / "canonicalization_diagnostics.jsonl"
        rows: list[dict[str, Any]] = []
        with diagnostics_jsonl.open("w", encoding="utf-8") as diagnostic_stream:
            configured_rounds = int(runner_yaml["general"]["training_rounds"]) + 1
            rounds = min(
                configured_rounds,
                int(os.environ.get("AASB_MAX_ROUNDS", configured_rounds)),
            )
            if rounds <= 0:
                raise ValueError("AASB_MAX_ROUNDS must be positive")
            for round_index in range(rounds):
                round_start = time.perf_counter()
                model = server_var.model_evaluator.model
                start = time.perf_counter()
                calibration_activations = _collect_activations(model, calibration_loader, device)
                calibration_seconds = time.perf_counter() - start
                start = time.perf_counter()
                heldout_activations = _collect_activations(model, functional_loader, device)
                heldout_collection_seconds = time.perf_counter() - start
                if ordering == "activation_aware":
                    aggregator.set_canonicalization_activation_inputs(calibration_activations)
                else:
                    aggregator.set_canonicalization_activation_inputs(None)
                aggregator._aasb_probe_activations = calibration_activations

                participants = runner.server_node.select_clients(runner.client_node_list)
                client_updates = list(strategy.simulate_client_local_training_process(participants))
                runner.server_node.receive_client_updates(client_updates)
                aggregation_start = time.perf_counter()
                runner.server_node.aggregation()
                aggregation_seconds = time.perf_counter() - aggregation_start
                runner.server_node.apply_weight()

                factor_metrics, layer_details = _factor_diagnostics(
                    server_var.model_weight,
                    calibration_activations,
                    heldout_activations,
                    aggregator,
                    method,
                    round_index,
                )
                prefix_metrics = _prefix_accuracies(
                    server_var.model_weight,
                    server_yaml["nn_model"],
                    test_loader,
                    device,
                )
                runner.server_node.broadcast()
                runner.server_node.evaluate()
                eval_results = dict(runner.server_node.eval_results)
                canonical_seconds = float(getattr(aggregator, "_aasb_canonical_seconds", 0.0))
                core_seconds = float(getattr(aggregator, "_aasb_core_seconds", aggregation_seconds))
                score_seconds = float(getattr(aggregator, "_aasb_score_seconds", 0.0))
                row = {
                    "communication_round": round_index + 1,
                    "internal_round": round_index,
                    "method": method,
                    "seed": seed,
                    "global_test_accuracy": float(eval_results["accuracy"]),
                    "global_test_loss": float(eval_results["average_loss"]),
                    "training_loss": _train_loss(client_updates),
                    "aggregation_time_seconds": aggregation_seconds,
                    "aggregation_core_time_seconds": core_seconds,
                    "canonicalization_time_seconds": canonical_seconds if enabled else 0.0,
                    "activation_collection_time_seconds": calibration_seconds,
                    "heldout_activation_collection_time_seconds": heldout_collection_seconds,
                    "activation_score_time_seconds": score_seconds,
                    "full_rank_probe_relative_error": float(
                        getattr(aggregator, "_aasb_full_rank_probe_relative_error", 0.0)
                    ),
                    "maximum_core_reconstruction_error": float(
                        aggregator.canonicalization_summary.get(
                            "maximum_core_reconstruction_error", 0.0
                        )
                    ),
                    "extra_server_time_seconds": (
                        (canonical_seconds if enabled else 0.0)
                        + (calibration_seconds if ordering == "activation_aware" else 0.0)
                    ),
                    **prefix_metrics,
                    **factor_metrics,
                }
                row["total_round_time_seconds"] = time.perf_counter() - round_start
                rows.append(row)
                for detail in layer_details:
                    diagnostic_stream.write(json.dumps(_jsonable(detail), allow_nan=True) + "\n")
                diagnostic_stream.flush()
                runner.server_node.eval_results.update(row)
                runner.server_node.record_evaluation()

        with round_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        server_var.training_logger.end()
        manifest = {
            "run_name": run_name,
            "method": method,
            "seed": seed,
            "communication_rounds": len(rows),
            "config": str(config_file),
            "device": str(device),
            "output_dir": str(output_dir),
            "status": "complete",
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        restore_timing()
        return output_dir
    finally:
        TrainingUtils.set_seed = staticmethod(original_set_seed)
        FedNodeVars.share_model = None


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m entries.aasb_mnist_experiment EXPERIMENT_YAML")
    output = run(sys.argv[1])
    print(f"AASB experiment complete: {output}")


if __name__ == "__main__":
    main()
