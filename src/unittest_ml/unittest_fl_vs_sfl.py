"""
Unit test: compare FL (FedAvg) and SFL global model weights using L2 norm.

Both FL and SFL are trained for the same number of rounds on identical data
and model architectures. After training we:
  1. Retrieve the server-side global model weights from each paradigm.
  2. Compute the L2 norm (torch.norm) of each individual weight tensor.
  3. Compute the L2 norm of the element-wise difference for shared keys.
  4. Report per-layer and overall statistics.

Initialization follows the same pattern as standard_sample_entry.py /
sfl_sample_entry.py (the authoritative reference in this repo):
  - FedNodeVars(yaml).prepare()  →  node.node_var = var  →  node.prepare_strategy()

FL  global weight  : server_node.node_var.model_weight
SFL global weight  : server_node.node_var.aggregated_weight  (front/client part)
                   + server_node.node_var.model.state_dict() (server/rear part)
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

# Init startup path
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from flex.fed_node import FedNodeVars, FedNodeEventArgs
from flex.fed_runner import FedRunner
from flex.ml_utils import ConfigLoader, console
from flex.fl_algorithms.noniid.noniid_data_generator import NoniidDataGenerator
from flex.ml_data_loader.dataset_loader_factory import DatasetLoaderFactory
from flex.ml_data_loader.dataset_loader_args import DatasetLoaderArgs


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Concrete FedRunner subclass (FedRunner is abstract)
# ---------------------------------------------------------------------------

class SimpleFedRunner(FedRunner):
    """Minimal concrete FedRunner that exposes create_nodes / create_run_strategy."""

    def run(self) -> None:                     # satisfy ABC
        super().run()


# ---------------------------------------------------------------------------
# No-op event handler (passed to node_var.attach_event so the system is
# consistent with how entries use it; we do not need custom behaviour here)
# ---------------------------------------------------------------------------

def _noop_handler(args: FedNodeEventArgs) -> None:
    pass


def _attach_noop_events(node_var: FedNodeVars) -> None:
    for event in (
        "on_prepare_data_loader",
        "on_prepare_model",
        "on_prepare_loss_func",
        "on_prepare_optimizer",
        "on_prepare_strategy",
        "on_prepare_extractor",
        "on_prepare_data_distribution",
        "on_prepare_data_handler",
        "on_prepare_client_selection",
        "on_prepare_trainer",
        "on_prepare_aggregation",
        "on_prepare_training_logger",
    ):
        node_var.attach_event(event, _noop_handler)


# ---------------------------------------------------------------------------
# L2-norm helpers
# ---------------------------------------------------------------------------

def state_dict_l2_norm(state_dict: Dict[str, torch.Tensor]) -> float:
    """Return the overall L2 norm of all parameters in a state_dict."""
    total_sq = 0.0
    for tensor in state_dict.values():
        if not isinstance(tensor, torch.Tensor):
            continue
        t = tensor.detach().to(device="cpu", dtype=torch.float32)
        total_sq += float(torch.sum(t * t).item())
    return math.sqrt(total_sq)


def state_dict_l2_distance(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute L2 distance ||a - b|| for each shared key and the overall norm.

    Returns
    -------
    overall_distance : float
    per_key_distances : dict[str, float]
    """
    common_keys = set(state_a.keys()) & set(state_b.keys())
    per_key: Dict[str, float] = {}
    total_sq = 0.0
    for key in sorted(common_keys):
        ta = state_a[key].detach().to(device="cpu", dtype=torch.float32)
        tb = state_b[key].detach().to(device="cpu", dtype=torch.float32)
        diff_sq = float(torch.sum((ta - tb) ** 2).item())
        per_key[key] = math.sqrt(diff_sq)
        total_sq += diff_sq
    return math.sqrt(total_sq), per_key


# ---------------------------------------------------------------------------
# Shared node initialisation helper
# (mirrors standard_sample_entry.py / sfl_sample_entry.py exactly)
# ---------------------------------------------------------------------------

def _build_and_run_federation(
    runner_yaml: dict,
    server_yaml: dict,
    client_yaml: dict,
    device: str = "cpu",
) -> FedRunner:
    """
    Initialise nodes, attach vars, call prepare(), wire strategies and run.

    This mirrors the authoritative pattern used by standard_sample_entry.py and
    sfl_sample_entry.py so that FedNodeServer.strategy is properly created
    before runner_strategy.run() calls server_node.prepare().

    Returns the fully-run FedRunner so callers can inspect server_node.node_var.
    """
    fed_runner = SimpleFedRunner()
    training_rounds = runner_yaml.get("general", {}).get("training_rounds", 3)
    fed_runner.training_rounds = training_rounds
    fed_runner.with_yaml(runner_yaml)
    fed_runner.create_nodes()
    fed_runner.create_run_strategy()

    # ---- Server node ----
    server_var = FedNodeVars(server_yaml)
    server_var.prepare()
    _attach_noop_events(server_var)
    server_var.owner_nodes = fed_runner.server_node   # two-way binding
    server_var.set_device(device)
    fed_runner.server_node.node_var = server_var
    fed_runner.server_node.prepare_strategy()
    fed_runner.server_node.node_var = server_var      # re-bind after prepare_strategy

    # ---- Distribute data across clients (non-IID split) ----
    train_loader = server_var.data_loader
    allocated_noniid_data = NoniidDataGenerator(
        train_loader.data_loader
    ).generate_noniid_data(
        distribution_config=server_yaml["data_distribution"]
    )
    for i, subset in enumerate(allocated_noniid_data):
        args = DatasetLoaderArgs({
            "name": "custom",
            "root": "../../../.dataset",
            "split": "",
            "batch_size": 64,
            "shuffle": True,
            "num_workers": 0,
            "is_download": True,
            "is_load_train_set": True,
            "is_load_test_set": True,
            "dataset": subset,
        })
        allocated_noniid_data[i] = DatasetLoaderFactory().create(args)

    # ---- Client nodes ----
    for index, node in enumerate(fed_runner.client_node_list):
        client_var = FedNodeVars(client_yaml, is_clone_dict=True)
        client_var.config_dict["nn_model"]["share_model"] = False
        # Pre-set data_loader so prepare() can resolve trainer dependencies
        client_var.data_loader = allocated_noniid_data[index]
        client_var.data_sample_num = client_var.data_loader.data_sample_num
        client_var.set_device(device)
        client_var.prepare()
        if client_var.trainer is not None:
            client_var.trainer.set_train_loader(client_var.data_loader)
        _attach_noop_events(client_var)
        client_var.owner_nodes = node                 # two-way binding
        node.node_var = client_var
        node.prepare_strategy()

    # ---- Run federation ----
    fed_runner.run()
    return fed_runner


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def run_fl_training(
    runner_yaml_path: str,
    client_yaml_path: str,
    server_yaml_path: str,
    training_rounds: int = 3,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Run Federated Learning (FedAvg) and return the server global model weight.

    Returns
    -------
    global_weight : Dict[str, torch.Tensor]  – server aggregated weight
    """
    console.out("\n" + "=" * 60)
    console.out("  Running FL (FedAvg) training")
    console.out("=" * 60)

    runner_yaml = ConfigLoader.load(runner_yaml_path)
    client_yaml = ConfigLoader.load(client_yaml_path)
    server_yaml = ConfigLoader.load(server_yaml_path)
    runner_yaml["general"]["training_rounds"] = training_rounds

    set_seed(42)
    fed_runner = _build_and_run_federation(runner_yaml, server_yaml, client_yaml, device)

    global_weight = getattr(fed_runner.server_node.node_var, "model_weight", None)
    if global_weight is None:
        raise RuntimeError("FL: server_node.node_var.model_weight is None after training.")

    console.ok(f"FL training done.  Global weight keys: {list(global_weight.keys())}")
    return global_weight


def run_sfl_training(
    runner_yaml_path: str,
    client_yaml_path: str,
    server_yaml_path: str,
    training_rounds: int = 3,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Run Split Federated Learning (SFL) and return the combined global model weight.

    The SFL global model is the union of:
      - aggregated client front weight  (server_node.node_var.aggregated_weight)
      - server rear model state_dict    (server_node.node_var.model.state_dict())

    Returns
    -------
    global_weight : Dict[str, torch.Tensor]  – combined front + rear weight
    """
    console.out("\n" + "=" * 60)
    console.out("  Running SFL training")
    console.out("=" * 60)

    runner_yaml = ConfigLoader.load(runner_yaml_path)
    client_yaml = ConfigLoader.load(client_yaml_path)
    server_yaml = ConfigLoader.load(server_yaml_path)
    runner_yaml["general"]["training_rounds"] = training_rounds

    set_seed(42)
    fed_runner = _build_and_run_federation(runner_yaml, server_yaml, client_yaml, device)

    node_var = fed_runner.server_node.node_var

    # Front part: aggregated client weights
    front_weight: Dict[str, torch.Tensor] = getattr(node_var, "aggregated_weight", None) or {}

    # Rear part: server-side model
    rear_model: nn.Module | None = getattr(node_var, "model", None)
    rear_weight: Dict[str, torch.Tensor] = rear_model.state_dict() if rear_model is not None else {}

    # Merge (rear keys should not overlap with front keys in a standard SFL split)
    combined: Dict[str, torch.Tensor] = {**front_weight, **rear_weight}
    if not combined:
        raise RuntimeError("SFL: no weights found on server node after training.")

    console.ok(f"SFL training done.  Combined weight keys: {list(combined.keys())}")
    return combined


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare_fl_vs_sfl(
    fl_weight: Dict[str, torch.Tensor],
    sfl_weight: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """
    Compare FL and SFL global model weights using L2 norm.

    Returns a result dict with:
      fl_l2_norm       : overall L2 norm of FL global model
      sfl_l2_norm      : overall L2 norm of SFL combined global model
      l2_distance      : ||fl - sfl|| on shared keys
      per_key_distance : per-layer L2 distances (shared keys only)
      shared_keys      : keys present in both models
      fl_only_keys     : keys only in FL model
      sfl_only_keys    : keys only in SFL model
    """
    console.out("\n" + "=" * 60)
    console.out("  FL vs SFL – Global Model L2-Norm Comparison")
    console.out("=" * 60)

    fl_norm  = state_dict_l2_norm(fl_weight)
    sfl_norm = state_dict_l2_norm(sfl_weight)

    shared_keys   = sorted(set(fl_weight.keys()) & set(sfl_weight.keys()))
    fl_only_keys  = sorted(set(fl_weight.keys()) - set(sfl_weight.keys()))
    sfl_only_keys = sorted(set(sfl_weight.keys()) - set(fl_weight.keys()))

    overall_dist, per_key_dist = state_dict_l2_distance(fl_weight, sfl_weight)

    # Per-layer report
    console.out(f"\n{'Layer':<50} {'FL L2':>12} {'SFL L2':>12} {'L2 Dist':>12}")
    console.out("-" * 90)
    for key in shared_keys:
        fl_k  = state_dict_l2_norm({key: fl_weight[key]})
        sfl_k = state_dict_l2_norm({key: sfl_weight[key]})
        d     = per_key_dist.get(key, float("nan"))
        console.out(f"  {key:<48} {fl_k:>12.6f} {sfl_k:>12.6f} {d:>12.6f}")

    if fl_only_keys:
        console.warn(f"\n  Keys only in FL  model: {fl_only_keys}")
    if sfl_only_keys:
        console.warn(f"  Keys only in SFL model: {sfl_only_keys}")

    console.out("\n" + "-" * 90)
    console.out(f"  FL  overall L2 norm  : {fl_norm:.6f}")
    console.out(f"  SFL overall L2 norm  : {sfl_norm:.6f}")
    console.out(f"  L2 distance (shared) : {overall_dist:.6f}")
    console.out("=" * 60)

    return {
        "fl_l2_norm":       fl_norm,
        "sfl_l2_norm":      sfl_norm,
        "l2_distance":      overall_dist,
        "per_key_distance": per_key_dist,
        "shared_keys":      shared_keys,
        "fl_only_keys":     fl_only_keys,
        "sfl_only_keys":    sfl_only_keys,
    }


# ---------------------------------------------------------------------------
# Main test entry
# ---------------------------------------------------------------------------

def test_fl_vs_sfl(
    fl_runner_yaml:  str = "./test_data/fl_runner.yaml",
    fl_client_yaml:  str = "./test_data/node_config_template_client.yaml",
    fl_server_yaml:  str = "./test_data/fl_server.yaml",
    sfl_runner_yaml: str = "./test_data/sfl_runner.yaml",
    sfl_client_yaml: str = "./test_data/sfl_client.yaml",
    sfl_server_yaml: str = "./test_data/sfl_server.yaml",
    training_rounds: int = 3,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    End-to-end test: train FL and SFL, then compare global model weights via L2 norm.
    """
    console.out("\n" + "#" * 70)
    console.out("  Unit Test: FL (FedAvg) vs SFL – Global Model L2-Norm Comparison")
    console.out("#" * 70)

    # --- Train FL ---
    fl_global_weight = run_fl_training(
        fl_runner_yaml, fl_client_yaml, fl_server_yaml,
        training_rounds=training_rounds, device=device,
    )

    # --- Train SFL ---
    sfl_global_weight = run_sfl_training(
        sfl_runner_yaml, sfl_client_yaml, sfl_server_yaml,
        training_rounds=training_rounds, device=device,
    )

    # --- Compare ---
    result = compare_fl_vs_sfl(fl_global_weight, sfl_global_weight)

    # --- Verdict ---
    console.out("\n" + "=" * 70)
    if result["shared_keys"]:
        console.ok(
            f"Comparison complete. "
            f"L2 distance between FL and SFL global models on "
            f"{len(result['shared_keys'])} shared layer(s): "
            f"{result['l2_distance']:.6f}"
        )
    else:
        console.warn(
            "No shared keys between FL and SFL models. "
            "The two architectures do not share parameter names – "
            "L2 norms are reported individually above."
        )
    console.out("=" * 70)

    return result


def main():
    """Run the FL vs SFL comparison test."""
    try:
        result = test_fl_vs_sfl(training_rounds=3, device="cpu")
        console.out(f"\nFinal L2 distance  : {result['l2_distance']:.6f}")
        console.out(f"FL  L2 norm        : {result['fl_l2_norm']:.6f}")
        console.out(f"SFL L2 norm        : {result['sfl_l2_norm']:.6f}")
    except Exception as exc:
        console.error(f"\nTest crashed: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
