"""Isolated seeded entry point for RBLA reference-frame robustness runs.

The ordinary :mod:`entries.lora` entry intentionally keeps its historical
``seed=42`` behaviour.  This module accepts experiment-only overrides through
environment variables so a seed matrix can be run without changing any
existing strategy or YAML.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

_ENTRIES_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_ENTRIES_DIR)
_TEST_DIR = os.path.join(_SRC_DIR, "test")
os.chdir(_TEST_DIR)
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, _SRC_DIR)

from entries.app.lora_entry import LoRAAppEntry
from flex.ml_utils.model_utils import ModelUtils
from flex.ml_utils.training_utils import TrainingUtils


def _required_environment() -> tuple[int, int, list[float], str]:
    training_seed = int(os.environ["RBLA_TRAINING_SEED"])
    rank_seed = int(os.environ["RBLA_RANK_SEED"])
    rank_ratios = [float(value) for value in json.loads(os.environ["RBLA_RANK_RATIOS"])]
    result_prefix = os.environ["RBLA_RESULT_PREFIX"]
    if len(rank_ratios) != 10:
        raise ValueError(f"Expected 10 rank ratios, received {len(rank_ratios)}")
    return training_seed, rank_seed, rank_ratios, result_prefix


def main(config_path: str) -> None:
    training_seed, rank_seed, rank_ratios, result_prefix = _required_environment()
    app = LoRAAppEntry()
    app.load_app_config(config_path)

    server_yaml = app.get_app_object("server_yaml")
    client_yaml = app.get_app_object("client_yaml")
    runner_yaml = app.get_app_object("runner")
    server_yaml["rank_distribution"]["rank_ratio_list"] = rank_ratios

    scaling_type = os.environ.get(
        "RBLA_SCALING_TYPE",
        str(server_yaml.get("support_scaling", {}).get("scaling_type", "q_power")),
    )
    gamma = float(
        os.environ.get(
            "RBLA_GAMMA",
            server_yaml.get("support_scaling", {}).get("gamma", 0.0),
        )
    )
    server_yaml.setdefault("support_scaling", {}).update(
        {"scaling_type": scaling_type, "gamma": gamma}
    )
    server_yaml.setdefault("aggregation", {}).update(
        {"scaling_type": scaling_type, "gamma": gamma}
    )

    try:
        git_commit = subprocess.check_output(
            ["git", "-C", _SRC_DIR, "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.run(
            ["git", "-C", _SRC_DIR, "diff", "--quiet"], check=False
        ).returncode != 0
    except (OSError, subprocess.SubprocessError):
        git_commit, dirty = "unknown", True

    # Persist the effective seeds/list in the CSV configuration header and use
    # an unambiguous output name.  These keys are diagnostics-only.
    seed_metadata = {
        "training_seed": training_seed,
        "rank_assignment_seed": rank_seed,
        "rank_ratio_list": rank_ratios,
    }
    server_yaml["reference_seed_matrix"] = seed_metadata
    server_yaml["reference_run_metadata"] = {
        **seed_metadata,
        "gamma": gamma,
        "scaling_type": scaling_type,
        "experiment_phase": os.environ.get("RBLA_EXPERIMENT_PHASE", "seed_matrix"),
        "git_commit": git_commit,
        "git_dirty": dirty,
        "yaml_path": os.path.abspath(config_path),
        "checkpoint_path": "recorded in *_run_metadata.json after training",
    }
    for config in (server_yaml, client_yaml, runner_yaml):
        config["training_logger"]["prefix"] = result_prefix

    TrainingUtils.set_seed_all(training_seed)

    # BaseStrategy and ModelTrainer preserve legacy reproducibility by calling
    # TrainingUtils.set_seed(42) in their constructors.  In this *isolated*
    # entry only, map that legacy default to the requested matrix seed.  Calls
    # with an explicit non-default seed keep their original meaning.
    original_set_seed = TrainingUtils.set_seed

    def experiment_set_seed(seed_input: int = 42) -> None:
        original_set_seed(training_seed if int(seed_input) == 42 else int(seed_input))

    TrainingUtils.set_seed = staticmethod(experiment_set_seed)
    try:
        app.run(ModelUtils.accelerator_device())
    finally:
        TrainingUtils.set_seed = staticmethod(original_set_seed)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m entries.lora_reference_seeded EXPERIMENT_YAML")
    main(sys.argv[1])
