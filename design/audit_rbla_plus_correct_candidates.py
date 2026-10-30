"""Read-only audit of current correct-assignment configs and RankScale logs.

Print JSON to stdout. Historical RBLA table values are explicitly distinguished
from revalidated raw logs. No models are downloaded and no training is started.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flex.ml_utils.app_entry import AppEntry
from flex.ml_models.nn_model_args import NNModelArgs
from flex.ml_models.lora._nn_model_cifar_lora_cnn import NNModel_CifarLoRACNN
from flex.ml_models.lora._nn_model_vgg_lora_cnn import NNModel_VGG9_LoRA

METHODS = ("rbla", "rbla_plus", "rbla_plus_rank_scale")


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            if key not in a or key not in b:
                result.append({"path": path + "." + key, "a": a.get(key), "b": b.get(key)})
            else:
                result.extend(differences(a[key], b[key], path + "." + key))
        return result
    return [] if a == b else [{"path": path, "a": a, "b": b}]


def table_baselines():
    text = (ROOT / "design/rbla_plus_tables_updated_2026-09-15.tex").read_text(encoding="utf-8")
    correct = text.split("\\label{tab:correct-rank-assignment}", 1)[1]
    names = {"CIFAR-10": "cifar10", "CINIC-10": "cinic10", "SVHN": "svhn",
             "MNLI": "mnli", "QQP": "qqp", "RTE": "rte", "SST-2": "sst2",
             "AG News": "agnews", "IMDB": "imdb"}
    result = {}
    for label, dataset in names.items():
        match = re.search(re.escape(label) + r"\s*&\s*0\.4\s*&\s*([0-9.]+)", correct)
        if match is None:
            raise ValueError(f"Missing historical RBLA table value: {label}")
        result[dataset] = float(match.group(1))
    return result


def audit():
    candidates = []
    experiment_root = ROOT / "src/test/rbla+/experiment/rank_match/dirichlet_0_4"
    for dataset, baseline in table_baselines().items():
        resolved, paths = {}, {}
        folder = "rte_correct_rerun" if dataset == "rte" else dataset
        for method in METHODS:
            path = experiment_root / folder / f"{dataset}_dirichlet_0_4_rank_correct_{method}.yaml"
            app = AppEntry()
            for key in ("runner", "client_yaml", "edge_yaml", "server_yaml"):
                app.app_objects.remove_object(key)
            app.load_app_config(str(path))
            resolved[method] = {key: copy.deepcopy(app.get_app_object(key))
                                for key in ("runner", "client_yaml", "server_yaml")}
            paths[method] = path.relative_to(ROOT).as_posix()

        logs = list((ROOT / "src/test/experiment_results").glob(
            f"{dataset}_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_*.csv"))
        path = max(logs, key=lambda p: re.search(r"\d{8}_\d{6}", p.name).group())
        raw = path.read_bytes()
        lines = raw.decode("utf-8-sig").splitlines()
        config = json.loads(lines[0].split(",", 1)[1])
        start = next(i for i, line in enumerate(lines) if line.startswith("round,"))
        rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
        best = max(rows, key=lambda row: float(row["accuracy"]))
        values = [float(row["accuracy"]) * 100 for row in rows]
        server = resolved["rbla_plus_rank_scale"]["server_yaml"]
        profiles = []
        for ratio in [1.0] + server["rank_distribution"]["rank_ratio_list"]:
            if dataset in {"cifar10", "cinic10", "svhn"}:
                args = NNModelArgs(dict(server["nn_model"], rank_ratio=ratio))
                factory = NNModel_VGG9_LoRA() if dataset == "svhn" else NNModel_CifarLoRACNN()
                model = factory.create_model(args)
                layers = [{"name": name, "rank": int(module.r), "scaling": float(module.scaling),
                           "a_shape": list(module.lora_A.shape), "b_shape": list(module.lora_B.shape),
                           "adapter_parameters": module.lora_A.numel() + module.lora_B.numel()}
                          for name, module in model.named_modules()
                          if hasattr(module, "lora_A") and hasattr(module, "lora_B")]
                profiles.append({"ratio": ratio, "source": "current_factory_instantiation",
                                 "layers": layers,
                                 "adapter_parameters": sum(layer["adapter_parameters"] for layer in layers)})
            else:
                rank = max(1, round(server["nn_model"]["lora_r"] * ratio))
                profiles.append({"ratio": ratio, "source": "current_factory_formula_not_instantiation",
                                 "rank": rank, "scaling": server["nn_model"]["lora_alpha"] / rank})
        keys = ("general", "data_loader", "nn_model", "loss_func", "optimizer", "training", "trainer",
                "tokenizer", "data_distribution", "client_selection", "aggregation", "canonicalization", "rank_distribution")
        historical = config["effective_yaml"]["server_yaml"]
        candidates.append({
            "dataset": dataset, "source_csv": path.relative_to(ROOT).as_posix(),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "peak_accuracy_percent": max(values), "peak_logged_round": int(best["round"]),
            "final_accuracy_percent": values[-1], "last10_mean_accuracy_percent": sum(values[-10:]) / len(values[-10:]),
            "final_logged_round": int(rows[-1]["round"]), "records": len(rows),
            "rbla_peak_from_supplied_table_not_revalidated_raw_log": baseline,
            "peak_gap_vs_supplied_rbla_table_pp": max(values) - baseline,
            "config_paths": paths, "current_resolved": resolved,
            "method_differences": {method: differences(resolved["rbla"], value)
                                   for method, value in resolved.items() if method != "rbla"},
            "historical_vs_current_config_differences": differences(
                {key: historical.get(key) for key in keys}, {key: server.get(key) for key in keys}),
            "current_model_profiles": profiles,
        })
    return {"audit_date": "2026-09-15", "scope": "correct assignment, historical screening, no new training",
            "baseline_table_source": "design/rbla_plus_tables_updated_2026-09-15.tex", "candidates": candidates}


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
