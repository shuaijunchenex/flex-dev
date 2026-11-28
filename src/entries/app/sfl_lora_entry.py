"""Fixed-cut LoRA-SFL experiments on Office-Caltech10."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter

import torch
import yaml

from flex.fed_node import FedNodeVars
from flex.fed_runner import FedRunner
from flex.ml_data_loader.image_benchmark_data import prepare_dataset, read_manifest
from flex.sfl.partition import build_sfl_manifest, build_client_views
from flex.ml_utils.training_utils import TrainingUtils
from flex.sfl.evaluation import SflModelEvaluator


def _write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _bind_node(node, model_config, role, device, optimizer, shard_size=0, *, training_logger=None):
    trainer_type = "split_client" if role == "client" else "split_server"
    var = FedNodeVars({
        "role": role, "strategy": {"role": role, "strategy_name": "sfl_lora"},
        "nn_model": model_config, "data_loader": {"task_type": "cv"},
        "loss_func": {"type": "CrossEntropyLoss"},
        "trainer": {"trainer_type": trainer_type},
    })
    # The runner supplies batches from the manifest, including its monitor/final split.
    var.set_device(device)
    var.data_sample_num = shard_size
    var.prepare_model()
    if any(p.requires_grad for p in var.model.parameters()):
        var.config_dict["optimizer"] = optimizer
        var.prepare_optimizer()
    var.prepare_loss_func()
    var.prepare_trainer()
    if training_logger is not None:
        var.config_dict["training_logger"] = training_logger
        var.prepare_training_logger()
    var.owner_nodes = node
    node.node_var = var
    node.prepare_strategy()


class SflLoRAAppEntry:
    def __init__(self, job, output_dir, smoke=False):
        self.job = job
        self.output_dir = Path(output_dir)
        self.smoke = smoke
        self.run_config = job["run"]
        self.protocol = job["protocol"]
        self.device = self.run_config["device"]
        self.seed = self.run_config["seed"]

    @classmethod
    def from_config(cls, path, output_dir, smoke=False):
        path = Path(path).resolve()
        job = yaml.safe_load(path.read_text(encoding="utf-8"))
        job["data"]["assets"] = str((path.parent / job["data"]["assets"]).resolve())
        if smoke:
            job["protocol"]["rounds"] = min(2, job["protocol"]["rounds"])
        return cls(job, output_dir, smoke)

    def build_nodes(self):
        job = self.job
        data = job["data"]
        assets = json.loads(Path(data["assets"]).read_text(encoding="utf-8"))
        dataset = assets["datasets"][data["dataset"]]
        catalog = prepare_dataset(data["dataset"], dataset["root"], download=False)
        base = read_manifest(dataset["manifest_path"], catalog)
        self.manifest = build_sfl_manifest(catalog, base, data["split_seed"],
                                           allocation=data.get("allocation", "domain"),
                                           data_distribution=job.get("data_distribution"))
        model_asset = assets["models"]["cifar_resnet20_lora"]
        checkpoint = Path(model_asset["path"])
        self.model_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if self.model_sha != model_asset["sha256"]:
            raise ValueError(f"Pretrained weight checksum mismatch: {checkpoint}")
        net = FedRunner()
        self.fed_runner = net
        net.with_yaml({
            "server_node": {"id": "server"},
            "client_nodes": {"office": {"number": len(self.manifest["clients"]),
                                        "id_prefix": "client", "link_to": "server"}},
            "general": {"training_rounds": self.protocol["rounds"]},
            "strategy": {"role": "runner", "strategy_name": "sfl_lora"},
        })
        net.create_nodes()
        if "client_ids" in data:
            selected = data["client_ids"]
            by_id = {c.node_id: c for c in net.client_node_list}
            net.client_node_list = [by_id[cid] for cid in selected]
        net.create_run_strategy()
        self.runner = net.runner_strategy
        self.nodes = {c.node_id: c for c in net.client_node_list}
        if not self.nodes or len(self.nodes) != len(net.client_node_list):
            raise ValueError("Select one or more distinct clients")
        self.views = build_client_views(catalog, self.manifest, list(self.nodes), self.seed,
                                        self.protocol["batch_size"], sample_limit=4 if self.smoke else None,
                                        train_subsample=job["data"].get("train_subsample"))
        adaptation = job["adaptation"]
        rank = adaptation["front_rank"]
        ranks = {cid: rank[cid] if isinstance(rank, dict) else rank for cid in self.nodes}
        if self.protocol["rear_mode"] == "private_adapter" and adaptation["rear_scope"] != "adapter":
            raise ValueError("private_adapter requires rear_scope=adapter")
        common = {**job["model"], **adaptation, "share_model": False,
                  "initialization_seed": self.seed, "checkpoint_path": str(checkpoint)}
        _bind_node(net.server_node, dict(common, part="rear", front_rank=max(ranks.values())),
                   "server", self.device, job["optimizer"],
                   training_logger={"path": str(self.output_dir), **job.get("training_logger", {})})
        for cid, c in self.nodes.items():
            _bind_node(c, dict(common, part="front", front_rank=ranks[cid]), "client", self.device,
                       job["optimizer"], self.views[cid].shard_size)
        self.ranks = ranks
        rear = net.server_node.node_var.model
        rear_params = sum(p.numel() for p in rear.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in rear.head.parameters() if p.requires_grad)
        owners = len(self.nodes) if self.protocol["rear_mode"] == "private_adapter" else 1
        self.budgets = {
            "front_trainable": {cid: sum(p.numel() for p in c.node_var.model.parameters() if p.requires_grad)
                                for cid, c in self.nodes.items()},
            "rear_trainable": rear_params, "head_trainable": head_params,
            "rear_owned_params": head_params + owners * (rear_params - head_params),
        }
        if not (any(self.budgets["front_trainable"].values()) or self.budgets["rear_trainable"]):
            raise ValueError("At least one side needs trainable parameters")
        TrainingUtils.set_seed_all(self.seed)

    def _evaluate(self, split):
        rows, summary = self.evaluator.evaluate(split)
        for row in rows:
            cid = row["client_id"]
            row["front_rank"] = self.ranks[cid]
            row["front_version"] = self.nodes[cid].strategy.front_version
            row["last_visit_round"] = self.runner.last_visit_round[cid]
            row["participated"] = self.runner.completed_round > 0 and row["last_visit_round"] == self.runner.completed_round
            row["rear_snapshot_round"] = self.runner.completed_round
            if self.protocol["rear_mode"] == "private_adapter":
                row["private_adapter_round"] = row["last_visit_round"]
        return rows, summary

    def _record_round(self, round_id, participants, updates):
        start = perf_counter()
        rows, summary = self._evaluate("monitor")
        visits, traffic = self.runner.train_rows, self.runner.traffic
        result = dict(round=round_id, participants=participants, clients=rows, monitor=summary,
                      budgets=self.budgets, eval_seconds=perf_counter() - start,
                      timings=self.runner.round_times, training=visits, traffic=traffic)
        if round_id == 0:
            mode = self.protocol["rear_mode"]
            result.update(device=str(self.device), torch_version=str(torch.__version__), smoke=self.smoke,
                          model_sha256=self.model_sha, manifest_sha256=self.manifest["manifest_sha256"],
                          rear_body_history={"private_adapter": "per_client", "round_average": "reset_each_visit"}
                          .get(mode, self.protocol.get("rear_optimizer_history", "shared")),
                          head_history="reset_each_visit" if mode == "round_average" else "shared")
            _write_json(self.output_dir / "initial_traffic.json", traffic)
            path = self.output_dir / "initial.json"
        else:
            if self.diagnostics is not None:
                result.update(self.diagnostics.after_round(round_id, participants, updates))
            path = self.output_dir / "rounds" / f"round_{round_id:04d}.json"
        _write_json(path, result)

        metrics = ("accuracy", "loss", "macro_f1", "client_mean_accuracy", "p10_accuracy", "worst_accuracy")
        counts = Counter()
        for client_counts in traffic.values():
            counts.update(client_counts)
        logger = self.fed_runner.server_node.node_var.training_logger
        logger.record({"round": round_id, **{key: summary[key] for key in metrics},
                       "steps": sum(row["steps"] for row in visits),
                       "n_examples": sum(row["n_examples"] for row in visits), **counts})
        logger.record_selected_clients(round_id, participants)
        logger.flush_selected_clients()

    def _deployment_weights(self):
        rear = self.runner.server.rear_state
        private_keys = rear.model.adapter_keys() if rear.mode == "private_adapter" else ()
        weights = {f"rear.shared.{k}": v for k, v in rear.model.state_dict().items() if k not in private_keys}
        for cid, node in self.nodes.items():
            weights.update({f"front.{cid}.{k}": v for k, v in node.node_var.model.state_dict().items()})
            if private_keys:
                state = rear.state_for_eval(cid)
                weights.update({f"rear.private.{cid}.{k}": state[k] for k in private_keys})
        return weights

    def run(self, *, final=False, diagnostics_factory=None):
        if min(self.protocol["rounds"], self.protocol["local_epochs"], self.protocol["batch_size"]) < 1:
            raise ValueError("rounds, local_epochs and batch_size must be positive")
        if self.run_config.get("recovery_steps", 0) and not self.run_config.get("probe_batches", 0):
            raise ValueError("Online recovery requires run.probe_batches > 0")
        if self.run_config.get("probe_batches", 0) and diagnostics_factory is None:
            raise ValueError("Use src/test/sfl_lora/diagnostics.py with the same CLI arguments for broadcast probes")
        # Reserve the run directory before writing. A retry keeps the earlier attempt's files.
        try:
            (self.output_dir / "rounds").mkdir(parents=True)
        except FileExistsError:
            self.output_dir = Path(mkdtemp(dir=self.output_dir, prefix="run_"))
            (self.output_dir / "rounds").mkdir()
        out = self.output_dir
        self.build_nodes()
        self.diagnostics = diagnostics_factory(self) if diagnostics_factory is not None else None
        self.runner.configure(self.job, self.views, on_round_end=self._record_round,
                              on_batch_end=self.diagnostics.after_batch if self.run_config.get("recovery_steps") else None)
        self.runner.initialize()
        self.evaluator = SflModelEvaluator(
            clients=self.nodes, server_vars=self.fed_runner.server_node.node_var,
            rear_state=self.runner.server.rear_state, views=self.views,
            device=self.device, num_classes=self.job["model"]["num_classes"])
        (out / "resolved.yaml").write_text(yaml.safe_dump(self.job, sort_keys=False), encoding="utf-8")
        _write_json(out / "manifest.json", self.manifest)
        logger = self.fed_runner.server_node.node_var.training_logger
        logger.begin({**self.job, "output_dir": str(out.resolve())})
        try:
            self._record_round(0, [], [])
            self.fed_runner.run()
            if final:
                rows, summary = self._evaluate("final")
                _write_json(out / "final.json", dict(round=self.runner.completed_round, clients=rows, final=summary))
            if logger.save_weights_enabled:
                logger.save_weights_if_enabled(self._deployment_weights(), self.runner.completed_round)
        finally:
            logger.end()
