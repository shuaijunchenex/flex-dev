"""Evaluate the actual front and rear deployed for each client."""
from __future__ import annotations

import copy
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import SequentialSampler

from flex.sfl.batch_observer import ModelEvaluatorBatchObserver


class SflModelEvaluator:
    def __init__(self, *, clients, server_vars, views, device, num_classes, rear_state=None):
        self.clients = clients
        self.server_vars = server_vars
        self.views = views
        self.device = device
        self.num_classes = num_classes
        # Owned by the SFL server strategy; injected so this module stays usable standalone.
        if rear_state is None:
            raise ValueError("SflModelEvaluator requires the rear_state of the SFL server strategy")
        self.rear_state = rear_state

    def model_for(self, client_id):
        # eval() may merge LoRA weights, and ModelEvaluator moves its model between devices.
        front = copy.deepcopy(self.clients[client_id].node_var.model)
        rear = copy.deepcopy(self.server_vars.model)
        rear.load_state_dict(self.rear_state.state_for_eval(client_id))
        return nn.Sequential(front, rear)

    def evaluate_client(self, client_id, split, *, collect_losses=False):
        if split not in ("monitor", "final"):
            raise ValueError(f"Unknown evaluation split: {split}")
        view = self.views[client_id]
        loader = view.monitor_loader if split == "monitor" else view.final_loader
        domains = getattr(loader.dataset, "domains", None)
        if domains is not None and not isinstance(loader.sampler, SequentialSampler):
            raise ValueError("Per-domain evaluation requires a sequential held-out loader")
        k = self.num_classes
        confusion = torch.zeros(k, k, dtype=torch.long)
        by_domain = defaultdict(lambda: dict(n=0, correct=0, loss_sum=0.0))
        sample_losses = []
        offset = 0

        def collect(labels, predictions, valid, losses):
            nonlocal offset
            cells = labels[valid] * k + predictions[valid]
            confusion.add_(torch.bincount(cells.cpu(), minlength=k * k).reshape(k, k))
            if domains is not None:
                indices = valid.nonzero(as_tuple=True)[0].tolist()
                correct = predictions[valid].eq(labels[valid]).tolist()
                for index, hit, loss in zip(indices, correct, losses.tolist()):
                    totals = by_domain[domains[offset + index]]
                    totals["n"] += 1
                    totals["correct"] += int(hit)
                    totals["loss_sum"] += loss
            if collect_losses:
                sample_losses.extend(losses.tolist())
            offset += len(labels)

        evaluator = ModelEvaluatorBatchObserver(
            self.model_for(client_id), loader, nn.CrossEntropyLoss(reduction="none"),
            self.device, legacy_eval_rng_compat=True, observer=collect)
        with torch.random.fork_rng(devices=[]):
            metrics = evaluator.evaluate()
        n = metrics["total_test_samples"]
        row = dict(client_id=client_id, split=split, domain=view.domain, n=n,
                   correct=int(confusion.diag().sum()), accuracy=metrics["accuracy"],
                   loss_sum=metrics["average_loss"] * n if n else 0.0,
                   confusion=confusion.tolist() if n else [], domain_totals=dict(by_domain))
        if collect_losses:
            row["sample_losses"] = sample_losses
        return row

    @torch.no_grad()
    def predict_proba(self, client_id, split):
        """Return (probabilities [n, K], labels [n]) for the deployed pair.

        ``model_for`` switches the copy to eval(), which merges the LoRA factors.
        The merged weights realise the same function, so these probabilities match
        the evaluation path; the trade-off is that no adapter gradient is available.
        """
        if split not in ("monitor", "final"):
            raise ValueError(f"Unknown evaluation split: {split}")
        view = self.views[client_id]
        loader = view.monitor_loader if split == "monitor" else view.final_loader
        model = self.model_for(client_id).to(self.device)
        model.eval()
        probs, labels = [], []
        for inputs, targets in loader:
            logits = model(inputs.to(self.device))
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            labels.append(targets.cpu().numpy())
        if not probs:
            return np.empty((0, self.num_classes), dtype=np.float32), np.empty(0, dtype=np.int64)
        return np.concatenate(probs, axis=0), np.concatenate(labels, axis=0)

    def evaluate(self, split):
        rows = [self.evaluate_client(cid, split) for cid in self.clients]
        return rows, summarize_clients(rows)


def summarize_clients(rows):
    """Pool raw counts for sample-level metrics; give each measured client one vote for tails."""
    measured = [row for row in rows if row["n"]]
    n = sum(row["n"] for row in measured)
    if n == 0:
        raise ValueError("No held-out samples in the selected roster")
    confusion = np.sum([row["confusion"] for row in measured], axis=0)
    support = confusion.sum(0) + confusion.sum(1)
    f1 = 2 * np.diag(confusion) / np.maximum(support, 1)
    accuracies = [row["accuracy"] for row in measured]
    worst = min(measured, key=lambda row: row["accuracy"])
    domains = defaultdict(list)
    for row in measured:
        for domain, totals in (row.get("domain_totals") or {row["domain"]: row}).items():
            domains[domain].append(totals)
    by_domain = {}
    for domain, members in domains.items():
        count = sum(r["n"] for r in members)
        by_domain[domain] = dict(n=count, accuracy=sum(r["correct"] for r in members) / count,
                                 loss=sum(r["loss_sum"] for r in members) / count)
    return dict(n=n, accuracy=sum(r["correct"] for r in measured) / n,
                loss=sum(r["loss_sum"] for r in measured) / n, macro_f1=float(f1.mean()),
                client_mean_accuracy=float(np.mean(accuracies)), p10_accuracy=float(np.quantile(accuracies, 0.1)),
                worst_client=worst["client_id"], worst_n=worst["n"], worst_accuracy=worst["accuracy"],
                measured_clients=len(measured), by_domain=by_domain)
