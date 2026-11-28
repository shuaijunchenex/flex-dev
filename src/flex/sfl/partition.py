"""Disjoint training and deployment views for the split Office experiment."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from flex.fl_algorithms.noniid.noniid_data_generator import NoniidDataGenerator
from flex.ml_data_loader.dataset_loader import DatasetLoader
from flex.ml_data_loader.image_benchmark_data import (
    ImageBenchmarkDataset, ImageCatalog, content_hash, image_transform,
)
from flex.ml_data_loader.image_epoch import iter_image_epoch


def _label_skew_counts(client_sizes, class_totals, label_order):
    """Intersect client quotas with the ordered class blocks. Columns are canonical label IDs."""
    edges = np.cumsum([0, *(class_totals[label] for label in label_order)])
    counts = np.zeros((len(client_sizes), len(class_totals)), dtype=int)
    start = 0
    for row, size in enumerate(client_sizes):
        stop = start + size
        counts[row, label_order] = np.maximum(0, np.minimum(stop, edges[1:]) - np.maximum(start, edges[:-1]))
        start = stop
    return counts


def _allocate_split_ids(catalog, sample_ids, node_ids, counts):
    """Allocate indices through FLEX's Non-IID generator, retaining empty holdout rows."""
    dataset = ImageBenchmarkDataset(catalog, sample_ids, transform=None)
    counts = np.asarray(counts)
    totals = np.bincount(dataset.targets, minlength=len(catalog.classes))
    if (counts.shape != (len(node_ids), len(totals)) or not np.issubdtype(counts.dtype, np.integer)
            or (counts < 0).any() or not np.array_equal(counts.sum(axis=0), totals)):
        raise ValueError("Office allocation must cover each class exactly once across the configured clients")
    result = {cid: [] for cid in node_ids}
    if sample_ids:
        allocator = NoniidDataGenerator(DataLoader(dataset, batch_size=64))
        subsets = allocator.generate_noniid_data(distribution="custom", verify_allocate=False,
                                                 data_volum_list=counts[:, allocator.labels_sorted].tolist())
        nonempty = [cid for cid, count in zip(node_ids, counts.sum(axis=1)) if count]
        for cid, subset in zip(nonempty, subsets, strict=True):
            result[cid] = [sample_ids[i] for i in sorted(subset.indices)]
    return result


def _allocate_office_splits(catalog, clients, rng, allocation, train_counts):
    node_order = list(clients)
    rng.shuffle(node_order)
    label_order = rng.permutation(len(catalog.classes)).tolist()
    for split in ("train", "monitor", "final"):
        if split != "train" and allocation == "domain":
            continue
        by_label = defaultdict(list)
        for row in clients.values():
            for sample_id in row[split]:
                by_label[catalog.records[sample_id].label].append(sample_id)
        pool = []
        for label in label_order:
            ids = sorted(by_label[label])
            rng.shuffle(ids)
            pool.extend(ids)
        if split == "train" and train_counts is not None:
            node_ids, counts = list(clients), train_counts
        else:
            node_ids = node_order
            counts = _label_skew_counts([len(clients[cid][split]) for cid in node_ids],
                                        [len(by_label[i]) for i in range(len(label_order))], label_order)
        for cid, ids in _allocate_split_ids(catalog, pool, node_ids, counts).items():
            clients[cid][split] = ids


def build_sfl_manifest(catalog: ImageCatalog, base_manifest: dict, seed: int = 42,
                       allocation: str = "domain", *, data_distribution=None) -> dict:
    """Partition the native Office holdout; optionally remap all splits by label."""
    if catalog.name != "office_caltech10" or base_manifest["source_sha256"] != catalog.source_sha256:
        raise ValueError("Expected an Office-Caltech10 manifest matching the image catalog")
    if base_manifest["splits"]["monitor"] != base_manifest["splits"]["final"]:
        raise ValueError("Expected the original Office shared holdout")

    clients = {}
    domain_clients = defaultdict(list)
    for row in base_manifest["clients"]:
        node_id = f"client.{row['id']}"
        clients[node_id] = dict(shard_id=row["id"], domain=row["domain"],
                                train=list(row["sample_ids"]), monitor=[], final=[])
        domain_clients[row["domain"]].append(node_id)

    groups = defaultdict(list)
    for sample_id in base_manifest["splits"]["monitor"]:
        sample = catalog.records[sample_id]
        groups[(sample.domain, sample.label)].append(sample_id)

    rng = np.random.default_rng(seed)
    offsets = defaultdict(int)
    for (domain, _), ids in sorted(groups.items()):
        ids.sort()
        rng.shuffle(ids)
        nodes = domain_clients[domain]
        # Rotate across class boundaries so small groups do not all go to client 1.
        start = offsets[domain]
        for i, sample_id in enumerate(ids, start):
            split = "monitor" if i % 2 == 0 else "final"
            clients[nodes[(i // 2) % len(nodes)]][split].append(sample_id)
        offsets[domain] += len(ids)

    if allocation not in ("domain", "label_skew"):
        raise ValueError(f"Unknown Office allocation: {allocation}")
    train_counts = None
    if data_distribution is not None:
        matrix = NoniidDataGenerator.resolve_distribution(distribution_config=data_distribution)
        train_counts = [list(row) for row in matrix]
    if allocation == "label_skew" or train_counts is not None:
        _allocate_office_splits(catalog, clients, rng, allocation, train_counts)
        for row in clients.values():
            domains = {catalog.records[s].domain for split in ("train", "monitor", "final") for s in row[split]}
            row["domain"] = next(iter(domains)) if train_counts is not None and len(domains) == 1 else "mixed"
    manifest = dict(version=2, dataset=catalog.name, source_sha256=catalog.source_sha256, allocation=allocation,
                    base_manifest_sha256=base_manifest["manifest_sha256"], split_seed=seed,
                    classes=list(catalog.classes), clients=clients)
    if train_counts is not None:
        manifest["data_distribution"] = dict(use=data_distribution.get("use", "mnist_lt"), train_counts=train_counts)
    manifest["splits"] = {
        split: sorted(s for row in clients.values() for s in row[split])
        for split in ("train", "monitor", "final")
    }
    validate_sfl_manifest(manifest, catalog)
    manifest["manifest_sha256"] = content_hash(manifest)
    return manifest


def validate_sfl_manifest(manifest: dict, catalog: ImageCatalog) -> None:
    assigned = []
    for split in ("train", "monitor", "final"):
        ids = [s for row in manifest["clients"].values() for s in row[split]]
        if sorted(ids) != manifest["splits"][split]:
            raise ValueError(f"Client assignments differ from the {split} split")
        assigned.extend(ids)
    if len(assigned) != len(set(assigned)) or set(assigned) != set(catalog.records):
        raise ValueError("Each Office sample must belong to exactly one client and split")
    for node_id, row in manifest["clients"].items():
        if node_id != f"client.{row['shard_id']}" or not row["train"]:
            raise ValueError(f"Invalid training shard for {node_id}")
        if manifest["allocation"] == "domain" and "data_distribution" not in manifest and any(
               catalog.records[s].domain != row["domain"] for split in ("train", "monitor", "final")
               for s in row[split]):
            raise ValueError(f"Unexpected sample domain for {node_id}")


@dataclass
class ClientViews:
    client_id: str
    domain: str
    train_loader: DataLoader
    monitor_loader: DataLoader
    final_loader: DataLoader
    seed: int

    @property
    def shard_size(self):
        return len(self.train_loader.dataset)

    @property
    def data_sample_num(self):
        return self.shard_size

    def train_batches(self, round_id, local_epoch):
        seed_parts = (self.seed, self.client_id, round_id, local_epoch)
        return iter_image_epoch(self.train_loader, seed_parts=seed_parts)


def _subsample_ids(ids, node_id, spec):
    """Keep a stable, seed-dependent fraction of one client's training ids in order."""
    ratio = float(spec["ratio"])
    if not 0 < ratio <= 1:
        raise ValueError("train_subsample ratio must be in (0, 1]")
    keep = max(1, int(round(ratio * len(ids))))
    if keep >= len(ids):
        return list(ids)
    digest = hashlib.sha256(f'{int(spec["seed"])}:{node_id}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    chosen = rng.choice(len(ids), size=keep, replace=False)
    return [ids[i] for i in sorted(chosen.tolist())]


def build_client_views(catalog, manifest, node_ids, seed, batch_size, *, sample_limit=None,
                       train_subsample=None):
    """Build views for the actual roster; sample_limit is for bounded smoke runs."""
    views = {}
    for node_id in node_ids:
        row = manifest["clients"][node_id]
        datasets = {}
        for split in ("train", "monitor", "final"):
            ids = row[split]
            if sample_limit is not None and len(ids) > sample_limit:
                # Spread smoke examples across the full ordered split.
                ids = [ids[i] for i in np.linspace(0, len(ids) - 1, sample_limit, dtype=int)]
            if split == "train" and train_subsample is not None:
                ids = _subsample_ids(ids, node_id, train_subsample)
            datasets[split] = ImageBenchmarkDataset(catalog, ids, image_transform(catalog.name, split == "train"))
        loaders = {split: DatasetLoader.create_loader(dataset, batch_size, shuffle=split == "train",
                                                             num_workers=0, generator=torch.Generator())
                   for split, dataset in datasets.items()}
        views[node_id] = ClientViews(node_id, row["domain"], loaders["train"],
                                     loaders["monitor"], loaders["final"], seed)
    return views
