"""Regression tests for indexed image partitioning; no dataset downloads."""
from collections import Counter

import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.datasets import QMNIST

from flex.fl_algorithms.noniid.noniid_data_generator import NoniidDataGenerator
from flex.ml_data_loader.dataset_loader_args import DatasetLoaderArgs
from flex.ml_data_loader.loader._dataset_loader_custom import DatasetLoader_Custom


class ChangingImages(Dataset):
    def __init__(self, labels=(0, 1, 0, 1, 0, 1), metadata="targets"):
        self._labels = list(labels)
        self.calls = 0
        setattr(self, metadata, np.asarray(labels))

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        self.calls += 1
        # A fresh view on every read makes the regression deterministic.
        return torch.tensor([index, self.calls]), self._labels[index]


def partition(loader, matrix):
    return NoniidDataGenerator(loader).generate_noniid_data(
        distribution="custom", data_volum_list=matrix, verify_allocate=False
    )


@pytest.mark.parametrize("metadata", ["targets", "labels"])
def test_client_reads_refresh_transforms_without_changing_allocation(metadata):
    source = ChangingImages(metadata=metadata)
    clients = partition(DataLoader(source, batch_size=2), [[2, 1], [1, 2]])
    assert source.calls == 0
    assert all(isinstance(client, Subset) and client.dataset is source for client in clients)
    assert Counter(clients[0].indices + clients[1].indices) == Counter(range(6))
    assert [[source._labels[i] for i in client.indices] for client in clients] == [
        [0, 0, 1], [0, 1, 1]
    ]

    # Use the actual client loader wrapper used by the FL entry.
    wrapper = DatasetLoader_Custom()
    wrapper._create_inner(DatasetLoaderArgs({
        "dataset": clients[0], "batch_size": 3, "shuffle": False, "num_workers": 0
    }))
    first, first_labels = next(iter(wrapper.data_loader))
    second, second_labels = next(iter(wrapper.data_loader))
    assert source.calls == 6
    assert torch.equal(first[:, 0], second[:, 0])
    assert not torch.equal(first[:, 1], second[:, 1])
    assert torch.equal(first_labels, second_labels)


def test_subset_concat_and_qmnist_annotation_labels():
    # Construct torchvision's real QMNIST object without downloading its files.
    qmnist = QMNIST.__new__(QMNIST)
    qmnist.compat = True
    qmnist.targets = torch.tensor([[0, 91, 92], [1, 93, 94]])
    qmnist.data = torch.zeros(2, 2, 2, dtype=torch.uint8)
    qmnist.transform = None
    qmnist.target_transform = None
    other = ChangingImages(labels=(1, 0), metadata="labels")
    source = Subset(ConcatDataset([qmnist, other]), [3, 0, 2, 1])
    generator = NoniidDataGenerator(DataLoader(source, batch_size=2))
    assert generator.y_train.tolist() == [0, 0, 1, 1]
    clients = generator.generate_noniid_data(
        distribution="custom", data_volum_list=[[1, 1], [1, 1]], verify_allocate=False
    )
    assert [client.indices for client in clients] == [[0, 2], [1, 3]]
    assert other.calls == 0
    qmnist.compat = False
    assert NoniidDataGenerator._dataset_labels(qmnist) is None


def test_batch_sampler_order_and_repeated_indices_are_preserved():
    source = ChangingImages()
    loader = DataLoader(source, batch_sampler=[[4, 1], [4, 3]])
    clients = partition(loader, [[2, 2]])
    assert clients[0].indices == [4, 4, 1, 3]
    assert source.calls == 0


def test_drop_last_and_seeded_sampler_are_preserved():
    source = ChangingImages(labels=(0, 1, 0, 1, 0, 1, 0))
    clients = partition(DataLoader(source, batch_size=2, drop_last=True), [[3, 3]])
    assert clients[0].indices == [0, 2, 4, 1, 3, 5]
    results = []
    for _ in range(2):
        loader = DataLoader(source, batch_size=2, shuffle=True,
                            generator=torch.Generator().manual_seed(7))
        results.append(partition(loader, [[2, 1], [2, 2]]))
    assert [client.indices for client in results[0]] == [client.indices for client in results[1]]


class PairText(Dataset):
    targets = [0, 1, 0, 1]

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return torch.arange(index + 1) + 10, self.targets[index]


def text_collate(batch):
    width = max(len(tokens) for tokens, _ in batch)
    ids = torch.full((len(batch), width), 99, dtype=torch.long)
    mask = torch.zeros_like(ids)
    segments = torch.zeros_like(ids)
    for row, (tokens, _) in enumerate(batch):
        ids[row, :len(tokens)] = tokens
        mask[row, :len(tokens)] = 1
        segments[row, :len(tokens)] = 1
    return {"input_ids": ids, "attention_mask": mask, "token_type_ids": segments}, torch.tensor(
        [label for _, label in batch]
    )


def test_text_custom_collate_still_preserves_encoded_fields_and_padding():
    generator = NoniidDataGenerator(DataLoader(PairText(), batch_size=2, collate_fn=text_collate))
    assert generator._indexed_dataset is None
    clients = generator.generate_noniid_data(
        distribution="custom", data_volum_list=[[1, 1], [1, 1]], verify_allocate=False
    )
    sample, label = clients[0][0]
    assert label == 0
    assert sample["input_ids"].tolist() == [10, 99, 99, 99]
    assert sample["attention_mask"].tolist() == [1, 0, 0, 0]
    assert sample["token_type_ids"].tolist() == [1, 0, 0, 0]


class RemappedLabels(ChangingImages):
    target_transform = staticmethod(lambda label: label + 5)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, self.target_transform(label)


def test_target_transform_uses_actual_loader_labels():
    source = RemappedLabels()
    generator = NoniidDataGenerator(DataLoader(source, batch_size=2))
    assert generator._indexed_dataset is None
    assert generator.labels_sorted == [5, 6]
