"""Keep QMNIST's extra training split out of evaluation, without downloads."""
from types import SimpleNamespace

from torch.utils.data import Dataset
from flex.ml_data_loader.loader import _dataset_loader_qmnist as qmnist_loader


def test_qmnist_train_and_test_are_disjoint(monkeypatch):
    calls = []

    class FakeQMNIST(Dataset):
        sizes = {"train": 60000, "test50k": 50000, "test10k": 10000}

        def __init__(self, *, root, what, transform, download, compat):
            self.what = what
            calls.append((what, transform, download))

        def __len__(self):
            return self.sizes[self.what]

        def __getitem__(self, index):
            if index >= len(self):
                raise IndexError(index)
            return (self.what, index), index % 10

    monkeypatch.setattr(qmnist_loader.datasets, "QMNIST", FakeQMNIST)
    train_transform, test_transform = object(), object()
    args = SimpleNamespace(
        root="unused-offline", transform=train_transform,
        test_transform=test_transform, is_download=False,
        batch_size=64, test_batch_size=32, shuffle=False, num_workers=0,
    )
    loader = qmnist_loader.DatasetLoader_Qmnist()
    loader._create_inner(args)
    assert calls == [
        ("train", train_transform, False),
        ("test50k", train_transform, False),
        ("test10k", test_transform, False),
    ]
    assert loader.data_sample_num == len(loader._dataset) == 110000
    assert len(loader._test_dataset) == 10000
    train_ids = {loader._dataset[i][0] for i in range(len(loader._dataset))}
    test_ids = {loader._test_dataset[i][0] for i in range(len(loader._test_dataset))}
    assert train_ids.isdisjoint(test_ids)
    assert {split for split, _ in train_ids} == {"train", "test50k"}
    assert {split for split, _ in test_ids} == {"test10k"}
    assert loader._data_loader.batch_size == 64
    assert loader._test_data_loader.batch_size == 32
