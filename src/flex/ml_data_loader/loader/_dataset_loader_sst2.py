from __future__ import annotations

import os
from functools import partial
from torch.utils.data import DataLoader, Dataset

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from ..dataset_loader_util import DatasetLoaderUtil

"""
Dataset loader for SST-2 (GLUE).
Uses HuggingFace `datasets` library (replaces the removed torchtext SST2 API).

HF split mapping:
  train_split="train"  → ds["train"]       (67 349 samples)
  test_split="dev"     → ds["validation"]  (872 labelled samples)
  test_split="test"    → ds["test"]        (1 821 unlabelled samples)

Each sample dict: {"sentence": str, "label": int (0/1), "idx": int}
"""

# HF split alias: yaml value → HF datasets split name
_SPLIT_ALIAS = {
    "train":      "train",
    "dev":        "validation",
    "validation": "validation",
    "test":       "test",
}


class _HFListDataset(Dataset):
    """Wraps a HuggingFace Dataset as a plain list-backed PyTorch Dataset so it
    can be iterated multiple times (unlike IterableDataset)."""
    def __init__(self, hf_dataset):
        self._data = [(row["sentence"], row["label"]) for row in hf_dataset]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class DatasetLoader_SST2(DatasetLoader):
    def __init__(self):
        super().__init__()

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        from datasets import load_dataset  # type: ignore

        root          = getattr(args, "root", ".dataset")
        batch_size    = getattr(args, "batch_size", 32)
        test_batch_size = getattr(args, "test_batch_size", None) or batch_size
        shuffle       = getattr(args, "shuffle", True)
        num_workers   = getattr(args, "num_workers", 0)
        train_split   = getattr(args, "train_split", "train")
        test_split    = getattr(args, "test_split", "dev")

        # Resolve cache dir: store alongside other datasets so offline reuse works
        cache_dir = os.path.join(root, "hf_datasets")

        hf_train_split = _SPLIT_ALIAS.get(train_split, train_split)
        hf_test_split  = _SPLIT_ALIAS.get(test_split,  "validation")

        ds = load_dataset("glue", "sst2", cache_dir=cache_dir)

        train_raw = ds[hf_train_split]
        test_raw  = ds[hf_test_split]

        # For the unlabelled "test" split, labels are -1; skip label distribution
        if hf_test_split == "test":
            test_raw = ds.get("validation", test_raw)  # fall back to validation for eval

        self._dataset      = _HFListDataset(train_raw)
        self._test_dataset = _HFListDataset(test_raw)

        self.train_label_distribution = DatasetLoaderUtil.count_label_distribution(
            self._dataset, "SST2-Train")
        self.test_label_distribution  = DatasetLoaderUtil.count_label_distribution(
            self._test_dataset, "SST2-Val")

        hf_tokenizer = getattr(args, "tokenizer", None)
        if hf_tokenizer is not None:
            args.vocab_size = getattr(hf_tokenizer, "vocab_size", None)

        collate = partial(
            DatasetLoaderUtil.text_collate_fn_hf,
            hf_tokenizer=hf_tokenizer,
            max_len=getattr(args, "max_len", 256),
            normalize_int_labels=False,
            tuple_format="text_label",  # each sample: (sentence, label)
        )

        self._data_loader = DataLoader(
            self._dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate,
        )

        self._test_data_loader = DataLoader(
            self._test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
        )

        self.task_type = "nlp"
        self.data_sample_num = len(self._dataset)
        return

    def get_dataset(self):
        if self._data_loader is not None:
            return self._data_loader.dataset
        raise ValueError("ERROR: DatasetLoader's data_loader is None.")
