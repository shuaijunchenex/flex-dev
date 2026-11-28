from __future__ import annotations

from functools import partial

from torch.utils.data import DataLoader, Dataset

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from ..dataset_loader_util import DatasetLoaderUtil
from ._hf_dataset import load_hf_dataset

"""
Dataset loader for CoLA (GLUE).

Backed by the HuggingFace `datasets` GLUE `cola` configuration instead of the
removed `torchtext.datasets.CoLA` API (torchtext's final release only supports
PyTorch 2.3, while this project runs PyTorch 2.6+).

HF split mapping:
  train_split="train"      -> train       (8 551 sentences)
  test_split="dev"         -> validation  (1 043 labelled sentences)
  test_split="validation"  -> validation
  test_split="test"        -> test        (1 063 sentences, labels are -1)

Each sample dict: {"sentence": str, "label": int (0/1), "idx": int}
0 = unacceptable, 1 = acceptable.
"""

# HF split alias: yaml value -> HF datasets split name
_SPLIT_ALIAS = {
    "train": "train",
    "dev": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
}


class _HFListDataset(Dataset):
    """Wraps a HuggingFace Dataset as a list-backed PyTorch Dataset so it can be
    iterated multiple times (unlike IterableDataset) and pickled by workers."""

    def __init__(self, hf_dataset):
        self._data = [(row["sentence"], row["label"]) for row in hf_dataset]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class DatasetLoader_CoLA(DatasetLoader):
    def __init__(self):
        super().__init__()

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        # args.root is resolved to an absolute project path by DatasetLoaderArgs;
        # args.get("root") would return the raw relative ".dataset" value.
        root = getattr(args, "root", ".dataset")
        is_download = args.get("is_download", True)
        batch_size = args.get("batch_size", 32)
        test_batch_size = args.get("test_batch_size", None) or batch_size
        shuffle = args.get("shuffle", True)
        num_workers = args.get("num_workers", 0)
        requested_train_split = args.get("train_split", "train")
        requested_test_split = args.get("test_split", "validation")

        hf_train_split = _SPLIT_ALIAS.get(requested_train_split, requested_train_split)
        hf_test_split = _SPLIT_ALIAS.get(requested_test_split, "validation")

        # Pinned revision + shared cache layout used by the other repaired loaders.
        # Offline mode cannot resolve a pinned revision through the Hub, so fall
        # back to reading the prepared Arrow files directly from the same cache.
        try:
            ds = load_hf_dataset("nyu-mll/glue", root, is_download, name="cola")
        except Exception:
            ds = load_hf_dataset("nyu-mll/glue", root, False, name="cola")

        train_raw = ds[hf_train_split]
        test_raw = ds[hf_test_split]

        self._dataset = _HFListDataset(train_raw)
        self._test_dataset = _HFListDataset(test_raw)

        self.train_label_distribution = DatasetLoaderUtil.count_label_distribution(
            self._dataset, "CoLA-Train")
        self.test_label_distribution = DatasetLoaderUtil.count_label_distribution(
            self._test_dataset, "CoLA-Validation")

        hf_tokenizer = getattr(args, "tokenizer", None)
        if hf_tokenizer is not None:
            args.vocab_size = getattr(hf_tokenizer, "vocab_size", None)

        collate = partial(
            DatasetLoaderUtil.text_collate_fn_hf,
            hf_tokenizer=hf_tokenizer,
            max_len=args.get("max_len", 256),
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
