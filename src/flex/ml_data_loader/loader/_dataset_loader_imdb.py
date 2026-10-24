from __future__ import annotations

# import json
# import os
from functools import partial
# from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from ..dataset_loader_util import DatasetLoaderUtil
from ._hf_dataset import load_hf_dataset

"""
Dataset loader for IMDB (binary sentiment, 25k train / 25k test).
Uses HuggingFace `datasets` library (torchtext IMDB API is deprecated).

HF dataset: "stanfordnlp/imdb"
  ds["train"]  → 25 000 samples  {"text": str, "label": int (0=neg, 1=pos)}
  ds["test"]   → 25 000 samples
"""


# Previous per-loader HF cache implementation; shared by all five loaders below.
# def _load_imdb(root: str, is_download: bool = True):
#     from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset  # type: ignore
#
#     cache_dir = os.path.join(root, "hf_datasets")
#     if is_download:
#         return load_dataset("stanfordnlp/imdb", cache_dir=cache_dir)
#
#     # Read prepared Arrow files directly: load_dataset may still request Hub metadata.
#     cached_versions = [
#         path.parent
#         for name in ("stanfordnlp___imdb", "imdb")
#         for path in (Path(cache_dir) / name).glob("*/*/*/dataset_info.json")
#     ]
#     if not cached_versions:
#         raise FileNotFoundError(f"IMDB cache is missing in {cache_dir}. Enable is_download to prepare it.")
#     if len(cached_versions) != 1:
#         raise RuntimeError(f"Multiple IMDB cache versions found in {cache_dir}; select one before offline loading.")
#
#     info = json.loads((cached_versions[0] / "dataset_info.json").read_text(encoding="utf-8"))
#     splits = {}
#     for split in ("train", "test"):
#         paths = sorted(cached_versions[0].glob(f"imdb-{split}*.arrow"))
#         if not paths:
#             raise FileNotFoundError(f"IMDB {split} cache is missing in {cached_versions[0]}")
#         shards = [Dataset.from_file(str(path)) for path in paths]
#         splits[split] = concatenate_datasets(shards) if len(shards) > 1 else shards[0]
#         expected = info["splits"][split]["num_examples"]
#         if len(splits[split]) != expected:
#             raise RuntimeError(f"Incomplete IMDB {split} cache: expected {expected} samples, found {len(splits[split])}.")
#     return DatasetDict(splits)


def _load_imdb(root: str, is_download: bool = True):
    return load_hf_dataset("stanfordnlp/imdb", root, is_download)


class _HFIMDBDataset(Dataset):
    """Wraps HuggingFace IMDB split as a plain list-backed PyTorch Dataset.
    Each item: (text: str, label: int)  where 0=neg, 1=pos.
    """
    def __init__(self, hf_dataset):
        self._data = [(row["text"], int(row["label"])) for row in hf_dataset]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class DatasetLoader_Imdb(DatasetLoader):
    def __init__(self):
        super().__init__()

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        # Previous loader:
        # from datasets import load_dataset  # type: ignore

        root            = getattr(args, "root", ".dataset")
        batch_size      = getattr(args, "batch_size", 32)
        test_batch_size = getattr(args, "test_batch_size", None) or batch_size
        shuffle         = getattr(args, "shuffle", True)
        num_workers     = getattr(args, "num_workers", 0)

        # cache_dir = os.path.join(root, "hf_datasets")
        # ds = load_dataset("imdb", cache_dir=cache_dir)
        ds = _load_imdb(root, getattr(args, "is_download", True))

        self._dataset      = _HFIMDBDataset(ds["train"])
        self._test_dataset = _HFIMDBDataset(ds["test"])
        self.tokenizer = getattr(args, "tokenizer", None)
        self.vocab = getattr(args, "vocab", None)

        self.train_label_distribution = DatasetLoaderUtil.count_label_distribution(
            self._dataset, "IMDB-Train")
        self.test_label_distribution  = DatasetLoaderUtil.count_label_distribution(
            self._test_dataset, "IMDB-Test")

        max_len = getattr(args, "max_len", 256)
        if self._is_hf_tokenizer(self.tokenizer):
            args.vocab_size = getattr(self.tokenizer, "vocab_size", None)
            collate = partial(
                DatasetLoaderUtil.text_collate_fn_hf,
                hf_tokenizer=self.tokenizer,
                max_len=max_len,
                normalize_int_labels=False,
                tuple_format="text_label",  # each sample: (text, label)
            )
        else:
            collate = self._collate_basic

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

    @staticmethod
    def _is_hf_tokenizer(tokenizer) -> bool:
        return tokenizer is not None and hasattr(tokenizer, "model_input_names")

    def _collate_basic(self, batch):
        if self.tokenizer is None:
            raise ValueError("IMDB basic tokenizer path requires tokenizer.")
        if self.vocab is None:
            raise ValueError("IMDB basic tokenizer path requires vocab.")
        max_len = getattr(self._args, "max_len", 256)
        return DatasetLoaderUtil.text_collate_fn(
            batch,
            tokenizer=self.tokenizer,
            vocab=self.vocab,
            max_len=max_len,
            normalize_int_labels=False,
            tuple_format="text_label",
        )

    def get_dataset(self) -> Dataset:
        if self._data_loader is not None:
            return self._data_loader.dataset
        raise ValueError("ERROR: DatasetLoader's data_loader is None.")
