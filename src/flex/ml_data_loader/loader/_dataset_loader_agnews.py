from __future__ import annotations

# import csv
# import os
from functools import partial
from torch.utils.data import DataLoader
# from torchtext.datasets import AG_NEWS
from torch.utils.data import IterDataPipe
# from torch.utils.data.datapipes.iter import FileOpener, IterableWrapper
# from torchvision.datasets.utils import check_integrity, download_url
from ._hf_dataset import load_hf_dataset
from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from ..dataset_loader_util import DatasetLoaderUtil

"""
Dataset loader for AG_NEWS (train/test splits)
"""


# _AGNEWS_URLS = {
#     "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
#     "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
# }
# _AGNEWS_MD5 = {
#     "train": "b1a00f826fdfbd249f79597b59e1dc12",
#     "test": "d52ea96a97a2d943681189a97654912d",
# }


# class _AgnewsRows(IterDataPipe):
#     def __init__(self, source_datapipe):
#         self.source_datapipe = source_datapipe
#
#     def __iter__(self):
#         for _, stream in self.source_datapipe:
#             try:
#                 yield from csv.reader(stream)
#             finally:
#                 stream.close()
class _AgnewsRows(IterDataPipe):
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        yield from self.rows


# def _agnews_record(row):
#     return int(row[0]), " ".join(row[1:])
def _agnews_record(row):
    label = int(row["label"])
    if label not in (0, 1, 2, 3):
        raise ValueError(f"AGNews label must be in 0..3, found {label}.")
    # Retain the existing 1..4 sample protocol and collator mapping.
    return label + 1, row["text"]


# def _load_agnews_split(root: str, split_name: str, is_download: bool = True):
#     if split_name not in _AGNEWS_URLS:
#         raise ValueError(f"Unsupported AGNews split: {split_name!r}")
#     cache_root = os.path.join(root, "datasets", "AG_NEWS")
#     filename = f"{split_name}.csv"
#     path = os.path.join(cache_root, filename)
#     if is_download:
#         download_url(_AGNEWS_URLS[split_name], cache_root, filename=filename, md5=_AGNEWS_MD5[split_name])
#     elif not check_integrity(path, _AGNEWS_MD5[split_name]):
#         raise FileNotFoundError(f"AGNews cache is missing or invalid: {path}. Enable is_download to prepare it.")
#
#     # Same raw CSV, labels 1..4, text join and shuffle/sharding graph as torchtext.
#     files = FileOpener(IterableWrapper([path]), encoding="utf-8")
#     return _AgnewsRows(files).map(_agnews_record).shuffle().set_shuffle(False).sharding_filter()
def _load_agnews_split(root: str, split_name: str, is_download: bool = True, cached=None):
    if split_name not in ("train", "test"):
        raise ValueError(f"Unsupported AGNews split: {split_name!r}")
    if cached is None:
        cached = load_hf_dataset("fancyzhx/ag_news", root, is_download)
    # Keep an unknown-length iterable and the original shuffle/sharding graph.
    return _AgnewsRows(cached[split_name]).map(_agnews_record).shuffle().set_shuffle(False).sharding_filter()


class DatasetLoader_Agnews(DatasetLoader):
    def __init__(self):
        super().__init__()

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        root = getattr(args, "root")
        is_download = getattr(args, "is_download", True)
        batch_size = getattr(args, "batch_size", 32)
        test_batch_size = getattr(args, "test_batch_size", None) or batch_size
        shuffle = getattr(args, "shuffle", True)
        num_workers = getattr(args, "num_workers", 0)
        train_split = getattr(args, "train_split", "train") or getattr(args, "split", "train")
        test_split = getattr(args, "test_split", "test")

        # def _load_split(split_name: str):
        #     """
        #     Compatible loader for both old and new torchtext AG_NEWS APIs.
        #     - New (>=0.10): AG_NEWS(root=root, split='train')
        #     - Old:          AG_NEWS(root=root) returns (train_iter, test_iter)
        #     """
        #     try:
        #         return AG_NEWS(root=root, split=split_name)
        #     except TypeError:
        #         # Old API: returns a tuple of (train_iter, test_iter)
        #         splits = AG_NEWS(root=root)
        #         mapping = {"train": 0, "test": 1}
        #         return splits[mapping.get(split_name, 0)]
        # def _load_split(split_name: str):
        #     return _load_agnews_split(root, split_name, is_download)
        cached = load_hf_dataset("fancyzhx/ag_news", root, is_download)

        def _load_split(split_name: str):
            return _load_agnews_split(root, split_name, is_download, cached)

        # if is_download:
        #     try:
        #         _load_split(train_split)
        #     except Exception:
        #         pass

        # Use temporary datasets for counting distribution to avoid consuming the main dataset (IterableDataset)
        _temp_train_dataset = _load_split(train_split)
        _temp_test_dataset  = _load_split(test_split)

        train_label_dist = DatasetLoaderUtil.count_label_distribution(_temp_train_dataset, "AGNews-Train")
        test_label_dist  = DatasetLoaderUtil.count_label_distribution(_temp_test_dataset,  "AGNews-Test")

        self._dataset      = _load_split(train_split)
        self._test_dataset = _load_split(test_split)
        
        self.train_label_distribution = dict(train_label_dist)
        self.test_label_distribution = dict(test_label_dist)

        hf_tokenizer = getattr(args, "tokenizer")
        args.vocab_size = getattr(hf_tokenizer, "vocab_size", None)

        collate = partial(
            DatasetLoaderUtil.text_collate_fn_hf,
            hf_tokenizer=hf_tokenizer,
            max_len=getattr(args, "max_len", 256),
            label_map={1: 0, 2: 1, 3: 2, 4: 3},
            normalize_int_labels=True,
        )

        self._data_loader = DataLoader(
            self._dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate,
        )

        self.task_type = "nlp"
        try:
            self.data_sample_num = len(self._dataset)
        except Exception:
            self.data_sample_num = None

        self._test_data_loader = DataLoader(
            self._test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
        )

        return

    def get_dataset(self):
        if self._data_loader is not None:
            return self._data_loader.dataset
        raise ValueError("ERROR: DatasetLoader's data_loader is None.")
