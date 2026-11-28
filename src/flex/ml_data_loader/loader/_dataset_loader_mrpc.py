from __future__ import annotations

# import csv
# import os
from functools import partial
from torch.utils.data import DataLoader
# from torchtext.datasets import MRPC
from torch.utils.data import IterDataPipe
# from torch.utils.data.datapipes.iter import FileOpener, IterableWrapper
# from torchvision.datasets.utils import check_integrity, download_url
from ._hf_dataset import load_hf_dataset

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from ..dataset_loader_util import DatasetLoaderUtil
from ...ml_algorithms.tokenizer_builder import TokenizerBuilder

"""
Dataset loader for MRPC (GLUE)
"""


# _MRPC_URLS = {
#     "train": "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt",
#     "test": "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt",
# }
# _MRPC_MD5 = {
#     "train": "793daf7b6224281e75fe61c1f80afe35",
#     "test": "e437fdddb92535b820fe8852e2df8a49",
# }


# class _MrpcRows(IterDataPipe):
#     def __init__(self, source_datapipe):
#         self.source_datapipe = source_datapipe
#
#     def __iter__(self):
#         for _, stream in self.source_datapipe:
#             try:
#                 next(stream, None)  # Header (including the optional UTF-8 BOM).
#                 yield from csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
#             finally:
#                 stream.close()
class _MrpcRows(IterDataPipe):
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        yield from self.rows


# def _mrpc_record(row):
#     return int(row[0]), row[3], row[4]
def _mrpc_record(row):
    return int(row["label"]), row["sentence1"], row["sentence2"]


# def _load_mrpc_split(root: str, split_name: str, is_download: bool = True):
#     if split_name not in _MRPC_URLS:
#         raise ValueError(f"Unsupported MRPC split: {split_name!r}")
#     cache_root = os.path.join(root, "datasets", "MRPC")
#     filename = os.path.basename(_MRPC_URLS[split_name])
#     path = os.path.join(cache_root, filename)
#     if is_download:
#         download_url(_MRPC_URLS[split_name], cache_root, filename=filename, md5=_MRPC_MD5[split_name])
#     elif not check_integrity(path, _MRPC_MD5[split_name]):
#         raise FileNotFoundError(f"MRPC cache is missing or invalid: {path}. Enable is_download to prepare it.")
#
#     # Preserve all 4076 training rows and 1725 labeled test rows in source order.
#     files = FileOpener(IterableWrapper([path]), encoding="utf-8")
#     return _MrpcRows(files).map(_mrpc_record).shuffle().set_shuffle(False).sharding_filter()
def _load_mrpc_split(root: str, split_name: str, is_download: bool = True, cached=None):
    if split_name not in ("train", "test"):
        raise ValueError(f"Unsupported MRPC split: {split_name!r}")
    if cached is None:
        cached = load_hf_dataset("nyu-mll/glue", root, is_download, name="mrpc")

    # GLUE separates 408 original training rows into validation; restore the
    # original 4076-row training set and its source order for existing budgets.
    rows = list(cached["train"]) + list(cached["validation"]) if split_name == "train" else list(cached["test"])
    rows.sort(key=lambda row: int(row["idx"]))
    expected = 4076 if split_name == "train" else 1725
    if [int(row["idx"]) for row in rows] != list(range(expected)):
        raise ValueError(f"MRPC {split_name} must contain unique original indices 0..{expected - 1}.")
    if any(int(row["label"]) not in (0, 1) for row in rows):
        raise ValueError(f"MRPC {split_name} must contain labeled binary examples.")
    return _MrpcRows(rows).map(_mrpc_record).shuffle().set_shuffle(False).sharding_filter()


class DatasetLoader_MRPC(DatasetLoader):
    def __init__(self):
        super().__init__()

    # def _warmup_download(self, root: str, splits: tuple[str, ...]):
    #     for sp in splits:
    #         try:
    #             # map common aliases to supported splits
    #             use_sp = "test" if sp in ("valid", "validation", "dev", "valid_matched", "valid_mismatched") else sp
    #             # it = iter(MRPC(root=root, split=use_sp))
    #             it = iter(_load_mrpc_split(root, use_sp))
    #             next(it)
    #         except StopIteration:
    #             continue
    #         except Exception:
    #             continue

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        root = getattr(args, "root")
        is_download = getattr(args, "is_download", True)
        batch_size = getattr(args, "batch_size", 32)
        test_batch_size = getattr(args, "test_batch_size", None) or batch_size
        shuffle = getattr(args, "shuffle", True)
        num_workers = getattr(args, "num_workers", 0)
        train_split = getattr(args, "train_split", "train")
        test_split = getattr(args, "test_split", "test")
        # map common aliases to torchtext MRPC-supported splits ('train', 'test')
        if test_split in ("valid", "validation", "dev"):
            test_split = "test"
        combine_fn = getattr(args, "text_pair_combine_fn", None)

        # if is_download:
        #     self._warmup_download(root, (train_split, test_split))
        cached = load_hf_dataset("nyu-mll/glue", root, is_download, name="mrpc")

        # Default: use torchtext MRPC dataset (iterable)
        # Read the same raw MRPC files without the torchtext/torchdata dependency.
        # Use Hugging Face data while retaining the original sample protocol.
        tokenizer = getattr(args, "tokenizer", None)
        use_hf = tokenizer is not None and (
            hasattr(tokenizer, "vocab_size") or hasattr(tokenizer, "pad_token_id") or hasattr(tokenizer, "__call__")
        )

        if use_hf:
            # Keep both sentences separate so the HF tokenizer creates proper
            # segment IDs instead of treating MRPC as a single sequence.
            # raw_train = MRPC(root=root, split=train_split)
            # raw_test = MRPC(root=root, split=test_split)
            # raw_train = _load_mrpc_split(root, train_split, is_download)
            # raw_test = _load_mrpc_split(root, test_split, is_download)
            raw_train = _load_mrpc_split(root, train_split, is_download, cached)
            raw_test = _load_mrpc_split(root, test_split, is_download, cached)

            def _normalize_pair(item, idx):
                # MRPC items can be tuple/list or dict
                if isinstance(item, dict):
                    s1 = item.get("sentence1") or item.get("text_a") or ""
                    s2 = item.get("sentence2") or item.get("text_b") or ""
                    lab = item.get("label")
                else:
                    # tuple/list: assume (label, sent1, sent2) or (label, sent1, sent2, idx)
                    if len(item) >= 3:
                        lab = item[0]
                        s1 = item[1]
                        s2 = item[2]
                    else:
                        lab = None
                        s1 = ""
                        s2 = ""
                return {
                    "label": lab,
                    "text_a": s1,
                    "text_b": s2,
                    "idx": idx,
                }

            train_list = list(raw_train)
            test_list = list(raw_test)
            self._dataset = [
                _normalize_pair(it, i) for i, it in enumerate(train_list)
            ]
            self._test_dataset = [
                _normalize_pair(it, i) for i, it in enumerate(test_list)
            ]

            train_label_dist = DatasetLoaderUtil.count_label_distribution(self._dataset, "MRPC-Train")
            test_label_dist = DatasetLoaderUtil.count_label_distribution(self._test_dataset, "MRPC-Test")

            self.train_label_distribution = dict(train_label_dist)
            self.test_label_distribution = dict(test_label_dist)

            args.vocab_size = getattr(tokenizer, "vocab_size", None)

            collate = partial(
                DatasetLoaderUtil.text_collate_fn_hf,
                hf_tokenizer=tokenizer,
                max_len=getattr(args, "max_len", 256),
            )

            self._data_loader = DataLoader(
                self._dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate,
            )
        else:
            # Legacy path using TokenizerBuilder and vocab
            # self._dataset = MRPC(root=root, split=train_split)
            # self._test_dataset = MRPC(root=root, split=test_split)
            # self._dataset = _load_mrpc_split(root, train_split, is_download)
            # self._test_dataset = _load_mrpc_split(root, test_split, is_download)
            self._dataset = _load_mrpc_split(root, train_split, is_download, cached)
            self._test_dataset = _load_mrpc_split(root, test_split, is_download, cached)

            train_label_dist = DatasetLoaderUtil.count_label_distribution(self._dataset, "MRPC-Train-Legacy")
            test_label_dist = DatasetLoaderUtil.count_label_distribution(self._test_dataset, "MRPC-Test-Legacy")

            self.train_label_distribution = dict(train_label_dist)
            self.test_label_distribution = dict(test_label_dist)

            self.vocab = getattr(args, "vocab", None)
            if self.vocab is None:
                self.vocab = TokenizerBuilder.build_vocab(self._dataset, tokenizer)

            args.vocab_size = len(self.vocab)

            collate = partial(
                DatasetLoaderUtil.text_pair_collate_fn,
                tokenizer=tokenizer,
                vocab=self.vocab,
                combine_fn=combine_fn,
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
