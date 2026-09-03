from __future__ import annotations

from functools import partial
from torch.utils.data import DataLoader
from datasets import load_dataset

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from ..dataset_loader_util import DatasetLoaderUtil

"""
Dataset loader for QQP (GLUE)
This loader preserves question1/question2 as a native sentence pair so each
Hugging Face tokenizer can add its own special tokens and segment metadata.
"""


class DatasetLoader_QQP(DatasetLoader):
    def __init__(self):
        super().__init__()

    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        root = getattr(args, "root")
        is_download = getattr(args, "is_download", True)
        batch_size = getattr(args, "batch_size", 32)
        test_batch_size = getattr(args, "test_batch_size", None) or batch_size
        shuffle = getattr(args, "shuffle", True)
        num_workers = getattr(args, "num_workers", 0)
        train_split = getattr(args, "train_split", "train")
        test_split = getattr(args, "test_split", "validation")

        # map common aliases
        if test_split in ("dev", "dev_matched", "dev_mismatched"):
            test_split = "validation"

        dataset = load_dataset("glue", "qqp", cache_dir=root) if is_download else load_dataset("glue", "qqp")

        hf_train = dataset.get(train_split, [])
        hf_test = dataset.get(test_split, [])

        def _to_pair_sample(item, idx):
            q1 = item.get("question1") or ""
            q2 = item.get("question2") or ""
            lab = item.get("label")
            return {
                "label": lab,
                "text_a": q1,
                "text_b": q2,
                "idx": idx,
            }

        self._dataset = [_to_pair_sample(it, i) for i, it in enumerate(hf_train)]
        self._test_dataset = [_to_pair_sample(it, i) for i, it in enumerate(hf_test)]

        train_label_dist = DatasetLoaderUtil.count_label_distribution(self._dataset, "QQP-Train")
        test_label_dist = DatasetLoaderUtil.count_label_distribution(self._test_dataset, "QQP-Test")

        self.train_label_distribution = dict(train_label_dist)
        self.test_label_distribution = dict(test_label_dist)

        hf_tokenizer = getattr(args, "tokenizer")
        args.vocab_size = getattr(hf_tokenizer, "vocab_size", None)

        collate = partial(
            DatasetLoaderUtil.text_collate_fn_hf,
            hf_tokenizer=hf_tokenizer,
            max_len=getattr(args, "max_len", 256),
            normalize_int_labels=False,
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
        try:
            self.data_sample_num = len(self._dataset)
        except Exception:
            self.data_sample_num = None

        return

    def get_dataset(self):
        if self._data_loader is not None:
            return self._data_loader.dataset
        raise ValueError("ERROR: DatasetLoader's data_loader is None.")
