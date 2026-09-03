from __future__ import annotations

from functools import partial

from torch.utils.data import DataLoader

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from ..dataset_loader_util import DatasetLoaderUtil


_SPLIT_ALIASES = {
    "dev": "validation",
    "valid": "validation",
}


class DatasetLoader_QNLI(DatasetLoader):
    """Dataset loader for GLUE QNLI.

    Hugging Face label ids are preserved: 0=entailment, 1=not_entailment.
    """

    def __init__(self):
        super().__init__()

    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        from datasets import load_dataset

        root = getattr(args, "root")
        is_download = getattr(args, "is_download", True)
        batch_size = getattr(args, "batch_size", 32)
        test_batch_size = args.get("test_batch_size", None) or batch_size
        shuffle = getattr(args, "shuffle", True)
        num_workers = getattr(args, "num_workers", 0)
        requested_train_split = args.get("train_split", "train")
        requested_test_split = args.get("test_split", "validation")
        train_split = _SPLIT_ALIASES.get(requested_train_split, requested_train_split)
        test_split = _SPLIT_ALIASES.get(requested_test_split, requested_test_split)

        dataset = (
            load_dataset("glue", "qnli", cache_dir=root)
            if is_download
            else load_dataset("glue", "qnli")
        )

        def _to_sample(item, fallback_idx):
            return {
                "label": item.get("label"),
                "text_a": item.get("question") or "",
                "text_b": item.get("sentence") or "",
                "idx": item.get("idx", fallback_idx),
            }

        self._dataset = [
            _to_sample(item, idx) for idx, item in enumerate(dataset[train_split])
        ]
        self._test_dataset = [
            _to_sample(item, idx) for idx, item in enumerate(dataset[test_split])
        ]

        self.train_label_distribution = dict(
            DatasetLoaderUtil.count_label_distribution(self._dataset, "QNLI-Train")
        )
        self.test_label_distribution = dict(
            DatasetLoaderUtil.count_label_distribution(self._test_dataset, "QNLI-Validation")
        )

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
        self.data_sample_num = len(self._dataset)

    def get_dataset(self):
        if self._data_loader is not None:
            return self._data_loader.dataset
        raise ValueError("ERROR: DatasetLoader's data_loader is None.")
