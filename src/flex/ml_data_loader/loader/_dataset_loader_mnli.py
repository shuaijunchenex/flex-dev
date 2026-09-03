from __future__ import annotations

from functools import partial
from torch.utils.data import DataLoader
from datasets import load_dataset

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from ..dataset_loader_util import DatasetLoaderUtil

"""
Dataset loader for MNLI (GLUE)
"""


class DatasetLoader_MNLI(DatasetLoader):
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
        train_split = getattr(args, "train_split", "train")
        test_split = getattr(args, "test_split", "dev_matched")

        # HuggingFace GLUE mnli splits: 'train', 'validation_matched', 'validation_mismatched'
        def _map_split(name: str) -> str:
            if name in ("dev_matched", "dev"):
                return "validation_matched"
            if name == "dev_mismatched":
                return "validation_mismatched"
            return name

        hf_train_split = _map_split(train_split)
        hf_test_split = _map_split(test_split)

        # load_dataset will download/cache data when called
        dataset = load_dataset("glue", "mnli", cache_dir=root) if is_download else load_dataset("glue", "mnli")

        # materialize hf splits
        try:
            hf_train = dataset[hf_train_split]
        except Exception:
            hf_train = dataset.get("train", [])

        try:
            hf_test = dataset[hf_test_split]
        except Exception:
            hf_test = dataset.get("validation_matched", [])

        tokenizer = getattr(args, "tokenizer", None)
        use_hf = tokenizer is not None and (
            hasattr(tokenizer, "vocab_size") or hasattr(tokenizer, "pad_token_id") or hasattr(tokenizer, "__call__")
        )

        def _to_pair_list(hf_ds):
            if hf_ds is None:
                return []
            out = []
            for item in hf_ds:
                prem = item.get("premise") or item.get("sentence1") or ""
                hyp = item.get("hypothesis") or item.get("sentence2") or ""
                lab = item.get("label")
                out.append({
                    "label": lab,
                    "text_a": prem,
                    "text_b": hyp,
                })
            return out

        if use_hf:
            # Preserve the pair so each tokenizer adds model-specific separators
            # and, for BERT-family models, token_type_ids.
            self._dataset = _to_pair_list(hf_train)
            self._test_dataset = _to_pair_list(hf_test)

            train_label_dist = DatasetLoaderUtil.count_label_distribution(self._dataset, "MNLI-Train")
            test_label_dist = DatasetLoaderUtil.count_label_distribution(self._test_dataset, "MNLI-Test")

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

            self._test_data_loader = DataLoader(
                self._test_dataset,
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate,
            )
        else:
            # Fallback: expose tuples (premise, hypothesis, label) as before
            def _to_tuple_list(hf_ds):
                if hf_ds is None:
                    return []
                out = []
                for item in hf_ds:
                    prem = item.get("premise") or item.get("sentence1") or ""
                    hyp = item.get("hypothesis") or item.get("sentence2") or ""
                    lab = item.get("label")
                    out.append((prem, hyp, lab))
                return out

            self._dataset = _to_tuple_list(hf_train)
            self._test_dataset = _to_tuple_list(hf_test)

            train_label_dist = DatasetLoaderUtil.count_label_distribution(self._dataset, "MNLI-Train-Legacy")
            test_label_dist = DatasetLoaderUtil.count_label_distribution(self._test_dataset, "MNLI-Test-Legacy")

            self.train_label_distribution = dict(train_label_dist)
            self.test_label_distribution = dict(test_label_dist)

            self._data_loader = DataLoader(
                self._dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            )

            self._test_data_loader = DataLoader(
                self._test_dataset,
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=num_workers,
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
