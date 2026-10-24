import torch
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

class DatasetLoaderUtil:
    """
    " DataLoader Util class
    """
    
    # torchtext datasets

    @staticmethod
    def _safe_map_label(label, label_map: Optional[Dict[Any, int]]):
        """
        Map a label using label_map with robust fallbacks.

        Fallback policy (in order):
        1) exact key in label_map
        2) int(label) key in label_map
        3) str(label) key in label_map
        4) if label is integer-like, return int(label) directly
        """
        if label_map is not None:
            if label in label_map:
                return label_map[label]

            # try cross-type key lookup
            try:
                il = int(label)
                if il in label_map:
                    return label_map[il]
            except Exception:
                pass

            sl = str(label)
            if sl in label_map:
                return label_map[sl]

        # passthrough if already integer-like (e.g., labels are already 0..K-1)
        try:
            return int(label)
        except Exception as e:
            keys_preview = [] if label_map is None else list(label_map.keys())[:8]
            raise KeyError(
                f"Cannot map label '{label}'. label_map preview keys={keys_preview}"
            ) from e

    @staticmethod
    def _map_int_labels_consistently(labels, label_map: Optional[Dict[Any, int]] = None):
        """
        Map a sequence of integer-like labels with batch-level consistency.

        Rules:
        - If label_map is None: create batch-local map to [0..K-1].
        - If all integer labels are covered by label_map keys (after int-casting keys), apply mapping.
        - Otherwise, treat labels as already-normalized and passthrough int(labels).
          (prevents partial remapping like [0,1] -> [0,0] when map is {1:0,2:1,...})
        """
        int_labels = [int(l) for l in labels]
        if label_map is None:
            uniq = sorted(set(int_labels))
            local_map = {l: i for i, l in enumerate(uniq)}
            return [local_map[l] for l in int_labels]

        int_key_map: Dict[int, int] = {}
        for k, v in label_map.items():
            try:
                int_key_map[int(k)] = int(v)
            except Exception:
                continue

        if int_key_map and all(l in int_key_map for l in int_labels):
            return [int_key_map[l] for l in int_labels]

        return int_labels

    @staticmethod
    def _ensure_label_ints(batch, label_map: Optional[Dict[Any, int]] = None, tuple_format: str = "auto", require_labels: bool = True):
        """
        Private helper to normalize string labels in a batch to integer ids.

        - Detects label position using `tuple_format` semantics (auto -> idx_label_text if len==3 else label_text).
        - If string labels are present and `label_map` is None, it creates a deterministic mapping:
            * Use known MNLI mapping if detected (entailment/neutral/contradiction).
            * Otherwise build a sorted unique-label mapping for determinism.
        - Returns (possibly modified) batch and the label_map used.
        """
        if not batch:
            return batch, label_map

        def _get_label(s):
            if isinstance(s, dict):
                return s.get("label", s.get("labels", None))
            if isinstance(s, (list, tuple)):
                n = len(s)
                fmt = tuple_format
                if fmt == "auto":
                    fmt = "idx_label_text" if n == 3 else "label_text"
                if fmt == "idx_label_text":
                    if n < 3:
                        return None
                    return s[1]
                if fmt == "text_label":
                    return s[-1] if n >= 2 else None
                # label_text
                return s[0] if n >= 1 else None
            return None

        labels = [_get_label(s) for s in batch]
        non_none = [l for l in labels if l is not None]
        if not non_none:
            return batch, label_map

        # If any label is string, ensure a label_map exists and convert labels
        if any(isinstance(l, str) for l in non_none):
            uniq = sorted(set([l for l in non_none if l is not None]))
            # known MNLI labels
            mnli_set = {"entailment", "neutral", "contradiction"}
            if label_map is None:
                if set(uniq) == mnli_set or mnli_set.issubset(set(uniq)):
                    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
                else:
                    label_map = {l: i for i, l in enumerate(uniq)}

            # apply mapping to batch
            new_batch = []
            for s in batch:
                if isinstance(s, dict):
                    lbl = s.get("label", s.get("labels", None))
                    if isinstance(lbl, str) and lbl in label_map:
                        s = dict(s)
                        s["label"] = label_map[lbl]
                    new_batch.append(s)
                    continue

                if isinstance(s, (list, tuple)):
                    n = len(s)
                    fmt = tuple_format
                    if fmt == "auto":
                        fmt = "idx_label_text" if n == 3 else "label_text"

                    if fmt == "idx_label_text":
                        if n >= 3:
                            lbl = s[1]
                            if isinstance(lbl, str) and lbl in label_map:
                                tmp = list(s)
                                tmp[1] = label_map[lbl]
                                new_batch.append(type(s)(tmp))
                                continue
                    elif fmt == "text_label":
                        if n >= 2:
                            lbl = s[-1]
                            if isinstance(lbl, str) and lbl in label_map:
                                tmp = list(s)
                                tmp[-1] = label_map[lbl]
                                new_batch.append(type(s)(tmp))
                                continue
                    else:  # label_text
                        if n >= 1:
                            lbl = s[0]
                            if isinstance(lbl, str) and lbl in label_map:
                                tmp = list(s)
                                tmp[0] = label_map[lbl]
                                new_batch.append(type(s)(tmp))
                                continue

                # fallback: append original
                new_batch.append(s)

            return new_batch, label_map

        return batch, label_map

    @staticmethod
    def text_collate_fn(
        batch,
        tokenizer=None,
        vocab=None,
        max_len: int = 256,
        pad_id: int = 0,
        unk_id: Optional[int] = None,
        # --- label handling ---
        label_map: Optional[Dict[Any, int]] = None,
        normalize_int_labels: bool = False,
        # --- tuple format handling ---
        tuple_format: str = "auto",  # "auto" | "label_text" | "idx_label_text"
        require_labels: bool = True,
    ):
        """
        Collate function for text datasets.

        Supports samples shaped as:
          - (label, text)                     -> tuple_format="label_text"
          - (idx, label, text)                -> tuple_format="idx_label_text"
          - dict with keys like {'label': ..., 'text': ...}
          - (idx, text) is NOT a supervised sample; if present and require_labels=True -> error

        Label mapping policy:
          - If labels are strings: you MUST provide label_map (dataset-level, stable).
          - If labels are ints:
              * default: keep as-is (no remap)
              * if normalize_int_labels=True: remap to [0..K-1] using label_map if given,
                else build a batch-local map (not recommended for training consistency).
        """
        if tokenizer is None or vocab is None:
            raise ValueError("text_collate_fn requires tokenizer and vocab.")
        if not batch:
            raise ValueError("Empty batch.")

        # Normalize string labels to integer ids for compatibility
        batch, label_map = DatasetLoaderUtil._ensure_label_ints(
            batch, label_map=label_map, tuple_format=tuple_format, require_labels=require_labels
        )

        # choose unk_id
        if unk_id is None:
            # try common patterns
            try:
                unk_id = vocab["<unk>"]
            except Exception:
                try:
                    unk_id = vocab.get("<unk>")  # type: ignore[attr-defined]
                except Exception:
                    unk_id = pad_id  # fallback

        def _vocab_lookup(token: str) -> int:
            # robust lookup with unk fallback
            try:
                return vocab[token]
            except Exception:
                try:
                    return vocab.get(token, unk_id)  # type: ignore[attr-defined]
                except Exception:
                    return unk_id

        def _extract_label_text(sample) -> Tuple[Any, str]:
            # Dict sample
            if isinstance(sample, dict):
                label = sample.get("label", sample.get("labels", None))
                text = sample.get("text", sample.get("sentence", sample.get("content", "")))
                return label, text if text is not None else ""

            # Tuple/list sample
            if isinstance(sample, (list, tuple)):
                n = len(sample)
                if n == 0:
                    return None, ""

                fmt = tuple_format
                if fmt == "auto":
                    fmt = "idx_label_text" if n == 3 else "label_text"

                if fmt == "idx_label_text":
                    if n < 3:
                        # e.g., (idx, text) -> not supervised
                        return None, str(sample[-1]) if n >= 1 else ""
                    label = sample[1]
                    text = sample[-1]
                    return label, "" if text is None else str(text)

                if fmt == "text_label":
                    if n < 2:
                        return None, str(sample[0]) if n == 1 else ""
                    text = sample[0]
                    label = sample[-1]
                    return label, "" if text is None else str(text)

                # fmt == "label_text"
                if n < 2:
                    # text-only or malformed
                    return None, str(sample[0]) if n == 1 else ""
                label = sample[0]
                text = sample[-1]
                return label, "" if text is None else str(text)

            # Otherwise treat as text-only
            return None, "" if sample is None else str(sample)

        labels_texts = [_extract_label_text(s) for s in batch]
        labels, texts = zip(*labels_texts)

        if require_labels and any(l is None for l in labels):
            raise ValueError(
                "Some samples have no label (e.g., (idx,text) or missing 'label'). "
                "Set require_labels=False if you intentionally want unlabeled batches."
            )

        # tokenize + map to ids
        tokenized = [tokenizer(t) for t in texts]
        ids = [[_vocab_lookup(tok) for tok in toks] for toks in tokenized]

        # pad / truncate
        batch_max_len = min(max((len(seq) for seq in ids), default=0), max_len)
        padded: List[List[int]] = []
        for seq in ids:
            if len(seq) > batch_max_len:
                seq = seq[:batch_max_len]
            else:
                seq = seq + [pad_id] * (batch_max_len - len(seq))
            padded.append(seq)

        input_ids = torch.tensor(padded, dtype=torch.long)

        # labels tensor
        if not require_labels and all(l is None for l in labels):
            # return dummy labels (or you can return None, depending on your pipeline)
            return input_ids, None

        # Map/encode labels
        first_non_none = next((l for l in labels if l is not None), None)

        if isinstance(first_non_none, str):
            if label_map is None:
                raise ValueError(
                    "String labels detected but label_map is None. "
                    "Provide a stable dataset-level label_map, e.g., {'neg':0,'pos':1}."
                )
            mapped = [DatasetLoaderUtil._safe_map_label(l, label_map) for l in labels]
        else:
            # integer-ish or other hashables
            if normalize_int_labels:
                mapped = DatasetLoaderUtil._map_int_labels_consistently(labels, label_map)
            else:
                mapped = list(labels)

        labels_tensor = torch.tensor(mapped, dtype=torch.long)
        return input_ids, labels_tensor

    @staticmethod
    def text_collate_fn_hf(
        batch,
        hf_tokenizer=None,
        max_len: int = 256,
        # --- label handling ---
        label_map: Optional[Dict[Any, int]] = None,
        normalize_int_labels: bool = False,
        # --- tuple format handling ---
        tuple_format: str = "auto",  # "auto" | "label_text" | "text_label" | "idx_label_text"
        require_labels: bool = True,
    ):
        """
        Collate function for HuggingFace tokenizers. Returns (encodings, labels_tensor|None).
        """
        if hf_tokenizer is None:
            raise ValueError("text_collate_fn_hf requires hf_tokenizer.")
        if not batch:
            raise ValueError("Empty batch.")

        # Normalize string labels to integer ids for compatibility
        batch, label_map = DatasetLoaderUtil._ensure_label_ints(
            batch, label_map=label_map, tuple_format=tuple_format, require_labels=require_labels
        )

        def _extract_label_text_pair(
            sample,
        ) -> Tuple[Any, str, Optional[str]]:
            if isinstance(sample, dict):
                label = sample.get("label", sample.get("labels", None))
                text_a = sample.get(
                    "text_a",
                    sample.get("sentence1", sample.get("sentence_a", None)),
                )
                text_b = sample.get(
                    "text_b",
                    sample.get("sentence2", sample.get("sentence_b", None)),
                )
                if text_a is not None or text_b is not None:
                    return (
                        label,
                        "" if text_a is None else str(text_a),
                        "" if text_b is None else str(text_b),
                    )
                text = sample.get("text", sample.get("sentence", sample.get("content", "")))
                return label, "" if text is None else str(text), None

            if isinstance(sample, (list, tuple)):
                n = len(sample)
                if n == 0:
                    return None, "", None

                fmt = tuple_format
                if fmt == "auto":
                    fmt = "idx_label_text" if n == 3 else "label_text"

                if fmt == "idx_label_text":
                    if n < 3:
                        return None, str(sample[-1]) if n >= 1 else "", None
                    label = sample[1]
                    text = sample[-1]
                    return label, "" if text is None else str(text), None

                if fmt == "text_label":
                    # SST-2 format: (text, label)
                    if n < 2:
                        return None, str(sample[0]) if n == 1 else "", None
                    text = sample[0]
                    label = sample[-1]
                    return label, "" if text is None else str(text), None

                # "label_text" (default)
                if n < 2:
                    return None, str(sample[0]) if n == 1 else "", None
                label = sample[0]
                text = sample[-1]
                return label, "" if text is None else str(text), None

            return None, "" if sample is None else str(sample), None

        labels_texts = [_extract_label_text_pair(s) for s in batch]
        labels, texts, text_pairs = zip(*labels_texts)

        if require_labels and any(l is None for l in labels):
            raise ValueError(
                "Some samples have no label (e.g., (idx,text) or missing 'label'). "
                "Set require_labels=False if you intentionally want unlabeled batches."
            )

        tokenizer_kwargs = {
            "add_special_tokens": True,
            "truncation": True,
            "max_length": max_len,
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }
        has_text_pair = any(text is not None for text in text_pairs)
        if has_text_pair:
            if not all(text is not None for text in text_pairs):
                raise ValueError(
                    "A batch cannot mix sentence-pair and single-sentence samples."
                )
            enc = hf_tokenizer(
                list(texts),
                list(text_pairs),
                **tokenizer_kwargs,
            )
        else:
            enc = hf_tokenizer(list(texts), **tokenizer_kwargs)

        if not require_labels and all(l is None for l in labels):
            return enc, None

        first_non_none = next((l for l in labels if l is not None), None)

        if isinstance(first_non_none, str):
            if label_map is None:
                raise ValueError("String labels detected but label_map is None.")
            mapped = [DatasetLoaderUtil._safe_map_label(l, label_map) for l in labels]
        else:
            if normalize_int_labels:
                mapped = DatasetLoaderUtil._map_int_labels_consistently(labels, label_map)
            else:
                mapped = [int(l) for l in labels]

        labels_tensor = torch.tensor(mapped, dtype=torch.long)
        return enc, labels_tensor

    @staticmethod
    def text_pair_collate_fn(batch, tokenizer=None, vocab=None, max_len=256, pad_id=0, combine_fn=None):
        """
        Collate function for paired-text datasets (e.g., MRPC).
        Merges (label, text_a, text_b) into a single text before tokenization.
        """
        if tokenizer is None or vocab is None:
            raise ValueError("text_pair_collate_fn requires tokenizer and vocab.")

        if combine_fn is None:
            def combine_fn(a, b):
                return f"{a} [SEP] {b}"

        labels, text_a, text_b = zip(*batch)
        merged = [combine_fn(a, b) for a, b in zip(text_a, text_b)]
        merged_batch = list(zip(labels, merged))

        return DatasetLoaderUtil.text_collate_fn(
            merged_batch,
            tokenizer=tokenizer,
            vocab=vocab,
            max_len=max_len,
            pad_id=pad_id,
        )


    # @staticmethod
    # def text_collate_fn(batch):

    #     """
    #     Collate function for text datasets.
    #     Merges a list of (label, text) tuples into lists.
    #     """

    #     labels, texts = zip(*batch)
    #     return list(labels), list(texts)

    def _load_data(self):
        """Load data from DataLoader. Handles both image tensors and text lists."""
        images_list, labels_list = [], []
        for images, labels in self.dataloader:
            # 1. 处理标签：确保是 Tensor 以便后续 unique/sorting
            if not torch.is_tensor(labels):
                labels = torch.as_tensor(labels)
            
            # 2. 处理数据（images 或 text）：
            # 如果是文本（字符串列表），不要调用 torch.as_tensor，否则会报 'too many dimensions str'
            # 只有当它是数值型数据时才转 Tensor
            if not torch.is_tensor(images):
                try:
                    images = torch.as_tensor(images)
                except (ValueError, TypeError):
                    # 如果报错（如文本任务），则保持原始列表/对象格式
                    pass
            
            images_list.append(images)
            labels_list.append(labels)

        # 3. 合并数据
        # 如果 images 是 Tensor，按 dim=0 合并；如果是 list（文本），用 list extend
        if torch.is_tensor(images_list[0]):
            self.x_train = torch.cat(images_list, dim=0)
        else:
            # 文本任务，合并为大列表
            self.x_train = []
            for b in images_list:
                self.x_train.extend(b) if isinstance(b, list) else self.x_train.append(b)
                
        self.y_train = torch.cat(labels_list, dim=0)

    @staticmethod
    def extract_label_from_sample(sample):
        """
        Robustly extract label from a sample.
        Supports: tuple/list, dict-like, or object with attributes.
        Returns label (int or original value).
        """
        # 1) dict-like
        if isinstance(sample, dict):
            for k in ("label", "labels", "y", "target"):
                if k in sample:
                    return sample[k]

        # 2) tuple/list: try common layouts
        if isinstance(sample, (tuple, list)):
            # Common patterns:
            # (text, label)
            # (label, text)
            # (idx, text, label) / (text, label, idx) etc.
            
            # Try to find something that looks like an integer label in common ranges
            for i, v in enumerate(sample):
                try:
                    iv = int(v)
                    # Broad range for common NLP tasks (4 for AG News, 2 for others, maybe more for some)
                    if 0 <= iv <= 100: 
                        return v
                except Exception:
                    pass

            # Fallback: many torchtext datasets put label at the first or last position
            try:
                return sample[0]
            except Exception:
                pass

        # 3) object with attributes
        for attr in ("label", "labels", "y", "target"):
            if hasattr(sample, attr):
                return getattr(sample, attr)

        return sample

    @staticmethod
    def count_label_distribution(dataset, name: str = "Dataset", debug_print_first: bool = True):
        """
        Count labels in dataset without assuming sample structure.
        """
        counter = Counter()

        it = iter(dataset)
        try:
            first = next(it)
        except StopIteration:
            return counter

        if debug_print_first:
            print(f"[{name}] first sample type={type(first)} value={first}")

        # For efficiency, determine an extraction strategy from the first sample
        def _get_strategy(s):
            if isinstance(s, dict):
                for k in ("label", "labels", "y", "target"):
                    if k in s: return lambda x: x[k]
            if isinstance(s, (tuple, list)):
                for i, v in enumerate(s):
                    try:
                        iv = int(v)
                        if 0 <= iv <= 100: return lambda x: x[i]
                    except Exception:
                        pass
                return lambda x: x[0]
            return lambda x: x

        strategy = _get_strategy(first)
        
        counter[strategy(first)] += 1
        for sample in it:
            counter[strategy(sample)] += 1

        return counter

        
