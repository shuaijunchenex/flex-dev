from __future__ import annotations
from typing import Any, Iterable, List, Dict, Optional, Callable
from collections import Counter
import re

from ..tokenizer_abc import Tokenizer
from ..tokenizer_args import TokenizerArgs


class BasicEnglishTokenizer(Tokenizer):
    """A small, dependency-free basic English tokenizer implementation.

    It tokenizes by finding word character sequences (\w+), supports simple
    lowercasing and truncation, and can build a simple vocab from data.
    """

    def __init__(self):
        super().__init__()
        self.lowercase = True
        self.base_tokenizer = None

    def _create_inner(self, args: Any) -> None:
        self.__build_basic_english(args)

    def __build_basic_english(self, args: Any) -> None:
        from torchtext.data.utils import get_tokenizer
        self.lowercase = args.get("lowercase", True) if hasattr(args, "get") else getattr(args, "lowercase", True)
        self.base_tokenizer = get_tokenizer("basic_english")
        self._args = args

    # override
    def tokenize(self, text: str) -> List[str]:
        if not self.is_created:
            raise RuntimeError("Tokenizer not created. Call create(args) first.")
        if text is None: return []
        proc = str(text).lower() if self.lowercase else str(text)
        return self.base_tokenizer(proc)

    def build_vocab(self, data_iter: Iterable, special_tokens: Optional[List[str]] = None,
                    max_size: Optional[int] = None, min_freq: int = 1) -> Dict[str, int]:
        counter = Counter()
        if not self.is_created:
            raise RuntimeError("Tokenizer not initialized; call create(args) first")

        for sample in data_iter:
            try:
                tokens = self.tokenize(sample)
            except Exception:
                tokens = self.tokenize(str(sample))
            counter.update(tokens)

        # apply min frequency
        items = [(t, c) for t, c in counter.items() if c >= min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))

        vocab: Dict[str, int] = {}
        specials = special_tokens or []
        for i, s in enumerate(specials):
            vocab[s] = i

        offset = len(specials)
        for i, (t, _) in enumerate(items):
            if max_size is not None and i + offset >= max_size:
                break
            vocab[t] = i + offset

        return vocab
