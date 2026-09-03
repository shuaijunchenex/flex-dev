from __future__ import annotations
import re
import torch
import torch.optim as optim
from typing import Callable, Any, Optional, Dict, List
from ..ml_utils import Handlers
from torch.nn.utils.rnn import pad_sequence
#from transformers import AutoTokenizer
from torchtext.vocab import build_vocab_from_iterator

class TokenizerBuilder(Handlers):
    """
    config 示例：
    {
      "tokenizer": {
        "type": "basic_english",   # 或 "bert"
        "lower": true,             # 仅对 basic_english
        "pretrained": "bert-base-uncased",  # 仅对 bert
        "use_fast": true,          # 仅对 bert
        "return_type": "tokens",   # tokens | ids  (bert 默认 tokens)
        "max_len": 256             # 仅对 bert 且 return_type=ids 时用于截断
      }
    }
    build() 返回一个可调用对象：
      - basic_english: fn(text) -> List[str]
      - bert:
          return_type='tokens' -> tok.tokenize(text) -> List[str]
          return_type='ids'    -> fn(text) -> List[int]  (包含 special tokens，按 max_len 截断)
    """

    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__()
        self.config = config_dict.get("tokenizer", config_dict)
        self._tokenizer_fn: Optional[Callable[[str], Any]] = None
        self.meta: Dict[str, Any] = {}  # 可选：记录 pad_id / hf_tokenizer 等

        self.register_handler("basic_english", self.__wrap_factory_create("basic_english"))
        self.register_handler("bert",           self.__wrap_factory_create("bert"))

    def __wrap_factory_create(self, t_type: str):
        def _build():
            from .tokenizer.tokenizer_factory import TokenizerFactory
            # TokenizerBuilder's config is passed as the args for creation
            tk = TokenizerFactory.create_tokenizer(t_type, self.config)

            # Record meta info (like pad_id) back to TokenizerBuilder
            if hasattr(tk, "meta"):
                self.meta.update(tk.meta)

            # Return the appropriate callable based on config
            return_type = str(self.config.get("return_type", "tokens")).lower()

            # Some tokenizers expose an internal `_encode_fn`; fall back to
            # the public `encode` if present.
            encode_available = getattr(tk, "_encode_fn", None) is not None or hasattr(tk, "encode")

            if return_type == "ids" and encode_available:
                return tk.encode

            return tk.tokenize

        return _build

    def build(self) -> Callable[[str], Any]:
        t = str(self.config.get("type", "basic_english")).lower()
        fn = self.invoke_handler(t)
        if fn is None:
            raise ValueError(f"Unsupported tokenizer type: {t}")
        self._tokenizer_fn = fn
        return fn

    @staticmethod
    def build_vocab(data_set, tokenizer, min_freq=2, max_vocab=30000):
        """Build a vocab from an iterable dataset.

        Compatible with torchtext datasets that may yield tuples of varying length
        (e.g., CoLA can yield (line_idx, label, text) depending on version).
        We only care about the text field, assumed to be the last element if a tuple/list,
        or the value of 'text'/'sentence' if a dict.
        """
        from collections import Counter
        from torchtext.vocab import Vocab
        
        train_samples = list(data_set)

        def _extract_text(sample):
            # Dict sample
            if isinstance(sample, dict):
                for key in ("text", "sentence", "content"):
                    if key in sample:
                        return sample[key]
                # fallback: first value
                return next(iter(sample.values()))
            # Tuple/list sample
            if isinstance(sample, (list, tuple)):
                if len(sample) == 0:
                    return ""
                # Assume last element is text
                return sample[-1]
            # String sample
            if isinstance(sample, str):
                return sample
            # Unknown structure
            return str(sample)

        PAD, CLS, UNK = "<pad>", "<cls>", "<unk>"
        specials = [PAD, CLS, UNK]
        counter = Counter()

        for sample in train_samples:
            text = _extract_text(sample)
            counter.update(tokenizer(text))

        cutoff = max_vocab - len(specials)
        most_common = counter.most_common(cutoff)
        counter = Counter(dict(most_common))

        def yield_tokens():
            for sample in train_samples:
                text = _extract_text(sample)
                yield tokenizer(text)

        # Build Vocab from counter for compatibility across torchtext versions
        cutoff = max_vocab - len(specials)
        try:
            vocab = Vocab(counter, min_freq=min_freq, specials=specials, max_size=cutoff)
        except TypeError:
            # Older/newer torchtext versions may use different arg names
            vocab = Vocab(counter, min_freq=min_freq, specials=specials)

        try:
            vocab.set_default_index(vocab[UNK])
        except Exception:
            # fallback: set default to index of UNK if available
            try:
                vocab.set_default_index(vocab.get_index(UNK))
            except Exception:
                pass

        return vocab

    # def get_imdb_iter(root: str, split: str):
    #     # Try new API first; fall back to legacy if TypeError (your error)
    #     try:
    #         it = _imdb_iter_new(root, split)
    #         # ensure it's materializable
    #         it = list(it)  # download on first call
    #         return it
    #     except TypeError:
    #         # legacy path
    #         return list(_imdb_iter_legacy(root, split))

    # def _imdb_iter_new(root: str, split: str):
    #     # New API: torchtext.datasets.IMDB(root=..., split='train'/'test')
    #     from torchtext.datasets import IMDB
    #     return IMDB(root=root, split=split)

    def __build_bert(self) -> Callable[[str], Any]:
        try:
            from transformers import AutoTokenizer
        except Exception as e:
            raise ImportError("transformers is required for 'bert' tokenizer.") from e

        name = self.config.get("pretrained", "bert-base-uncased")
        use_fast = self.config.get("use_fast", True)
        return_type = str(self.config.get("return_type", "tokens")).lower()
        max_len = int(self.config.get("max_len", 512))

        tok = AutoTokenizer.from_pretrained(name, use_fast=use_fast)
        self.meta["hf_tokenizer"] = tok
        if tok.pad_token_id is not None:
            self.meta["pad_id"] = int(tok.pad_token_id)

        if return_type == "ids":
            def fn_ids(s: str):
                out = tok(s, add_special_tokens=True, truncation=True, max_length=max_len)
                return list(map(int, out["input_ids"]))
            return fn_ids
        else:
            return tok.tokenize