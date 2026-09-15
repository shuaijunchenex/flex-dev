from __future__ import annotations
from typing import Any, List, Callable, Optional

from ..tokenizer_abc import Tokenizer
from flex.ml_utils import KeyValueArgs
from transformers import AutoTokenizer

class BertTokenizer(Tokenizer):
    """
    BERT Tokenizer using HuggingFace Transformers.
    """

    def __init__(self):
        super().__init__()
        self.tok: Any = None
        self.return_type: str = "tokens"
        self.max_len: int = 512
        self.meta: dict[str, Any] = {}

    def _create_inner(self, args: Any) -> None:
        self.__build_bert(args)

    def __build_bert(self, args: Any) -> None:
        try:
            from transformers import AutoTokenizer
        except Exception as e:
            raise ImportError("transformers is required for 'bert' tokenizer.") from e

        name = args.get("pretrained", "bert-base-uncased")
        use_fast = args.get("use_fast", True)
        self.return_type = str(args.get("return_type", "tokens")).lower()
        self.max_len = int(args.get("max_len", 512))

        try:
            self.tok = AutoTokenizer.from_pretrained(name, use_fast=use_fast)
        except (ValueError, OSError):
            # Fast tokenizer not available — try slow tokenizer first.
            try:
                self.tok = AutoTokenizer.from_pretrained(name, use_fast=False)
            except (ValueError, OSError):
                # Model has no tokenizer files at all (e.g. prajjwal1/bert-tiny only ships
                # weights; its vocab is identical to bert-base-uncased).
                # Fall back to bert-base-uncased which covers all BERT-family tiny models.
                import warnings
                warnings.warn(
                    f"[BertTokenizer] '{name}' has no tokenizer files. "
                    "Falling back to 'bert-base-uncased' (same vocab)."
                )
                self.tok = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
        self.meta["hf_tokenizer"] = self.tok
        if self.tok.pad_token_id is not None:
            self.meta["pad_id"] = int(self.tok.pad_token_id)
        
        self._args = args

    # override
    def tokenize(self, text: str) -> List[str]:
        if not self.is_created:
             raise RuntimeError("Tokenizer not created. Call create(args) first.")
        return self.tok.tokenize(text)

    # override
    def encode(self, text: str) -> List[int]:
        if not self.is_created:
             raise RuntimeError("Tokenizer not created. Call create(args) first.")
        
        if self.return_type == "ids":
            out = self.tok(text, add_special_tokens=True, truncation=True, max_length=self.max_len)
            return list(map(int, out["input_ids"]))
        
        return super().encode(text)
