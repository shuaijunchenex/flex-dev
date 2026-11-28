from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Optional
from packaging import version
import torch
import torch.nn as nn
from ... import AbstractNNModel, NNModel, NNModelArgs
from flex.ml_utils import console

"""
Generic Encoder-only Transformer for Sequence Classification.
Compatible with SST2, MRPC, DBpedia, IMDB, etc.
"""

class NNModel_TransformerClassification(NNModel):
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.pad_id: Optional[int] = None  # use model/tokenizer pad id when possible

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        from transformers import AutoModelForSequenceClassification, AutoConfig, BertConfig

        model_name = args.get("pretrained_model", "bert-base-uncased")
        num_labels = args.get("num_classes", 2)

        # Prefer offline models placed under <project_root>/../hf_models/<model_name>. If not present, download into that folder.
        base_dir = Path(__file__).resolve().parents[5].parent
        local_dir = base_dir / "hf_models" / model_name
        local_dir.mkdir(parents=True, exist_ok=True)

        # Detect whether usable local weights exist (both weights AND config file required)
        has_safetensors = (local_dir / "model.safetensors").exists()
        has_bin = (local_dir / "pytorch_model.bin").exists()
        has_config = (local_dir / "config.json").exists()
        has_local = (has_safetensors or has_bin) and has_config
        model_source = local_dir if has_local else model_name

        def _load_patched_bert_config_if_needed() -> Optional[object]:
            """Best-effort fallback for legacy/corrupted TinyBERT config missing `model_type`."""
            if "bert" not in model_name.lower():
                return None

            candidate_paths: list[Path] = []
            direct_cfg = local_dir / "config.json"
            if direct_cfg.exists():
                candidate_paths.append(direct_cfg)

            # HF cache layout: <cache_dir>/models--org--repo/snapshots/<sha>/config.json
            candidate_paths.extend(local_dir.glob("models--*--*/snapshots/*/config.json"))

            for cfg_path in candidate_paths:
                try:
                    cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
                    if isinstance(cfg_dict, dict) and "model_type" not in cfg_dict:
                        cfg_dict["model_type"] = "bert"
                    cfg_dict["num_labels"] = num_labels
                    return BertConfig.from_dict(cfg_dict)
                except Exception:
                    continue
            return None

        try:
            config = AutoConfig.from_pretrained(
                model_source,
                num_labels=num_labels,
                cache_dir=local_dir,
                local_files_only=has_local,
            )
        except ValueError as e:
            if "model_type" not in str(e):
                raise
            patched = _load_patched_bert_config_if_needed()
            if patched is None:
                raise
            config = patched

        torch_ver = version.parse(torch.__version__.split("+")[0])
        need_safetensors_only = torch_ver < version.parse("2.6.0")

        # Strategy:
        # 1) Always try safetensors first (works with torch<2.6 under new transformers).
        # 2) Only if torch>=2.6, allow fallback to default loader (.bin may be used).
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_source,
                config=config,
                use_safetensors=True,
                cache_dir=local_dir,
                local_files_only=has_local,
            )
        except Exception as e:
            if need_safetensors_only:
                raise ValueError(
                    "Current environment uses torch<2.6 and transformers now blocks "
                    "torch.load(.bin) due to CVE-2025-32434. "
                    "Please use a model checkpoint that provides safetensors "
                    "(or place model.safetensors under local hf_models cache), "
                    "or upgrade torch to >=2.6. "
                    f"Original error: {e}"
                ) from e

            console.warn(
                "[TransformerClassification] safetensors load failed, fallback to default loader. "
                f"source={model_source}"
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_source,
                config=config,
                use_safetensors=has_safetensors if has_local else None,
                cache_dir=local_dir,
                local_files_only=has_local,
            )

        # Prefer model-config pad token id (more reliable than args default)
        self.pad_id = getattr(self.model.config, "pad_token_id", None)
        # fallback to args if config missing
        if self.pad_id is None:
            self.pad_id = args.get("pad_id", 0)

        return self

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        **kwargs: Any
    ):
        # Allow calling with HF BatchEncoding / dict / tokenizers.Encoding
        try:
            from transformers.tokenization_utils_base import BatchEncoding  # type: ignore
        except Exception:
            BatchEncoding = tuple()  # type: ignore

        if isinstance(input_ids, BatchEncoding):
            if attention_mask is None and "attention_mask" in input_ids:
                attention_mask = input_ids["attention_mask"]
            if "input_ids" in input_ids:
                input_ids = input_ids["input_ids"]
            else:
                first = next((v for v in input_ids.values() if torch.is_tensor(v)), None)
                if first is None:
                    raise TypeError("BatchEncoding missing tensor values for input_ids")
                input_ids = first
        elif isinstance(input_ids, dict):
            if attention_mask is None and "attention_mask" in input_ids:
                attention_mask = input_ids["attention_mask"]
            if "input_ids" in input_ids:
                input_ids = input_ids["input_ids"]
            elif "ids" in input_ids:
                input_ids = input_ids["ids"]
        elif hasattr(input_ids, "ids"):
            input_ids = torch.as_tensor(getattr(input_ids, "ids"))

        if not torch.is_tensor(input_ids):
            input_ids = torch.as_tensor(input_ids)
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        model_device = next(self.model.parameters()).device
        input_ids = input_ids.to(model_device)

        if attention_mask is None:
            if self.pad_id is None:
                raise ValueError("pad_id is None; provide attention_mask explicitly.")
            attention_mask = (input_ids != int(self.pad_id)).long()
        else:
            if not torch.is_tensor(attention_mask):
                attention_mask = torch.as_tensor(attention_mask)
            if attention_mask.dtype != torch.long:
                attention_mask = attention_mask.long()
            attention_mask = attention_mask.to(model_device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return outputs.logits
