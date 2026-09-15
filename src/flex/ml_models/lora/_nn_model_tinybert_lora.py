from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Optional, List
from packaging import version

from .. import AbstractNNModel, NNModel, NNModelArgs
from ...ml_algorithms.lora.lora_utils import LoRAUtils
from ...ml_utils import console


class NNModel_TinyBERTLoRA(NNModel):
    """
    TinyBERT (prajjwal1/bert-tiny) with LoRA-enabled layers.

    Uses HuggingFace transformers to load the pretrained TinyBERT checkpoint,
    then replaces all nn.Linear → MSLoRALinear and nn.Embedding → MSEmbedding
    so that LoRA fine-tuning can be applied directly.

    Configurable via NNModelArgs:
        - pretrained_model: str = "prajjwal1/bert-tiny"
        - num_classes: int = 2
        - lora_r: int = 8
        - lora_alpha: int = 16
        - lora_dropout: float = 0.1
        - merge_weights: bool = True
        - lora_target_modules: list[str] | None = None  (e.g. ["query", "value"]; None = all Linear)
        - lora_embedding: bool = True  (whether to also replace Embedding layers)
    """

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.pad_id: Optional[int] = None
        self._lora_mode: str = "standard"

    # ------------------------------------------------------------------
    # override
    # ------------------------------------------------------------------
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        from transformers import AutoModelForSequenceClassification, AutoConfig, BertConfig

        # ---- 1. Basic config -------------------------------------------------
        model_name = args.get("pretrained_model", "prajjwal1/bert-tiny")
        num_labels = int(args.get("num_classes", 2))

        # LoRA hyper-parameters (use args.get() for DictPath-backed fields)
        rank_ratio   = float(args.get("rank_ratio", 1))
        base_lora_r  = int(args.get("lora_r", 8))
        lora_r       = int(max(1, round(base_lora_r * rank_ratio)))
        if base_lora_r == 1 and rank_ratio != 1.0:
            console.warn(
                f"[TinyBERT LoRA] rank_ratio={rank_ratio} is configured but base lora_r=1, "
                f"effective rank will always be 1 regardless of rank_ratio. "
                f"Consider increasing lora_r in the YAML config (e.g., lora_r: 8)."
            )
        lora_alpha   = int(args.get("lora_alpha", 16))
        lora_dropout = float(args.get("lora_dropout", 0.1))
        merge_weights = bool(args.get("merge_weights", True))
        lora_target_modules = args.get("lora_target_modules", None)
        lora_embedding = bool(args.get("lora_embedding", True))

        # ---- 2. Resolve local / remote model path ---------------------------
        # Download to <flex-src>/../hf_models/ (sibling of project root)
        base_dir = Path(__file__).resolve().parents[4].parent
        local_dir = base_dir / "hf_models" / model_name
        local_dir.mkdir(parents=True, exist_ok=True)

        has_safetensors = (local_dir / "model.safetensors").exists()
        has_bin = (local_dir / "pytorch_model.bin").exists()
        has_config = (local_dir / "config.json").exists()
        has_local = (has_safetensors or has_bin) and has_config
        model_source = str(local_dir) if has_local else model_name

        # ---- 3. Build HF config ---------------------------------------------
        import json

        def _load_patched_bert_config_if_needed() -> Optional[object]:
            """Fallback for legacy/corrupted TinyBERT config missing `model_type`."""
            candidate_paths: list[Path] = [local_dir / "config.json"]
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
                cache_dir=str(local_dir),
                local_files_only=has_local,
            )
        except ValueError as e:
            if "model_type" not in str(e):
                raise
            patched = _load_patched_bert_config_if_needed()
            if patched is None:
                raise
            config = patched

        # ---- 4. Load pretrained model ---------------------------------------
        torch_ver = version.parse(torch.__version__.split("+")[0])
        need_safetensors_only = torch_ver < version.parse("2.6.0")

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_source,
                config=config,
                use_safetensors=True,
                cache_dir=str(local_dir),
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
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_source,
                config=config,
                use_safetensors=has_safetensors if has_local else None,
                cache_dir=str(local_dir),
                local_files_only=has_local,
            )

        # ---- 5. Replace layers with LoRA variants ---------------------------
        LoRAUtils.replace_with_lora_linear_embedding(
            self.model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            target_module_names=lora_target_modules,
            replace_embedding=lora_embedding,
        )

        # ---- 6. Pad token id ------------------------------------------------
        self.pad_id = getattr(self.model.config, "pad_token_id", None)
        if self.pad_id is None:
            self.pad_id = int(args.get("pad_id", 0))

        # ---- 7. Declare LoRA configuration for aggregation system -----------
        self._lora_config = {
            "suffix_A": "lora_A",
            "suffix_B": "lora_B",
            "sp_suffix": "sp_aggregated",
            "has_non_lora_params": True,
        }

        return self

    # ------------------------------------------------------------------
    # override
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        **kwargs: Any,
    ):
        # ---- Normalise input types ------------------------------------------
        try:
            from transformers.tokenization_utils_base import BatchEncoding
        except Exception:
            BatchEncoding = tuple()

        if isinstance(input_ids, BatchEncoding):
            if attention_mask is None and "attention_mask" in input_ids:
                attention_mask = input_ids["attention_mask"]
            if token_type_ids is None and "token_type_ids" in input_ids:
                token_type_ids = input_ids["token_type_ids"]
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
            if token_type_ids is None and "token_type_ids" in input_ids:
                token_type_ids = input_ids["token_type_ids"]
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

        if attention_mask is None:
            if self.pad_id is None:
                raise ValueError("pad_id is None; provide attention_mask explicitly.")
            attention_mask = (input_ids != int(self.pad_id)).long()
        else:
            if not torch.is_tensor(attention_mask):
                attention_mask = torch.as_tensor(attention_mask)
            if attention_mask.dtype != torch.long:
                attention_mask = attention_mask.long()

        if token_type_ids is not None:
            if not torch.is_tensor(token_type_ids):
                token_type_ids = torch.as_tensor(token_type_ids)
            if token_type_ids.dtype != torch.long:
                token_type_ids = token_type_ids.long()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        return outputs.logits

    # ------------------------------------------------------------------
    # LoRA mode control
    # ------------------------------------------------------------------
    def set_lora_mode(self, mode: str):
        """
        Set LoRA mode for all LoRA-enabled layers in the model.

        Args:
            mode: One of "standard", "lora_only", "lora_disabled", "scaling".
        """
        if mode not in ("standard", "lora_only", "lora_disabled", "scaling"):
            raise ValueError(f"Unsupported lora_mode: {mode}")
        self._lora_mode = mode
        for m in self.model.modules():
            if hasattr(m, "lora_mode"):
                m.lora_mode = mode


