from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Optional

from .. import AbstractNNModel, NNModel, NNModelArgs
from ...ml_algorithms.lora.lora_utils import LoRAUtils
from ...ml_utils import console


class NNModel_Qwen2_5_0_5BLoRA(NNModel):
    """Qwen2.5-0.5B with LoRA-enabled layers for sequence classification.

    Uses HuggingFace transformers to load the pretrained Qwen2.5-0.5B
    checkpoint, then replaces target nn.Linear → MSLoRALinear.

    Configurable via NNModelArgs:
        - pretrained_model: str = "Qwen/Qwen2.5-0.5B"
        - num_classes: int = 2
        - lora_r: int = 16
        - lora_alpha: int = 16
        - lora_dropout: float = 0.1
        - merge_weights: bool = True
        - lora_target_modules: list[str] | None = None  (None = all Linear)
        - lora_embedding: bool = False
    """

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.pad_id: Optional[int] = None

    # ------------------------------------------------------------------
    # override
    # ------------------------------------------------------------------
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        from transformers import AutoModelForSequenceClassification, AutoConfig

        # ---- 1. Basic config -------------------------------------------------
        model_name = args.get("pretrained_model", "Qwen/Qwen2.5-0.5B")
        num_labels = int(args.get("num_classes", 2))

        # LoRA hyper-parameters
        rank_ratio   = float(args.get("rank_ratio", 1))
        base_lora_r  = int(args.get("lora_r", 16))
        lora_r       = int(max(1, round(base_lora_r * rank_ratio)))
        if base_lora_r == 1 and rank_ratio != 1.0:
            console.warn(
                f"[Qwen2.5 LoRA] rank_ratio={rank_ratio} is configured but base lora_r=1, "
                f"effective rank will always be 1 regardless of rank_ratio. "
                f"Consider increasing lora_r in the YAML config (e.g., lora_r: 8)."
            )
        lora_alpha   = int(args.get("lora_alpha", 16))
        lora_dropout = float(args.get("lora_dropout", 0.1))
        merge_weights = bool(args.get("merge_weights", True))
        lora_target_modules = args.get("lora_target_modules", None)
        lora_embedding = bool(args.get("lora_embedding", False))

        # ---- 2. Resolve local / remote model path ---------------------------
        base_dir = Path(__file__).resolve().parents[4].parent
        local_name = model_name.replace("/", "--")
        local_dir = base_dir / "hf_models" / local_name
        local_dir.mkdir(parents=True, exist_ok=True)

        has_safetensors = (local_dir / "model.safetensors").exists()
        has_bin = (local_dir / "pytorch_model.bin").exists()
        has_config = (local_dir / "config.json").exists()
        has_local = (has_safetensors or has_bin) and has_config
        model_source = str(local_dir) if has_local else model_name

        # ---- 3. Build HF config ---------------------------------------------
        config = AutoConfig.from_pretrained(
            model_source,
            num_labels=num_labels,
            cache_dir=str(local_dir),
            local_files_only=has_local,
        )

        # Qwen2.5 has no native pad_token; use eos_token as pad_token
        if config.pad_token_id is None:
            config.pad_token_id = getattr(config, "eos_token_id", 151643)

        # ---- 4. Load pretrained model ---------------------------------------
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_source,
                config=config,
                use_safetensors=True,
                cache_dir=str(local_dir),
                local_files_only=has_local,
            )
        except Exception:
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
            self.pad_id = int(args.get("pad_id", getattr(config, "eos_token_id", 151643)))

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
        **kwargs: Any,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
