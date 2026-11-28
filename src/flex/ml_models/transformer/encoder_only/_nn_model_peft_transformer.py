from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn

from ... import AbstractNNModel, NNModel, NNModelArgs
from ...ml_utils import console

"""
PEFT LoRA-wrapped Transformer encoder for text classification.
Uses HuggingFace transformers + peft to inject LoRA adapters.
"""

class NNModel_PeftTransformer(NNModel):
    def __init__(self):
        super().__init__()
        self.model: nn.Module | None = None
        self.pad_id: int = 0

    # override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        from transformers import AutoModelForSequenceClassification
        from peft import LoraConfig, get_peft_model

        model_name = getattr(args, "pretrained_model", "bert-base-uncased")
        num_labels = getattr(args, "num_classes", 2)
        base_lora_rank = getattr(args, "lora_rank", 8)
        rank_ratio = getattr(args, "rank_ratio", 1)
        lora_r = int(max(1, round(base_lora_rank * rank_ratio)))
        if base_lora_rank == 1 and rank_ratio != 1.0:
            console.warn(
                f"[PEFT Transformer] rank_ratio={rank_ratio} is configured but base lora_rank=1, "
                f"effective rank will always be 1 regardless of rank_ratio. "
                f"Consider increasing lora_rank in the config (e.g., lora_rank: 8)."
            )
        lora_alpha = getattr(args, "lora_alpha", 16)
        lora_dropout = getattr(args, "lora_dropout", 0.05)
        lora_bias = getattr(args, "lora_bias", "none")
        target_modules = getattr(args, "lora_target_modules", ("query", "value"))
        task_type = getattr(args, "lora_task_type", "SEQ_CLS")

        base = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

        peft_cfg = LoraConfig(
            task_type=task_type,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(target_modules) if isinstance(target_modules, Iterable) else target_modules,
            bias=lora_bias,
        )

        self.model = get_peft_model(base, peft_cfg)
        self.pad_id = getattr(args, "pad_id", 0)
        return self

    # override
    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor | None = None, **kwargs: Any):
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs[0]
