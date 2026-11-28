from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from ... import AbstractNNModel, NNModel, NNModelArgs

"""
Decoder-only Transformer (GPT-style) for IMDB text classification.
"""

class NNModel_DecoderOnlyIMDB(NNModel):
    def __init__(self):
        super().__init__()
        self.model: nn.Module | None = None
        self.pad_id: int = 50256 # Default GPT2 pad_token_id

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        from transformers import AutoModelForSequenceClassification, AutoConfig

        model_name = getattr(args, "pretrained_model", "gpt2")
        num_labels = getattr(args, "num_classes", 2)
        
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        # GPT2 does not have a pad token by default, we often use eos_token
        config.pad_token_id = config.eos_token_id
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
        self.pad_id = config.pad_token_id
        return self

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor | None = None, **kwargs: Any):
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs[0]
