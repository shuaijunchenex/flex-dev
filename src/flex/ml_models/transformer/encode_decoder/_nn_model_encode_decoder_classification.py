from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from ... import AbstractNNModel, NNModel, NNModelArgs

"""
Generic Encoder-Decoder Transformer for Sequence Classification.
Can be used for SST2, MRPC, DBpedia, etc.
"""

class NNModel_EncodeDecoderClassification(NNModel):
    def __init__(self):
        super().__init__()
        self.model: nn.Module | None = None
        self.pad_id: int = 1

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        from transformers import AutoModelForSequenceClassification

        model_name = getattr(args, "pretrained_model", "facebook/bart-base")
        num_labels = getattr(args, "num_classes", 2)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.pad_id = getattr(args, "pad_id", 1)
        return self

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor | None = None, **kwargs: Any):
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs[0]
