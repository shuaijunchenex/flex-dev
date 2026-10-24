from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from ... import AbstractNNModel, NNModel, NNModelArgs

"""
Hybrid Transformer model (Conv + Transformer) for IMDB classification.
"""

class NNModel_HybridIMDB(NNModel):
    def __init__(self):
        super().__init__()
        self.model: nn.Module | None = None
        self.pad_id: int = 0

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        
        vocab_size = getattr(args, "vocab_size", 30000)
        embed_dim = getattr(args, "embed_dim", 128)
        num_classes = getattr(args, "num_classes", 2)
        num_heads = getattr(args, "num_heads", 4)
        num_layers = getattr(args, "num_layers", 2)
        
        class SimpleHybrid(nn.Module):
            def __init__(self, vocab_size, embed_dim, num_classes, num_heads, num_layers):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                # Hybrid part: Conv1D to extract local features before transformer
                self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
                encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.fc = nn.Linear(embed_dim, num_classes)
                
            def forward(self, input_ids, attention_mask=None):
                x = self.embedding(input_ids) # [B, L, D]
                x = x.transpose(1, 2) # [B, D, L]
                x = torch.relu(self.conv(x))
                x = x.transpose(1, 2) # [B, L, D]
                
                # Transformer expects [B, L, D] if batch_first=True
                x = self.transformer(x)
                # Global average pooling
                x = x.mean(dim=1)
                return self.fc(x)

        self.model = SimpleHybrid(vocab_size, embed_dim, num_classes, num_heads, num_layers)
        self.pad_id = getattr(args, "pad_id", 0)
        return self

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor | None = None, **kwargs: Any):
        return self.model(input_ids)
