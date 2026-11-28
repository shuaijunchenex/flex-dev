from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from ... import AbstractNNModel, NNModel, NNModelArgs

"""
A Tiny Transformer from scratch (Non-pre-trained) 
Specifically designed to be under 5M parameters.
Default params: Vocab 20k, Dim 128, 2 Layers, 4 Heads -> ~3.5M params.
"""

class NNModel_TinyScratchIMDB(NNModel):
    def __init__(self):
        super().__init__()
        self.model: nn.Module | None = None
        self.pad_id: int = 0

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        
        # 严格控制参数量的默认值
        vocab_size = getattr(args, "vocab_size", 20000) 
        embed_dim = getattr(args, "embed_dim", 128)     
        num_heads = getattr(args, "num_heads", 4)
        num_layers = getattr(args, "num_layers", 2)
        num_classes = getattr(args, "num_classes", 2)
        max_seq_len = getattr(args, "max_seq_len", 256)
        
        class TinyTransformer(nn.Module):
            def __init__(self, v_size, e_dim, n_heads, n_layers, n_classes, max_len):
                super().__init__()
                self.embedding = nn.Embedding(v_size, e_dim)
                self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, e_dim))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=e_dim, 
                    nhead=n_heads, 
                    dim_feedforward=e_dim*4, 
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.classifier = nn.Linear(e_dim, n_classes)
                
            def forward(self, input_ids):
                # input_ids: [B, L]
                seq_len = input_ids.size(1)
                x = self.embedding(input_ids) # [B, L, E]
                x = x + self.pos_embedding[:, :seq_len, :]
                
                x = self.transformer(x)
                # 取第一个 token (<cls>) 或者做平均
                x = x.mean(dim=1) 
                return self.classifier(x)

        self.model = TinyTransformer(vocab_size, embed_dim, num_heads, num_layers, num_classes, max_seq_len)
        self.pad_id = getattr(args, "pad_id", 0)
        return self

    def forward(self, input_ids: torch.LongTensor, **kwargs: Any):
        # 这种从零训练的模型通常不接受 attention_mask 等复杂输入
        return self.model(input_ids)
