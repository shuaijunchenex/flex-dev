import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .. import AbstractNNModel, NNModelArgs, NNModel
from ...ml_algorithms.lora import MSLoRALinear, MSLoRAConv2d, MSEmbedding
from ...ml_algorithms.lora.impl.lora_imdb_transformer import SinusoidalPositionalEncoding, EncoderBlock
from ...ml_utils import console

class NNModel_ImdbMSLoRATransformer(NNModel):
    """
    Transformer-based text classifier with LoRA-enabled layers.
    Inherits from NNModel to be compatible with the framework.
    """

    def __init__(self):
        super().__init__()
        self.lora_mode = "standard"

    # override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        vocab_size = getattr(args, "vocab_size")
        embed_dim = getattr(args, "embed_dim", 256)
        depth = getattr(args, "depth", 6)
        heads = getattr(args, "heads", 4)
        mlp_ratio = getattr(args, "mlp_ratio", 4.0)
        max_len = getattr(args, "max_len", 256)
        num_classes = getattr(args, "num_classes", 2)
        qkv_bias = getattr(args, "qkv_bias", True)
        drop_rate = getattr(args, "drop_rate", 0.1)
        attn_drop = getattr(args, "attn_drop", 0.1)

        base_lora_r = getattr(args, "lora_rank", 8)
        rank_ratio = getattr(args, "rank_ratio", 1)
        effective_r = int(base_lora_r * rank_ratio)
        if base_lora_r == 1 and rank_ratio != 1.0:
            console.warn(
                f"[IMDB LoRA Transformer] rank_ratio={rank_ratio} is configured but base lora_rank=1, "
                f"effective rank will always be <=1 regardless of rank_ratio. "
                f"Consider increasing lora_rank in the config (e.g., lora_rank: 8)."
            )
        lora_alpha = getattr(args, "lora_alpha", 16)
        lora_dropout = getattr(args, "lora_dropout", 0.0)

        # embedding
        self.embed = MSEmbedding(
            vocab_size, embed_dim, r=effective_r, lora_alpha=lora_alpha,
            merge_weights=True, padding_idx=None
        )
        self.pos = SinusoidalPositionalEncoding(embed_dim, max_len=max_len)
        self.drop = nn.Dropout(drop_rate)

        # transformer blocks
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, heads, mlp_ratio, qkv_bias, attn_drop, drop_rate,
                         effective_r, lora_alpha, lora_dropout) for _ in range(depth)
        ])

        # normalization + classifier head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = MSLoRALinear(
            embed_dim, num_classes, r=effective_r, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout, merge_weights=True
        )

        # Declare LoRA configuration for aggregation system
        self._lora_config = {
            "suffix_A": "lora_A",
            "suffix_B": "lora_B",
            "sp_suffix": "sp_aggregated",
            "has_non_lora_params": True,   # LayerNorm layers are non-LoRA
        }

        return self

    # override
    def forward(self, x_ids: torch.LongTensor, attn_mask: Optional[torch.LongTensor] = None):
        x = self.embed(x_ids)
        x = self.pos(x)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, mask=attn_mask)
        x = self.norm(x)
        cls = x[:, 0, :]  # [CLS] token representation
        return self.head(cls)

    def set_lora_mode(self, mode: str):
        """
        Set LoRA mode for all LoRA-enabled layers.
        """
        if mode not in ["standard", "lora_only", "lora_disabled", "scaling"]:
            raise ValueError(f"Unsupported lora_mode: {mode}")
        self.lora_mode = mode

        # apply to embedding and head
        self.embed.lora_mode = mode
        self.head.lora_mode = mode

        # apply to each block (need EncoderBlock to propagate mode)
        for blk in self.blocks:
            if hasattr(blk, "set_lora_mode"):
                blk.set_lora_mode(mode)
            else:
                # if EncoderBlock doesn't have set_lora_mode, assign manually to its LoRA sublayers
                for m in blk.modules():
                    if hasattr(m, "lora_mode"):
                        m.lora_mode = mode
