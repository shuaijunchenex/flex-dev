import torch
import torch.nn as nn

from ..transformer._nn_model_transformer_encoder import TransformerEncoder
from ...ml_algorithms.lora.impl.lora_cifar_vit import PatchEmbed, Attention, MLP, Block
from ...ml_algorithms.lora import MSLoRALinear
from .. import AbstractNNModel, NNModelArgs, NNModel
from ...ml_utils import console
from typing import Any, Optional

class NNModel_ViTMSLoRACIFAR10(NNModel):
    def __init__(self):
        super().__init__()
        self.lora_mode = "standard"
        self._model: Optional[nn.Module] = None

    # override
    def create_model(self, args: "NNModelArgs") -> "AbstractNNModel":
        super().create_model(args)

        rank = getattr(args, "lora_rank", 4)
        scaling = getattr(args, "lora_scaling", 0.5)
        rank_ratio = getattr(args, "rank_ratio", 1)
        use_bias = getattr(args, "use_bias", True)

        # ViT 
        embed_dim = getattr(args, "embed_dim", 256)
        depth = getattr(args, "depth", 6)
        num_heads = getattr(args, "num_heads", 4)
        patch_size = getattr(args, "patch_size", 4)
        num_classes = getattr(args, "num_classes", 10)

        lora_r = int(max(0, round(rank * rank_ratio)))
        if rank == 1 and rank_ratio != 1.0:
            console.warn(
                f"[ViT LoRA] rank_ratio={rank_ratio} is configured but base lora_rank=1, "
                f"effective rank will always be <=1 regardless of rank_ratio. "
                f"Consider increasing lora_rank in the config (e.g., lora_rank: 8)."
            )
        lora_alpha = int(max(1, round(rank * scaling)))
        lora_dropout = float(getattr(args, "lora_dropout", 0.0))

        self._model = ViT_MSLoRA_CIFAR10(
            img_size=32,
            patch_size=patch_size,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )

        # Declare LoRA configuration for aggregation system
        self._lora_config = {
            "suffix_A": "lora_A",
            "suffix_B": "lora_B",
            "sp_suffix": "sp_aggregated",
            "has_non_lora_params": True,   # cls_token, pos_embed, LayerNorm are non-LoRA
        }

        return self

    # override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def set_lora_mode(self, mode: str):
        if mode not in ["standard", "lora_only", "lora_disabled", "scaling"]:
            raise ValueError(f"Unsupported lora_mode: {mode}")
        self.lora_mode = mode
        if self._model is not None:
            self._model.set_lora_mode(mode)

class ViT_MSLoRA_CIFAR10(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        lora_r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        )
        num_patches = self.patch_embed.num_patches

        # cls token & pos embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # head
        self.head = MSLoRALinear(
            embed_dim, num_classes,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            merge_weights=True,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, C)
        cls_tok = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
        x = torch.cat((cls_tok, x), dim=1)          # (B, 1+N, C)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_feat = x[:, 0]  # 取 cls token
        logits = self.head(cls_feat)
        return logits