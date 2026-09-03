from torch import nn
import torch
from .lora_ms import MSLoRALinear, MSLoRAConv2d
from typing import Optional

class PatchEmbed(nn.Module):
    """
    用 MSLoRAConv2d 做 ViT 的 Patch Embedding：
    - 等价于 kernel=patch_size, stride=patch_size 的卷积
    - 输出 shape: (B, N, C) 其中 N = (H/patch)*(W/patch), C=embed_dim
    """
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 256,
        lora_r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = MSLoRAConv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=bias,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=True,
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.proj(x)               # (B, C, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        lora_r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = MSLoRALinear(
            dim, dim * 3,
            bias=qkv_bias,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            merge_weights=True,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = MSLoRALinear(
            dim, dim,
            bias=True,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            merge_weights=True,
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)  # (B, heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
        lora_r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = MSLoRALinear(
            in_features, hidden_features,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            merge_weights=True,
        )
        self.act = nn.GELU()
        self.fc2 = MSLoRALinear(
            hidden_features, out_features,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            merge_weights=True,
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        lora_r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x