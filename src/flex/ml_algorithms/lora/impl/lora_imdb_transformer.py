from torch import nn
import torch.nn.functional as F
import torch
import math
from .lora_ms import MSLoRALinear, MSLoRAConv2d
from typing import Optional

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x):  # (B,L,C)
        return x + self.pe[:x.size(1)].unsqueeze(0)

class MSLoRA_MHA(nn.Module):
    def __init__(self, dim, heads, qkv_bias, attn_drop, proj_drop, lora_r, lora_alpha, lora_dropout):
        super().__init__()
        assert dim % heads == 0
        self.h = heads; self.d = dim // heads; self.scale = self.d ** -0.5
        self.qkv = MSLoRALinear(dim, dim*3, bias=qkv_bias, r=lora_r, lora_alpha=lora_alpha,
                                lora_dropout=lora_dropout, merge_weights=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = MSLoRALinear(dim, dim, bias=True, r=lora_r, lora_alpha=lora_alpha,
                                 lora_dropout=lora_dropout, merge_weights=True)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x, attn_mask=None):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.h,self.d).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        if attn_mask is not None:
            mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,N)
            attn = attn.masked_fill(mask==0, float("-inf"))
        attn = F.softmax(attn, dim=-1); attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x); x = self.proj_drop(x)
        return x

class MSLoRA_FFN(nn.Module):
    def __init__(self, dim, mlp_ratio, lora_r, lora_alpha, lora_dropout, drop):
        super().__init__()
        h = int(dim*mlp_ratio)
        self.fc1 = MSLoRALinear(dim, h, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=True)
        self.act = nn.GELU()
        self.fc2 = MSLoRALinear(h, dim, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=True)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x); return x

class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, qkv_bias, attn_drop, drop, lora_r, lora_alpha, lora_dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.mha = MSLoRA_MHA(dim, heads, qkv_bias, attn_drop, drop, lora_r, lora_alpha, lora_dropout)
        self.n2 = nn.LayerNorm(dim)
        self.ffn = MSLoRA_FFN(dim, mlp_ratio, lora_r, lora_alpha, lora_dropout, drop)
    def forward(self, x, mask=None):
        x = x + self.mha(self.n1(x), attn_mask=mask)
        x = x + self.ffn(self.n2(x)); return x
