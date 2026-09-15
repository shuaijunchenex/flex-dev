import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from flex.ml_models.transformer._nn_model_multi_head_self_attention import MultiHeadSelfAttention

#from ._nn_model_multi_head_self_attention import MultiHeadSelfAttention

class TransformerEncoder(nn.Module):
    """
    A basic Transformer Encoder block consisting of:
    - LayerNorm
    - Multi-Head Self-Attention
    - Feedforward MLP
    Each sub-block uses residual connections.
    """
    def __init__(self, feature_dim: int, mlp_hidden_dim: int, num_heads: int = 8, dropout: float = 0.0):
        """
        :param feature_dim: Dimension of input token embeddings
        :param mlp_hidden_dim: Hidden dimension in the MLP block
        :param num_heads: Number of attention heads
        :param dropout: Dropout rate
        """
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(feature_dim)
        self.attention = MultiHeadSelfAttention(feature_dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape (batch_size, sequence_length, feature_dim)
        :return: Output tensor with the same shape
        """
        # Apply LayerNorm, Attention, and add residual connection
        attention_out = self.attention(self.norm1(x)) + x
        # Apply LayerNorm, MLP, and another residual connection
        output = self.mlp(self.norm2(attention_out)) + attention_out
        return output

