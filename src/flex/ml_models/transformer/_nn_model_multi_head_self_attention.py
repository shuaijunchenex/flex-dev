import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import AbstractNNModel, NNModelArgs, NNModel


class MultiHeadSelfAttention(NNModel):
    """
    Multi-Head Self-Attention mechanism.
    Projects input into Q/K/V, computes scaled dot-product attention,
    and merges multiple heads.
    """
    def __init__(self):
        super().__init__()
        self.dropout = None
        self.out_proj = None
        self.value_proj = None
        self.key_proj = None
        self.query_proj = None
        self.scale = None
        self.head_dim = None
        self.feature_dim = None
        self.num_heads = None
        return

    # override
    def create_args(self):
        args = super().create_args()
        args.feature_dim = 0
        args.num_heads = 8
        args.dropout = 0.0
        return args

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        self.num_heads = args.num_heads
        self.feature_dim = args.feature_dim
        self.head_dim = args.feature_dim // args.num_heads
        self.scale = self.head_dim ** 0.5

        # Linear projections for query, key, and value
        self.query_proj = nn.Linear(args.feature_dim, args.feature_dim)
        self.key_proj = nn.Linear(args.feature_dim, args.feature_dim)
        self.value_proj = nn.Linear(args.feature_dim, args.feature_dim)
        self.out_proj = nn.Linear(args.feature_dim, args.feature_dim)
        self.dropout = nn.Dropout(args.dropout)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape (batch_size, seq_len, feature_dim)
        :return: Output tensor of same shape
        """
        batch_size, seq_len, _ = x.size()

        # Project and reshape for multi-head
        query = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention scores
        attention_scores = torch.einsum("bhqd, bhkd -> bhqk", query, key) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights to values
        attended_values = torch.einsum("bhqk, bhvd -> bhqd", attention_weights, value)

        # Concatenate attention heads and project
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim)
        output = self.dropout(self.out_proj(attended_values))
        return output



