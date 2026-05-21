"""
GPT-2 building blocks — self-contained for chess-gpt.

Imports only from the local src package.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention.mhsa import FusedMultiHeadSelfAttention


class LayerNorm(nn.Module):
    """Pre-norm LayerNorm wrapper (bias=True, eps=1e-5)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


class GELU(nn.Module):
    """Exact GELU (as used in GPT-2)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class CausalSelfAttention(nn.Module):
    """
    Causal (masked) multi-head self-attention.
    Generates a causal mask so each position attends only to past tokens.

    Args:
        d_model:     model dimension
        n_heads:     number of attention heads
        max_seq_len: max context length (pre-allocates mask buffer)
        dropout:     dropout rate
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.attn = FusedMultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        # Upper triangular = True → block this position (causal mask)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        mask = self.causal_mask[:seq_len, :seq_len]
        return self.attn(x, mask)


class GPTFeedForward(nn.Module):
    """
    GPT-2 feed-forward network: Linear → GELU → Linear → Dropout.

    d_ff is typically 4 × d_model.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1     = nn.Linear(d_model, d_ff)
        self.act     = GELU()
        self.fc2     = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))
