"""
Scaled Dot-Product Attention (Vaswani et al., 2017).

The fundamental attention operation. All other attention variants
build on top of or modify this.
"""

import torch.nn.functional as F
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention.

    Input:
        Q: (batch, ..., seq_q, d_k)
        K: (batch, ..., seq_k, d_k)
        V: (batch, ..., seq_k, d_v)
        mask: optional, broadcastable to (batch, ..., seq_q, seq_k)
              positions with True / 1 are masked (set to -inf before softmax)

    Output: (batch, ..., seq_q, d_v)
    """

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:

        d_k = K.shape[-1]
        attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill_(mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        self.attn_weights = attn_weights.detach()
        output = attn_weights @ V

        return output


        
