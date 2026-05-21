import torch
import torch.nn as nn
from .scaled_dot_product import ScaledDotProductAttention

class FusedMultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) using a single fused QKV projection.
    
    Instead of maintaining 3 separate linear layers for Q, K, and V, it uses
    a single massive layer (d_model -> 3 * d_model) to maximize GPU parallelization.
    This is the standard for Encoder-only (ViT/BERT) and Decoder-only (GPT) models.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Fused QKV projection: projects to 3 * d_model in one go
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 1. Project x all at once using self.qkv_proj
        # 3. Apply self.attention(q, k, v, mask)
        # 4. Recombine heads, apply out_proj, and apply dropout
        

        qkv = self.qkv_proj(x) # fused projection: more efficient
        B, seq, _ = qkv.shape

        # Separate the 3 (Q/K/V) BEFORE distributing to heads
        # (B, seq, 3*d_model) -> (B, seq, 3, n_heads, d_k) -> (3, B, n_heads, seq, d_k)
        qkv = qkv.reshape(B, seq, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, n_heads, seq, d_k)

        attn = self.attention(q, k, v, mask)
        attn = attn.permute(0, 2, 1, 3).reshape(B, seq, self.d_model)
        output = self.dropout(self.out_proj(attn))

        return output
        





        


