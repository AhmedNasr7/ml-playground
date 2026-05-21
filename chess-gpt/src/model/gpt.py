"""
GPT-2 model — self-contained for chess-gpt.

Architecture:
    token_emb + pos_emb  →  dropout
    → N × GPTBlock (pre-norm residual)
    → LayerNorm
    → lm_head  (weight-tied with token_emb)

Input:  (B, T)  token IDs
Output: (B, T, vocab_size)  logits
"""

import torch
import torch.nn as nn

from .config import GPT2Config
from .layers import CausalSelfAttention, GPTFeedForward, LayerNorm


class GPTBlock(nn.Module):
    """
    Single GPT-2 transformer block.

    Pre-norm residual:
        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.norm1 = LayerNorm(config.d_model)
        self.attn  = CausalSelfAttention(config.d_model, config.n_heads, config.max_seq_len, config.dropout)
        self.norm2 = LayerNorm(config.d_model)
        self.ffn   = GPTFeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GPT2(nn.Module):
    """
    GPT-2 language model.

    Input:  (B, T)  token IDs
    Output: (B, T, vocab_size)  logits

    Weight tying: lm_head.weight == token_emb.weight  (saves params, improves perplexity).
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config    = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb   = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop      = nn.Dropout(config.dropout)
        self.blocks    = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layers)])
        self.norm      = LayerNorm(config.d_model)
        self.lm_head   = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        tok_emb = self.token_emb(token_ids)                                          # (B, T, D)
        pos_idx = torch.arange(T, dtype=torch.long, device=token_ids.device)
        x = self.drop(tok_emb + self.pos_emb(pos_idx))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)                                                        # (B, T, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """
        Autoregressive next-token generation.

        Args:
            prompt_ids:    (1, T) starting token IDs
            max_new_tokens: how many new tokens to append
            temperature:   sampling temperature (lower = more greedy)
            top_k:         if set, sample from top-k logits only

        Returns:
            (1, T + max_new_tokens) generated token IDs
        """
        seq = prompt_ids
        for _ in range(max_new_tokens):
            # Crop to context window
            ctx = seq[:, -self.config.max_seq_len:]
            logits = self.forward(ctx)[:, -1, :] / temperature   # (1, vocab)

            if top_k is not None:
                # Zero out all but top-k logits
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float('-inf')

            probs      = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            seq        = torch.cat([seq, next_token], dim=1)
        return seq

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
