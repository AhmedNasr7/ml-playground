from dataclasses import dataclass


@dataclass
class GPT2Config:
    """Configuration for GPT-2."""

    # --- Vocabulary (set by your tokenizer after training) ---
    vocab_size: int = 10_000
    pad_id: int = 0

    # --- Architecture ---
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048            # typically 4 * d_model
    max_seq_len: int = 512
    dropout: float = 0.1

    # --- Training ---
    batch_size: int = 32
    lr: float = 3e-4
    epochs: int = 10
    warmup_steps: int = 2000

    device = 'cuda'
