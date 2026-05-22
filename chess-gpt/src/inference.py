"""
src/inference.py — Decoupled Chess-GPT inference module

Provides a clean interface to load a trained Chess-GPT model and generate
constrained (always-legal) moves. Can be imported by any app.

Usage:
    from src.inference import load_chessgpt, get_gpt_move

    engine = load_chessgpt("artifacts/chessgpt_tiny_300k_best.pt")
    san = get_gpt_move(board, engine, temperature=0.8, top_k=10)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chess
import torch

from src.model import GPT2, GPT2Config
from src.tokenizers import MoveTokenizer


# ── Preset architectures ──────────────────────────────────────────────────────

MODEL_PRESETS: dict[str, dict] = {
    'nano':   dict(d_model=64,  n_heads=2, n_layers=2, d_ff=128),
    'tiny':   dict(d_model=128, n_heads=4, n_layers=4, d_ff=512),
    'small':  dict(d_model=256, n_heads=8, n_layers=6, d_ff=1024),
    'medium': dict(d_model=512, n_heads=8, n_layers=8, d_ff=2048),
}


@dataclass
class ChessGPTEngine:
    """Self-contained engine: model + tokenizer + config. Pass this around."""
    model:     GPT2
    tokenizer: MoveTokenizer
    config:    GPT2Config
    device:    str
    n_params:  int


# ── Loading ───────────────────────────────────────────────────────────────────

def load_chessgpt(
    checkpoint: str | Path,
    model_preset: str = 'tiny',
    tok_path: str | Path = 'artifacts/move_tok.pkl',
    seq_len: int = 128,
    device: Optional[str] = None,
) -> ChessGPTEngine:
    """
    Load a Chess-GPT checkpoint and return a ready-to-use engine.

    Args:
        checkpoint:    Path to .pt file.
        model_preset:  Architecture preset (nano/tiny/small/medium).
                       Ignored if a .json config file exists alongside the checkpoint.
        tok_path:      Path to tokenizer .pkl file.
                       Ignored if a .json config file specifies tok_path.
        seq_len:       Context window length. Ignored if .json config present.
        device:        'cuda' / 'cpu' / 'mps' or None (auto-detect).

    Returns:
        ChessGPTEngine with model in eval mode.
    """
    if device is None:
        device = (
            'cuda' if torch.cuda.is_available() else
            'mps'  if torch.backends.mps.is_available() else
            'cpu'
        )

    ckpt_path = Path(checkpoint)
    cfg_path  = ckpt_path.with_suffix('.json')

    if cfg_path.exists():
        cfg        = json.loads(cfg_path.read_text())
        preset     = {k: cfg[k] for k in ('d_model', 'n_heads', 'n_layers', 'd_ff')}
        vocab_size = cfg['vocab_size']
        pad_id     = cfg['pad_id']
        seq_len    = cfg['max_seq_len']
        tok_path   = Path(cfg.get('tok_path', tok_path))
        print(f'[inference] Config  ← {cfg_path}')
    else:
        preset     = MODEL_PRESETS[model_preset]
        tok_path   = Path(tok_path)
        vocab_size = None
        pad_id     = 0
        print(f'[inference] No .json found, using preset={model_preset}')

    mtok = MoveTokenizer.load(tok_path)
    if vocab_size is None:
        vocab_size = mtok.vocab_size

    config = GPT2Config(
        vocab_size  = vocab_size,
        pad_id      = pad_id,
        d_model     = preset['d_model'],
        n_heads     = preset['n_heads'],
        n_layers    = preset['n_layers'],
        d_ff        = preset['d_ff'],
        max_seq_len = seq_len,
        dropout     = 0.0,
    )
    config.device = device

    model = GPT2(config).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f'[inference] Loaded   params={n_params:,}  device={device}  vocab={vocab_size}')

    return ChessGPTEngine(
        model=model, tokenizer=mtok, config=config,
        device=device, n_params=n_params,
    )


# ── Move generation ───────────────────────────────────────────────────────────

@torch.no_grad()
def get_gpt_move(
    board: chess.Board,
    engine: ChessGPTEngine,
    temperature: float = 0.8,
    top_k: int = 10,
) -> str:
    """
    Generate a legal chess move for the current board position.

    Uses constrained decoding: only legal moves receive non-zero probability.
    Always returns a valid SAN string.

    Args:
        board:        Current chess.Board (will not be modified).
        engine:       Loaded ChessGPTEngine.
        temperature:  Sampling temperature (higher = more random).
        top_k:        Keep only top-K legal moves by logit (0 = all legal).

    Returns:
        SAN string of chosen move (e.g. 'e4', 'Nf3', 'O-O').
    """
    model, tokenizer, config = engine.model, engine.tokenizer, engine.config

    # ── Legal move mask ───────────────────────────────────────────────────────
    legal_sans = [board.san(m) for m in board.legal_moves]
    legal_ids  = [tokenizer.move_to_id[s] for s in legal_sans
                  if s in tokenizer.move_to_id]

    if not legal_ids:
        return board.san(random.choice(list(board.legal_moves)))

    # ── Build token context ───────────────────────────────────────────────────
    tmp, san_history = chess.Board(), []
    for move in board.move_stack:
        san_history.append(tmp.san(move))
        tmp.push(move)

    ids    = tokenizer.encode(' '.join(san_history), add_special=False) if san_history else []
    tensor = (torch.tensor([ids], dtype=torch.long, device=config.device) if ids
              else torch.zeros((1, 1), dtype=torch.long, device=config.device))
    ctx    = tensor[:, -config.max_seq_len:]

    # ── Forward pass + constrained sampling ──────────────────────────────────
    logits = model(ctx)[:, -1, :] / max(temperature, 1e-6)

    mask = torch.full_like(logits, float('-inf'))
    mask[0, legal_ids] = logits[0, legal_ids]

    if top_k > 0 and top_k < len(legal_ids):
        topk_vals, _ = torch.topk(mask, top_k)
        mask[mask < topk_vals[:, -1:]] = float('-inf')

    probs      = torch.softmax(mask, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    move_san   = tokenizer.id_to_move.get(next_token, '')

    if move_san in legal_sans:
        return move_san

    # ── Fallback: argmax over legal ids ──────────────────────────────────────
    best_id  = max(legal_ids, key=lambda i: logits[0, i].item())
    best_san = tokenizer.id_to_move.get(best_id, '')
    if best_san in legal_sans:
        return best_san

    return board.san(random.choice(list(board.legal_moves)))
