"""
Chess-GPT evaluation metrics.

Two families:

  Batch-level  (tensor ops, used every training step)
  ─────────────────────────────────────────────────────
  top_k_accuracy   — how often is the true next-move token in the top-k logits?
  move_rank        — median rank of the true token in the sorted distribution

  Game-level  (python-chess, used on generated games)
  ─────────────────────────────────────────────────────
  parse_game_legality   — per-game: legal count, total, first-error index
  legal_move_rate       — mean % of moves that are legal (before first error)
  game_completion_rate  — % of games with zero illegal moves
  avg_legal_length      — mean number of legal half-moves before first error
"""

from __future__ import annotations

from typing import List, Dict

import torch


# ── Batch-level metrics ───────────────────────────────────────────────────────

def top_k_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pad_id: int,
    k: int = 1,
) -> float:
    """
    Fraction of non-padding positions where the true token is in the top-k predictions.

    Args:
        logits : (B, T, V)
        labels : (B, T)
        pad_id : token ID to ignore
        k      : top-k window

    Returns:
        float in [0, 1]
    """
    B, T, V = logits.shape
    flat_logits = logits.reshape(-1, V)          # (B*T, V)
    flat_labels = labels.reshape(-1)             # (B*T,)

    mask = flat_labels != pad_id
    if mask.sum() == 0:
        return 0.0

    topk_preds = torch.topk(flat_logits, k, dim=-1).indices   # (B*T, k)
    correct    = (topk_preds == flat_labels.unsqueeze(-1)).any(dim=-1)
    return (correct & mask).sum().item() / mask.sum().item()


def move_rank(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pad_id: int,
) -> float:
    """
    Median rank of the true next-move token in the model's sorted distribution.

    Rank 1 = the model's top prediction was correct.
    Lower is better.

    Args:
        logits : (B, T, V)
        labels : (B, T)
        pad_id : token ID to ignore

    Returns:
        float — median rank across all non-padding positions
    """
    B, T, V = logits.shape
    flat_logits = logits.reshape(-1, V)
    flat_labels = labels.reshape(-1)

    mask = flat_labels != pad_id
    if mask.sum() == 0:
        return float('nan')

    # argsort descending → rank of true token
    sorted_idx = torch.argsort(flat_logits, dim=-1, descending=True)   # (B*T, V)
    ranks = (sorted_idx == flat_labels.unsqueeze(-1)).nonzero(as_tuple=False)[:, 1] + 1

    valid_ranks = ranks[mask]
    return float(valid_ranks.float().median().item())


# ── Game-level metrics ────────────────────────────────────────────────────────

def parse_game_legality(game_str: str) -> Dict:
    """
    Walk through a generated PGN-like string and validate moves.

    Returns dict:
        legal        — consecutive legal moves from the start
        total        — total non-trivial tokens examined (up to and including first error)
        legal_pct    — legal / total * 100
        completed    — True if every move was legal (no error encountered)
        first_error  — SAN of the first illegal token (or None)
    """
    try:
        import chess
    except ImportError:
        raise ImportError("pip install chess")

    board = chess.Board()
    tokens = game_str.split()
    legal = 0
    total = 0
    first_error = None

    for tok in tokens:
        if tok.endswith('.') or tok in ('1-0', '0-1', '1/2-1/2', '*'):
            continue
        try:
            move = board.parse_san(tok)
            board.push(move)
            legal += 1
            total += 1
        except Exception:
            first_error = tok
            total += 1
            break

    return {
        'legal':       legal,
        'total':       total,
        'legal_pct':   legal / max(total, 1) * 100,
        'completed':   first_error is None,
        'first_error': first_error,
    }


def legal_move_rate(games: List[str]) -> float:
    """Mean legal_pct across a list of generated games."""
    if not games:
        return 0.0
    return sum(parse_game_legality(g)['legal_pct'] for g in games) / len(games)


def game_completion_rate(games: List[str]) -> float:
    """Fraction of games with zero illegal moves (100% legal)."""
    if not games:
        return 0.0
    return sum(1 for g in games if parse_game_legality(g)['completed']) / len(games)


def avg_legal_length(games: List[str]) -> float:
    """Mean number of consecutive legal half-moves before the first error."""
    if not games:
        return 0.0
    return sum(parse_game_legality(g)['legal'] for g in games) / len(games)


# ── Convenience: compute all game-level metrics at once ───────────────────────

def game_metrics(games: List[str]) -> Dict:
    """
    Run all game-level metrics on a list of generated games.

    Returns dict with:
        legal_move_rate      — mean % legal moves per game
        game_completion_rate — % fully legal games
        avg_legal_length     — mean legal moves before first error
        n_games              — number of games evaluated
    """
    results = [parse_game_legality(g) for g in games]
    n = len(results)
    if n == 0:
        return {'legal_move_rate': 0.0, 'game_completion_rate': 0.0,
                'avg_legal_length': 0.0, 'n_games': 0}
    return {
        'legal_move_rate':      sum(r['legal_pct']  for r in results) / n,
        'game_completion_rate': sum(r['completed']  for r in results) / n * 100,
        'avg_legal_length':     sum(r['legal']      for r in results) / n,
        'n_games':              n,
    }
