"""
src/search.py — Minimax search for Chess-GPT

Uses the GPT policy to sample candidate moves at each node, then applies
minimax with alpha-beta pruning and a material-balance leaf evaluator.

Completely decoupled from the UI — works with any ChessGPTEngine.

Usage:
    from src.inference import load_chessgpt
    from src.search   import minimax_move

    engine = load_chessgpt("artifacts/chessgpt_tiny_300k_best.pt")
    san    = minimax_move(board, engine, k=5, depth=2)
"""

from __future__ import annotations

import chess
import torch
from loguru import logger

from src.inference import ChessGPTEngine


# ── Piece values (centipawns) ─────────────────────────────────────────────────

PIECE_VALUES: dict[int, int] = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:     0,
}


# ── Evaluator ─────────────────────────────────────────────────────────────────

def material_score(board: chess.Board) -> int:
    """
    Static material-balance evaluation.

    Returns a score in centipawns from the perspective of the side to move:
    positive = side to move is ahead, negative = behind.
    """
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score if board.turn == chess.WHITE else -score


# ── Policy: top-K legal moves by raw logit ────────────────────────────────────

@torch.no_grad()
def get_top_k_moves(
    board: chess.Board,
    engine: ChessGPTEngine,
    k: int = 5,
) -> list[str]:
    """
    Return up to k legal SAN moves ranked by the model's raw logit (no sampling).

    Args:
        board:   Current position (not modified).
        engine:  Loaded ChessGPTEngine.
        k:       Number of candidates to return.

    Returns:
        List of SAN strings, best-first by logit.
    """
    model, tokenizer, config = engine.model, engine.tokenizer, engine.config

    legal_sans = [board.san(m) for m in board.legal_moves]
    legal_ids  = [(tokenizer.move_to_id[s], s) for s in legal_sans
                  if s in tokenizer.move_to_id]

    if not legal_ids:
        import random
        return [board.san(m) for m in list(board.legal_moves)[:k]]

    # Build context from move history
    tmp, san_history = chess.Board(), []
    for move in board.move_stack:
        san_history.append(tmp.san(move))
        tmp.push(move)

    ids    = tokenizer.encode(' '.join(san_history), add_special=False) if san_history else []
    tensor = (torch.tensor([ids], dtype=torch.long, device=config.device) if ids
              else torch.zeros((1, 1), dtype=torch.long, device=config.device))
    ctx    = tensor[:, -config.max_seq_len:]

    logits = model(ctx)[0, -1, :]   # (vocab_size,)

    ranked = sorted(legal_ids, key=lambda x: logits[x[0]].item(), reverse=True)
    top = [san for _, san in ranked[:k]]
    logger.debug('Top-{} candidates: {}', k, top)
    return top


# ── Minimax with alpha-beta pruning ───────────────────────────────────────────

def _minimax(
    board: chess.Board,
    depth: int,
    maximizing: bool,
    engine: ChessGPTEngine,
    k: int,
    alpha: float,
    beta: float,
) -> float:
    """Recursive minimax helper. Returns score from the root player's perspective."""
    if board.is_game_over():
        result = board.result()
        if result == '1-0':
            return 100_000 if board.turn == chess.BLACK else -100_000
        if result == '0-1':
            return 100_000 if board.turn == chess.WHITE else -100_000
        return 0   # draw / stalemate

    if depth == 0:
        return material_score(board)

    candidates = get_top_k_moves(board, engine, k)

    if maximizing:
        value = -float('inf')
        for san in candidates:
            board.push_san(san)
            value = max(value, _minimax(board, depth - 1, False, engine, k, alpha, beta))
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break   # beta cut-off
        return value
    else:
        value = float('inf')
        for san in candidates:
            board.push_san(san)
            value = min(value, _minimax(board, depth - 1, True, engine, k, alpha, beta))
            board.pop()
            beta = min(beta, value)
            if beta <= alpha:
                break   # alpha cut-off
        return value


# ── Public entry point ────────────────────────────────────────────────────────

def minimax_move(
    board: chess.Board,
    engine: ChessGPTEngine,
    k: int = 5,
    depth: int = 2,
) -> str:
    """
    Pick the best move using minimax over GPT-sampled candidates.

    At each node the model's top-k moves (by logit) are expanded, and leaf
    positions are scored by material balance. Alpha-beta pruning keeps it fast.

    Complexity: O(k^depth) forward passes — default 5^2 = 25, very fast.

    Args:
        board:   Current position (not modified).
        engine:  Loaded ChessGPTEngine.
        k:       Branching factor — top-k candidates per node.
        depth:   Search depth in plies (2 = look 1 move ahead for each side).

    Returns:
        SAN string of the chosen move.
    """
    candidates = get_top_k_moves(board, engine, k)
    if not candidates:
        import random
        return board.san(random.choice(list(board.legal_moves)))

    best_san, best_score = candidates[0], -float('inf')

    logger.debug('Minimax search  depth={}  k={}  candidates={}', depth, k, candidates)
    for san in candidates:
        board.push_san(san)
        score = _minimax(
            board, depth - 1, maximizing=False,
            engine=engine, k=k,
            alpha=-float('inf'), beta=float('inf'),
        )
        board.pop()
        logger.debug('  {} → score={}', san, score)
        if score > best_score:
            best_score, best_san = score, san

    logger.info('Minimax picked: {}  (score={}, depth={}, k={})', best_san, best_score, depth, k)
    return best_san
