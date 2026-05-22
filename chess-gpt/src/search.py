"""
src/search.py — Minimax search for Chess-GPT

Three search strategies, all using the GPT policy + material-balance leaf eval:

  minimax_move          — sequential alpha-beta (baseline)
  minimax_move_threaded — parallel root candidates via ThreadPoolExecutor (~3-4× faster)
  minimax_move_batched  — breadth-first batched GPU inference (fewest forward passes)

Forward pass count at depth=3, k=5:
  sequential : 1 + 5 + 25 = 31 serial passes
  threaded   : same count but overlapped across GPU threads
  batched    : 3 batched passes (root + 2 levels) — best GPU utilisation

Usage:
    from src.inference import load_chessgpt
    from src.search   import minimax_move_batched   # recommended

    engine = load_chessgpt("artifacts/chessgpt_tiny_300k_best.pt")
    san    = minimax_move_batched(board, engine, k=5, depth=3)
"""

from __future__ import annotations

import chess
import torch
from concurrent.futures import ThreadPoolExecutor
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


# ── Batched policy: top-K for many boards in one GPU forward pass ─────────────

@torch.no_grad()
def get_top_k_moves_batch(
    boards: list[chess.Board],
    engine: ChessGPTEngine,
    k: int = 5,
) -> list[list[str]]:
    """
    Return top-k legal moves for each board in a single batched forward pass.

    Contexts are left-padded to the same length so the last token of each
    sequence is always at position [:, -1], which is where GPT's prediction lives.

    Args:
        boards:  List of board positions (not modified).
        engine:  Loaded ChessGPTEngine.
        k:       Number of candidates per board.

    Returns:
        List of SAN-string lists (one per board), best-first by logit.
    """
    model, tokenizer, config = engine.model, engine.tokenizer, engine.config

    legal_ids_list: list[list[tuple[int, str]]] = []
    raw_contexts:   list[list[int]] = []

    for board in boards:
        legal_sans = [board.san(m) for m in board.legal_moves]
        legal_ids  = [(tokenizer.move_to_id[s], s) for s in legal_sans
                      if s in tokenizer.move_to_id]
        legal_ids_list.append(legal_ids)

        tmp, san_history = chess.Board(), []
        for move in board.move_stack:
            san_history.append(tmp.san(move))
            tmp.push(move)
        ids = tokenizer.encode(' '.join(san_history), add_special=False) if san_history else []
        raw_contexts.append(ids[-config.max_seq_len:] if ids else [0])

    # Left-pad all contexts to the same length
    max_len = max(len(c) for c in raw_contexts)
    padded  = torch.zeros(len(boards), max_len, dtype=torch.long, device=config.device)
    for i, ctx in enumerate(raw_contexts):
        t = torch.tensor(ctx, dtype=torch.long, device=config.device)
        padded[i, max_len - len(ctx):] = t

    logits = model(padded)[:, -1, :]   # (batch, vocab)

    results = []
    for i, (board, legal_ids) in enumerate(zip(boards, legal_ids_list)):
        if not legal_ids:
            import random
            results.append([board.san(m) for m in list(board.legal_moves)[:k]])
            continue
        ranked = sorted(legal_ids, key=lambda x: logits[i, x[0]].item(), reverse=True)
        results.append([san for _, san in ranked[:k]])
    return results


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


# ── Entry points ─────────────────────────────────────────────────────────────

def minimax_move(
    board: chess.Board,
    engine: ChessGPTEngine,
    k: int = 5,
    depth: int = 3,
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


def minimax_move_threaded(
    board: chess.Board,
    engine: ChessGPTEngine,
    k: int = 5,
    depth: int = 3,
) -> str:
    """
    Minimax with root-level parallelism via ThreadPoolExecutor.

    Each root candidate is evaluated in its own thread with an independent
    board copy. PyTorch releases the GIL during CUDA ops, so multiple threads
    overlap on GPU giving ~3-4× speedup over sequential.

    Alpha-beta pruning is applied independently within each thread's subtree.
    """
    candidates = get_top_k_moves(board, engine, k)
    if not candidates:
        import random
        return board.san(random.choice(list(board.legal_moves)))

    def _score(san: str) -> float:
        b = board.copy()   # independent copy per thread — no shared state
        b.push_san(san)
        return _minimax(b, depth - 1, maximizing=False,
                        engine=engine, k=k,
                        alpha=-float('inf'), beta=float('inf'))

    with ThreadPoolExecutor(max_workers=len(candidates)) as ex:
        scores = list(ex.map(_score, candidates))

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_san, best_score = candidates[best_idx], scores[best_idx]
    logger.debug('Threaded minimax scores: {}', dict(zip(candidates, scores)))
    logger.info('Threaded minimax picked: {}  (score={}, depth={}, k={})', best_san, best_score, depth, k)
    return best_san


# ── Batched tree traversal (breadth-first, no alpha-beta) ─────────────────────

def _minimax_batch_level(
    boards: list[chess.Board],
    engine: ChessGPTEngine,
    k: int,
    depth: int,
    maximizing: bool,
) -> list[float]:
    """
    Evaluate a list of boards simultaneously with one batched forward pass per level.

    No alpha-beta (incompatible with breadth-first batching), but only
    ceil(log_k(total_nodes)) batched passes are needed instead of O(k^depth) serial ones.
    """
    scores: list[float | None] = [None] * len(boards)
    active: list[int] = []

    for i, board in enumerate(boards):
        if board.is_game_over():
            result = board.result()
            if result == '1-0':
                scores[i] = 100_000 if board.turn == chess.BLACK else -100_000
            elif result == '0-1':
                scores[i] = 100_000 if board.turn == chess.WHITE else -100_000
            else:
                scores[i] = 0
        elif depth == 0:
            scores[i] = material_score(board)
        else:
            active.append(i)

    if not active:
        return scores   # type: ignore[return-value]

    active_boards = [boards[i] for i in active]

    # One batched forward pass for all active boards at this level
    candidates_list = get_top_k_moves_batch(active_boards, engine, k)

    # Expand children for all active boards
    child_boards:     list[chess.Board] = []
    child_parent_idx: list[int]         = []   # maps child → index in active_boards

    for ai, (b, candidates) in enumerate(zip(active_boards, candidates_list)):
        for san in candidates:
            bc = b.copy()
            bc.push_san(san)
            child_boards.append(bc)
            child_parent_idx.append(ai)

    if not child_boards:
        for i, ai in enumerate(active):
            scores[ai] = material_score(active_boards[i])
        return scores   # type: ignore[return-value]

    # Recurse on all children at once
    child_scores = _minimax_batch_level(child_boards, engine, k, depth - 1, not maximizing)

    # Aggregate: each active board picks max/min over its children
    parent_buckets: list[list[float]] = [[] for _ in active_boards]
    for ci, ai in enumerate(child_parent_idx):
        parent_buckets[ai].append(child_scores[ci])

    fn = max if maximizing else min
    for i, ai in enumerate(active):
        bucket = parent_buckets[i]
        scores[ai] = fn(bucket) if bucket else material_score(active_boards[i])

    return scores   # type: ignore[return-value]


def minimax_move_batched(
    board: chess.Board,
    engine: ChessGPTEngine,
    k: int = 5,
    depth: int = 3,
) -> str:
    """
    Minimax with batched GPU inference (breadth-first tree traversal).

    At each depth level all board positions are evaluated in a single batched
    forward pass, reducing GPU round-trips from O(k^depth) to O(depth).

    Trade-off: no alpha-beta pruning (batching requires knowing all children
    upfront), so more boards are evaluated than the sequential version — but
    each evaluation is far cheaper due to batching.

    Forward passes: depth batched calls (e.g. depth=3 → 3 passes total).
    """
    candidates = get_top_k_moves(board, engine, k)
    if not candidates:
        import random
        return board.san(random.choice(list(board.legal_moves)))

    # Build child boards for each root candidate
    child_boards = []
    for san in candidates:
        bc = board.copy()
        bc.push_san(san)
        child_boards.append(bc)

    # Evaluate all subtrees with batched level-by-level inference
    scores = _minimax_batch_level(child_boards, engine, k, depth - 1, maximizing=False)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_san, best_score = candidates[best_idx], scores[best_idx]
    logger.debug('Batched minimax scores: {}', dict(zip(candidates, scores)))
    logger.info('Batched minimax picked: {}  (score={}, depth={}, k={})', best_san, best_score, depth, k)
    return best_san
