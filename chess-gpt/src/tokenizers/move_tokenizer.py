"""
MoveTokenizer — one token per chess move.

Each unique SAN move string (e.g. 'e4', 'Nf3', 'O-O') is a single token.
Move numbers ('1.', '2.') and results ('1-0', '0-1', etc.) are stripped.

Why simpler than BPE for constrained decoding:
    - Token IDs map 1-to-1 with SAN strings
    - At generation time: get board.legal_moves → convert to SAN → look up IDs
    - Mask everything else to -inf before softmax  ← trivial
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import List, Union


_RESULT_RE = re.compile(r'^(1-0|0-1|1/2-1/2|\*)$')
_MOVNUM_RE = re.compile(r'^\d+\.+$')


class MoveTokenizer:
    """
    Vocabulary: special tokens + every unique SAN move seen during build().

    Special token IDs (always at positions 0-3):
        PAD = 0
        BOS = 1   (start of game)
        EOS = 2   (end of game)
        UNK = 3   (unseen move at inference — rare)
    """

    PAD = '<PAD>'
    BOS = '<BOS>'
    EOS = '<EOS>'
    UNK = '<UNK>'
    _SPECIALS = [PAD, BOS, EOS, UNK]

    def __init__(self):
        self.move_to_id: dict[str, int] = {}
        self.id_to_move: dict[int, str] = {}

    # ── build ────────────────────────────────────────────────────────────────

    def build(self, games: List[str]) -> 'MoveTokenizer':
        """Scan all games and assign an ID to every unique move string."""
        move_set: set[str] = set()
        for game in games:
            for m in self._split_moves(game):
                move_set.add(m)

        all_tokens = self._SPECIALS + sorted(move_set)
        self.move_to_id = {t: i for i, t in enumerate(all_tokens)}
        self.id_to_move = {i: t for t, i in self.move_to_id.items()}
        return self

    # ── vocab properties ─────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self.move_to_id)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    # ── encode / decode ──────────────────────────────────────────────────────

    def encode(self, game: str, add_special: bool = False) -> List[int]:
        ids = [self.move_to_id.get(m, self.unk_id) for m in self._split_moves(game)]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def encode_batch(self, games: List[str], add_special: bool = False) -> List[List[int]]:
        return [self.encode(g, add_special) for g in games]

    def decode(self, token_ids: List[int], with_move_numbers: bool = True) -> str:
        """Reconstruct a PGN-like string from token IDs."""
        specials = {self.pad_id, self.bos_id, self.eos_id}
        moves = [self.id_to_move[i] for i in token_ids
                 if i not in specials and i in self.id_to_move and self.id_to_move[i] != self.UNK]
        if not with_move_numbers:
            return ' '.join(moves)
        parts = []
        for i, move in enumerate(moves):
            if i % 2 == 0:
                parts.append(f'{i // 2 + 1}. {move}')
            else:
                parts.append(move)
        return ' '.join(parts)

    def decode_move_list(self, token_ids: List[int]) -> List[str]:
        """Return plain list of SAN strings (no move numbers)."""
        specials = {self.pad_id, self.bos_id, self.eos_id}
        return [self.id_to_move[i] for i in token_ids
                if i not in specials and i in self.id_to_move and self.id_to_move[i] != self.UNK]

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _split_moves(game: str) -> List[str]:
        """Strip move numbers, results; return bare SAN move strings."""
        out = []
        for tok in game.split():
            if _MOVNUM_RE.match(tok):
                continue
            if _RESULT_RE.match(tok):
                continue
            out.append(tok)
        return out

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'move_to_id': self.move_to_id, 'id_to_move': self.id_to_move}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(f'[MoveTokenizer] saved → {path}  vocab_size={self.vocab_size}')

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'MoveTokenizer':
        tok = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        tok.move_to_id = data['move_to_id']
        tok.id_to_move = {int(k): v for k, v in data['id_to_move'].items()}
        print(f'[MoveTokenizer] loaded ← {path}  vocab_size={tok.vocab_size}')
        return tok
