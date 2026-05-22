#!/usr/bin/env python3
"""
play_chessgpt.py — Chess-GPT interactive web app (FastAPI + chessboard.js)

Drag-and-drop chess board. FastAPI serves the page and handles GPT inference.
All move validation is done server-side with python-chess.

Usage:
    python play_chessgpt.py
    python play_chessgpt.py --checkpoint artifacts/chessgpt_tiny_300k_best.pt
    python play_chessgpt.py --port 8000 --color black
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import chess
import chess.svg
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from loguru import logger
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from src.inference import load_chessgpt, get_gpt_move, ChessGPTEngine
from src.search   import minimax_move, minimax_move_threaded, minimax_move_batched


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', default='artifacts/chessgpt_tiny_300k_best.pt')
    p.add_argument('--model',      default='tiny')
    p.add_argument('--tok_path',   default='artifacts/move_tok.pkl')
    p.add_argument('--port',       type=int, default=8000)
    p.add_argument('--host',       default='0.0.0.0')
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def board_from_history(history_uci: List[str]) -> chess.Board:
    board = chess.Board()
    for uci in history_uci:
        board.push_uci(uci)
    return board


def get_pgn(history_uci: List[str]) -> str:
    board = chess.Board()
    parts = []
    for i, uci in enumerate(history_uci):
        move = chess.Move.from_uci(uci)
        san  = board.san(move)
        if i % 2 == 0:
            parts.append(f'{i // 2 + 1}. {san}')
        else:
            parts.append(san)
        board.push(move)
    return ' '.join(parts)


def board_status(board: chess.Board) -> str:
    if board.is_checkmate():
        return 'checkmate'
    if board.is_stalemate():
        return 'stalemate'
    if board.is_insufficient_material():
        return 'insufficient_material'
    if board.is_fifty_moves():
        return 'fifty_moves'
    if board.is_repetition(3):
        return 'repetition'
    return 'ongoing'


# ── FastAPI app ────────────────────────────────────────────────────────────────

args         = parse_args()
app          = FastAPI(title='Chess-GPT')
engine: Optional[ChessGPTEngine] = None
GAME_HISTORY: List[dict] = []   # last 5 completed/abandoned games
MAX_HISTORY = 5

# chessboard.js piece name → python-chess Piece
_PIECE_MAP = {
    'wK': chess.Piece(chess.KING,   chess.WHITE),
    'wQ': chess.Piece(chess.QUEEN,  chess.WHITE),
    'wR': chess.Piece(chess.ROOK,   chess.WHITE),
    'wB': chess.Piece(chess.BISHOP, chess.WHITE),
    'wN': chess.Piece(chess.KNIGHT, chess.WHITE),
    'wP': chess.Piece(chess.PAWN,   chess.WHITE),
    'bK': chess.Piece(chess.KING,   chess.BLACK),
    'bQ': chess.Piece(chess.QUEEN,  chess.BLACK),
    'bR': chess.Piece(chess.ROOK,   chess.BLACK),
    'bB': chess.Piece(chess.BISHOP, chess.BLACK),
    'bN': chess.Piece(chess.KNIGHT, chess.BLACK),
    'bP': chess.Piece(chess.PAWN,   chess.BLACK),
}


@app.on_event('startup')
def startup():
    global engine
    logger.info(f'Loading model from {args.checkpoint}')
    engine = load_chessgpt(args.checkpoint, model_preset=args.model, tok_path=args.tok_path)
    logger.info(f'Model ready — params={engine.n_params:,}  device={engine.device}  vocab={engine.config.vocab_size}')
    logger.info('Piece images will be served via chess.svg (no CDN needed)')


# ── API schemas ────────────────────────────────────────────────────────────────

class NewGameRequest(BaseModel):
    human_color:  str   = 'white'
    temperature:  float = 0.8
    top_k:        int   = 10
    search_mode:  str   = 'greedy'   # greedy | minimax | threaded | batched
    minimax_k:    int   = 5
    minimax_depth: int  = 3

class MoveRequest(BaseModel):
    history_uci:  List[str]
    move_uci:     str
    human_color:  str   = 'white'
    temperature:  float = 0.8
    top_k:        int   = 10
    search_mode:  str   = 'greedy'
    minimax_k:    int   = 5
    minimax_depth: int  = 3

class SaveGameRequest(BaseModel):
    history_uci: List[str]
    result: str = 'abandoned'
    human_color: str = 'white'


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post('/api/new_game')
def api_new_game(req: NewGameRequest):
    logger.info(f'New game | human_color={req.human_color}')
    board       = chess.Board()
    history_uci = []
    gpt_san     = None
    gpt_uci     = None

    if req.human_color == 'black':
        san     = _gpt_move(board, req.temperature, req.top_k, req.search_mode, req.minimax_k, req.minimax_depth)
        move    = board.parse_san(san)
        gpt_uci = move.uci()
        gpt_san = san
        board.push(move)
        history_uci.append(gpt_uci)
        logger.info(f'GPT opens with {gpt_san}')

    return {
        'fen':         board.fen(),
        'history_uci': history_uci,
        'gpt_san':     gpt_san,
        'gpt_uci':     gpt_uci,
        'status':      board_status(board),
        'pgn':         get_pgn(history_uci),
        'orientation': 'white' if req.human_color == 'white' else 'black',
    }


@app.post('/api/move')
def api_move(req: MoveRequest):
    logger.info(f'Move request | move={req.move_uci}  history_len={len(req.history_uci)}')

    # Reconstruct board from history
    try:
        board = board_from_history(req.history_uci)
    except Exception as e:
        logger.error(f'Invalid history: {e}')
        return {'ok': False, 'error': f'Invalid game history: {e}'}

    # Validate human move
    try:
        human_move = chess.Move.from_uci(req.move_uci)
        if human_move not in board.legal_moves:
            logger.warning(f'Illegal move {req.move_uci} in position {board.fen()}')
            return {'ok': False, 'error': f'Illegal move: {req.move_uci}'}
    except Exception as e:
        logger.error(f'Bad UCI move {req.move_uci}: {e}')
        return {'ok': False, 'error': str(e)}

    human_san   = board.san(human_move)
    board.push(human_move)
    history_uci = req.history_uci + [req.move_uci]
    logger.info(f'Human played {human_san} | fen={board.fen()}')

    status = board_status(board)
    if status != 'ongoing':
        logger.info(f'Game over: {status}')
        return {
            'ok': True, 'human_san': human_san,
            'gpt_san': None, 'gpt_uci': None,
            'fen': board.fen(), 'history_uci': history_uci,
            'status': status, 'pgn': get_pgn(history_uci),
        }

    # GPT responds
    gpt_san  = _gpt_move(board, req.temperature, req.top_k, req.search_mode, req.minimax_k, req.minimax_depth)
    gpt_move = board.parse_san(gpt_san)
    gpt_uci  = gpt_move.uci()
    board.push(gpt_move)
    history_uci.append(gpt_uci)
    logger.info(f'GPT played  {gpt_san} | fen={board.fen()}')

    status = board_status(board)
    if status != 'ongoing':
        logger.info(f'Game over after GPT move: {status}')
        _save_game(history_uci, status, req.human_color)

    return {
        'ok': True, 'human_san': human_san,
        'gpt_san': gpt_san, 'gpt_uci': gpt_uci,
        'fen': board.fen(), 'history_uci': history_uci,
        'status': status, 'pgn': get_pgn(history_uci),
    }


def _save_game(history_uci: List[str], result: str, human_color: str):
    if not history_uci:
        return
    pgn = get_pgn(history_uci)
    entry = {
        'id':          len(GAME_HISTORY) + 1,
        'pgn':         pgn,
        'result':      result,
        'moves':       len(history_uci),
        'human_color': human_color,
        'time':        datetime.now().strftime('%H:%M'),
    }
    GAME_HISTORY.append(entry)
    if len(GAME_HISTORY) > MAX_HISTORY:
        GAME_HISTORY.pop(0)
    logger.info(f'Game saved: {result}  moves={len(history_uci)}')


@app.post('/api/save_game')
def api_save_game(req: SaveGameRequest):
    _save_game(req.history_uci, req.result, req.human_color)
    return {'ok': True, 'history': list(reversed(GAME_HISTORY))}


@app.get('/api/history')
def api_history():
    return {'history': list(reversed(GAME_HISTORY))}


_SEARCH_FN = {
    'greedy':   None,
    'minimax':  minimax_move,
    'threaded': minimax_move_threaded,
    'batched':  minimax_move_batched,
}

def _gpt_move(
    board: chess.Board,
    temperature: float,
    top_k: int,
    search_mode: str = 'greedy',
    minimax_k: int = 5,
    minimax_depth: int = 3,
) -> str:
    legal_sans = [board.san(m) for m in board.legal_moves]
    fn = _SEARCH_FN.get(search_mode)
    if fn is not None:
        logger.info(f'Search mode={search_mode}  k={minimax_k}  depth={minimax_depth}')
        san = fn(board, engine, k=minimax_k, depth=minimax_depth)
    else:
        san = get_gpt_move(board, engine, temperature=temperature, top_k=top_k)
    if san not in legal_sans:
        import random
        san = board.san(random.choice(list(board.legal_moves)))
        logger.warning(f'GPT fallback to random: {san}')
    return san


# ── HTML frontend ─────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Chess-GPT</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #1a1a2e; color: #eee; font-family: 'Segoe UI', sans-serif; min-height: 100vh; }
  h1 { text-align: center; padding: 18px 0 8px; font-size: 1.8rem; color: #a8d8ea; letter-spacing: 2px; }
  #app { display: flex; justify-content: center; gap: 32px; padding: 16px; flex-wrap: wrap; }
  #board-wrap { width: 480px; }
  #board { width: 480px; }
  #sidebar { width: 280px; display: flex; flex-direction: column; gap: 14px; max-height: 90vh; overflow-y: auto; padding-right: 4px; }
  .panel { background: #16213e; border-radius: 10px; padding: 16px; }
  .panel h3 { color: #a8d8ea; margin-bottom: 10px; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1px; }
  #status { font-size: 1.1rem; font-weight: bold; color: #f0f0f0; min-height: 28px; }
  #status.check { color: #ff9f43; }
  #status.over { color: #ff6b6b; }
  #pgn { font-family: monospace; font-size: 0.82rem; color: #ccc; line-height: 1.7; max-height: 200px; overflow-y: auto; word-break: break-word; }
  label { font-size: 0.85rem; color: #aaa; display: block; margin-bottom: 4px; }
  select, input[type=range] { width: 100%; background: #0f3460; color: #eee; border: 1px solid #334; border-radius: 6px; padding: 6px 8px; }
  input[type=range] { padding: 4px 0; accent-color: #a8d8ea; }
  .slider-row { display: flex; justify-content: space-between; align-items: center; }
  .slider-val { color: #a8d8ea; font-size: 0.85rem; min-width: 28px; text-align: right; }
  button { width: 100%; padding: 10px; border: none; border-radius: 8px; cursor: pointer; font-size: 0.95rem; font-weight: bold; }
  #btn-new { background: #a8d8ea; color: #1a1a2e; }
  #btn-new:hover { background: #7ec8e3; }
  .highlight-white { box-shadow: inset 0 0 3px 3px rgba(255,255,100,0.7) !important; }
  .highlight-black { box-shadow: inset 0 0 3px 3px rgba(255,255,100,0.7) !important; }
  .dot-hint::after { content: ''; display: block; width: 30%; height: 30%; border-radius: 50%; background: rgba(0,200,0,0.45); margin: 35% auto; pointer-events: none; }
  .capture-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
  .capture-label { font-size: 0.8rem; color: #888; min-width: 70px; }
  .capture-pieces { font-size: 1.2rem; letter-spacing: 1px; line-height: 1.4; flex-wrap: wrap; display: flex; gap: 1px; }
  .capture-count { font-size: 0.75rem; color: #a8d8ea; font-weight: bold; margin-left: 4px; }
  .game-btn { width: 100%; background: #0f3460; color: #ccc; border: 1px solid #334; border-radius: 6px; padding: 7px 10px; margin-bottom: 6px; cursor: pointer; text-align: left; font-size: 0.8rem; line-height: 1.4; }
  .game-btn:hover { background: #1a4a80; color: #fff; }
  .game-btn .gb-result { font-weight: bold; color: #a8d8ea; }
  #no-history { color: #555; font-size: 0.82rem; }
  #pgn-modal { background: #0f3460; border-radius: 8px; padding: 12px; margin-top: 8px; font-family: monospace; font-size: 0.78rem; color: #ccc; line-height: 1.6; display: none; max-height: 160px; overflow-y: auto; word-break: break-word; }
</style>
</head>
<body>
<h1>♟ Chess-GPT</h1>
<div id="app">
  <div id="board-wrap">
    <div id="board"></div>
  </div>
  <div id="sidebar">
    <div class="panel">
      <h3>Status</h3>
      <div id="status">Loading model…</div>
    </div>
    <div class="panel">
      <h3>Settings</h3>
      <label>Play as</label>
      <select id="color-sel">
        <option value="white">White</option>
        <option value="black">Black</option>
      </select>
      <br><br>
      <label>Temperature <span class="slider-val" id="temp-val">0.1</span></label>
      <input type="range" id="temp" min="0.1" max="2.0" step="0.1" value="0.1">
      <br><br>
      <label>Top-K <span class="slider-val" id="topk-val">5</span></label>
      <input type="range" id="topk" min="0" max="30" step="1" value="5">
      <br><br>
      <label>AI Mode</label>
      <select id="search-mode">
        <option value="greedy">Greedy sampling</option>
        <option value="minimax">Minimax (alpha-beta)</option>
        <option value="threaded">Minimax Threaded</option>
        <option value="batched" selected>Minimax Batched</option>
      </select>
      <div id="minimax-opts" style="display:none;background:#0f3460;border-radius:8px;padding:10px;margin-top:8px;margin-bottom:10px">
        <label>Search Depth <span class="slider-val" id="mm-depth-val">3</span></label>
        <input type="range" id="mm-depth" min="1" max="4" step="1" value="3">
        <br><br>
        <label>Candidates (k) <span class="slider-val" id="mm-k-val">5</span></label>
        <input type="range" id="mm-k" min="1" max="10" step="1" value="5">
        <br>
        <div style="font-size:0.75rem;color:#888;margin-top:6px" id="mm-cost-line">~<span id="mm-cost">125</span> forward passes/move</div>
      </div>
      <button id="btn-new">⟳ New Game</button>
    </div>
    <div class="panel">
      <h3>Captured Pieces</h3>
      <div class="capture-row">
        <span class="capture-label">⬜ Lost</span>
        <div class="capture-pieces" id="white-captured"><span style="color:#555">—</span></div>
      </div>
      <div class="capture-row">
        <span class="capture-label">⬛ Lost</span>
        <div class="capture-pieces" id="black-captured"><span style="color:#555">—</span></div>
      </div>
    </div>
    <div class="panel">
      <h3>PGN</h3>
      <div id="pgn">(no moves yet)</div>
    </div>
    <div class="panel">
      <h3>Past Games</h3>
      <div id="history-list"><span id="no-history">No games yet.</span></div>
      <div id="pgn-modal"></div>
    </div>
  </div>
</div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
<script>
var board, game;
var historyUci = [];
var humanColor = 'white';
var busy = false;

function getTemp()        { return parseFloat($('#temp').val()); }
function getTopK()        { return parseInt($('#topk').val()); }
function getSearchMode()  { return $('#search-mode').val(); }
function getMMDepth()     { return parseInt($('#mm-depth').val()); }
function getMMK()         { return parseInt($('#mm-k').val()); }

function updateMMCost() {
  var k = getMMK(), d = getMMDepth(), mode = getSearchMode();
  var cost = (mode === 'batched') ? d : Math.pow(k, d);
  $('#mm-cost').text(cost);
  $('#mm-cost-line').html('~<span id="mm-cost">' + cost + '</span> ' +
    (mode === 'batched' ? 'batched passes/move' : 'forward passes/move'));
}

function setStatus(msg, cls) {
  var el = $('#status');
  el.attr('class', cls || '');
  el.text(msg);
}

function updatePgn(pgn) {
  $('#pgn').text(pgn || '(no moves yet)');
}

function resultLabel(r) {
  var map = {checkmate:'Checkmate',stalemate:'Stalemate',insufficient_material:'Draw',fifty_moves:'Draw',repetition:'Draw',abandoned:'Abandoned'};
  return map[r] || r;
}

function renderHistory(history) {
  var el = $('#history-list');
  if (!history || !history.length) { el.html('<span id="no-history">No games yet.</span>'); return; }
  var html = '';
  history.forEach(function(g, i) {
    var label = resultLabel(g.result);
    var side  = g.human_color === 'white' ? '⬜' : '⬛';
    html += '<button class="game-btn" onclick="showGamePgn(' + i + ')">' +
      side + ' <span class="gb-result">' + label + '</span>' +
      ' &nbsp;·&nbsp; ' + g.moves + ' moves &nbsp;·&nbsp; ' + g.time +
      '</button>';
  });
  el.html(html);
  window._gameHistory = history;
}

function showGamePgn(idx) {
  var g = window._gameHistory[idx];
  var modal = $('#pgn-modal');
  if (modal.is(':visible') && modal.data('idx') === idx) { modal.hide(); return; }
  modal.text(g.pgn || '(empty)').data('idx', idx).show();
}

function loadHistory() {
  $.get('/api/history', function(res) { renderHistory(res.history); });
}

function updateCaptures() {
  var fen = game.fen().split(' ')[0];
  var counts = {w:{p:0,n:0,b:0,r:0,q:0}, b:{p:0,n:0,b:0,r:0,q:0}};
  for (var i = 0; i < fen.length; i++) {
    var c = fen[i];
    if      (c==='P') counts.w.p++; else if (c==='N') counts.w.n++;
    else if (c==='B') counts.w.b++; else if (c==='R') counts.w.r++;
    else if (c==='Q') counts.w.q++; else if (c==='p') counts.b.p++;
    else if (c==='n') counts.b.n++; else if (c==='b') counts.b.b++;
    else if (c==='r') counts.b.r++; else if (c==='q') counts.b.q++;
  }
  var start = {p:8,n:2,b:2,r:2,q:1};
  var wLost  = {p:start.p-counts.w.p, n:start.n-counts.w.n, b:start.b-counts.w.b, r:start.r-counts.w.r, q:start.q-counts.w.q};
  var bLost  = {p:start.p-counts.b.p, n:start.n-counts.b.n, b:start.b-counts.b.b, r:start.r-counts.b.r, q:start.q-counts.b.q};
  var wSym   = {q:'♕',r:'♖',b:'♗',n:'♘',p:'♙'};
  var bSym   = {q:'♛',r:'♜',b:'♝',n:'♞',p:'♟'};
  function render(lost, sym) {
    var html = ''; var total = 0;
    ['q','r','b','n','p'].forEach(function(pt) {
      for (var i = 0; i < lost[pt]; i++) { html += sym[pt]; total++; }
    });
    if (!total) return '<span style="color:#555">—</span>';
    return '<span>' + html + '</span><span class="capture-count">×' + total + '</span>';
  }
  $('#white-captured').html(render(wLost, wSym));
  $('#black-captured').html(render(bLost, bSym));
}

function removeHighlights() {
  $('#board .square-55d63').removeClass('highlight-white highlight-black dot-hint');
}

function highlightSquare(sq) {
  var cls = (game.turn() === 'w') ? 'highlight-white' : 'highlight-black';
  $('#board .square-' + sq).addClass(cls);
}

function showDots(moves) {
  moves.forEach(function(m) {
    $('#board .square-' + m.to).addClass('dot-hint');
  });
}

function onDragStart(source, piece) {
  if (busy) return false;
  if (game.game_over()) return false;
  var myColor = (humanColor === 'white') ? 'w' : 'b';
  if (game.turn() !== myColor) return false;
  if (piece.search(myColor === 'w' ? /^b/ : /^w/) !== -1) return false;

  removeHighlights();
  highlightSquare(source);
  showDots(game.moves({ square: source, verbose: true }));
  return true;
}

function onDrop(source, target) {
  removeHighlights();
  if (source === target) return 'snapback';

  // Client-side pre-check
  var move = game.move({ from: source, to: target, promotion: 'q' });
  if (move === null) return 'snapback';
  game.undo(); // undo — server is authoritative

  var moveUci = source + target + (move.promotion || '');

  busy = true;
  setStatus('Thinking…');

  $.ajax({
    type: 'POST',
    url: '/api/move',
    contentType: 'application/json',
    data: JSON.stringify({
      history_uci: historyUci,
      move_uci: moveUci,
      human_color: humanColor,
      temperature: getTemp(),
      top_k: getTopK(),
      search_mode: getSearchMode(),
      minimax_k: getMMK(),
      minimax_depth: getMMDepth()
    }),
    success: function(res) {
      busy = false;
      if (!res.ok) {
        setStatus('❌ ' + res.error, 'check');
        board.position(game.fen());
        return;
      }
      // Apply human move
      game.move({ from: source, to: target, promotion: 'q' });
      // Apply GPT move if any
      if (res.gpt_uci) {
        var from = res.gpt_uci.slice(0,2), to = res.gpt_uci.slice(2,4), promo = res.gpt_uci[4] || undefined;
        game.move({ from: from, to: to, promotion: promo });
      }
      historyUci = res.history_uci;
      board.position(game.fen());
      updatePgn(res.pgn);
      updateCaptures();

      if (res.status !== 'ongoing') {
        loadHistory();
        var msgs = {
          checkmate: '♚ Checkmate!',
          stalemate: '½ Stalemate',
          insufficient_material: '½ Insufficient material',
          fifty_moves: '½ Fifty-move rule',
          repetition: '½ Threefold repetition'
        };
        setStatus(msgs[res.status] || 'Game over', 'over');
      } else {
        var turn = game.turn() === 'w' ? 'White' : 'Black';
        var inCheck = game.in_check() ? ' — Check!' : '';
        if ((humanColor === 'white' && game.turn() === 'w') || (humanColor === 'black' && game.turn() === 'b')) {
          setStatus('Your turn' + inCheck, inCheck ? 'check' : '');
        } else {
          setStatus('GPT played ' + (res.gpt_san || '') + (inCheck ? ' — Check!' : ''), inCheck ? 'check' : '');
        }
      }
    },
    error: function(xhr) {
      busy = false;
      setStatus('Server error: ' + xhr.status, 'over');
      board.position(game.fen());
    }
  });
}

function onSnapEnd() {
  board.position(game.fen());
}

function newGame() {
  humanColor = $('#color-sel').val();
  // Save current game as abandoned if it has moves
  if (historyUci.length > 0) {
    $.ajax({
      type: 'POST', url: '/api/save_game', contentType: 'application/json',
      data: JSON.stringify({ history_uci: historyUci, result: 'abandoned', human_color: humanColor }),
      success: function(res) { renderHistory(res.history); }
    });
  }
  historyUci = [];
  busy = true;
  setStatus('Starting…');
  $.ajax({
    type: 'POST',
    url: '/api/new_game',
    contentType: 'application/json',
    data: JSON.stringify({ human_color: humanColor, temperature: getTemp(), top_k: getTopK(), search_mode: getSearchMode(), minimax_k: getMMK(), minimax_depth: getMMDepth() }),
    success: function(res) {
      busy = false;
      historyUci = res.history_uci;
      game = Chess(res.fen);
      board.orientation(res.orientation);
      board.position(res.fen);
      updatePgn(res.pgn);
      updateCaptures();
      if (res.gpt_san) {
        setStatus('GPT opened with ' + res.gpt_san + '. Your turn.');
      } else {
        setStatus('Your turn.');
      }
    },
    error: function() { busy = false; setStatus('Failed to start game', 'over'); }
  });
}

$(document).ready(function() {
  game = Chess();

  board = Chessboard('board', {
    draggable: true,
    position: 'start',
    onDragStart: onDragStart,
    onDrop: onDrop,
    onSnapEnd: onSnapEnd,
    pieceTheme: '/pieces/{piece}.svg',
    snapbackSpeed: 200,
    snapSpeed: 50,
  });

  $('#temp').on('input', function() { $('#temp-val').text($(this).val()); });
  $('#topk').on('input', function() { $('#topk-val').text($(this).val()); });
  $('#search-mode').on('change', function() {
    var isSearch = $(this).val() !== 'greedy';
    $('#minimax-opts').toggle(isSearch);
    updateMMCost();
  });
  $('#mm-depth').on('input', function() { $('#mm-depth-val').text($(this).val()); updateMMCost(); });
  $('#mm-k').on('input', function() { $('#mm-k-val').text($(this).val()); updateMMCost(); });
  $('#btn-new').on('click', newGame);

  loadHistory();
  newGame();
});
</script>
</body>
</html>"""


@app.get('/pieces/{piece}', response_class=Response)
def serve_piece(piece: str):
    name = piece.replace('.svg', '').replace('.png', '')
    if name not in _PIECE_MAP:
        raise HTTPException(status_code=404, detail=f'Unknown piece: {piece}')
    svg = chess.svg.piece(_PIECE_MAP[name], size=60)
    return Response(content=svg, media_type='image/svg+xml')


@app.get('/', response_class=HTMLResponse)
def index():
    return HTML


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logger.info(f'Starting Chess-GPT server on http://{args.host}:{args.port}')
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')
