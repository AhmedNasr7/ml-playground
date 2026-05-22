#!/usr/bin/env python3
"""
train_chessgpt.py — standalone training script for Chess-GPT

Usage examples:
    python train_chessgpt.py
    python train_chessgpt.py --model small --max_games 500000 --epochs 5
    python train_chessgpt.py --model tiny  --max_games 50000  --epochs 10 --amp
    python train_chessgpt.py --model nano  --max_games 10000  --batch_size 64 --no_tb
    python train_chessgpt.py --help
"""

import argparse
import math
import random
import re
import sys
from pathlib import Path

import torch
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.logging_utils import log_epoch_metrics, make_tb_writer, setup_loguru
from src.metrics import top_k_accuracy
from src.model import GPT2, GPT2Config
from src.tokenizers import MoveTokenizer


# ── Model presets ─────────────────────────────────────────────────────────────

MODEL_PRESETS: dict[str, dict] = {
    'nano':   dict(d_model=64,  n_heads=2, n_layers=2, d_ff=128),
    'tiny':   dict(d_model=128, n_heads=4, n_layers=4, d_ff=512),
    'small':  dict(d_model=256, n_heads=8, n_layers=6, d_ff=1024),
    'medium': dict(d_model=512, n_heads=8, n_layers=8, d_ff=2048),
}


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Train Chess-GPT move predictor',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    dg = p.add_argument_group('Data')
    dg.add_argument('--pgn_path',    default='data/lichess_2014_07.pgn',
                    help='Path to input PGN file')
    dg.add_argument('--max_games',   type=int, default=500_000,
                    help='Maximum number of games to load')
    dg.add_argument('--train_split', type=float, default=0.9,
                    help='Fraction of games used for training (rest = val)')
    dg.add_argument('--seq_len',     type=int, default=128,
                    help='Token sequence length (context window in moves)')
    dg.add_argument('--seed',        type=int, default=42,
                    help='Random seed for reproducibility')

    # Model
    mg = p.add_argument_group('Model')
    mg.add_argument('--model',   choices=list(MODEL_PRESETS), default='small',
                    help='Model size preset')
    mg.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate')

    # Training
    tg = p.add_argument_group('Training')
    tg.add_argument('--epochs',       type=int,   default=5,
                    help='Number of training epochs')
    tg.add_argument('--batch_size',   type=int,   default=None,
                    help='Batch size (omit to auto-compute from dataset size)')
    tg.add_argument('--lr',           type=float, default=3e-4,
                    help='Peak learning rate (AdamW)')
    tg.add_argument('--weight_decay', type=float, default=0.01,
                    help='AdamW weight decay')
    tg.add_argument('--grad_clip',    type=float, default=1.0,
                    help='Gradient norm clipping (0 = disabled)')
    tg.add_argument('--warmup_steps', type=int,   default=200,
                    help='Linear LR warmup steps (0 = no warmup)')
    tg.add_argument('--early_stop',   type=int,   default=0,
                    help='Early stopping patience in epochs (0 = disabled)')
    tg.add_argument('--amp',          action='store_true',
                    help='Enable automatic mixed precision (CUDA only)')

    # Tokenizer
    tok = p.add_argument_group('Tokenizer')
    tok.add_argument('--tok_path',    default='artifacts/move_tok.pkl',
                     help='Path to save / load the MoveTokenizer cache')
    tok.add_argument('--rebuild_tok', action='store_true',
                     help='Force rebuild tokenizer even if a cache exists')

    # Resume
    rg = p.add_argument_group('Resume')
    rg.add_argument('--checkpoint', default=None,
                    help='Path to a .pt checkpoint to resume weights from')

    # Output & logging
    og = p.add_argument_group('Output')
    og.add_argument('--run_name',   default=None,
                    help='Run name (default: auto from model + game count)')
    og.add_argument('--out_dir',    default='artifacts',
                    help='Directory for model checkpoints')
    og.add_argument('--log_dir',    default='artifacts/logs',
                    help='Directory for log files')
    og.add_argument('--tb_dir',     default='artifacts/runs',
                    help='TensorBoard log directory')
    og.add_argument('--no_tb',      action='store_true',
                    help='Disable TensorBoard logging')
    og.add_argument('--save_every', type=int, default=0,
                    help='Save a checkpoint every N epochs (0 = only save best)')

    return p.parse_args()


# ── PGN parsing ───────────────────────────────────────────────────────────────

def parse_pgn_moves(pgn_path: str, max_games: int = 50_000) -> list[str]:
    games: list[str] = []
    current_moves: list[str] = []
    in_moves = False

    result_re = re.compile(r'(1-0|0-1|1/2-1/2|\*)\s*$')
    clock_re  = re.compile(r'\{[^}]*\}')
    eol_re    = re.compile(r'\s+')

    with open(pgn_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line.startswith('['):
                if in_moves and current_moves:
                    games.append(' '.join(current_moves))
                    current_moves = []
                    if len(games) >= max_games:
                        break
                in_moves = False
                continue
            if not line:
                in_moves = True
                continue
            if in_moves:
                line = clock_re.sub('', line)
                line = result_re.sub('', line)
                line = eol_re.sub(' ', line).strip()
                if line:
                    current_moves.append(line)

    if current_moves and len(games) < max_games:
        games.append(' '.join(current_moves))

    return games


# ── Dataset ───────────────────────────────────────────────────────────────────

class MoveLevelDataset(Dataset):
    """Sliding-window dataset over a flat token stream of encoded games."""

    def __init__(self, games: list[str], tokenizer: MoveTokenizer, max_len: int = 128):
        all_ids = tokenizer.encode_batch(games, add_special=True)
        flat: list[int] = []
        for ids in all_ids:
            flat.extend(ids)
        self.tokens  = flat
        self.max_len = max_len
        self.n       = (len(flat) - 1) // max_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.max_len
        end   = start + self.max_len
        return {
            'input_ids': torch.tensor(self.tokens[start:end],     dtype=torch.long),
            'labels':    torch.tensor(self.tokens[start+1:end+1], dtype=torch.long),
        }


# ── Scaling law report ────────────────────────────────────────────────────────

def print_scaling_report(
    model: GPT2,
    config: GPT2Config,
    train_tokens: int,
    n_train_games: int,
) -> None:
    n_params     = model.n_params
    embed_params = config.vocab_size * config.d_model
    pos_params   = config.max_seq_len * config.d_model
    n_backbone   = n_params - embed_params - pos_params

    tok_per_total    = train_tokens / n_params
    tok_per_backbone = train_tokens / max(n_backbone, 1)
    chilla_opt       = train_tokens / 20
    chilla_needed    = n_params * 20
    avg_tok_per_game = train_tokens / max(n_train_games, 1)
    chilla_games     = int(chilla_needed / max(avg_tok_per_game, 1))

    def _verdict(r: float) -> str:
        if r > 5:     return f'{r:.0f}× over-parametrised → data starvation'
        elif r > 2:   return f'{r:.1f}× over-parametrised → train longer or add games'
        elif r > 0.5: return f'well-matched  (ratio={r:.2f}×)'
        else:         return f'{1/r:.1f}× UNDER-parametrised → model too small for this data'

    SEP = '─' * 62
    logger.info(SEP)
    logger.info(f'  Total parameters          : {n_params:>14,}')
    logger.info(f'    · token embedding        : {embed_params:>14,}  ({embed_params/n_params*100:.0f}%)')
    logger.info(f'    · positional embedding   : {pos_params:>14,}')
    logger.info(f'    · transformer backbone   : {n_backbone:>14,}')
    logger.info(SEP)
    logger.info(f'  Training tokens           : {train_tokens:>14,}')
    logger.info(f'  Tokens / total param      : {tok_per_total:>14.1f}')
    logger.info(f'  Tokens / backbone param   : {tok_per_backbone:>14.1f}')
    logger.info(SEP)
    logger.info(f'  Chinchilla-optimal total  : {chilla_opt:>14,.0f}  (D / 20)')
    logger.info(f'  Tokens needed (full model): {chilla_needed:>14,.0f}  ≈ {chilla_games:,} games')
    logger.info(SEP)
    logger.info(f'  Verdict (total params)    : {_verdict(n_params / chilla_opt)}')
    logger.info(f'  Verdict (backbone only)   : {_verdict(n_backbone / chilla_opt)}')
    logger.info(SEP)


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(
    model: GPT2,
    loader: DataLoader,
    config: GPT2Config,
    criterion: torch.nn.Module,
    optimizer: AdamW | None = None,
    scheduler: LinearLR | None = None,
    scaler=None,
    writer=None,
    global_step: int = 0,
    grad_clip: float = 1.0,
    train: bool = True,
    epoch: int = 0,
) -> tuple[dict, int]:
    model.train() if train else model.eval()
    total_loss = total_top1 = total_top5 = 0.0
    n = 0
    phase = 'train' if train else 'val'
    ctx = torch.enable_grad() if train else torch.no_grad()

    bar = tqdm(loader, desc=f'Epoch {epoch:>3} [{phase}]', leave=False, dynamic_ncols=True)

    with ctx:
        for batch in bar:
            ids    = batch['input_ids'].to(config.device)
            labels = batch['labels'].to(config.device)

            if train and scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = model(ids)
                    loss   = criterion(logits.reshape(-1, config.vocab_size), labels.reshape(-1))
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(ids)
                loss   = criterion(logits.reshape(-1, config.vocab_size), labels.reshape(-1))
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            if train:
                if scheduler is not None:
                    scheduler.step()
                if writer is not None:
                    writer.add_scalar('step/train_loss', loss.item(), global_step)
                global_step += 1

            total_loss += loss.item()
            with torch.no_grad():
                t1 = top_k_accuracy(logits, labels, config.pad_id, k=1)
                t5 = top_k_accuracy(logits, labels, config.pad_id, k=5)
                total_top1 += t1
                total_top5 += t5
            n += 1
            bar.set_postfix(
                loss=f'{total_loss/n:.3f}',
                top1=f'{total_top1/n*100:.1f}%',
                top5=f'{total_top5/n*100:.1f}%',
            )

    avg_loss = total_loss / n
    return {
        'loss': avg_loss,
        'ppl':  math.exp(min(avg_loss, 20)),
        'top1': total_top1 / n * 100,
        'top5': total_top5 / n * 100,
    }, global_step


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = (
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu'
    )

    # ── Run name & directories ────────────────────────────────────────────────
    run_name = args.run_name or f'chessgpt_{args.model}_{args.max_games // 1000}k'
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────────────
    setup_loguru(log_dir=args.log_dir, run_name=run_name)
    writer = None if args.no_tb else make_tb_writer(log_dir=args.tb_dir, run_name=run_name)
    logger.info(
        f'Run={run_name}  device={device}  model={args.model}  '
        f'max_games={args.max_games:,}  epochs={args.epochs}  lr={args.lr}  amp={args.amp}'
    )

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # ── Parse PGN ─────────────────────────────────────────────────────────────
    pgn_path = Path(args.pgn_path)
    if not pgn_path.exists():
        logger.error(f'PGN file not found: {pgn_path}')
        sys.exit(1)

    logger.info(f'Parsing PGN ({pgn_path.stat().st_size / 1e6:.0f} MB) …')
    games = parse_pgn_moves(str(pgn_path), max_games=args.max_games)
    logger.info(f'Loaded {len(games):,} games')

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_path = Path(args.tok_path)
    if tok_path.exists() and not args.rebuild_tok:
        logger.info(f'Loading tokenizer ← {tok_path}')
        mtok = MoveTokenizer.load(tok_path)
    else:
        logger.info('Building tokenizer…')
        mtok = MoveTokenizer().build(games)
        mtok.save(tok_path)
    logger.info(f'Vocab size: {mtok.vocab_size}  PAD={mtok.pad_id}  BOS={mtok.bos_id}  EOS={mtok.eos_id}')

    # ── Train / val split ─────────────────────────────────────────────────────
    random.shuffle(games)
    split_idx   = int(len(games) * args.train_split)
    train_games = games[:split_idx]
    val_games   = games[split_idx:]
    logger.info(f'Train: {len(train_games):,} games  |  Val: {len(val_games):,} games')

    # ── Datasets ──────────────────────────────────────────────────────────────
    logger.info('Tokenising datasets…')
    train_ds = MoveLevelDataset(train_games, mtok, max_len=args.seq_len)
    val_ds   = MoveLevelDataset(val_games,   mtok, max_len=args.seq_len)
    logger.info(f'Train: {len(train_ds.tokens):,} tokens  {len(train_ds):,} windows')
    logger.info(f'Val  : {len(val_ds.tokens):,} tokens  {len(val_ds):,} windows')

    # ── Batch size ────────────────────────────────────────────────────────────
    if args.batch_size is None:
        batch_size = int(2 ** round(math.log2(max(32, len(train_ds) ** 0.5))))
        batch_size = min(batch_size, 512)
        logger.info(f'Auto batch size: {batch_size}  (sqrt heuristic, capped at 512)')
    else:
        batch_size = args.batch_size
        logger.info(f'Batch size: {batch_size}')

    n_workers = min(4, os.cpu_count() or 1) if (device != 'mps') else 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=(device == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, pin_memory=(device == 'cuda'),
    )
    logger.info(f'Train batches: {len(train_loader):,}  |  Val batches: {len(val_loader):,}')

    # ── Model ─────────────────────────────────────────────────────────────────
    preset = MODEL_PRESETS[args.model]
    config = GPT2Config(
        vocab_size  = mtok.vocab_size,
        pad_id      = mtok.pad_id,
        d_model     = preset['d_model'],
        n_heads     = preset['n_heads'],
        n_layers    = preset['n_layers'],
        d_ff        = preset['d_ff'],
        max_seq_len = args.seq_len,
        dropout     = args.dropout,
        batch_size  = batch_size,
        lr          = args.lr,
        epochs      = args.epochs,
    )
    config.device = device

    model = GPT2(config).to(device)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            logger.error(f'Checkpoint not found: {ckpt_path}')
            sys.exit(1)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        logger.info(f'Resumed weights ← {ckpt_path}')

    print_scaling_report(model, config, len(train_ds.tokens), len(train_games))

    # ── Save config JSON so play_gpt.py can auto-load architecture ────────────
    import json
    cfg_dict = {
        'model':      args.model,
        'vocab_size': mtok.vocab_size,
        'pad_id':     mtok.pad_id,
        'd_model':    config.d_model,
        'n_heads':    config.n_heads,
        'n_layers':   config.n_layers,
        'd_ff':       config.d_ff,
        'max_seq_len':config.max_seq_len,
        'dropout':    config.dropout,
        'tok_path':   str(Path(args.tok_path).resolve()),
    }
    cfg_path = Path(args.out_dir) / f'{run_name}.json'
    cfg_path.write_text(json.dumps(cfg_dict, indent=2))
    logger.info(f'Config saved → {cfg_path}')

    # ── Optimizer, scheduler, scaler ─────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=config.pad_id)

    scheduler = None
    if args.warmup_steps > 0:
        scheduler = LinearLR(optimizer, start_factor=1e-3, total_iters=args.warmup_steps)
        logger.info(f'LR warmup: {args.warmup_steps} steps')

    scaler = torch.amp.GradScaler('cuda') if (args.amp and device == 'cuda') else None
    if scaler:
        logger.info('AMP enabled')

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info(
        f'Starting training: {args.epochs} epochs  '
        f'batch={batch_size}  lr={args.lr}  grad_clip={args.grad_clip}'
    )

    history: list[dict] = []
    global_step   = 0
    best_val_loss = float('inf')
    patience      = 0
    best_ckpt     = Path(args.out_dir) / f'{run_name}_best.pt'

    for epoch in range(1, args.epochs + 1):
        tr, global_step = run_epoch(
            model, train_loader, config, criterion,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            writer=writer, global_step=global_step,
            grad_clip=args.grad_clip, train=True, epoch=epoch,
        )
        vl, _ = run_epoch(
            model, val_loader, config, criterion,
            writer=writer, global_step=global_step,
            grad_clip=0.0, train=False, epoch=epoch,
        )
        history.append({'epoch': epoch, 'train': tr, 'val': vl})

        if writer:
            log_epoch_metrics(writer, epoch, tr, vl)
        else:
            logger.info(
                f'Epoch {epoch:>3} | '
                f'tr_loss={tr["loss"]:.4f}  tr_ppl={tr["ppl"]:.1f}  tr_top1={tr["top1"]:.1f}%  |  '
                f'vl_loss={vl["loss"]:.4f}  vl_ppl={vl["ppl"]:.1f}  vl_top1={vl["top1"]:.1f}%  '
                f'vl_top5={vl["top5"]:.1f}%'
            )

        # ── Save best ─────────────────────────────────────────────────────────
        if vl['loss'] < best_val_loss:
            best_val_loss = vl['loss']
            torch.save(model.state_dict(), best_ckpt)
            best_ckpt.with_suffix('.json').write_text(json.dumps(cfg_dict, indent=2))
            logger.info(f'  ✓ New best  val_loss={best_val_loss:.4f}  →  {best_ckpt}')
            patience = 0
        else:
            patience += 1
            logger.info(f'  No improvement ({patience}/{args.early_stop or "∞"})')

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt = Path(args.out_dir) / f'{run_name}_epoch{epoch:03d}.pt'
            torch.save(model.state_dict(), ckpt)
            logger.info(f'  Checkpoint → {ckpt}')

        # ── Early stopping ────────────────────────────────────────────────────
        if args.early_stop > 0 and patience >= args.early_stop:
            logger.info(f'Early stopping triggered at epoch {epoch} (patience={args.early_stop})')
            break

    # ── Final save & hparam logging ───────────────────────────────────────────
    final_ckpt = Path(args.out_dir) / f'{run_name}_final.pt'
    torch.save(model.state_dict(), final_ckpt)

    if writer:
        writer.add_hparams(
            {
                'model':       args.model,
                'd_model':     config.d_model,
                'n_layers':    config.n_layers,
                'n_heads':     config.n_heads,
                'lr':          args.lr,
                'batch_size':  batch_size,
                'max_games':   args.max_games,
                'seq_len':     args.seq_len,
            },
            {
                'hparam/val_loss': history[-1]['val']['loss'],
                'hparam/val_ppl':  history[-1]['val']['ppl'],
                'hparam/val_top1': history[-1]['val']['top1'],
            },
        )
        writer.flush()
        writer.close()

    logger.success(f'Done.  best_val_loss={best_val_loss:.4f}')
    logger.success(f'Best checkpoint   →  {best_ckpt}')
    logger.success(f'Final checkpoint  →  {final_ckpt}')


if __name__ == '__main__':
    import os
    main()
