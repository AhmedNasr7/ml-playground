"""
Logging utilities for Chess-GPT.

Provides:
    setup_loguru()        — configure loguru sink (console + file)
    make_tb_writer()      — create a TensorBoard SummaryWriter
    log_epoch_metrics()   — write one epoch of metrics to both TB and loguru
    log_game_metrics()    — write game-level eval metrics to both TB and loguru
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

from loguru import logger
from torch.utils.tensorboard import SummaryWriter


# ── Loguru setup ──────────────────────────────────────────────────────────────

def setup_loguru(log_dir: str | Path = "artifacts/logs", run_name: str = "chess_gpt") -> None:
    """
    Configure loguru to write to stderr (with colours) and a rotating log file.

    Call once at the top of your notebook / script.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()   # drop the default handler

    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )

    log_file = log_dir / f"{run_name}.log"
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        rotation="10 MB",
        retention=3,
        compression="zip",
    )

    logger.info(f"Loguru initialised  →  {log_file}")


# ── TensorBoard setup ─────────────────────────────────────────────────────────

def make_tb_writer(log_dir: str | Path = "artifacts/runs", run_name: str = "chess_gpt") -> SummaryWriter:
    """
    Create and return a TensorBoard SummaryWriter.

    Launch TensorBoard with:
        tensorboard --logdir artifacts/runs
    """
    run_dir = Path(log_dir) / run_name
    writer = SummaryWriter(log_dir=str(run_dir))
    logger.info(f"TensorBoard writer  →  {run_dir}  (tensorboard --logdir {log_dir})")
    return writer


# ── Per-epoch logging ─────────────────────────────────────────────────────────

def log_epoch_metrics(
    writer: SummaryWriter,
    epoch: int,
    train: Dict[str, float],
    val: Dict[str, float],
) -> None:
    """
    Log one epoch of train/val metrics to TensorBoard and loguru.

    Expected keys in train/val dicts:
        loss, ppl, top1, top5, rank
    """
    for split, metrics in (("train", train), ("val", val)):
        for key, value in metrics.items():
            writer.add_scalar(f"{split}/{key}", value, global_step=epoch)

    logger.info(
        f"Epoch {epoch:>3} | "
        f"tr_loss={train['loss']:.4f}  tr_ppl={train['ppl']:.1f}  tr_top1={train['top1']:.1f}%  |  "
        f"vl_loss={val['loss']:.4f}  vl_ppl={val['ppl']:.1f}  vl_top1={val['top1']:.1f}%  "
        f"vl_top5={val['top5']:.1f}%  vl_rank={val['rank']:.1f}"
    )


# ── Game-level eval logging ───────────────────────────────────────────────────

def log_game_metrics(
    writer: SummaryWriter,
    metrics: Dict[str, float],
    step: int,
    tag_prefix: str = "eval",
) -> None:
    """
    Log game-level metrics (legal_move_rate, game_completion_rate, avg_legal_length)
    to TensorBoard and loguru.

    Args:
        writer:     SummaryWriter instance
        metrics:    dict from src.metrics.game_metrics()
        step:       global step (e.g. epoch number)
        tag_prefix: TB tag prefix, e.g. 'eval/constrained' or 'eval/unconstrained'
    """
    for key, value in metrics.items():
        if key == "n_games":
            continue
        writer.add_scalar(f"{tag_prefix}/{key}", value, global_step=step)

    logger.info(
        f"[{tag_prefix}] "
        f"legal_rate={metrics.get('legal_move_rate', 0):.1f}%  "
        f"completion={metrics.get('game_completion_rate', 0):.1f}%  "
        f"avg_legal_len={metrics.get('avg_legal_length', 0):.1f}  "
        f"(n={metrics.get('n_games', 0)})"
    )
