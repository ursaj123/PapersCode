"""
trainer.py — Universal PyTorch Trainer
---------------------------------------
Designed to be paper-agnostic. Override `compute_loss()` and optionally
`training_step()` / `validation_step()` in a subclass for each paper.

Features:
  - Mixed precision (bfloat16 / float16 / float32)
  - Gradient clipping (norm or value)
  - Gradient accumulation
  - Early stopping (loss or metric, min or max)
  - LR scheduling (step, cosine, plateau, linear warmup+cosine)
  - Checkpointing (best + last + every-n-epochs)
  - Multi-GPU: DataParallel (simple) or DistributedDataParallel (scalable)
  - Structured logging → console + rotating log file (no W&B required)
  - Optional W&B and TensorBoard integrations
  - tqdm progress bars (epoch bar + per-batch bar with live metrics)
  - EMA (Exponential Moving Average) of weights
  - Reproducibility (seed everything)
  - Callback hooks: on_epoch_start/end, on_batch_start/end
"""

import os
import sys
import time
import math
import copy
import json
import random
import logging
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

log = logging.getLogger("Trainer")


# ─────────────────────────────────────────────
# 0. TQDM-SAFE LOGGING HANDLER
# ─────────────────────────────────────────────

class _TqdmLoggingHandler(logging.StreamHandler):
    """
    Routes log records through tqdm.write() so log messages never
    corrupt an active progress bar. Falls back to plain print when
    tqdm is not installed or no bar is active.
    """
    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            if _TQDM_AVAILABLE:
                tqdm.write(msg, file=sys.stdout)
            else:
                print(msg, file=sys.stdout)
            self.flush()
        except Exception:
            self.handleError(record)


# ─────────────────────────────────────────────
# 1. TRAINING CONFIG
# ─────────────────────────────────────────────

@dataclass
class TrainerConfig:
    # ── Paths ──────────────────────────────────
    output_dir: str = "runs/experiment"
    run_name: str = "run"

    # ── Duration ───────────────────────────────
    max_epochs: int = 100
    max_steps: Optional[int] = None          # overrides max_epochs if set

    # ── Precision ──────────────────────────────
    precision: str = "float32"               # "float32" | "float16" | "bfloat16"
    # Note: bfloat16 recommended for modern GPUs (A100, H100, RTX 30xx+)
    # float16 needs GradScaler; bfloat16 does not

    # ── Gradient handling ──────────────────────
    grad_clip_norm: Optional[float] = 1.0    # max gradient norm (None = disabled)
    grad_clip_value: Optional[float] = None  # clip each grad to [-v, v] (None = disabled)
    grad_accumulation_steps: int = 1         # simulate larger batch size

    # ── Optimizer ──────────────────────────────
    optimizer: str = "adamw"                 # "adam" | "adamw" | "sgd" | "rmsprop"
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9                    # SGD only
    betas: Tuple[float, float] = (0.9, 0.999)  # Adam/AdamW only

    # ── LR Scheduler ───────────────────────────
    scheduler: Optional[str] = None
    # Options:
    #   "step"          → StepLR (step_size, gamma)
    #   "cosine"        → CosineAnnealingLR (eta_min)
    #   "plateau"       → ReduceLROnPlateau (patience, factor)
    #   "linear_warmup_cosine" → warmup then cosine decay
    #   None            → constant LR

    scheduler_step_size: int = 10            # for "step"
    scheduler_gamma: float = 0.5            # for "step" / "plateau"
    scheduler_eta_min: float = 0.0          # for "cosine"
    scheduler_patience: int = 5             # for "plateau"
    scheduler_warmup_steps: int = 0         # for "linear_warmup_cosine"

    # ── Early Stopping ─────────────────────────
    early_stopping: bool = False
    early_stopping_monitor: str = "val_loss"  # metric name to watch
    early_stopping_mode: str = "min"          # "min" or "max"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    # ── Checkpointing ─────────────────────────
    save_best: bool = True
    save_last: bool = True
    save_every_n_epochs: Optional[int] = None
    checkpoint_monitor: str = "val_loss"
    checkpoint_mode: str = "min"

    # ── Multi-GPU ──────────────────────────────
    strategy: str = "auto"
    # "auto"  → detect and use DataParallel if >1 GPU, else single
    # "dp"    → force DataParallel
    # "ddp"   → DistributedDataParallel (launched via torchrun)
    # "cpu"   → force CPU

    # ── EMA ───────────────────────────────────
    ema: bool = False
    ema_decay: float = 0.999

    # ── Logging ────────────────────────────────
    log_every_n_steps: int = 10
    log_level: str = "INFO"

    # File logging
    use_log_file: bool = True              # write trainer.log inside output_dir
    log_file_name: str = "trainer.log"
    log_file_max_bytes: int = 10_485_760   # 10 MB before rotating
    log_file_backup_count: int = 3         # keep last 3 rotated files
    log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    log_date_format: str = "%Y-%m-%d %H:%M:%S"

    # External loggers (optional, both can be off)
    use_wandb: bool = False
    use_tensorboard: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

    # Progress bars
    progress_bar: bool = True              # tqdm bar per batch (requires tqdm)
    progress_bar_epochs: bool = True       # outer epoch-level bar

    # ── Reproducibility ────────────────────────
    seed: Optional[int] = 42
    deterministic: bool = False              # torch.use_deterministic_algorithms

    # ── Misc ───────────────────────────────────
    val_every_n_epochs: int = 1


# ─────────────────────────────────────────────
# 2. CALLBACKS
# ─────────────────────────────────────────────

class Callback:
    """Base callback. Override any hook you need."""
    def on_train_start(self, trainer): pass
    def on_train_end(self, trainer): pass
    def on_epoch_start(self, trainer, epoch: int): pass
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict): pass
    def on_batch_start(self, trainer, batch, batch_idx: int): pass
    def on_batch_end(self, trainer, loss: float, batch_idx: int): pass
    def on_val_start(self, trainer): pass
    def on_val_end(self, trainer, metrics: Dict): pass


class PrintCallback(Callback):
    """Logs a summary line at epoch end (works with or without tqdm)."""
    def on_epoch_end(self, trainer, epoch, metrics):
        parts = [f"Epoch {epoch:04d}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        msg = " | ".join(parts)
        log.info(msg)
        # also write a blank line to visually separate epochs in the log file
        for handler in log.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.stream.write("\n")


# ─────────────────────────────────────────────
# 3. EMA WRAPPER
# ─────────────────────────────────────────────

class ModelEMA:
    """Exponential Moving Average of model weights.
    Use ema.model for inference after training.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)


# ─────────────────────────────────────────────
# 4. EARLY STOPPING
# ─────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, monitor: str, mode: str, patience: int, min_delta: float):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float("inf") if mode == "min" else float("-inf")
        self._is_better = (lambda a, b: a < b - min_delta) if mode == "min" \
                     else (lambda a, b: a > b + min_delta)
        self.should_stop = False

    def step(self, metrics: Dict) -> bool:
        value = metrics.get(self.monitor)
        if value is None:
            warnings.warn(f"EarlyStopping: '{self.monitor}' not in metrics. Skipping.")
            return False
        if self._is_better(value, self.best):
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                log.info(f"Early stopping triggered (patience={self.patience}).")
                self.should_stop = True
        return self.should_stop


# ─────────────────────────────────────────────
# 5. UTILS
# ─────────────────────────────────────────────

def seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(strategy: str) -> torch.device:
    if strategy == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(model: nn.Module, cfg: TrainerConfig) -> optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    name = cfg.optimizer.lower()
    if name == "adam":
        return optim.Adam(params, lr=cfg.lr, betas=cfg.betas,
                          weight_decay=cfg.weight_decay)
    elif name == "adamw":
        return optim.AdamW(params, lr=cfg.lr, betas=cfg.betas,
                           weight_decay=cfg.weight_decay)
    elif name == "sgd":
        return optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                         weight_decay=cfg.weight_decay)
    elif name == "rmsprop":
        return optim.RMSprop(params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                             momentum=cfg.momentum)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def build_scheduler(optimizer: optim.Optimizer, cfg: TrainerConfig,
                    total_steps: int):
    if cfg.scheduler is None:
        return None
    name = cfg.scheduler.lower()
    if name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma)
    elif name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epochs, eta_min=cfg.scheduler_eta_min)
    elif name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=cfg.checkpoint_mode, factor=cfg.scheduler_gamma,
            patience=cfg.scheduler_patience)
    elif name == "linear_warmup_cosine":
        warmup = cfg.scheduler_warmup_steps
        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


def is_plateau_scheduler(scheduler) -> bool:
    return isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)


# ─────────────────────────────────────────────
# 6. METRICS TRACKER
# ─────────────────────────────────────────────

class MetricsTracker:
    """Accumulates scalar values, returns epoch-level averages."""
    def __init__(self):
        self._sums = defaultdict(float)
        self._counts = defaultdict(int)

    def update(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self._sums[k] += v * n
            self._counts[k] += n

    def compute(self) -> Dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}

    def reset(self):
        self._sums.clear()
        self._counts.clear()


# ─────────────────────────────────────────────
# 7. BASE TRAINER
# ─────────────────────────────────────────────

class Trainer(ABC):
    """
    Usage
    -----
    Subclass and implement `compute_loss(batch) -> (loss, metrics_dict)`.
    Optionally override `training_step()` or `validation_step()` for
    more control (e.g. multiple optimizers, custom forward logic).

    Example
    -------
    class MyTrainer(Trainer):
        def compute_loss(self, batch):
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(-1) == y).float().mean()
            return loss, {"loss": loss.item(), "acc": acc.item()}

    trainer = MyTrainer(model, cfg)
    trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainerConfig,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.cfg = cfg
        self.callbacks = callbacks or [PrintCallback()]
        self.global_step = 0
        self.current_epoch = 0
        self._history: List[Dict] = []

        # ── Logging setup ─────────────────────
        # (done after output_dir is known so the log file lands there)
        self._log_file_path: Optional[Path] = None

        # ── Reproducibility ───────────────────
        if cfg.seed is not None:
            seed_everything(cfg.seed, cfg.deterministic)

        # ── Output dir ────────────────────────
        self.output_dir = Path(cfg.output_dir) / cfg.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

        # ── Logger (console + optional file) ──
        self._setup_logger()

        # ── Device & model wrapping ───────────
        self.device = get_device(cfg.strategy)
        self.model = model.to(self.device)
        self.model = self._wrap_model(self.model)

        # ── Precision & scaler ─────────────────
        self.amp_dtype = None
        self.scaler = None
        if cfg.precision == "float16":
            self.amp_dtype = torch.float16
            self.scaler = GradScaler()
        elif cfg.precision == "bfloat16":
            self.amp_dtype = torch.bfloat16
            # bfloat16 doesn't need GradScaler

        self._autocast_ctx = (
            torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
            if self.amp_dtype is not None else nullcontext()
        )

        # ── Optimizer & scheduler (init later when total_steps is known) ──
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler = None

        # ── EMA ───────────────────────────────
        self.ema: Optional[ModelEMA] = None
        if cfg.ema:
            raw = self.model.module if hasattr(self.model, "module") else self.model
            self.ema = ModelEMA(raw, decay=cfg.ema_decay)

        # ── Early stopping ────────────────────
        self._early_stopper: Optional[EarlyStopping] = None
        if cfg.early_stopping:
            self._early_stopper = EarlyStopping(
                monitor=cfg.early_stopping_monitor,
                mode=cfg.early_stopping_mode,
                patience=cfg.early_stopping_patience,
                min_delta=cfg.early_stopping_min_delta,
            )

        # ── Best checkpoint tracking ──────────
        self._best_metric = float("inf") if cfg.checkpoint_mode == "min" else float("-inf")
        self._is_better_ckpt = (
            (lambda a, b: a < b) if cfg.checkpoint_mode == "min"
            else (lambda a, b: a > b)
        )

        # ── External loggers ──────────────────
        self._wandb = None
        self._tb_writer = None
        if cfg.use_wandb:
            self._init_wandb()
        if cfg.use_tensorboard:
            self._init_tensorboard()

    # ── Abstract interface ────────────────────────────────────────────────

    @abstractmethod
    def compute_loss(self, batch: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass + loss computation.
        Args:
            batch: whatever your DataLoader yields
        Returns:
            loss  : scalar Tensor (un-reduced), must support .backward()
            metrics: dict of floats to log (must include 'loss' key)
        """
        ...

    # ── Optional overrides ────────────────────────────────────────────────

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, float]:
        """
        Override for multi-optimizer setups (e.g. GANs, SpectralNet phases).
        Default: single optimizer step.
        """
        with self._autocast_ctx:
            loss, metrics = self.compute_loss(batch)

        scaled_loss = loss / self.cfg.grad_accumulation_steps

        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return metrics

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, float]:
        """Override to customize val logic. Default: same compute_loss, no grad."""
        with torch.no_grad(), self._autocast_ctx:
            _, metrics = self.compute_loss(batch)
        return metrics

    def on_before_optimizer_step(self):
        """Called right before optimizer.step(). Override for custom clipping."""
        cfg = self.cfg
        raw_model = self.model.module if hasattr(self.model, "module") else self.model

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        if cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip_norm)
        if cfg.grad_clip_value is not None:
            nn.utils.clip_grad_value_(raw_model.parameters(), cfg.grad_clip_value)

    # ── Main fit loop ─────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None,
        extra_epochs: Optional[int] = None,
        finetune_from: Optional[str] = None,
    ):
        """
        Train the model.

        Two distinct modes for starting from an existing checkpoint:

        ── RESUME (`resume_from`) ────────────────────────────────────────────
        Continue an interrupted training run. Restores EVERYTHING:
          weights, optimizer state (momentum/adam buffers), scheduler state,
          scaler, EMA, global_step, epoch counter, history, best metric.
        Training continues from where it left off — logs are continuous.

          trainer.fit(train_loader, val_loader,
                      resume_from="runs/exp/last.pt")

          # Train 20 more epochs on top of whatever epoch was saved:
          trainer.fit(train_loader, val_loader,
                      resume_from="runs/exp/last.pt", extra_epochs=20)

        ── FINETUNE (`finetune_from`) ────────────────────────────────────────
        Start fresh training using pretrained weights as initialization.
        Restores ONLY the model weights. Everything else is fresh:
          optimizer (no momentum history), scheduler (restarts from epoch 0),
          history (empty), epoch counter (0), best metric (reset).
        Use this when: changing LR, changing dataset, transfer learning,
        running an ablation from the same pretrained base.

          trainer.fit(train_loader, val_loader,
                      finetune_from="runs/pretrain/best.pt")

        Args:
            train_loader:   training DataLoader
            val_loader:     optional validation DataLoader
            resume_from:    checkpoint path for RESUME (full state restored)
            extra_epochs:   when resuming, train this many MORE epochs on top.
                            e.g. saved at epoch 30, extra_epochs=20 → runs to 50.
                            Ignored when finetune_from is used.
            finetune_from:  checkpoint path for FINETUNE (weights only)
        """
        assert not (resume_from and finetune_from), \
            "Pass either resume_from OR finetune_from, not both."

        # Build optimizer + scheduler (fresh every time)
        self.optimizer = build_optimizer(self.model, self.cfg)
        steps_per_epoch = len(train_loader) // self.cfg.grad_accumulation_steps
        total_steps = steps_per_epoch * self.cfg.max_epochs
        self.scheduler = build_scheduler(self.optimizer, self.cfg, total_steps)

        if resume_from:
            # ── RESUME: restore full state ─────────────────────────────────
            self._resume_checkpoint(resume_from)
            if extra_epochs is not None:
                self.cfg.max_epochs = self.current_epoch + extra_epochs
                log.info(f"extra_epochs={extra_epochs} → "
                         f"training until epoch {self.cfg.max_epochs}")
            mode_str = f"Resuming from epoch {self.current_epoch}"

        elif finetune_from:
            # ── FINETUNE: weights only, everything else stays fresh ────────
            self._finetune_checkpoint(finetune_from)
            mode_str = "Finetuning from pretrained weights (fresh optimizer + history)"

        else:
            mode_str = "Starting fresh"

        self._fire("on_train_start")
        log.info(f"{mode_str}. "
                 f"Epochs {self.current_epoch}→{self.cfg.max_epochs}. "
                 f"Output: {self.output_dir}")
        log.info(f"Device: {self.device} | Precision: {self.cfg.precision} | "
                 f"Strategy: {self.cfg.strategy}")
        if self._log_file_path:
            log.info(f"Log file: {self._log_file_path}")

        train_start = time.time()

        for epoch in range(self.current_epoch, self.cfg.max_epochs):
            self.current_epoch = epoch
            self._fire("on_epoch_start", epoch)

            # ── Train epoch ───────────────────
            train_metrics = self._run_epoch(train_loader, training=True)
            train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}

            # ── Val epoch ─────────────────────
            val_metrics = {}
            if val_loader is not None and (epoch + 1) % self.cfg.val_every_n_epochs == 0:
                val_metrics = self._run_epoch(val_loader, training=False)
                val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

            all_metrics = {**train_metrics, **val_metrics,
                           "lr": self.optimizer.param_groups[0]["lr"],
                           "epoch": epoch}
            self._history.append(all_metrics)
            self._fire("on_epoch_end", epoch, all_metrics)

            # ── Scheduler step ────────────────
            if self.scheduler is not None:
                if is_plateau_scheduler(self.scheduler):
                    monitor_val = all_metrics.get(self.cfg.early_stopping_monitor,
                                                  all_metrics.get("val_loss"))
                    if monitor_val is not None:
                        self.scheduler.step(monitor_val)
                else:
                    self.scheduler.step()

            # ── Checkpointing ─────────────────
            self._handle_checkpointing(all_metrics, epoch)

            # ── Early stopping ────────────────
            if self._early_stopper is not None:
                if self._early_stopper.step(all_metrics):
                    log.info(f"Early stopping at epoch {epoch}.")
                    break

            # ── Max steps override ─────────────
            if self.cfg.max_steps is not None and self.global_step >= self.cfg.max_steps:
                log.info(f"Reached max_steps={self.cfg.max_steps}. Stopping.")
                break

        elapsed = time.time() - train_start
        log.info(f"Training complete in {elapsed/60:.1f} min "
                 f"({elapsed:.0f} s).")
        self._fire("on_train_end")

        if self._wandb is not None:
            self._wandb.finish()
        if self._tb_writer is not None:
            self._tb_writer.close()

        self._save_history()
        return self

    # ── Epoch runner ──────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, training: bool) -> Dict[str, float]:
        self.model.train(training)
        tracker = MetricsTracker()

        phase = "Train" if training else "Val  "
        desc  = f"{phase} ep{self.current_epoch:04d}"

        use_pbar = self.cfg.progress_bar and _TQDM_AVAILABLE
        pbar = self._make_pbar(loader, desc=desc, total=len(loader))

        if not training:
            self._fire("on_val_start")

        accum_steps = self.cfg.grad_accumulation_steps
        accum_count = 0

        for batch_idx, batch in enumerate(pbar):
            if training:
                self._fire("on_batch_start", batch, batch_idx)
                metrics = self.training_step(batch, batch_idx)
                accum_count += 1

                if accum_count == accum_steps:
                    self.on_before_optimizer_step()
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    self.global_step += 1

                    if self.ema is not None:
                        raw = self.model.module if hasattr(self.model, "module") else self.model
                        self.ema.update(raw)

                    if self.global_step % self.cfg.log_every_n_steps == 0:
                        self._log_step(metrics)

                self._fire("on_batch_end", metrics.get("loss", 0.0), batch_idx)
            else:
                metrics = self.validation_step(batch, batch_idx)

            tracker.update(metrics, n=1)

            # postfix updated without forcing refresh — tqdm's natural
            # __next__ advance controls when to redraw, preventing double-writes
            if use_pbar:
                running = tracker.compute()
                pbar.set_postfix(
                    {k: f"{v:.4f}" for k, v in running.items() if isinstance(v, float)},
                    refresh=False,
                )

        if use_pbar:
            pbar.close()

        epoch_metrics = tracker.compute()
        if not training:
            self._fire("on_val_end", epoch_metrics)
        return epoch_metrics

    # ── Checkpointing ─────────────────────────────────────────────────────

    def _handle_checkpointing(self, metrics: Dict, epoch: int):
        cfg = self.cfg

        if cfg.save_last:
            self._save_checkpoint("last.pt", epoch, metrics)

        monitor_val = metrics.get(cfg.checkpoint_monitor)
        if cfg.save_best and monitor_val is not None:
            if self._is_better_ckpt(monitor_val, self._best_metric):
                self._best_metric = monitor_val
                self._save_checkpoint("best.pt", epoch, metrics)
                log.info(f"  ✓ New best {cfg.checkpoint_monitor}={monitor_val:.4f}")

        if cfg.save_every_n_epochs and (epoch + 1) % cfg.save_every_n_epochs == 0:
            self._save_checkpoint(f"epoch_{epoch:04d}.pt", epoch, metrics)

    def _save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        raw = self.model.module if hasattr(self.model, "module") else self.model
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": raw.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": asdict(self.cfg),
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()

        path = self.output_dir / filename
        torch.save(state, path)

    def _resume_checkpoint(self, path: str):
        """
        RESUME — restore full training state.
        After this call, training continues as if it was never interrupted:
          - model weights
          - optimizer state (Adam/SGD momentum buffers, step counts)
          - scheduler state (last_epoch, base_lrs, internal counters)
          - GradScaler state (float16 only)
          - EMA weights
          - epoch counter → training loop starts from the next epoch
          - global_step
          - history → new epochs append to the old timeline
          - best_metric → checkpointing stays consistent with previous run
        """
        log.info(f"[RESUME] Loading full state from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Model
        raw = self.model.module if hasattr(self.model, "module") else self.model
        raw.load_state_dict(ckpt["model"])

        # Optimizer — restores Adam/SGD internal buffers (momentum, exp_avg, etc.)
        if self.optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])

        # Scheduler — restores last_epoch and all internal LR state
        # This means LR continues exactly where it left off, not from epoch 0
        if self.scheduler and "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])

        # GradScaler (float16 AMP only)
        if self.scaler and "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])

        # EMA shadow weights
        if self.ema and "ema" in ckpt:
            self.ema.load_state_dict(ckpt["ema"])

        # Step/epoch counters — loop starts from the NEXT epoch
        self.current_epoch = ckpt.get("epoch", 0) + 1
        self.global_step   = ckpt.get("global_step", 0)

        # History — find history.json next to checkpoint, then fall back to output_dir
        for history_dir in [Path(path).parent, self.output_dir]:
            history_path = history_dir / "history.json"
            if history_path.exists():
                with open(history_path) as f:
                    self._history = json.load(f)
                log.info(f"[RESUME] Restored {len(self._history)} epoch history "
                         f"entries from {history_path}")
                break
        else:
            log.warning("[RESUME] history.json not found — history will restart "
                        "from the resumed epoch")

        # Best metric — so save_best logic is consistent with previous run
        if "metrics" in ckpt:
            monitor_val = ckpt["metrics"].get(self.cfg.checkpoint_monitor)
            if monitor_val is not None:
                self._best_metric = monitor_val
                log.info(f"[RESUME] Restored best "
                         f"{self.cfg.checkpoint_monitor}={monitor_val:.4f}")

        log.info(f"[RESUME] Continuing from epoch {self.current_epoch} "
                 f"(global_step={self.global_step})")

    def _finetune_checkpoint(self, path: str):
        """
        FINETUNE — load pretrained weights only, start everything else fresh.
        After this call:
          - model weights are loaded from the checkpoint
          - optimizer: fresh (no momentum history from pretraining)
          - scheduler: fresh (starts from epoch 0 of the new LR schedule)
          - scaler: fresh
          - EMA: re-initialized from the loaded weights
          - epoch counter: 0 (new training run)
          - global_step: 0
          - history: empty (new run's own history)
          - best_metric: reset (new run competes against itself)

        Use when: transfer learning, changing dataset, changing LR after
        pretraining, running ablations from a shared pretrained base.
        """
        log.info(f"[FINETUNE] Loading weights only from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        raw = self.model.module if hasattr(self.model, "module") else self.model
        raw.load_state_dict(ckpt["model"])

        # Re-initialize EMA from the newly loaded weights (not from ckpt EMA)
        # because the new training may use a different decay or different LR regime
        if self.ema is not None:
            self.ema = ModelEMA(raw, decay=self.cfg.ema_decay)

        # Everything else stays at __init__ defaults:
        # current_epoch=0, global_step=0, _history=[], _best_metric=inf/-inf
        # optimizer/scheduler were already built fresh in fit() before this call

        saved_epoch = ckpt.get("epoch", "?")
        log.info(f"[FINETUNE] Weights loaded (checkpoint was at epoch {saved_epoch}). "
                 f"Optimizer, scheduler, and history are fresh.")

    def load_best(self):
        """Load best checkpoint weights into model (for inference). Weights only."""
        self._finetune_checkpoint(str(self.output_dir / "best.pt"))
        return self

    # ── Multi-GPU ─────────────────────────────────────────────────────────

    def _wrap_model(self, model: nn.Module) -> nn.Module:
        strategy = self.cfg.strategy
        n_gpus = torch.cuda.device_count()

        if strategy == "ddp":
            # Assumes torchrun / torch.distributed already initialized
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            model = model.to(local_rank)
            return DDP(model, device_ids=[local_rank])
        elif strategy in ("dp", "auto") and n_gpus > 1:
            log.info(f"Using DataParallel across {n_gpus} GPUs.")
            return nn.DataParallel(model)
        return model

    # ── Logger setup ──────────────────────────────────────────────────────

    def _setup_logger(self):
        """
        Configures the 'Trainer' logger with:
          - A StreamHandler (console) using a clean, coloured-friendly format.
          - An optional RotatingFileHandler writing to output_dir/trainer.log.

        Uses the module-level `log` so all log.info() calls inside the
        class automatically go to both sinks.
        """
        cfg = self.cfg
        level = getattr(logging, cfg.log_level.upper(), logging.INFO)
        formatter = logging.Formatter(fmt=cfg.log_format, datefmt=cfg.log_date_format)

        # Avoid adding duplicate handlers if Trainer is instantiated more than once
        log.handlers.clear()
        log.setLevel(level)
        log.propagate = False  # don't double-print via root logger

        # Console handler — routes through tqdm.write so bars aren't broken
        ch = _TqdmLoggingHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        log.addHandler(ch)

        # File handler
        if cfg.use_log_file:
            self._log_file_path = self.output_dir / cfg.log_file_name
            fh = RotatingFileHandler(
                filename=str(self._log_file_path),
                maxBytes=cfg.log_file_max_bytes,
                backupCount=cfg.log_file_backup_count,
                encoding="utf-8",
            )
            fh.setLevel(logging.DEBUG)   # file always gets full detail
            fh.setFormatter(formatter)
            log.addHandler(fh)
            log.debug(f"Log file: {self._log_file_path}")

    # ── External loggers ──────────────────────────────────────────────────

    def _log_step(self, metrics: Dict):
        """Push step-level metrics to W&B / TensorBoard if enabled."""
        if self._wandb is not None:
            self._wandb.log({**metrics, "step": self.global_step})
        if self._tb_writer is not None:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(k, v, self.global_step)
        # Always mirror to the file logger at DEBUG level
        metric_str = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in metrics.items())
        log.debug(f"[step {self.global_step:07d}] {metric_str}")

    def _init_wandb(self):
        try:
            import wandb
            self._wandb = wandb.init(
                project=self.cfg.wandb_project or "trainer",
                entity=self.cfg.wandb_entity,
                name=self.cfg.run_name,
                config=asdict(self.cfg),
            )
        except ImportError:
            warnings.warn("wandb not installed. Skipping W&B logging.")

    def _init_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(log_dir=str(self.output_dir / "tb"))
        except ImportError:
            warnings.warn("TensorBoard not installed. Skipping TB logging.")

    # ── Progress bar helpers ──────────────────────────────────────────────
    #
    # Design: ONE bar per phase (train / val), leave=True so it freezes as a
    # summary line when done. No position= kwarg — that forces multi-line mode
    # which requires a real TTY and breaks in SSH/subprocess/VS Code terminals.
    # Logs go to stdout via tqdm.write; bars go to stderr — separate streams.

    def _make_pbar(self, iterable, desc: str, total: int):
        if self.cfg.progress_bar and _TQDM_AVAILABLE:
            return tqdm(
                iterable,
                desc=desc,
                total=total,
                leave=True,               # freeze as summary line when done
                disable=False,
                ncols=90,                 # fixed — dynamic_ncols=True returns 0 in non-TTY
                file=sys.stderr,
                miniters=1,
                mininterval=0.1,          # redraw at most 10×/sec
                bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
            )
        return iterable

    def _make_epoch_pbar(self, total: int):
        # Kept for API compat but returns None — epoch bar removed (caused
        # position= conflicts). Epoch progress is visible from the frozen bars.
        return None

    def _tqdm_log(self, msg: str):
        """Safe log line that won't corrupt an active bar."""
        if _TQDM_AVAILABLE:
            tqdm.write(msg, file=sys.stdout)
        else:
            print(msg, file=sys.stdout)

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _fire(self, hook: str, *args, **kwargs):
        for cb in self.callbacks:
            getattr(cb, hook)(self, *args, **kwargs)

    # ── History ───────────────────────────────────────────────────────────

    def _save_history(self):
        path = self.output_dir / "history.json"
        path.write_text(json.dumps(self._history, indent=2))

    @property
    def history(self) -> List[Dict]:
        return self._history


# ─────────────────────────────────────────────
# 8. MULTI-PHASE TRAINER (for papers with staged training, e.g. SpectralNet)
# ─────────────────────────────────────────────

class MultiPhaseTrainer:
    """
    Wraps multiple Trainer subclasses that must run sequentially,
    passing state (e.g. trained encoder weights) between phases.

    Usage
    -----
    phases = [
        SiameseTrainer(siamese_model, phase1_cfg),
        SpectralTrainer(spectral_model, siamese_model, phase2_cfg),
    ]
    MultiPhaseTrainer(phases).run(train_loader, val_loader)
    """
    def __init__(self, phases: List[Trainer]):
        self.phases = phases

    def run(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        for i, trainer in enumerate(self.phases):
            log.info(f"\n{'='*50}")
            log.info(f"  Phase {i+1}/{len(self.phases)}: {type(trainer).__name__}")
            log.info(f"{'='*50}")
            trainer.fit(train_loader, val_loader)
        return self


# ─────────────────────────────────────────────
# 9. MINIMAL USAGE EXAMPLE  (not executed on import)
# ─────────────────────────────────────────────

def _example():
    """
    How to use this trainer for a simple classification task.
    Delete or replace this when adapting to your paper.
    """
    import torch.nn.functional as F
    from torchvision import datasets, transforms

    class SimpleClassifierTrainer(Trainer):
        def compute_loss(self, batch):
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x.view(x.size(0), -1))
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(-1) == y).float().mean()
            return loss, {"loss": loss.item(), "acc": acc.item()}

    cfg = TrainerConfig(
        output_dir="runs",
        run_name="mnist_test",
        max_epochs=5,
        lr=1e-3,
        optimizer="adamw",
        precision="bfloat16",           # bfloat16 AMP
        grad_clip_norm=1.0,
        scheduler="cosine",
        early_stopping=True,
        early_stopping_monitor="val_loss",
        early_stopping_patience=3,
        save_best=True,
        ema=True,
        ema_decay=0.999,
        seed=42,
        # Logging
        use_log_file=True,              # → runs/mnist_test/trainer.log
        log_level="INFO",
        # Progress bars (requires: pip install tqdm)
        progress_bar=True,              # per-batch bar with live loss
        progress_bar_epochs=True,       # outer epoch bar
    )

    model = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )

    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("data", train=True,  download=True, transform=tf)
    val_ds   = datasets.MNIST("data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=4)

    trainer = SimpleClassifierTrainer(model, cfg)
    trainer.fit(train_loader, val_loader)
    trainer.load_best()
    print("Best val_loss:", trainer._best_metric)


if __name__ == "__main__":
    _example()