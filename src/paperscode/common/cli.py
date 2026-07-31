import argparse

from paperscode.common.trainer import TrainerConfig


def add_trainer_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("Trainer")

    # ==========================================================
    # Experiment
    # ==========================================================
    g.add_argument("--output-dir", type=str, default="runs")
    g.add_argument("--run-name", type=str, default="experiment")

    # ==========================================================
    # Training
    # ==========================================================
    g.add_argument("--max-epochs", type=int, default=5)
    g.add_argument("--max-steps", type=int, default=None)
    g.add_argument("--batch-size", type=int, default=256)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    # ==========================================================
    # Precision
    # ==========================================================
    g.add_argument(
        "--precision",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )

    # ==========================================================
    # Optimizer
    # ==========================================================
    g.add_argument(
        "--optimizer",
        choices=["adam", "adamw", "sgd", "rmsprop"],
        default="adamw",
    )

    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--momentum", type=float, default=0.9)
    g.add_argument("--beta1", type=float, default=0.9)
    g.add_argument("--beta2", type=float, default=0.999)

    # ==========================================================
    # Gradient
    # ==========================================================
    g.add_argument("--grad-clip-norm", type=float, default=1.0)
    g.add_argument("--grad-clip-value", type=float, default=None)
    g.add_argument("--grad-accumulation-steps", type=int, default=1)

    # ==========================================================
    # Scheduler
    # ==========================================================
    g.add_argument(
        "--scheduler",
        choices=[
            "none",
            "step",
            "cosine",
            "plateau",
            "linear_warmup_cosine",
        ],
        default="none",
    )

    g.add_argument("--scheduler-step-size", type=int, default=10)
    g.add_argument("--scheduler-gamma", type=float, default=0.5)
    g.add_argument("--scheduler-eta-min", type=float, default=0.0)
    g.add_argument("--scheduler-patience", type=int, default=5)
    g.add_argument("--scheduler-warmup-steps", type=int, default=0)

    # ==========================================================
    # Early Stopping
    # ==========================================================
    g.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    g.add_argument("--early-stopping-monitor", default="val_loss")

    g.add_argument(
        "--early-stopping-mode",
        choices=["min", "max"],
        default="min",
    )

    g.add_argument("--early-stopping-patience", type=int, default=10)
    g.add_argument("--early-stopping-min-delta", type=float, default=1e-4)

    # ==========================================================
    # Checkpointing
    # ==========================================================
    g.add_argument(
        "--save-best",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    g.add_argument(
        "--save-last",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    g.add_argument("--save-every-n-epochs", type=int, default=None)

    g.add_argument("--checkpoint-monitor", default="val_loss")

    g.add_argument(
        "--checkpoint-mode",
        choices=["min", "max"],
        default="min",
    )

    # ==========================================================
    # Multi-GPU
    # ==========================================================
    g.add_argument(
        "--strategy",
        choices=["auto", "cpu", "dp", "ddp"],
        default="auto",
    )

    # ==========================================================
    # EMA
    # ==========================================================
    g.add_argument(
        "--ema",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    g.add_argument("--ema-decay", type=float, default=0.999)

    # ==========================================================
    # Logging
    # ==========================================================
    g.add_argument("--log-every-n-steps", type=int, default=10)

    g.add_argument(
        "--use-log-file",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    g.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )

    g.add_argument("--log-file-name", default="trainer.log")
    g.add_argument("--log-file-max-bytes", type=int, default=10_485_760)
    g.add_argument("--log-file-backup-count", type=int, default=3)

    # ==========================================================
    # W&B / TensorBoard
    # ==========================================================
    g.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    g.add_argument(
        "--use-tensorboard",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    g.add_argument("--wandb-project", default=None)
    g.add_argument("--wandb-entity", default=None)

    # ==========================================================
    # Progress Bars
    # ==========================================================
    g.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    g.add_argument(
        "--progress-bar-epochs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # ==========================================================
    # Validation
    # ==========================================================
    g.add_argument("--val-every-n-epochs", type=int, default=1)

    # ==========================================================
    # Data Loading
    # ==========================================================
    g.add_argument("--num-workers", type=int, default=4)

    g.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # ==========================================================
    # Resume / Finetune
    # ==========================================================
    g.add_argument("--resume", type=str, default=None)
    g.add_argument("--finetune", type=str, default=None)
    g.add_argument("--extra-epochs", type=int, default=None)

    return parser


def trainer_config_from_args(args) -> TrainerConfig:
    return TrainerConfig(
        # Paths
        output_dir=args.output_dir,
        run_name=args.run_name,

        # Training
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,

        # Precision
        precision=args.precision,

        # Gradient
        grad_clip_norm=args.grad_clip_norm,
        grad_clip_value=args.grad_clip_value,
        grad_accumulation_steps=args.grad_accumulation_steps,

        # Optimizer
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        betas=(args.beta1, args.beta2),

        # Scheduler
        scheduler=None if args.scheduler == "none" else args.scheduler,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        scheduler_eta_min=args.scheduler_eta_min,
        scheduler_patience=args.scheduler_patience,
        scheduler_warmup_steps=args.scheduler_warmup_steps,

        # Early stopping
        early_stopping=args.early_stopping,
        early_stopping_monitor=args.early_stopping_monitor,
        early_stopping_mode=args.early_stopping_mode,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,

        # Checkpointing
        save_best=args.save_best,
        save_last=args.save_last,
        save_every_n_epochs=args.save_every_n_epochs,
        checkpoint_monitor=args.checkpoint_monitor,
        checkpoint_mode=args.checkpoint_mode,

        # Strategy
        strategy=args.strategy,

        # EMA
        ema=args.ema,
        ema_decay=args.ema_decay,

        # Logging
        log_every_n_steps=args.log_every_n_steps,
        log_level=args.log_level,
        use_log_file=args.use_log_file,
        log_file_name=args.log_file_name,
        log_file_max_bytes=args.log_file_max_bytes,
        log_file_backup_count=args.log_file_backup_count,

        # External loggers
        use_wandb=args.use_wandb,
        use_tensorboard=args.use_tensorboard,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,

        # Progress
        progress_bar=args.progress_bar,
        progress_bar_epochs=args.progress_bar_epochs,

        # Reproducibility
        seed=args.seed,
        deterministic=args.deterministic,

        # Validation
        val_every_n_epochs=args.val_every_n_epochs,
    )