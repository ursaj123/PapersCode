import argparse
from paperscode.common.trainer import Trainer, TrainerConfig


def add_trainer_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Adds all common TrainerConfig arguments to an ArgumentParser.
    Paper-specific arguments should be added separately.
    """

    g = parser.add_argument_group("Trainer")

    # ==========================================================
    # Experiment
    # ==========================================================
    g.add_argument("--output-dir", type=str, default="runs")
    g.add_argument("--run-name", type=str, default="mnist_test")

    # ==========================================================
    # Training
    # ==========================================================
    g.add_argument("--max-epochs", type=int, default=5)
    g.add_argument("--batch-size", type=int, default=256)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--seed", type=int, default=42)

    # ==========================================================
    # Optimizer
    # ==========================================================
    g.add_argument(
        "--optimizer",
        choices=["adam", "adamw", "sgd", "rmsprop"],
        default="adamw",
    )

    g.add_argument("--weight-decay", type=float, default=0.0)

    # ==========================================================
    # Precision
    # ==========================================================
    g.add_argument(
        "--precision",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )

    # ==========================================================
    # Gradient
    # ==========================================================
    g.add_argument("--grad-clip-norm", type=float, default=1.0)
    g.add_argument("--grad-clip-value", type=float, default=None)

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
        default="cosine",
    )

    # ==========================================================
    # Early Stopping
    # ==========================================================
    g.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    g.add_argument(
        "--early-stopping-monitor",
        type=str,
        default="val_loss",
    )

    g.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
    )

    # ==========================================================
    # Checkpointing
    # ==========================================================
    g.add_argument(
        "--save-best",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # ==========================================================
    # EMA
    # ==========================================================
    g.add_argument(
        "--ema",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    g.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
    )

    # ==========================================================
    # Logging
    # ==========================================================
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
    # Data Loading
    # ==========================================================
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)

    # ==========================================================
    # Device
    # ==========================================================
    g.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )

    g.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to resume training from.",
    )

    return parser



def trainer_config_from_args(args) -> TrainerConfig:
    return TrainerConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        max_epochs=args.max_epochs,
        lr=args.lr,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        precision=args.precision,
        grad_clip_norm=args.grad_clip_norm,
        grad_clip_value=args.grad_clip_value,
        scheduler=None if args.scheduler == "none" else args.scheduler,
        early_stopping=args.early_stopping,
        early_stopping_monitor=args.early_stopping_monitor,
        early_stopping_patience=args.early_stopping_patience,
        save_best=args.save_best,
        ema=args.ema,
        ema_decay=args.ema_decay,
        seed=args.seed,
        use_log_file=args.use_log_file,
        log_level=args.log_level,
        progress_bar=args.progress_bar,
        progress_bar_epochs=args.progress_bar_epochs,
    )
