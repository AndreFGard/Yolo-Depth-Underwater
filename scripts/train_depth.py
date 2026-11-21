#!/usr/bin/env python3
"""Train a YOLO depth model, resuming an existing named run when possible.

Example:
    python scripts/train_depth.py --data data.yaml --epochs 30 --fraction 0.1
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from ultralytics import YOLO

DEFAULT_FREEZE = 11
"""Top-level YOLO26 modules 0-10 are the backbone; 11 freezes all of it."""


def _named_run_dir(project: str | None, name: str) -> Path:
    from ultralytics.utils import RUNS_DIR

    project_path = Path(project) if project else Path()
    if project_path.is_absolute():
        base = project_path
    else:
        base = Path(RUNS_DIR) / "depth" / project_path
    return (base / name).resolve()


def _write_run_doc(run_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        f"# Run: {run_dir.name}",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Command | `{shlex.join(sys.argv)}` |",
        f"| Dataset | `{args.data}` |",
        f"| Starting model | `{args.model}` |",
        f"| Epochs | {args.epochs} |",
        f"| Frozen layers | first {args.freeze} |",
        f"| Train fraction | {args.fraction:g} |",
        f"| Image size | {args.imgsz} |",
        f"| Batch size | {args.batch} |",
        f"| Device | `{args.device}` |",
        f"| Save period | {args.save_period} epochs |",
        "",
        "Artifacts: `args.yaml`, `results.csv`, `weights/best.pt`, `weights/last.pt`.",
    ]
    (run_dir / "RUN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True, help="Dataset YAML")
    parser.add_argument("--model", default="yolo26n-depth.pt", help="Initial weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--fraction", type=float, default=1.0, help="Fraction of the train split to use"
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument(
        "--save-period",
        type=int,
        default=2,
        help="Save an extra checkpoint every N epochs; -1 disables it",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=DEFAULT_FREEZE,
        help="Freeze the first N top-level modules (default: 11 = whole backbone; 0 = none)",
    )
    parser.add_argument("--project", help="Directory for training runs")
    parser.add_argument("--name", help="Name of this run")
    args = parser.parse_args(argv)

    existing_run = False
    resume = False
    model = None
    if args.name:
        run_dir = _named_run_dir(args.project, args.name)
        existing_run = run_dir.exists()
        last_checkpoint = run_dir / "weights" / "last.pt"
        if last_checkpoint.is_file():
            model = YOLO(str(last_checkpoint))
            checkpoint = model.ckpt or {}
            checkpoint_epoch = checkpoint.get("epoch", -1)
            checkpoint_args = checkpoint.get("train_args", {})
            checkpoint_epochs = checkpoint_args.get("epochs", args.epochs)
            if checkpoint.get("optimizer") is None or checkpoint_epoch < 0:
                parser.error(
                    f"{last_checkpoint} is not a resumable training checkpoint; "
                    "use a new --name to start another run"
                )
            if checkpoint_epoch + 1 >= checkpoint_epochs:
                parser.error(
                    f"{last_checkpoint} already completed {checkpoint_epochs} epochs; "
                    "use a new --name to start another run"
                )
            resume = True
            print(f"Resuming existing run from {last_checkpoint}")
        elif existing_run:
            print(f"No checkpoint in {run_dir}; starting over in that run directory")

    if model is None:
        model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        fraction=args.fraction,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        freeze=args.freeze,
        save_period=args.save_period,
        project=args.project,
        name=args.name,
        exist_ok=existing_run and not resume,
        resume=resume,
    )
    _write_run_doc(Path(model.trainer.save_dir), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
