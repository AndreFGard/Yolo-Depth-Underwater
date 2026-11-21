#!/usr/bin/env python3
"""Train a YOLO depth model.

Example:
    python scripts/train_depth.py --data data.yaml --epochs 30 --fraction 0.1
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from ultralytics import YOLO


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
    parser.add_argument("--project", help="Directory for training runs")
    parser.add_argument("--name", help="Name of this run")
    args = parser.parse_args(argv)

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        fraction=args.fraction,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        save_period=args.save_period,
        project=args.project,
        name=args.name,
    )
    _write_run_doc(Path(model.trainer.save_dir), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
