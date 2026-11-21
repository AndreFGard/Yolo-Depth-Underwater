#!/usr/bin/env python3
"""Train a YOLO depth model on a dataset fraction and document the run.

Usage (new run or warm start from previous weights):
    python scripts/train_depth.py --data datasets/usod10k-unidepth/data.yaml \\
        --name usod-s1-30e --epochs 30 --fraction 0.1

    python scripts/train_depth.py --data datasets/usod10k-unidepth/data.yaml \\
        --model runs/depth/usod-s1-30e/weights/best.pt \\
        --name usod-s2-20e --epochs 20 --fraction 0.1

Usage (resume an unfinished run):
    python scripts/train_depth.py --resume usod-s1-30e

A new run requires ``--name``. ``--resume`` takes the name of an existing run
and inherits every training setting from that run's checkpoint; combining it
with training arguments such as ``--epochs``, ``--data`` or ``--model`` is an
error. ``RUN.md`` records the command, environment, model provenance, actual
training subset and final metrics.

The ROCm workaround for gfx1030 boards is applied before torch is imported.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

_ENV_SETUP = {
    "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
}
_RUN_FACTS: dict = {}
_ROCm_PATHS = ("/opt/rocm",)


def _apply_environment() -> None:
    """Set backend workarounds that must be in place before torch is imported."""
    if any(Path(p).exists() for p in _ROCm_PATHS):
        for key, value in _ENV_SETUP.items():
            os.environ.setdefault(key, value)


def _git_commit() -> str:
    """Return the current git commit, or 'unknown' when unavailable."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _resolve_save_dir(project: str | None, name: str, runs_dir: Path) -> Path:
    """Mirror ultralytics' save-directory resolution for a depth training run."""
    if project:
        base = Path(project)
        if not base.is_absolute():
            base = runs_dir / "depth" / base
    else:
        base = runs_dir / "depth"
    return (base / name).resolve()


def _model_provenance(model: str, ckpt: dict | None) -> str:
    """One-line description of --model for RUN.md."""
    if not ckpt:
        return f"{model} (named pretrained asset)"
    args = ckpt.get("train_args") or {}
    details = [
        f"epoch={ckpt.get('epoch', -1)}",
        f"opt={'yes' if ckpt.get('optimizer') is not None else 'stripped'}",
    ]
    details += [f"src_{key}={args[key]}" for key in ("epochs", "fraction") if key in args]
    return f"{model} ({', '.join(details)})"


def _dump_subset(trainer) -> None:
    """Write the exact training image list and capture run facts."""
    dataset = trainer.train_loader.dataset
    root = Path(trainer.data["path"]).resolve()
    files = [Path(f) for f in dataset.im_files]

    def relative(path: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    listing = trainer.save_dir / "subset_train_images.txt"
    listing.write_text("\n".join(relative(f) for f in files) + "\n")
    _RUN_FACTS.update(
        train_images=len(files),
        val_images=len(trainer.test_loader.dataset),
        device=str(trainer.device),
        batch=trainer.args.batch,
        imgsz=trainer.args.imgsz,
        subset_listing=listing,
    )
    print(f"training subset: {len(files)} images -> {listing}")


def _metric_row(rows: list[dict], key: str) -> dict:
    """Return the results.csv row with the largest value for ``key``."""
    return max(rows, key=lambda r: float(r[key]))


def _write_run_doc(
    save_dir: Path,
    args: argparse.Namespace,
    data_yaml: Path,
    started: datetime,
    model_provenance: str,
) -> None:
    """Write RUN.md with the command, environment, subset and final metrics."""
    import torch
    import ultralytics

    lines = [
        f"# Run: {save_dir.name}",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Command | `{shlex.join(sys.argv)}` |",
        f"| Dataset | `{data_yaml}` |",
        f"| Model | `{args.model}` |",
        f"| Model provenance | `{model_provenance}` |",
        f"| Epochs | {args.epochs} |",
        f"| Fraction | {args.fraction:g} |",
    ]
    facts = _RUN_FACTS
    if facts:
        lines.extend(
            [
                f"| Train images used | {facts['train_images']} |",
                f"| Val images | {facts['val_images']} |",
                f"| Subset listing | `{Path(facts['subset_listing']).name}` |",
                f"| Batch / imgsz | {facts['batch']} / {facts['imgsz']} |",
                f"| Device | `{facts['device']}` |",
            ]
        )
    lines.extend(
        [
            f"| Started (UTC) | {started.strftime('%Y-%m-%d %H:%M:%S')} |",
            f"| Host | {platform.node()} ({platform.platform()}) |",
            (
                f"| Python / torch / ultralytics | {platform.python_version()} / "
                f"{torch.__version__} / {ultralytics.__version__} |"
            ),
            f"| Git commit | `{_git_commit()}` |",
            f"| Depth provenance | `{data_yaml.parent / 'README.md'}` |",
            "",
            (
                "Artifacts: `args.yaml`, `results.csv`, `results.png`, `labels.jpg`, "
                "`weights/best.pt`, `weights/last.pt` in this directory."
            ),
        ]
    )

    results_csv = save_dir / "results.csv"
    if results_csv.is_file():
        with results_csv.open(newline="") as file:
            rows = list(csv.DictReader(file))
        if rows:
            best = _metric_row(rows, "metrics/delta1")
            lines.extend(
                [
                    "",
                    "## Final metrics",
                    "",
                    (
                        "Best epoch by val delta1 (checkpoint selection): **epoch "
                        f"{best['epoch']}**"
                    ),
                    "",
                    "| metric | best epoch | last epoch |",
                    "|---|---|---|",
                ]
            )
            last = rows[-1]
            for key, label in (
                ("metrics/delta1", "delta1"),
                ("metrics/delta2", "delta2"),
                ("metrics/delta3", "delta3"),
                ("metrics/abs_rel", "abs_rel"),
                ("metrics/rmse", "rmse"),
                ("metrics/silog", "silog"),
            ):
                lines.append(
                    f"| {label} | {float(best[key]):.4f} | {float(last[key]):.4f} |"
                )
            lines.extend(
                [
                    "",
                    (
                        "`best.pt` was auto-calibrated on the validation split by the "
                        "trainer (log-affine `cal_a`/`cal_b`)."
                    ),
                ]
            )
    (save_dir / "RUN.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--name", help="Name of a NEW run (use a fresh name for warm starts too)"
    )
    mode.add_argument(
        "--resume",
        metavar="NAME",
        help="Name of an EXISTING run to resume; training args come from its checkpoint",
    )
    parser.add_argument(
        "--data", type=Path, default=None, help="Ultralytics data YAML (new runs only)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Starting weights (new runs only; default: yolo26n-depth.pt)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Training epochs (new runs only)"
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Fraction of the train split to use (new runs only; val/test stay full)",
    )
    parser.add_argument("--imgsz", type=int, default=None, help="Image size (new runs only)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (new runs only)")
    parser.add_argument(
        "--project",
        default=None,
        help="Run group used to locate/create NAME (default: depth)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string, e.g. 0 or cpu (new runs only)",
    )
    args = parser.parse_args(argv)

    new_only = ("data", "model", "epochs", "fraction", "imgsz", "batch", "device")
    if args.resume is not None:
        conflicting = [f"--{key}" for key in new_only if getattr(args, key) is not None]
        if conflicting:
            parser.error(
                "--resume cannot be combined with "
                + ", ".join(conflicting)
                + "; all training settings are inherited from the checkpoint"
            )
        name = args.resume
    else:
        if args.data is None:
            parser.error("--data is required for a new run")
        args.model = args.model or "yolo26n-depth.pt"
        args.epochs = 100 if args.epochs is None else args.epochs
        args.fraction = 1.0 if args.fraction is None else args.fraction
        args.imgsz = 640 if args.imgsz is None else args.imgsz
        args.batch = 16 if args.batch is None else args.batch
        name = args.name

    _apply_environment()

    from ultralytics import YOLO
    from ultralytics.utils import RUNS_DIR

    save_dir = _resolve_save_dir(args.project, name, RUNS_DIR)
    started = datetime.now(UTC)
    data_yaml: Path | None = None
    model_provenance = ""
    resumed_from: Path | None = None

    if args.resume is not None:
        if not save_dir.is_dir():
            print(
                f"error: run directory {save_dir} does not exist; check --resume/--project",
                file=sys.stderr,
            )
            return 1

        resumed_from = save_dir / "weights" / "last.pt"
        if not resumed_from.is_file():
            print(
                f"error: run directory {save_dir} has no weights/last.pt",
                file=sys.stderr,
            )
            return 1

        model = YOLO(str(resumed_from))
        ckpt = model.ckpt or {}
        epoch = ckpt.get("epoch", -1)
        if epoch < 0 or ckpt.get("optimizer") is None:
            print(
                f"error: {resumed_from} is not a resumable training checkpoint; "
                f"start a new run with --name <new-name> --model {resumed_from}",
                file=sys.stderr,
            )
            return 1

        total_epochs = (ckpt.get("train_args") or {}).get("epochs")
        if isinstance(total_epochs, int) and epoch + 1 >= total_epochs:
            print(
                f"error: {resumed_from} already completed {total_epochs} epochs; "
                f"start a new run with --name <new-name> --model {resumed_from}",
                file=sys.stderr,
            )
            return 1

        overrides = {"resume": True, "save_dir": str(save_dir)}
    else:
        if save_dir.exists():
            print(
                f"error: run directory {save_dir} already exists; choose another "
                f"--name, or resume this run with --resume {name}",
                file=sys.stderr,
            )
            return 1

        data_yaml = args.data.resolve()
        model = YOLO(args.model)
        model_provenance = _model_provenance(args.model, model.ckpt)
        overrides = {
            "data": str(data_yaml),
            "epochs": args.epochs,
            "fraction": args.fraction,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "project": args.project,
            "name": name,
        }

    if args.device is not None:
        overrides["device"] = args.device
    model.add_callback("on_train_start", _dump_subset)
    print(f"run directory: {save_dir}")
    model.train(**overrides)

    trainer_save_dir = Path(model.trainer.save_dir)
    if resumed_from is None:
        assert data_yaml is not None  # set in the new-run branch
        _write_run_doc(trainer_save_dir, args, data_yaml, started, model_provenance)
        print(f"run documented in {trainer_save_dir / 'RUN.md'}")
    else:
        run_doc = trainer_save_dir / "RUN.md"
        if run_doc.is_file():
            with run_doc.open("a") as file:
                file.write(
                    f"\n## Resume ({started.strftime('%Y-%m-%d %H:%M:%S')} UTC)\n\n"
                    f"- Command: `{shlex.join(sys.argv)}`\n"
                    f"- Checkpoint: `{resumed_from}`\n"
                    "- Training settings inherited from the checkpoint.\n"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
