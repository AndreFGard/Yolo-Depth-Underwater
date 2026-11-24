#!/usr/bin/env python3
"""Score one YOLO training run's checkpoints on the FLSea evaluation frames."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

from yolo_depth_underwater import evaluation, flsea

LOGGER = logging.getLogger(__name__)

MANIFEST = Path("configs/flsea_eval_frames.csv")
SUBSET = flsea.Subset.CAI_METZLER
LOCAL_ROOT = Path("/guaxim/.cache/kagglehub/datasets/viseaonlab/flsea-vi/versions/5")
DEVICE = "cuda"
BASELINE_WEIGHTS = "yolo26n-depth.pt"
CHECKPOINT_FILES = (
     [f"epoch{e}.pt" for e in (2, 4, 6, 8, 12, 16, 20, 24,28, 32,36, 40, 44, 48,52,56,60, 64,68,72,76, 78)]
    #[f"epoch{e}.pt" for e in (2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 78)]
    + ["best.pt", "last.pt"]
)

LADDER_EXTRA = (
    "run",
    "checkpoint",
    "epoch",
    "step",
    "weights_sha256",
    "cal_a",
    "cal_b",
)
LADDER_COLUMNS = (*LADDER_EXTRA, *evaluation.ROW_FIELDS)


def _checkpoint_meta_and_provenance(path: Path) -> dict:
    """Provenance of one weight file: optimizer step and depth-head calibration."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch")
    step = ckpt.get("updates")
    train_args = ckpt.get("train_args") or {}
    total = train_args.get("epochs")
    model = ckpt.get("ema") or ckpt.get("model")
    state = model.state_dict()
    cal_a = next(v for k, v in state.items() if k.endswith("cal_a"))
    cal_b = next(v for k, v in state.items() if k.endswith("cal_b"))
    return {
        "epoch": int(epoch) if epoch is not None else -1,
        "step": int(step) if step is not None else -1,
        "total_epochs": int(total) if total is not None else -1,
        "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cal_a": float(cal_a.reshape(-1)[0]),
        "cal_b": float(cal_b.reshape(-1)[0]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from yolo_depth_underwater.encoding import write_csv
    from yolo_depth_underwater.yolo_model import YoloModel

    name = args.run_dir.name
    out = args.out or Path("artifacts") / f"flsea_ladder_{name}.csv"

    frames = flsea.read_manifest(MANIFEST, flsea.SUBSETS[SUBSET])
    frames = flsea.attach_local_root(frames, LOCAL_ROOT)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    all_rows: list[dict] = []
    if out.exists():
        print(f"overwriting {out}: resume disabled, scoring every checkpoint fresh")

    ladder = [("baseline", None)] + [(Path(f).stem, f) for f in CHECKPOINT_FILES]
    for label, filename in ladder:
        weight = (
            Path(BASELINE_WEIGHTS)
            if filename is None
            else args.run_dir / "weights" / filename
        )
        if not weight.is_file():
            LOGGER.warning("skip %s/%s: missing %s", name, label, weight)
            continue
        rows, _ = evaluation.evaluate_model(
            YoloModel(None, str(weight), DEVICE), frames
        )
        if not rows or "ssi" not in {r["alignment"] for r in rows}:
            raise SystemExit(f"{name}/{label}: eval produced no usable rows")
        meta = _checkpoint_meta_and_provenance(weight)
        if filename is None:
            meta["epoch"], meta["step"] = 0, 0
        elif label.startswith("epoch"):
            meta["epoch"] = int(label.removeprefix("epoch"))
        elif label == "last" and meta["epoch"] < 0 and meta["total_epochs"] > 0:
            meta["epoch"] = meta["total_epochs"]
        del meta["total_epochs"]
        for row in rows:
            row.update({"run": name, "checkpoint": label, **meta})
        all_rows.extend(rows)
        write_csv(all_rows, out, columns=list(LADDER_COLUMNS))
        ssi_abs = sum(r["AbsRel"] for r in rows if r["alignment"] == "ssi")
        ssi_n = sum(1 for r in rows if r["alignment"] == "ssi")
        print(
            f"{name}/{label}: {ssi_n} frames -> {out} (ssi AbsRel={ssi_abs / ssi_n:.4f})"
        )
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
