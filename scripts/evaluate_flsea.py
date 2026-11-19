#!/usr/bin/env python3
"""Evaluate a depth model on the FLSea evaluation frames.

Every model is scored on the same frames (``configs/flsea_eval_frames.csv``) with the
same alignment and metrics (see ``yolo_depth_underwater.metrics``).

Examples:
    uv run --extra cu --extra flsea --extra depth-anything \\
        python scripts/evaluate_flsea.py --model depth-anything

    # the six sequences of arXiv:2507.02148 only
    uv run --extra cu --extra flsea python scripts/evaluate_flsea.py \\
        --model yolo --variant n --subset arxiv-2507.02148

    # dataset already on disk (e.g. a Kaggle notebook)
    ... --local-root /kaggle/input/flsea-vi

Kaggle credentials (KAGGLE_USERNAME + KAGGLE_KEY, or KAGGLE_API_TOKEN) are needed to
download frames. See docs/flsea_eval.md.
"""

import argparse
import logging
from pathlib import Path

from yolo_depth_underwater import evaluation, flsea
from yolo_depth_underwater.inference_model import InferenceModel

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "configs" / "flsea_eval_frames.csv"


def build_model(name: str, variant: str | None, device: str | None) -> InferenceModel:
    """Models are imported lazily so each one only needs its own extra."""
    if name == "depth-anything":
        from yolo_depth_underwater.depth_anything_model import DepthAnythingV2Model

        return DepthAnythingV2Model(variant=variant or "small", device=device)
    if name == "yolo":
        from yolo_depth_underwater.yolo_model import YoloModel

        # No USOD10K DepthDataset is needed to call ``predict``.
        return YoloModel(None, variant or "n", device)  # type: ignore[arg-type]
    if name == "unidepth":
        from yolo_depth_underwater.unidepth_model import UniDepthV2Model

        return UniDepthV2Model(None, device)  # type: ignore[arg-type]
    raise ValueError(f"unknown model {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["depth-anything", "yolo", "unidepth"])
    parser.add_argument("--variant", help="depth-anything: small|base|large; yolo: n|s|x")
    parser.add_argument("--device", help="e.g. cuda, cpu (default: cuda if available)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--subset", choices=sorted(flsea.SUBSETS), default="all")
    parser.add_argument("--local-root", type=Path, help="use a local copy of the dataset instead of downloading")
    parser.add_argument("--min-valid-frac", type=float, default=evaluation.DEFAULT_MIN_VALID_FRAC)
    parser.add_argument("--include-raw", action="store_true",
                        help="also score depth-output models without alignment (only meaningful for metric models)")
    parser.add_argument("--workers", type=int, default=2, help="parallel downloads (keep low: Kaggle rate-limits)")
    parser.add_argument("--out", type=Path, help="per-frame CSV (default: results/flsea_<model>[_<variant>].csv)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    frames = flsea.read_manifest(args.manifest, flsea.SUBSETS[args.subset])
    print(f"{len(frames)} frames from the manifest (subset={args.subset})")
    if args.local_root:
        frames = flsea.attach_local_root(frames, args.local_root)
    else:
        frames = flsea.download_frames(frames, workers=args.workers)

    model = build_model(args.model, args.variant, args.device)
    rows, skipped = evaluation.evaluate_model(
        model, frames, min_valid_frac=args.min_valid_frac, include_raw=args.include_raw
    )

    out = args.out or Path("results") / (
        f"flsea_{args.model}" + (f"_{args.variant}" if args.variant else "") + ".csv"
    )
    evaluation.write_rows(rows, out)
    n_frames = len({(r["sequence"], r["frame_id"]) for r in rows})
    print(evaluation.format_summary(evaluation.summarize(rows), n_frames))
    print(f"skipped {len(skipped)}: {[(f.sequence, f.frame_id, why) for f, why in skipped][:5]}")
    print(f"per-frame results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
