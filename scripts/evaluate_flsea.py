#!/usr/bin/env python3
"""Evaluate a depth model on the FLSea evaluation frames.

Every model is scored on the same frames (``configs/flsea_eval_frames.csv``) with the
same alignment and metrics (see ``yolo_depth_underwater.metrics``). The default subset
is the six FLSea sequences of arXiv:2507.02148 (120 frames); use ``--subset all`` to
score all 12 sequences (240 frames).

Examples:
    uv run --extra cu --extra flsea --extra depth-anything \\
        python scripts/evaluate_flsea.py --model depth-anything

    # all 12 sequences (default is the six sequences of arXiv:2507.02148)
    uv run --extra cu --extra flsea python scripts/evaluate_flsea.py \\
        --model yolo --variant path/to/yolo.pt --subset all

    # dataset already on disk (e.g. a Kaggle notebook)
    ... --local-root /kaggle/input/flsea-vi

    # also save one scaled uint16 depth PNG per scored frame
    ... --save-predictions results/flsea_yolo_n_depth

Kaggle credentials (KAGGLE_USERNAME + KAGGLE_KEY, or KAGGLE_API_TOKEN) are needed to
download frames. See docs/flsea_eval.md.
"""

import argparse
import logging
from pathlib import Path

from yolo_depth_underwater import evaluation, flsea
from yolo_depth_underwater.inference_model import InferenceModel

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "configs" / "flsea_eval_frames.csv"
DEFAULT_SUBSET = flsea.Subset.CAI_METZLER


def build_model(name: str, variant: str | None, device: str | None) -> InferenceModel:
    """Models are imported lazily so each one only needs its own extra."""
    if name == "depth-anything":
        from yolo_depth_underwater.depth_anything_model import DepthAnythingV2Model

        return DepthAnythingV2Model(variant=variant or "small", device=device)
    if name == "yolo":
        from yolo_depth_underwater.yolo_model import YoloModel

        # No USOD10K DepthDataset is needed to call ``predict``.
        return YoloModel(None, variant or "yolo26n-depth.pt", device)
    if name == "unidepth":
        from yolo_depth_underwater.unidepth_model import UniDepthV2Model

        return UniDepthV2Model(None, device)
    raise ValueError(f"unknown model {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["depth-anything", "yolo", "unidepth"])
    parser.add_argument("--variant", help="depth-anything: small|base|large; yolo: path/to/yolo.pt")
    parser.add_argument("--device", help="e.g. cuda, cpu (default: cuda if available)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--subset", type=flsea.Subset, choices=list(flsea.Subset), default=DEFAULT_SUBSET
    )
    parser.add_argument("--local-root", type=Path, help="use a local copy of the dataset instead of downloading")
    parser.add_argument("--min-valid-frac", type=float, default=evaluation.DEFAULT_MIN_VALID_FRAC)
    parser.add_argument("--include-raw", action="store_true",
                        help="also score depth-output models without alignment (only meaningful for metric models)")
    parser.add_argument("--workers", type=int, default=2, help="parallel downloads (keep low: Kaggle rate-limits)")
    parser.add_argument("--out", type=Path, help="per-frame CSV (default: results/flsea_<model>[_<variant>].csv)")
    parser.add_argument(
        "--save-predictions", type=Path, metavar="DIR",
        help="save <frame_id>.png for each scored frame in DIR (uint16 depth x1000); "
             "depth models: raw metres, disparity models: oracle ssi-aligned metres",
    )
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
        model, frames, min_valid_frac=args.min_valid_frac, include_raw=args.include_raw,
        predictions_dir=args.save_predictions,
    )

    out = args.out or Path("results") / (
        f"flsea_{args.model}" + (f"_{args.variant}" if args.variant else "") + ".csv"
    )
    evaluation.write_rows(rows, out)
    n_frames = len({(r["sequence"], r["frame_id"]) for r in rows})
    print(evaluation.format_summary(evaluation.summarize(rows), n_frames))
    print(f"skipped {len(skipped)}: {[(f.sequence, f.frame_id, why) for f, why in skipped][:5]}")
    print(f"per-frame results: {out}")
    if args.save_predictions:
        print(f"prediction PNGs: {args.save_predictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
