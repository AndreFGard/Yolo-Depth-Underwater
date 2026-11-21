#!/usr/bin/env python3
"""Build a FLSea frame manifest: N frames evenly spaced in time per sequence.

You do NOT need this to reproduce the reported numbers: use the committed
``configs/flsea_eval_frames.csv``. This script documents how it was made.

It lists the whole Kaggle dataset (a few minutes; needs Kaggle credentials) and
caches the listing in ``flsea_files.json``.

Usage:
    uv run --extra flsea python scripts/make_flsea_manifest.py --n-per-seq 20 \\
        --out configs/flsea_eval_frames.csv
"""

import argparse
import logging
from pathlib import Path

from yolo_depth_underwater import flsea


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-per-seq", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("configs/flsea_eval_frames.csv"))
    parser.add_argument("--cache", type=Path, default=Path("flsea_files.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    names = flsea.list_remote_files(args.cache)
    frames = [fr for seq in flsea.SEQUENCES for fr in flsea.pick_frames(names, seq, args.n_per_seq)]
    flsea.write_manifest(frames, args.out)
    print(f"wrote {len(frames)} frames to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
