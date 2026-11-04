#!/usr/bin/env python3
"""Print an Ultralytics data YAML, its split sizes, and per-split depth stats (in DEPTH_SCALE units).

Usage:
    python scripts/check_dataset.py datasets/USDO10k-COCO/data.yaml

"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from yolo_depth_underwater.dataset import DepthDataset
from yolo_depth_underwater.split import Split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_yaml",
        type=Path,
        default="datasets/usod10k_new_script/data.yaml",
        help="Path to an Ultralytics data YAML",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=Split,
        choices=list(Split),
        default=[Split.VAL],
        help="Split(s) to check, e.g. --splits train val (default: val)",
    )
    args = parser.parse_args()

    data = DepthDataset(args.data_yaml)

    print(args.data_yaml.read_text().rstrip())
    print()

    rows = []
    for split in args.splits:
        if not data.data.get(split.value):
            continue
        # one (p25, p50, p75) triple per image, over that image's valid pixels.
        per_image = []
        invalid = 0
        for s in tqdm(
            iter(data.depth_samples(split)),
            total=len(data.indices(split)),
            desc=split.value,
            unit="img",
        ):
            valid = s.valid_depth_mask
            if valid.any():
                per_image.append(np.percentile(s.depth[valid], [25, 50, 75]))
            else:
                per_image.append(
                    (np.nan,) * 3
                )  # no valid pixel: excluded from the stats
                invalid += 1
        rows.append((split.value, len(per_image), invalid, np.array(per_image)))

    header = f"{'split':<8}{'files':>8}{'invalid':>9}{'p25':>16}{'p50':>16}{'p75':>16}"
    print(header)
    print("-" * len(header))
    for name, count, invalid, per_image in rows:
        cells = [
            f"{np.nanmedian(per_image[:, i]):.1f} ± {np.nanstd(per_image[:, i]):.1f}"
            for i in range(3)
        ]
        print(f"{name:<8}{count:>8}{invalid:>9}" + "".join(f"{c:>16}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
