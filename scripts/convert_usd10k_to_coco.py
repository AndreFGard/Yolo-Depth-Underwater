#!/usr/bin/env python3
"""
- Ignores the saliency
- annotations.json  has an empty annotations list, it only has depth files for each image.

## Disparity
USOD10k depth maps are DPT pseudo-depth, stored as per-image min-max normalized disparity, and they are inverted relative to the COCO convention. A higher disparity is closer.

This converter remaps to relative depth using this formula, for a smoother inversion:

    depth_rel = eps / (d + eps),   d = raw / 65535

Where `eps` is the assumed far-end disparity as a fraction of the full range. meaning that 0 (largest distance) maps to 1 (max distance), while 30k maps to ~0.5.
Default eps = 0.25. Depth ratios are preserved and the Output range is [round(65535·eps/(1+eps)), 65535]; 0 is reserved. Max representable far/near ratio ~= (1+eps)/eps ≈ 5.
"""

import argparse
import json
import logging
import os
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

LOGGER = logging.getLogger("convert_usd10k")

SPLIT_MAP = {"TR": "train", "VAL": "val", "TE": "test"}
DEFAULT_EPS = 0.25
DEPTH_SCALE = 1000


@dataclass
class COCOImage:
    file_name: str
    width: int
    height: int
    split: str
    depth_file_name: str
    id: int = 0


def disparity_to_relative_depth(disp: np.ndarray, eps: float) -> np.ndarray:
    """
    USOD10k DPT disparity (uint16, higher = closer) -> relative depth (uint16,
    higher = farther).  depth_rel = eps / (d + eps), d = disp / 65535.
    """
    d = disp.astype(np.float64) / 65535.0
    return np.rint(eps / (d + eps) * 65535.0).astype(np.uint16)


def depth_value_range(eps: float) -> tuple[int, int]:
    lo = round(65535 * eps / (1 + eps))
    return lo, 65535


def _convert_disparity_to_depth(args: tuple[str, str, float]) -> None:
    # tuple param necessary due to being a job
    src_str, dst_str, eps = args
    src = Path(src_str)
    dst = Path(dst_str)
    dst.parent.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(Image.open(src), dtype=np.uint16)
    converted = disparity_to_relative_depth(raw, eps)
    Image.fromarray(converted).save(dst, format="PNG")


def hardlink_images(usod10k_rgb: Path, dest_img: Path) -> None:
    """Hardlink every rgb PNG from one USOD split into the coco image dir."""
    dest_img.mkdir(parents=True, exist_ok=True)
    for f in sorted(usod10k_rgb.iterdir()):
        if f.suffix.lower() != ".png":
            continue
        dst = dest_img / f.name
        if dst.exists():
            raise FileExistsError(f"refusing to overwrite: {dst}")
        try:
            os.link(str(f), str(dst))
        except OSError as exc:
            raise RuntimeError(
                f"hardlink failed for {f} -> {dst}: {exc}"
            ) from exc


def convert_disparity_maps(usod10k_depth: Path, dest_depth: Path, eps: float) -> None:
    """Read disparity PNGs from one split, apply the eps remap, write as uint16 PNG."""
    dest_depth.mkdir(parents=True, exist_ok=True)
    jobs = []
    for f in sorted(usod10k_depth.iterdir()):
        if f.suffix.lower() != ".png":
            continue
        dst = dest_depth / f.name
        if dst.exists():
            raise FileExistsError(f"refusing to overwrite: {dst}")
        jobs.append((str(f), str(dst), eps))

    assert jobs, "no disparity maps found to convert"
    LOGGER.info("depth: %d maps to convert", len(jobs))
    workers = min(os.cpu_count() or 1, 8)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for _ in tqdm(
            ex.map(_convert_disparity_to_depth, jobs, chunksize=32),
            total=len(jobs), unit="map", desc="disparity",
        ):
            pass


def collect_split_images(dest_img: Path, split_name: str, id_seq: Iterator[int]) -> list[COCOImage]:
    """Collect COCO image entries for one split, assigning sequential ids."""
    images: list[COCOImage] = []
    if not dest_img.is_dir():
        return images
    for f in sorted(dest_img.iterdir()):
        if f.suffix.lower() != ".png":
            continue
        with Image.open(f) as im:
            w, h = im.size
        images.append(
            COCOImage(
                file_name=f"images/{split_name}/{f.name}",
                width=w,
                height=h,
                split=split_name,
                depth_file_name=f"depth/{split_name}/{f.name}",
                id=next(id_seq),
            )
        )
    return images


def write_annotations(ann_path: Path, images: list[COCOImage], eps: float) -> None:
    """Write annotations/annotations.json (COCO-depth manifest)."""
    lo, hi = depth_value_range(eps)
    manifest = {
        "info": {
            "description": "USOD10k RGB + relative depth (DPT pseudo-depth, disparity corrected)",
            "version": "1.0",
            "year": 2021,
            "source": {
                "name": "USOD10k",
                "url": "https://github.com/DCCover/USOD10k",
                "license": "free for academic research",
            },
            "depth": {
                "convention": "relative depth, higher = farther",
                "encoding": "uint16 PNG (mode I;16)",
                "depth_scale": DEPTH_SCALE,
                "unit": "arbitrary relative units; NOT meters",
                "conversion": f"depth_rel = eps/(d + eps), d = raw/65535, eps={eps}",
                "value_range": [lo, hi],
                "max_far_near_ratio": round((1 + eps) / eps, 2),
                "metric": False,
            },
        },
        "licenses": [],
        "categories": [],
        "images": [asdict(img) for img in images],
        "annotations": [],
    }

    ann_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ann_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    LOGGER.info("wrote COCO manifest: %s (%d images)", ann_path, len(images))


def write_coco_yaml(yaml_path: Path, dataset_root: Path, sizes: dict[str, int]) -> None:
    """Write data.yaml for ultralytics depth training."""
    content = f"""
path: {dataset_root}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: depth

depth_scale: {DEPTH_SCALE}

source:
  name: USOD10k
  url: https://github.com/DCCover/USOD10k
  gt_nature: DPT monocular-network pseudo-depth (per-image min-max normalized
    disparity), not sensor-captured. Cross-image comparisons are meaningless.
  splits: {{train: {sizes.get('train', 0)}, val: {sizes.get('val', 0)}, test: {sizes.get('test', 0)}}}
"""
    yaml_path.write_text(content)
    LOGGER.info("wrote data.yaml: %s", yaml_path)


def run_pipeline(usod10k_root: Path, output_root: Path, eps: float, yaml_path: Path) -> None:
    """Orchestrate the full USOD10k -> COCO-depth conversion."""
    all_images: list[COCOImage] = []
    sizes: dict[str, int] = {}
    id_seq = count(1)

    for split_key, split_name in SPLIT_MAP.items():
        hardlink_images(usod10k_root / split_key / "RGB", output_root / "images" / split_name)
        convert_disparity_maps(usod10k_root / split_key / "depth", output_root / "depth" / split_name, eps)
        split_images = collect_split_images(output_root / "images" / split_name, split_name, id_seq)
        all_images.extend(split_images)
        sizes[split_name] = len(split_images)

    write_annotations(output_root / "annotations" / "annotations.json", all_images, eps)
    write_coco_yaml(yaml_path, output_root, sizes)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--usod10k-root", type=Path, required=True,
                    help="Root of extracted USOD10k")
    p.add_argument("--output-root", type=Path, required=True,
                    help="Destination for the COCO-depth dataset")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = parse_args(argv)
    eps = DEFAULT_EPS
    yaml_path = cfg.output_root / "data.yaml"

    LOGGER.info("USOD10k -> COCO-depth: %s -> %s with eps=%.2f",
                cfg.usod10k_root, cfg.output_root, eps)

    run_pipeline(cfg.usod10k_root, cfg.output_root, eps, yaml_path)

    LOGGER.info("finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
