#!/usr/bin/env python3
"""Build a COCO Depth training dataset from FLSea-VI (zip, extracted tree or Kaggle).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from yolo_depth_underwater import flsea
from yolo_depth_underwater.encoding import DEPTH_SCALE, MAX_ENCODABLE, save_depth_png

LOGGER = logging.getLogger("build_flsea")

MANIFEST = Path("configs/flsea_eval_frames.csv")
LISTING = Path("flsea_files.json")
SPLITS = ("train", "val", "test")
JPEG_QUALITY = 95


@dataclass
class Row:
    """One converted frame, as written to ``audit.csv``."""

    split: str
    sequence: str
    frame_id: str
    valid_frac: float
    depth_min: float
    depth_max: float
    clipped_px: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--zip", type=Path, help="FLSea-VI .zip, read without extracting")
    source.add_argument("--root", type=Path, help="extracted FLSea-VI root directory")
    parser.add_argument("--out", type=Path, required=True, help="destination dataset")
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=list(flsea.TRAINING_SCENES),
        help="default: every sequence the evaluation does not score",
    )
    parser.add_argument("--test-per-scene", type=int, default=50)
    parser.add_argument("--val-per-scene", type=int, default=167)
    parser.add_argument(
        "--image-format", choices=("jpg", "png"), default="jpg", help="default: jpg"
    )
    parser.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    parser.add_argument("--depth-scale", type=float, default=DEPTH_SCALE)
    parser.add_argument(
        "--min-valid-frac",
        type=float,
        default=0.0,
        help="drop frames below this reference coverage (0 keeps everything, as MAU/WaterMono do)",
    )
    parser.add_argument("--limit", type=int, help="only the first N frames per split")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the split and stop"
    )
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"{args.out} exists and is not empty")
    return args


def remote_names(args: argparse.Namespace) -> list[str]:
    """Remote listing of the dataset, from whichever source was given."""
    if args.zip is not None:
        with flsea.ArchiveFrames.open(args.zip) as archive:
            counts = archive.sequences()
            LOGGER.info("archive %s: %d sequences", args.zip, len(counts))
            for sequence, (images, depth) in counts.items():
                LOGGER.info("  %-28s %6d images %6d depth", sequence, images, depth)
            return archive.names()
    if args.root is not None:
        names: list[str] = []
        for sequence in flsea.SEQUENCES:
            for kind in ("imgs", "depth"):
                names += [
                    f"{sequence}/{kind}/{path.name}"
                    for path in (args.root / sequence / kind).glob("*.tif*")
                ]
        if not names:
            raise SystemExit(f"no <sequence>/imgs|depth/*.tif* under {args.root}")
        return names
    if LISTING.is_file():
        LOGGER.info("using cached listing %s", LISTING)
        return json.loads(LISTING.read_text())
    return flsea.list_remote_files()


def present_scenes(
    names: list[str], requested: tuple[str, ...], strict: bool
) -> tuple[str, ...]:
    """Keep the scenes the archive holds pairs for; ``strict`` rejects a partial archive
    (used when ``--scenes`` was spelled out, so a typo cannot shrink the dataset silently)."""
    present = tuple(s for s in requested if flsea.scene_frame_ids(names, s))
    missing = [s for s in requested if s not in present]
    if missing:
        message = "no image/depth pairs for %s in this archive; using %s" % (
            missing,
            list(present),
        )
        if strict:
            raise SystemExit(message)
        LOGGER.warning(message)
    return present


class Source:
    """Loads ``(image, depth)`` floats from the zip, a tree, or Kaggle."""

    def __init__(self, args: argparse.Namespace):
        self._archive = (
            flsea.ArchiveFrames.open(args.zip) if args.zip is not None else None
        )

    def load(self, frame: flsea.FLSeaFrame) -> tuple[np.ndarray, np.ndarray]:
        if self._archive is not None:
            return self._archive.load_frame(frame)
        downloaded = flsea.download_frames([frame]) if frame.image_path is None else [frame]
        if not downloaded:
            raise RuntimeError(f"could not fetch {frame.sequence}/{frame.frame_id}")
        return flsea.load_frame(downloaded[0])

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()


def write_image(path: Path, image_bgr: np.ndarray, fmt: str, quality: int) -> None:
    """Write a BGR uint8 image as JPEG (quality) or lossless PNG."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = Image.fromarray(np.ascontiguousarray(image_bgr[..., ::-1]))
    if fmt == "jpg":
        rgb.save(path, format="JPEG", quality=quality)
    else:
        rgb.save(path, format="PNG")


def convert(
    frame: flsea.FLSeaFrame,
    split: str,
    source: Source,
    args: argparse.Namespace,
) -> Row:
    """Convert one frame into ``images/<split>`` + ``depth/<split>`` and audit it."""
    image, depth = source.load(frame)
    valid = np.isfinite(depth) & (depth > 0)
    clipped = int(np.count_nonzero(valid & (depth > MAX_ENCODABLE * args.depth_scale / DEPTH_SCALE)))
    stem = frame.frame_id
    write_image(
        args.out / "images" / split / f"{stem}.{args.image_format}",
        image,
        args.image_format,
        args.jpeg_quality,
    )
    save_depth_png(args.out / "depth" / split / f"{stem}.png", depth, args.depth_scale)
    return Row(
        split=split,
        sequence=frame.sequence,
        frame_id=stem,
        valid_frac=float(valid.mean()),
        depth_min=float(np.nanmin(depth[valid])) if valid.any() else 0.0,
        depth_max=float(np.nanmax(depth[valid])) if valid.any() else 0.0,
        clipped_px=clipped,
    )


def write_metadata(
    args: argparse.Namespace, split: dict[str, list[flsea.FLSeaFrame]], rows: list[Row]
) -> None:
    """Write data.yaml, annotations.json and README.md."""
    sizes = {name: len(frames) for name, frames in split.items()}
    yaml = [
        f"# COCO Depth dataset: {args.out.name}",
        "# Images and labels from FLSea-VI, built by scripts/build_flsea_coco.py.",
        "",
        f"path: {args.out.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "nc: 1",
        "names:",
        "  0: depth",
        "",
        f"depth_scale: {args.depth_scale:g}",
        "",
        "source:",
        "  name: FLSea-VI",
        "  url: https://www.kaggle.com/datasets/viseaonlab/flsea-vi",
        "  gt_nature: 'SfM (Agisoft Metashape) metric depth, scaled to scene scale with"
        " in-scene objects of known size; float32 metres encoded as uint16 with"
        " depth_scale, 0 = invalid'",
        f"  splits: {{train: {sizes['train']}, val: {sizes['val']}, test: {sizes['test']}}}",
        f"  scenes: {json.dumps(sorted(args.scenes))}",
        "  split_rule: 'per scene, chronological: first test_per_scene frames -> test,"
        " next val_per_scene -> val, remainder -> train'",
    ]
    (args.out / "data.yaml").write_text("\n".join(yaml) + "\n")

    images = [
        {
            "file_name": f"images/{row.split}/{row.frame_id}.{args.image_format}",
            "depth_file_name": f"depth/{row.split}/{row.frame_id}.png",
            "split": row.split,
            "id": index,
        }
        for index, row in enumerate(rows)
    ]
    annotations = {
        "info": {
            "description": f"FLSea-VI RGB + SfM depth, scenes {sorted(args.scenes)}",
            "source": {
                "name": "FLSea-VI",
                "url": "https://www.kaggle.com/datasets/viseaonlab/flsea-vi",
            },
            "depth": {
                "convention": "metric depth, higher = farther",
                "encoding": "uint16 PNG, value = round(depth_m * depth_scale)",
                "depth_scale": args.depth_scale,
                "unit": "metres",
                "invalid": 0,
                "ceiling_m": MAX_ENCODABLE * args.depth_scale / DEPTH_SCALE,
                "metric": True,
                "provenance": "SfM (Agisoft Metashape) scaled with objects of known size",
            },
        },
        "licenses": [],
        "categories": [],
        "images": images,
        "annotations": [],
    }
    (args.out / "annotations").mkdir(parents=True, exist_ok=True)
    (args.out / "annotations" / "annotations.json").write_text(
        json.dumps(annotations, indent=2) + "\n"
    )

    valid = np.array([row.valid_frac for row in rows])
    readme = [
        f"# COCO Depth dataset: {args.out.name}",
        "",
        "Built by `scripts/build_flsea_coco.py` from FLSea-VI.",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Source | {args.zip or args.root or 'Kaggle (per file)'} |",
        f"| Scenes | {', '.join(sorted(args.scenes))} |",
        f"| Split rule | first {args.test_per_scene} -> test, next {args.val_per_scene} -> val, rest -> train |",
        f"| Images | {sizes['train']} train / {sizes['val']} val / {sizes['test']} test |",
        f"| Image format | {args.image_format} (quality {args.jpeg_quality}) |",
        f"| Depth | uint16 PNG, `depth_scale: {args.depth_scale:g}`, metres, 0 = invalid |",
        f"| Reference coverage | median {np.median(valid):.2f}, p5 {np.percentile(valid, 5):.2f},"
        f" min {valid.min():.2f} |",
        f"| Frames below 2 % reference | {int((valid < 0.02).sum())} |",
        f"| Pixels above the {MAX_ENCODABLE * args.depth_scale / DEPTH_SCALE:.2f} m ceiling |"
        f" {sum(row.clipped_px for row in rows)} |",
        "",
        "## Caveats",
        "",
        "* Test/val come from the first frames of each dive, temporally adjacent to training",
        "  frames; `best.pt`'s own-val numbers are not a generalisation estimate.",
        "* The first frames of a dive hold the scale reference objects, so val/test may look",
        "  easier than train.",
        "* No frame filtering: frames with almost no valid reference are kept and counted above.",
        "* 120 frames of `configs/flsea_eval_frames.csv` fall inside these six scenes, so",
        "  `--subset all` (238 frames) is **not** zero-shot for models trained here; use the",
        "  default 119-frame Cai-Metzler subset.",
    ]
    (args.out / "README.md").write_text("\n".join(readme) + "\n")


def write_splits_csv(
    args: argparse.Namespace, split: dict[str, list[flsea.FLSeaFrame]]
) -> None:
    """Record the assignment before anything is converted."""
    with open(args.out / "splits.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "frame_id", "split", "index_in_scene"])
        for name in SPLITS:
            per_scene: dict[str, int] = {}
            for frame in split[name]:
                index = per_scene.get(frame.sequence, 0)
                per_scene[frame.sequence] = index + 1
                writer.writerow([frame.sequence, frame.frame_id, name, index])


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    names = remote_names(args)
    scenes = present_scenes(
        names,
        tuple(args.scenes),
        strict=tuple(args.scenes) != flsea.TRAINING_SCENES,
    )
    if not scenes:
        raise SystemExit("none of the requested scenes has an image/depth pair")
    args.scenes = list(scenes)
    split = flsea.training_split(
        names,
        scenes=scenes,
        test_per_scene=args.test_per_scene,
        val_per_scene=args.val_per_scene,
    )
    scored = {(row["sequence"], row["frame_id"]) for row in csv.DictReader(open(MANIFEST))}
    assigned = [(frame.sequence, frame.frame_id) for frames in split.values() for frame in frames]
    leaked = len(set(assigned) & scored)
    LOGGER.info(
        "split: %s (frames also in the eval manifest: %d)",
        {name: len(frames) for name, frames in split.items()},
        leaked,
    )
    if args.dry_run:
        for name in SPLITS:
            per_scene: dict[str, int] = {}
            for frame in split[name]:
                per_scene[frame.sequence] = per_scene.get(frame.sequence, 0) + 1
            LOGGER.info("%-5s %s", name, per_scene)
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    write_splits_csv(args, split)

    source = Source(args)
    rows: list[Row] = []
    encoding_log = logging.getLogger("yolo_depth_underwater.encoding")
    previous_level = encoding_log.level
    encoding_log.setLevel(logging.ERROR)  # per-frame clip warnings; audit.csv has them
    try:
        for name in SPLITS:
            frames = split[name]
            if args.limit is not None:
                frames = frames[: args.limit]
            base = args.out / "depth" / name
            base.mkdir(parents=True, exist_ok=True)
            for frame in tqdm(frames, desc=name, unit="frame"):
                row = convert(frame, name, source, args)
                if row.valid_frac < args.min_valid_frac:
                    for path in (
                        args.out / "images" / name / f"{row.frame_id}.{args.image_format}",
                        base / f"{row.frame_id}.png",
                    ):
                        path.unlink(missing_ok=True)
                    continue
                rows.append(row)
    finally:
        source.close()
        encoding_log.setLevel(previous_level)

    with open(args.out / "audit.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Row.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(row.__dict__ for row in rows)

    write_metadata(args, split, rows)

    valid = np.array([row.valid_frac for row in rows])
    total = sum(path.stat().st_size for path in args.out.rglob("*") if path.is_file())
    LOGGER.info(
        "wrote %d pairs (%.1f GB) | valid frac median %.2f p5 %.2f | clipped px %d | below 2%%: %d",
        len(rows),
        total / 1e9,
        float(np.median(valid)),
        float(np.percentile(valid, 5)),
        sum(row.clipped_px for row in rows),
        int((valid < 0.02).sum()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
