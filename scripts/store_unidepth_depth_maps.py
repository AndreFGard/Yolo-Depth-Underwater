#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from yolo_depth_underwater import split
from yolo_depth_underwater.dataset import DepthDataset
from yolo_depth_underwater.encoding import save_depth_png
from yolo_depth_underwater.store_predictions import _read_image
from yolo_depth_underwater.unidepth_model import UniDepthV2Model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True, help="COCO Depth yaml")
    parser.add_argument("--out", type=Path, required=True, help="output root")
    parser.add_argument("--device", help="e.g. cuda, cpu (default: cuda if available)")
    parser.add_argument(
        "--limit", type=int, help="only the first N images of each split"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="redo maps that already exist"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ds = DepthDataset(args.data)
    model = UniDepthV2Model(ds, device=args.device)

    for spl in list(split.Split):
        paths = ds.image_paths(spl)
        if args.limit is not None:
            paths = paths[: args.limit]
        base = args.out / spl.value
        base.mkdir(parents=True, exist_ok=True)
        pending = [p for p in paths if args.overwrite or not (base / p.name).exists()]
        logging.info(
            "%s: %d/%d maps to write in %s", spl.value, len(pending), len(paths), base
        )
        predictions = model.predict(_read_image(p) for p in pending)
        for path, pred in tqdm(
            zip(pending, predictions), total=len(pending), desc=spl.value
        ):
            save_depth_png(base / path.with_suffix(".png").name, pred)

    logging.info("metric depth maps: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
