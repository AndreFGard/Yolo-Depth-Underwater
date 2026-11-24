#!/usr/bin/env python3
"""Build the compact zero-shot model comparison table for the FLSea report."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "report" / "tables" / "models_on_flsea.csv"

BASELINE = ("YOLO26n-Depth (zero-shot)", "yolo_baseline_evaluation_flsea.csv")
MODELS = (
    ("UniDepth V2 (ViT-L)", "unidepth_evaluation_flsea.csv", True),
    ("Depth Anything V2 (ViT-L)", "depth_anything_v2large_evaluation_flsea.csv", False),
    ("UDepth-RMI", "udepth_evaluation_flsea.csv", False),
)
ALIGNMENTS = ("raw", "ssi")
COLUMNS = ("Model", "raw AbsRel↓", "raw δ₁↑", "SSI AbsRel↓", "SSI δ₁↑")
Row = dict[str, str]
FrameKey = tuple[str, str]


def _load(path: Path) -> list[Row]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: no evaluation rows")
    return rows


def _aligned(rows: list[Row], alignment: str) -> list[Row]:
    return [row for row in rows if row["alignment"] == alignment]


def _frame_keys(rows: list[Row]) -> set[FrameKey]:
    return {(row["sequence"], row["frame_id"]) for row in rows}


def _mean(rows: list[Row], metric: str) -> str:
    return f"{sum(float(row[metric]) for row in rows) / len(rows):.2f}"


def _validate_frames(reference: list[Row], candidate: list[Row], name: str) -> None:
    if _frame_keys(reference) != _frame_keys(candidate):
        raise ValueError(f"{name}: evaluated frames differ from the YOLO baseline")


def _metrics(rows: list[Row]) -> tuple[str, str]:
    return _mean(rows, "AbsRel"), _mean(rows, "d1")


def build() -> None:
    baseline_name, baseline_path = BASELINE
    baseline_rows = _load(ARTIFACTS / baseline_path)
    baseline_by_alignment = {
        alignment: _aligned(baseline_rows, alignment) for alignment in ALIGNMENTS
    }
    for alignment, rows in baseline_by_alignment.items():
        if not rows:
            raise ValueError(f"{baseline_path}: missing {alignment!r} rows")

    output = [COLUMNS]
    entries = ((baseline_name, None, None),) + tuple(
        (name, path, has_metric_scale) for name, path, has_metric_scale in MODELS
    )
    for name, path, has_metric_scale in entries:
        rows = baseline_rows if path is None else _load(ARTIFACTS / path)
        raw_rows = _aligned(rows, "raw")
        ssi_rows = _aligned(rows, "ssi")
        _validate_frames(baseline_by_alignment["ssi"], ssi_rows, name)

        if has_metric_scale or path is None:
            _validate_frames(baseline_by_alignment["raw"], raw_rows, name)
            raw_abs_rel, raw_d1 = _metrics(raw_rows)
        else:
            raw_abs_rel = raw_d1 = "—"

        ssi_abs_rel, ssi_d1 = _metrics(ssi_rows)
        output.append((name, raw_abs_rel, raw_d1, ssi_abs_rel, ssi_d1))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(output)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


def main() -> None:
    build()


if __name__ == "__main__":
    main()
