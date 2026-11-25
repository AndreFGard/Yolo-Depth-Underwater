#!/usr/bin/env python3
"""Build the compact table comparing the YOLO fine-tuned checkpoints on FLSea.

Transposed (metrics as rows, models as columns) so that an extra alignment adds two
rows instead of two columns.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "report" / "tables" / "finetuned_models_on_flsea.csv"
BASELINE = "yolo_baseline_evaluation_flsea.csv"
CHECKPOINT = "best"
RUNS = (
    ("YOLO-DA", "flsea_ladder_DA_LARGE.csv", False),
    ("YOLO-USOD", "flsea_ladder_usodk10_frozen_80.csv", False),
    ("YOLO-UniDepth", "flsea_ladder_unidepth-headneck-80.csv", True),
)
ALIGNMENTS = ("raw", "ssi", "median")
ALIGNMENT_LABELS = {"raw": "raw", "ssi": "SSI", "median": "Mediana"}
METRICS = (("AbsRel", "AbsRel↓"), ("d1", "δ₁↑"))
MISSING = "—"
Row = dict[str, str]
FrameKey = tuple[str, str]


def _load(path: Path) -> list[Row]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: no evaluation rows")
    return rows


def _select(
    rows: list[Row], alignment: str, checkpoint: str | None = None
) -> list[Row]:
    return [
        row
        for row in rows
        if row["alignment"] == alignment
        and (checkpoint is None or row["checkpoint"] == checkpoint)
    ]


def _frame_keys(rows: list[Row]) -> set[FrameKey]:
    return {(row["sequence"], row["frame_id"]) for row in rows}


def _mean(rows: list[Row], metric: str) -> str:
    return f"{sum(float(row[metric]) for row in rows) / len(rows):.2f}"


def _metrics(rows: list[Row]) -> dict[str, str]:
    return {metric: _mean(rows, metric) for metric, _ in METRICS}


def _validate_frames(reference: list[Row], candidate: list[Row], name: str) -> None:
    if _frame_keys(reference) != _frame_keys(candidate):
        raise ValueError(f"{name}: evaluated frames differ from the YOLO baseline")


def build() -> None:
    baseline = _load(ARTIFACTS / BASELINE)
    values: dict[str, dict[tuple[str, str], str]] = {
        "Baseline": {
            (alignment, metric): value
            for alignment in ALIGNMENTS
            for metric, value in _metrics(_select(baseline, alignment)).items()
        }
    }

    for name, filename, has_metric_scale in RUNS:
        rows = _load(ARTIFACTS / filename)
        values[name] = {}
        for alignment in ALIGNMENTS:
            if alignment == "raw" and not has_metric_scale:
                values[name].update(
                    {(alignment, metric): MISSING for metric, _ in METRICS}
                )
                continue
            selected = _select(rows, alignment, CHECKPOINT)
            _validate_frames(_select(baseline, alignment), selected, name)
            values[name].update(
                {
                    (alignment, metric): value
                    for metric, value in _metrics(selected).items()
                }
            )

    names = ["Baseline", *(name for name, _, _ in RUNS)]
    output: list[tuple[str, ...]] = [("Métrica", *names)]
    for alignment in ALIGNMENTS:
        for metric, label in METRICS:
            output.append(
                (
                    f"{ALIGNMENT_LABELS[alignment]} {label}",
                    *(values[name][(alignment, metric)] for name in names),
                )
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(output)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


def main() -> None:
    build()


if __name__ == "__main__":
    main()
