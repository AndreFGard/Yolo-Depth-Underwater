#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

from yolo_depth_underwater.encoding import write_csv

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "artifacts" / "unidepth_evaluation_flsea.csv"
OUTPUT = ROOT / "report" / "tables" / "models_on_flsea.csv"

MODEL = "UniDepth V2 (ViT-L)"
METRICS = (
    ("AbsRel↓", "AbsRel"),
    ("RMSE↓", "RMSE"),
    ("δ₁↑", "d1"),
    ("SILog↓", "SILog"),
)
COLUMNS = ("Model", *(label for label, _ in METRICS))
ALIGNMENT = "ssi"


def load_rows() -> list[dict[str, str]]:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = {"alignment", *(field for _, field in METRICS)}
        missing = fields.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{INPUT}: missing columns: {', '.join(sorted(missing))}")
        rows = [row for row in reader if row["alignment"] == ALIGNMENT]

    if not rows:
        raise ValueError(f"{INPUT}: no {ALIGNMENT!r} rows")
    return rows


def main() -> None:
    """Write the formatted table CSV."""
    rows = load_rows()
    record = {"Model": MODEL}
    for label, field in METRICS:
        record[label] = sum(float(row[field]) for row in rows) / len(rows)

    write_csv([record], OUTPUT, columns=COLUMNS, float_places=4)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
