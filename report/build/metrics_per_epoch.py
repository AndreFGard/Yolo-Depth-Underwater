#!/usr/bin/env python3

from __future__ import annotations

import pandas as pd
import plotly.express as px

px.defaults.height = 600  # type: ignore
px.defaults.width = 800  # type: ignore

LADDERS = {
    "YOLO-DA": "artifacts/flsea_ladder_DA_LARGE.csv",
    "YOLO-UniDepth": "artifacts/flsea_ladder_unidepth-headneck-80.csv",
    "YOLO-USOD": "artifacts/flsea_ladder_usodk10_frozen_80.csv",
}

FLSEA_LADDERS = {
    "YOLO-F-GT": "artifacts/flsea_ladder_YOLO-GT-FLSea.csv",
    "YOLO-F-UniDepth": "artifacts/flsea_ladder_YOLO-UniDepth-FLSea.csv",
    "YOLO-F-DA": "artifacts/flsea_ladder_YOLO-DA-FLSea.csv",
}


def _load_ladders(paths: dict[str, str]) -> pd.DataFrame:
    dfs = []
    for model, path in paths.items():
        tdf = pd.read_csv(path, usecols=["checkpoint", "epoch", "alignment", "d1"])
        tdf["model"] = model
        dfs.append(tdf)
    return pd.concat(dfs)


def _plot(
    paths: dict[str, str],
    alignment: str,
    output: str,
    title: str,
    *,
    mark_best: bool = False,
) -> None:
    ladders = _load_ladders(paths)
    rows = ladders[ladders["alignment"] == alignment]
    per_epoch = (
        rows[rows["epoch"] >= 0].groupby(["epoch", "model"])["d1"].mean().reset_index()
    )
    fig = px.line(
        per_epoch, x="epoch", y="d1", color="model", title=title
    ).update_traces(textfont_size=16)

    fig.add_hline(
        y=rows[rows["checkpoint"] == "baseline"]["d1"].mean(),
        annotation_text="YOLO Baseline",
    )
    if mark_best:
        for i, trace in enumerate(fig.data):
            best = rows[(rows["model"] == trace.name) & (rows["checkpoint"] == "best")]
            fig.add_hline(
                y=best["d1"].mean(),
                line_dash="dot",
                line_color=trace.line.color,
                annotation_text=f"{trace.name} best.pt",
                annotation_position="top left" if i % 2 == 0 else "bottom left",
            )
    fig.write_image(output)


def main() -> None:
    for alignment in ("raw", "ssi"):
        _plot(
            LADDERS,
            alignment,
            f"report/figures/D1_per_epoch_{alignment}.png",
            f"D1 no FLSea a cada época de treinamento ({alignment})",
            mark_best=alignment == "raw",
        )
    _plot(
        FLSEA_LADDERS,
        "median",
        "report/figures/D1_per_epoch_median_flsea.png",
        "D1 no FLSea a cada época (mediana) — braços treinados no próprio FLSea",
    )


if __name__ == "__main__":
    main()
