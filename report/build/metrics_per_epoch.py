from matplotlib.pyplot import ylim
import pandas as pd
import plotly.express as px

px.defaults.height = 600  # type: ignore
px.defaults.width = 800  # type: ignore

LADDERS = {
    "YOLO-DA": "artifacts/flsea_ladder_DA_LARGE.csv",
    "YOLO-UniDepth": "artifacts/flsea_ladder_unidepth-headneck-80.csv",
    "YOLO-USOD": "artifacts/flsea_ladder_usodk10_frozen_80.csv",
}


def _load_ladders(paths: dict[str, str]) -> pd.DataFrame:
    dfs = []
    for model, path in paths.items():
        tdf = pd.read_csv(path, usecols=["checkpoint", "epoch", "alignment", "d1"])
        tdf["model"] = model
        dfs.append(tdf)
    return pd.concat(dfs)


if __name__ == "__main__":
    ladders = _load_ladders(LADDERS)

    for alignment in ("raw", "ssi"):
        rows = ladders[ladders["alignment"] == alignment]
        per_epoch = (
            rows[rows["epoch"] >= 0]
            .groupby(["epoch", "model"])["d1"]
            .mean()
            .reset_index()
        )
        fig = px.line(
            per_epoch,
            x="epoch",
            y="d1",
            color="model",
            title=f"D1 no FLSea a cada época de treinamento ({alignment})",
        ).update_traces(textfont_size=16)

        fig.add_hline(
            y=rows[rows["checkpoint"] == "baseline"]["d1"].mean(),
            annotation_text="YOLO Baseline",
        )
        if alignment == "raw":
            for i, trace in enumerate(fig.data):
                best = rows[
                    (rows["model"] == trace.name) & (rows["checkpoint"] == "best")
                ]
                fig.add_hline(
                    y=best["d1"].mean(),
                    line_dash="dot",
                    line_color=trace.line.color,
                    annotation_text=f"{trace.name} best.pt",
                    annotation_position="top left" if i % 2 == 0 else "bottom left",
                )
        fig.write_image(f"report/figures/D1_per_epoch_{alignment}.png")
