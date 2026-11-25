import pandas as pd
import plotly.express as px

px.defaults.height = 600  # type: ignore
px.defaults.width = 800  # type: ignore

RESULTS = {
    "YOLO-DA": "artifacts/da_large_80_training_results.csv",
    "YOLO-UniDepth": "artifacts/unidepth_headneck_80_training_results.csv",
    "YOLO-USOD": "artifacts/usodk10_frozen_80_training_results.csv",
}


def _load_results(paths: dict[str, str]) -> pd.DataFrame:
    dfs = []
    for model, path in paths.items():
        tdf = pd.read_csv(path, usecols=["epoch", "metrics/delta1"])
        tdf["model"] = model
        dfs.append(tdf)
    return pd.concat(dfs)


if __name__ == "__main__":
    results = _load_results(RESULTS)

    fig = px.line(
        results,
        x="epoch",
        y="metrics/delta1",
        color="model",
        title="D1 no conjunto de validação (USOD) a cada época de treinamento",
    )
    for trace in fig.data:
        curve = results[results["model"] == trace.name]
        selected = curve["metrics/delta1"].idxmax()
        fig.add_scatter(
            x=[curve.loc[selected, "epoch"]],
            y=[curve.loc[selected, "metrics/delta1"]],
            mode="markers",
            marker={"color": trace.line.color, "size": 11, "symbol": "x"},
            name=f"{trace.name} best.pt",
        )
    fig.write_image("report/figures/D1_per_epoch_val.png")
