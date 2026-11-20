# FLSea evaluation

Scores any `InferenceModel` on the same 240 FLSea-VI frames (20 per sequence, 12 sequences,
evenly spaced in time) with the same protocol, so models can go in one table.

## Setup
```toml
# pyproject.toml, [project.optional-dependencies]
flsea = ["kagglehub", "kaggle>=1.7", "tifffile", "imagecodecs"]
depth-anything = ["transformers>=4.45"]
```
Kaggle credentials (`KAGGLE_USERNAME` + `KAGGLE_KEY`, or `KAGGLE_API_TOKEN`, or `~/.kaggle/kaggle.json`).
The dataset is 92 GB: only the 480 files of the manifest are downloaded (kagglehub caches them).

## Run
```bash
uv run --extra cu --extra flsea --extra depth-anything python scripts/evaluate_flsea.py --model depth-anything
uv run --extra cu --extra flsea python scripts/evaluate_flsea.py --model yolo --variant n
uv run --extra cu --extra flsea python scripts/evaluate_flsea.py --model unidepth
# only the 6 sequences of arXiv:2507.02148
... --subset arxiv-2507.02148
# dataset already on disk
... --local-root /kaggle/input/flsea-vi
```
Output: per-frame CSV in `results/` and a summary table (mean ± std over frames).
Keep `--workers` at 2: Kaggle answers HTTP 429 when asked too fast.

## Protocol
* Reference: SfM depth (metric, float32), valid = finite and > 0; frames with < 2% valid pixels are skipped.
* Predictions are resized to the reference resolution.
* Models that output **disparity** (higher = closer; Depth Anything V2) set `output_kind = "disparity"`;
  otherwise the output is treated as depth.
* Per-image oracle alignment in disparity space: `ssi` (scale + shift, least squares) and `median` (scale only);
  aligned disparity is clipped to >= 1/20 m and inverted. The same is applied to every model, metric ones included.
  The per-image scale fit means these scores measure **structure only**: they cannot detect wrong absolute scale
  or frame-to-frame scale drift. `--include-raw` scores metric models without alignment.
* `constant` = best flat map; a model must beat it.
* Metrics: AbsRel, RMSE, delta1/2/3 (1.25^i), SILog (lambda = 0.85), on valid pixels.
* Oracle alignment uses the reference, so numbers are an upper bound for relative models and are **not comparable**
  with the arXiv:2507.02148 tables (no alignment there; only the sequence choice is shared). For claims about
  physical meters, use `--include-raw` (possibly after calibrating on a separate split), not the aligned rows.

## Reference numbers (Depth Anything V2 Small, zero-shot, Colab T4)
| subset | alignment | AbsRel | RMSE | d1 | d2 | d3 | SILog |
|---|---|---|---|---|---|---|---|
| all (238) | ssi | 0.0674 | 0.5654 | 0.9601 | 0.9912 | 0.9961 | 10.19 |
| all (238) | median | 0.1495 | 1.1858 | 0.8481 | 0.9564 | 0.9820 | 18.33 |
| all (238) | constant | 0.3127 | 1.5550 | 0.4151 | 0.7311 | 0.8866 | 42.44 |
| arxiv subset (119) | ssi | 0.0656 | 0.6305 | 0.9647 | 0.9896 | 0.9948 | 10.34 |

2 of 240 frames are discarded (too little reference). Reproduction should match up to GPU numerics.

## Adding a model
Implement `InferenceModel.predict(images)` (BGR uint8 in, one float map per image out), set
`output_kind = "disparity"` if needed, and add it to `build_model` in `scripts/evaluate_flsea.py`.
Note: `YoloModel` loads a hardcoded `yolo26{variant}-depth.pt`; to score fine-tuned weights it needs a weights-path option.

`configs/flsea_eval_frames.csv` was made by `scripts/make_flsea_manifest.py` (not needed to reproduce).
