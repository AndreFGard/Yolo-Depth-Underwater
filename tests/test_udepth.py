import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from yolo_depth_underwater import evaluation, flsea, metrics
from yolo_depth_underwater import udepth_model as adapter
from yolo_depth_underwater.inference_model import OutputKind
from yolo_depth_underwater.udepth_runtime import inverse_to_relative

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def model_factory(monkeypatch, tmp_path):
    class Network:
        protocol = {"weights_sha256": "fixture", "torch": "fixture", "device": "cpu"}
        device = "cpu"
        calls = 0
        inputs = []

        def predict(self, rgb, output_hw):
            self.calls += 1
            self.inputs.append(rgb.copy())
            raw = np.linspace(0.001, 0.002, np.prod(output_hw), dtype=np.float32)
            raw = raw.reshape(output_hw)
            return inverse_to_relative(raw)[0], raw, {}

    network = Network()
    monkeypatch.setattr(adapter, "prepare_assets", lambda path: tmp_path)
    monkeypatch.setattr(adapter, "UDepthRelative", lambda *a, **kw: network)

    def create():
        return adapter.UDepthRMIModel(assets=tmp_path, output=tmp_path / "cache")

    return create, network


def test_bgr_input_and_verified_cache(model_factory):
    create, network = model_factory
    model = create()
    bgr = np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8)
    first = next(model.predict(iter([bgr])))
    resumed = create()
    np.testing.assert_array_equal(next(resumed.predict(iter([bgr]))), first)
    assert model.output_kind is OutputKind.DISPARITY
    assert network.calls == 1
    assert resumed.n_cache_hits == 1
    np.testing.assert_array_equal(network.inputs[0][0, 0], [30, 20, 10])
    record = next(iter(model.records.values()))
    relative = Path(record["relative"])
    np.testing.assert_array_equal(np.load(relative), 1 / first)
    relative.write_bytes(b"corrupted")
    next(resumed.predict(iter([bgr])))
    assert network.calls == 2
    np.testing.assert_array_equal(np.load(relative), 1 / first)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1])
def test_inverse_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        inverse_to_relative(np.array([[bad]], dtype=np.float32))


def test_relative_depth_is_unquantized_and_farther_is_larger():
    inverse = np.array([[0.0, 0.001, 0.002]], dtype=np.float32)
    relative, clamped = inverse_to_relative(inverse)
    assert relative.dtype == np.float32
    assert clamped == 1
    assert relative[0, 0] > relative[0, 1] > relative[0, 2] > 65.535


def test_cli_separates_relative_npy_from_gt_aligned_png(
    model_factory, monkeypatch, tmp_path
):
    create, network = model_factory
    model = create()
    frames = [flsea.FLSeaFrame("canyons/flatiron", str(i)) for i in range(3)]
    bgr = np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8)
    inverse = np.linspace(0.001, 0.002, 6, dtype=np.float32).reshape(2, 3)
    gt = (1 / (100 * inverse)).astype(np.float32)

    def load(frame):
        reference = np.full_like(gt, np.nan) if frame.frame_id == "2" else gt
        return bgr, reference

    monkeypatch.setattr(flsea, "read_manifest", lambda *a: frames)
    monkeypatch.setattr(flsea, "attach_local_root", lambda frames, root: frames)
    monkeypatch.setattr(flsea, "load_frame", load)
    monkeypatch.setattr(evaluation, "load_frame", load)
    monkeypatch.setattr(adapter, "UDepthRMIModel", lambda device=None: model)
    spec = importlib.util.spec_from_file_location(
        "evaluate_flsea_cli", ROOT / "scripts/evaluate_flsea.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    out, predictions = tmp_path / "metrics.csv", tmp_path / "predictions"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(ROOT / "scripts/evaluate_flsea.py"),
            "--model",
            "udepth",
            "--local-root",
            str(tmp_path),
            "--device",
            "cpu",
            "--out",
            str(out),
            "--save-predictions",
            str(predictions),
        ],
    )
    assert cli.main() == 0
    with out.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert {r["alignment"] for r in rows} == {"ssi", "median", "constant", "global-ssi"}
    assert {r["frame_id"] for r in rows} == {"0", "1"}
    assert network.calls == 1
    assert len(list((predictions / "relativas").rglob("*.npy"))) == 2
    relative = np.load(predictions / "relativas/canyons/flatiron/0.npy")
    np.testing.assert_array_equal(relative, 1 / inverse)
    assert relative.dtype == np.float32
    assert relative.min() > 65.535
    from PIL import Image

    aligned = np.asarray(Image.open(predictions / "alinhadas_ssi/0.png")) / 1000
    np.testing.assert_allclose(aligned, gt, atol=0.00051)
    metadata = json.loads((predictions / "formato_predicoes.json").read_text())
    assert metadata["relative_predictions"]["gt_used"] is False
    assert metadata["relative_predictions"]["n_maps"] == 2
    assert metadata["aligned_auxiliary"]["gt_used"] is True
    assert not (predictions / "relativas/canyons/flatiron/2.npy").exists()


def test_results_match_group_frames_masks_and_constant_control():
    def read(path):
        with path.open() as stream:
            return list(csv.DictReader(stream))

    rows = read(ROOT / "artifacts/udepth_evaluation_flsea.csv")
    key = lambda r: (r["sequence"], r["frame_id"])
    expected = {key(r) for r in rows if r["alignment"] == "ssi"}
    manifest = flsea.read_manifest(
        ROOT / "configs/flsea_eval_frames.csv", flsea.SUBSETS[flsea.Subset.CAI_METZLER]
    )
    assert len(manifest) == 120
    assert len(expected) == 119
    assert len(rows) == 476
    excluded = json.loads(
        (ROOT / "artifacts/udepth_flsea_excluded_frames.json").read_text()
    )
    assert {(f.sequence, f.frame_id) for f in manifest} - expected == {
        key(r) for r in excluded
    }
    assert {r["alignment"] for r in rows} == {"ssi", "median", "constant", "global-ssi"}
    own = {key(r): r for r in rows if r["alignment"] == "constant"}
    for name in ["yolo_baseline", "yolo_unfrozen_80_unidepth", "unidepth"]:
        peer = read(ROOT / f"artifacts/{name}_evaluation_flsea.csv")
        assert {key(r) for r in peer if r["alignment"] == "ssi"} == expected
        for row in peer:
            if row["alignment"] == "constant":
                for field in ["valid_frac", *metrics.METRIC_NAMES]:
                    assert float(row[field]) == pytest.approx(
                        float(own[key(row)][field]), abs=2e-6
                    )
