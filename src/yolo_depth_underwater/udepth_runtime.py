from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps

COMMIT = "baa8fcf7893fc4a485f115ff567a39481f9f24b6"

WEIGHTS_SHA256 = "f97343d3356feca7b8cca1a8951cea38ce1f17eeb3a797a63d98f2bfae764c89"

SOURCE_HASHES = {
    "model/__init__.py": "54da650c7287da112586c260424734abb14e2e8deaf3eb337063a64775958026",
    "model/layers.py": "8cadc145953ce511307e4b35a664aea501d9e3e94f7129366bd561784a0ce47a",
    "model/miniViT.py": "8f9d66e7dbc9e7289f29a90ad28991f038b6ac423fe92fe3627aa6cf21f98cbb",
    "model/udepth.py": "5d041ff40b8e25edb08d60934a7ef393f231f21f97663ec6260564887f0a7dbf",
}

INPUT_HW = (480, 640)

EPS = 1e-6


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)


def save_npy(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temp.replace(path)


def inverse_to_relative(raw, eps=EPS):
    """Conversão numérica fixa. Não confundir com trocar/inverter o mapa de cores."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or not raw.size:
        raise ValueError("A saída bruta precisa ser escalar [altura, largura].")
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Saída bruta contém valores negativos ou não finitos.")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps deve ser positivo.")
    depth = 1.0 / np.maximum(raw, np.float32(eps))
    return depth.astype(np.float32), int((raw <= eps).sum())


def _download(url, destination, expected_sha=None):
    destination = Path(destination)
    if destination.is_file() and (
        expected_sha is None or sha256_file(destination) == expected_sha
    ):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "UDepth-FLSea-research/1.0"}
            )
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temp.open("wb") as output,
            ):
                while block := response.read(1024 * 1024):
                    output.write(block)
            if expected_sha and sha256_file(temp) != expected_sha:
                raise ValueError(f"Hash inesperado: {destination.name}.")
            temp.replace(destination)
            return destination
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2 ** (attempt + 1), 16))


def prepare_assets(folder):
    """Baixa e verifica as fontes e os pesos oficiais do UDepth-RMI."""
    folder = Path(folder).resolve()
    base = f"https://raw.githubusercontent.com/uf-robopi/UDepth/{COMMIT}/"
    if not SOURCE_HASHES:
        raise RuntimeError("Hashes das fontes oficiais não configurados.")
    for relative, expected in SOURCE_HASHES.items():
        _download(base + relative, folder / relative, expected)
    _download(
        base + "saved_model/model_RMI.pth",
        folder / "saved_model/model_RMI.pth",
        WEIGHTS_SHA256,
    )
    return folder


class UDepthRelative:
    """Pesos congelados. Sem otimizador, backward, treino ou ajuste no FLSea."""

    def __init__(self, assets, device="auto", input_hw=INPUT_HW):
        import torch
        import torch.nn.functional as F
        import torchvision

        self.torch, self.F = torch, F
        self.assets = Path(assets).resolve()
        h, w = map(int, input_hw)
        if h % 32 or w % 32 or h < 320 or w < 320 or (h // 32) * (w // 32) > 500:
            raise ValueError(
                "Use dimensões múltiplas de 32, >=320 e até 500 patches; padrão 480x640."
            )
        self.input_hw = (h, w)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA não está disponível. Use device="cpu" ou reconecte uma GPU.'
            )
        self.device = torch.device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        for relative, expected in SOURCE_HASHES.items():
            if sha256_file(self.assets / relative) != expected:
                raise ValueError(f"Fonte oficial alterada: {relative}.")
        weight = self.assets / "saved_model/model_RMI.pth"
        if sha256_file(weight) != WEIGHTS_SHA256:
            raise ValueError("Checkpoint UDepth diferente do protocolo.")
        package = "_udepth_flsea_official"
        if package not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                package,
                self.assets / "model/__init__.py",
                submodule_search_locations=[str(self.assets / "model")],
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[package] = module
            spec.loader.exec_module(module)
        official = sys.modules[package].UDepth
        state = torch.load(weight, map_location="cpu", weights_only=True)
        mobilenet = torchvision.models.mobilenet_v2

        with patch.object(
            torchvision.models, "mobilenet_v2", lambda *a, **kw: mobilenet(weights=None)
        ):
            self.model = official(
                None,
                n_bins=int(state["conv_out.0.weight"].shape[0]),
                min_val=0.001,
                max_val=1.0,
                norm="linear",
            )
        self.model.load_state_dict(state, strict=True)
        self.model.requires_grad_(False).eval().to(self.device)
        self.protocol = {
            "model": "UDepth-RMI, saída da rede sem filtro CPD/guidedfilter",
            "commit": COMMIT,
            "weights_sha256": WEIGHTS_SHA256,
            "runtime_sha256": sha256_file(__file__),
            "input_hw": list(self.input_hw),
            "input": "R, max(G,B), luminância PIL",
            "preprocessing": "RGB para RMI antes do resize PIL BICUBIC; float32 / 255",
            "output_convention": "saída bruta tratada como inversa relativa, fixa antes do teste",
            "conversion": "depth_relative = 1 / max(inverse_raw, 1e-6)",
            "eps": EPS,
            "resize_output": "bilinear na inversa, align_corners=False, antes da conversão",
            "unit": "relative; sem calibração em metros",
            "training": False,
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "device": str(self.device),
        }

    def predict(self, rgb, output_hw=None):
        torch = self.torch
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("A entrada deve ser uma imagem RGB.")
        image = Image.fromarray(rgb)
        a = np.asarray(image)
        rmi = np.stack(
            (
                a[..., 0],
                np.maximum(a[..., 1], a[..., 2]),
                np.asarray(ImageOps.grayscale(image)),
            ),
            axis=-1,
        )
        resized = Image.fromarray(rmi).resize(
            (self.input_hw[1], self.input_hw[0]), Image.Resampling.BICUBIC
        )
        x = (
            torch.from_numpy(np.asarray(resized, dtype=np.float32).copy()).permute(
                2, 0, 1
            )[None]
            / 255.0
        )
        with torch.inference_mode():
            if self.model.training or any(
                p.requires_grad for p in self.model.parameters()
            ):
                raise RuntimeError(
                    "UDepth deve permanecer em eval, com pesos congelados."
                )
            raw = self.model(x.to(self.device))[1].float()
            raw = (
                self.F.interpolate(
                    raw,
                    size=tuple(output_hw or rgb.shape[:2]),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                .cpu()
                .numpy()
            )
        depth, n_clamped = inverse_to_relative(raw)
        return (
            depth,
            raw.astype(np.float32),
            {"n_eps_clamped": n_clamped, "output_hw": list(raw.shape)},
        )
