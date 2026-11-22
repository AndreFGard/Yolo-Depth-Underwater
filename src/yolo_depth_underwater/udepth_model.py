import hashlib
import json
import os
from pathlib import Path

import numpy as np

from yolo_depth_underwater.inference_model import InferenceModel, OutputKind
from yolo_depth_underwater.udepth_runtime import (
    UDepthRelative,
    digest_json,
    prepare_assets,
    save_npy,
    sha256_file,
    write_json,
)


class UDepthRMIModel(InferenceModel):
    @property
    def output_kind(self):
        return OutputKind.DISPARITY

    def __init__(self, device=None, assets=None, output=None):
        assets = Path(assets or os.environ.get("UDEPTH_ASSETS", "udepth_assets"))
        self.output = Path(
            output or os.environ.get("UDEPTH_CACHE", "results/udepth_cache")
        )
        self.output.mkdir(parents=True, exist_ok=True)
        self.network = UDepthRelative(prepare_assets(assets), device=device or "auto")
        self.protocol = {
            **self.network.protocol,
            "adapter": "UDepthRMIModel-v1",
            "interface_input": "BGR uint8 from repository flsea.load_frame; converted to RGB before official RMI preprocessing",
            "interface_output": "relative inverse depth; larger means closer; not calibrated stereo disparity",
            "saved_relative_map": "1/max(raw,1e-6), float32 NPY; larger means farther",
            "flsea_training": False,
            "minmax_normalization": False,
        }
        self.signature = digest_json(self.protocol)
        self.n_inferences = 0
        self.n_cache_hits = 0
        self.records = {}
        write_json(self.output / "protocolo_udepth.json", self.protocol)

    def predict(self, images):
        for image in images:
            image = np.asarray(image)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    "O script do grupo deve fornecer BGR uint8 de três canais."
                )
            digest = hashlib.sha256(
                np.ascontiguousarray(image).tobytes() + str(image.shape).encode()
            ).hexdigest()
            folder = self.output / "mapas_relativos" / digest[:24]
            folder.mkdir(parents=True, exist_ok=True)
            inverse_path = folder / "inversa_relativa.npy"
            relative_path = folder / "profundidade_relativa.npy"
            metadata_path = folder / "registro.json"
            try:
                record = (
                    json.loads(metadata_path.read_text())
                    if metadata_path.exists()
                    else {}
                )
            except (ValueError, OSError):
                record = {}
            reuse = (
                record.get("signature") == self.signature
                and record.get("image_sha256") == digest
            )
            reuse = reuse and all(
                (folder / name).is_file() and sha256_file(folder / name) == sha
                for name, sha in record.get("hashes", {}).items()
            )
            reuse = reuse and set(record.get("hashes", {})) == {
                inverse_path.name,
                relative_path.name,
            }
            if reuse:
                inverse = np.load(inverse_path, allow_pickle=False)
                self.n_cache_hits += 1
            else:
                rgb = np.ascontiguousarray(image[..., ::-1])
                relative, raw, _ = self.network.predict(rgb, output_hw=image.shape[:2])
                inverse = np.maximum(raw, 1e-6).astype(np.float32)
                if not np.isfinite(inverse).all() or (inverse <= 0).any():
                    raise FloatingPointError("UDepth produziu valores inválidos.")
                save_npy(inverse_path, inverse)
                save_npy(relative_path, relative)
                record = {
                    "signature": self.signature,
                    "image_sha256": digest,
                    "runtime": {
                        "device": str(self.network.device),
                        "torch": self.protocol["torch"],
                    },
                    "hashes": {
                        inverse_path.name: sha256_file(inverse_path),
                        relative_path.name: sha256_file(relative_path),
                    },
                }
                write_json(metadata_path, record)
                self.n_inferences += 1
            self.records[digest] = {
                "inverse": str(inverse_path),
                "relative": str(relative_path),
                **record,
            }
            write_json(self.output / "indice_mapas.json", self.records)
            write_json(
                self.output / "estado_inferencia.json",
                {
                    "n_new_inferences": self.n_inferences,
                    "n_cache_hits": self.n_cache_hits,
                    "n_images": len(self.records),
                    "training": False,
                    "device": str(self.network.device),
                },
            )
            yield inverse

    def predict_split(self, split):
        raise NotImplementedError(
            "Este adaptador usa os frames fixos de evaluate_flsea.py."
        )


def export_relative_predictions(model, frames, rows, destination):
    import csv
    import shutil

    from yolo_depth_underwater.flsea import load_frame

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    scored = {
        (r["sequence"], str(r["frame_id"])) for r in rows if r["alignment"] == "ssi"
    }
    exported = []
    for frame in frames:
        if (frame.sequence, frame.frame_id) not in scored:
            continue
        image, _ = load_frame(frame)
        digest = hashlib.sha256(
            np.ascontiguousarray(image).tobytes() + str(image.shape).encode()
        ).hexdigest()
        record = model.records[digest]
        source = Path(record["relative"])
        expected = record["hashes"]["profundidade_relativa.npy"]
        if sha256_file(source) != expected:
            raise ValueError("Mapa relativo alterado antes da exportação.")
        target = destination / "relativas" / frame.sequence / (frame.frame_id + ".npy")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or sha256_file(target) != expected:
            temporary = target.with_suffix(".npy.partial")
            shutil.copyfile(source, temporary)
            temporary.replace(target)
        if sha256_file(target) != expected:
            raise ValueError("Cópia do mapa relativo não confere.")
        exported.append(
            {
                "sequence": frame.sequence,
                "frame_id": frame.frame_id,
                "relative_npy": str(target.relative_to(destination)),
                "sha256": expected,
                "unit": "relative",
                "dtype": "float32",
            }
        )
    if len(exported) != len(scored):
        raise ValueError("Exportação incompleta de mapas relativos.")
    with (destination / "indice_relativas.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sequence",
                "frame_id",
                "relative_npy",
                "sha256",
                "unit",
                "dtype",
            ],
        )
        writer.writeheader()
        writer.writerows(exported)
    write_json(
        destination / "formato_predicoes.json",
        {
            "relative_predictions": {
                "folder": "relativas",
                "format": "NPY float32",
                "unit": "relative, not calibrated meters",
                "n_maps": len(exported),
                "gt_used": False,
                "direction": "larger means farther",
                "definition": "1/max(raw_inverse,1e-6)",
                "uint16_clipping": False,
                "minmax_normalization": False,
            },
            "aligned_auxiliary": {
                "folder": "alinhadas_ssi",
                "format": "uint16 PNG x1000",
                "gt_used": True,
                "purpose": "only SSI evaluation/visualization; not the original relative prediction",
            },
        },
    )
    print(
        f"Previsões relativas exportadas: {len(exported)} NPY float32 em {destination / 'relativas'}"
    )
    return exported
