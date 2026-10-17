# yolo-depth-underwater

YOLO26 experiments for underwater depth estimation.

# Setup
On colab, import your notebook or similar and setup the git repo:

```bash
!git clone https://github.com/AndreFGard/Yolo-Depth-Underwater.git
%cd Yolo-Depth-Underwater
!uv pip install --system -e ".[cu]"
import yolo_depth_underwater
```

# Docs

## Setup

To support both AMD and Nvidia gpus, you must **always** use uv --extra so it doesnt overwrite the packages for your specific hardware.
### AMD GPUs (ROCm)

```sh
uv sync --extra rocm
```

Installs torch 2.8.0 + torchvision 0.23.0 from the ROCm 6.4 wheel.

### NVIDIA GPUs (CUDA)

```sh
uv sync --extra cu
```

Installs the latest torch/torchvision from PyPI (CUDA support is bundled on
Linux).

If on windows, you must add a`cu126`/`cu130` extra pointing at the matching
> `download.pytorch.org/whl/cuXXX` index (same pattern as the `rocm` extra).


[more information](https://docs.astral.sh/uv/guides/integration/pytorch/#configuring-accelerators-with-optional-dependencies)

## Running

You must always pass --extra to stop uv from overwriting your packages.

```sh
uv run --extra rocm python script.py
```

NVIDIA users:

```sh
uv run --extra cu python script.py
```

## Notes

- The `rocm` and `cu` extras are declared as conflicting in
  `[tool.uv] conflicts`, so a single `uv.lock` covers both variants without
  version collisions. Do not enable both at once.
- Use `uv add`/`uv remove` for dependency changes rather than `uv pip
  install`, so the lockfile stays authoritative (mixing the two interfaces is
  how the venv can drift out of sync with `pyproject.toml`).
