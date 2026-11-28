# yolo-depth-underwater

>   A estimação de profundidade submarina é de extrema importância para operações industriais e de veículos autônomos não tripulados, e a estimação monocular de profundidade (MDE) é interessante em aplicações com recursos limitados que dificultam outras técnicas, como visão estéreo, além de fornecer mapas densos, ao contrário de técnicas baseadas em _feature matching_@udepth. No entanto, um desafio frequente é a baixa variedade de conjuntos de dados com profundidade confiável e de tamanho suficiente para o treinamento destes modelos nesse cenário, sendo frequente o uso de dados sintéticos para contornar esse problema. Além disso, em aplicações reais, em que dados com profundidade reais são raros, rótulos sintéticos de profundidade possibilitam a adaptação de imagens verdadeiras do domínio da aplicação para o possível ajuste de modelos.
> 
>   Dessa maneira, esse trabalho procura avaliar se é possível melhorar o desempenho de um modelo computacionalmente eficiente da família YOLO26 para o cenário de MDE submarina, com imagens reais, e rótulos sintéticos produzidos por modelos mais capazes, adotando-os como professores.


# Setup
On colab, import your notebook or similar and setup the git repo:

```bash
!git clone https://github.com/AndreFGard/Yolo-Depth-Underwater.git
%cd Yolo-Depth-Underwater
!pip install -e ".[cu]"
import sys
import os
sys.path.append(os.path.abspath('src'))

import yolo_depth_underwater
```

## Other
- Do not use cv2.imread or imwrite, because it may quantize the uint16 maps into uint8. Instead, use encoding.py's utils.
- Raw depth maps may have very small variance and therefore be near impossibe to discern. Use visualization.py utils instead.


## Setup
Install git lfs.

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
- Use the `encoding` module to store csvs on disk with the standardized float rounding.
- read report/README.md for the convention on keeping csvs and such versioned.
