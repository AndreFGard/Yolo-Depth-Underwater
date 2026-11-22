# UDepth-RMI no FLSea

Esta é a parte de inferência e avaliação do UDepth no projeto. Usa os pesos oficiais
pré-treinados, congelados, sem treinamento ou ajuste no FLSea. A interface integra
o modelo diretamente ao `scripts/evaluate_flsea.py`; `evaluation.py`, `metrics.py`
e o manifesto comum permanecem iguais aos usados pelos outros modelos.

## Executar

Com Python >= 3.13, na raiz do repositório:

```bash
uv run --extra cu --extra flsea --extra udepth python scripts/evaluate_flsea.py \
  --model udepth --subset arxiv-2507.02148 \
  --rescaled --global-ssi --no-include-raw \
  --out results/udepth_evaluation_flsea.csv \
  --save-predictions results/udepth_predictions
```

Para AMD, substitua `--extra cu` por `--extra rocm`. Para executar em CPU,
acrescente `--device cpu`. Com o FLSea já disponível, acrescente
`--local-root /caminho/flsea-vi/versions/5`. A raiz precisa conter
`canyons/flatiron/imgs/`, `canyons/flatiron/depth/` e as demais sequências.

O download usa somente os arquivos do manifesto, sem baixar o ZIP inteiro.
Os resultados versionados usam o FLSea-VI **v5**: para reprodução exata, use
uma cópia local dessa versão. O downloader comum do projeto usa o handle Kaggle
sem versão fixa. No experimento original, o download público desses pares funcionou
sem credenciais; se o Kaggle pedir autenticação, configure-a fora do Git.

`UDEPTH_ASSETS` define a pasta de fontes/pesos oficiais, por padrão `udepth_assets/`.
`UDEPTH_CACHE` define o cache das inferências, por padrão `results/udepth_cache/`.
O cache só é reutilizado quando protocolo, imagem e hashes dos dois mapas conferem.
Assim, a segunda passagem de `rescaled` e `global-ssi` pode reutilizar a inferência.

As dependências próprias ficam no extra `udepth`. Os outros modelos continuam
importados sob demanda; UniDepth e Depth Anything não são necessários para UDepth.
O [notebook executado](../notebooks/Udepth_FLSea.ipynb) chama esse mesmo script,
fixa o commit usado na execução e verifica os hashes dos 120 pares FLSea-VI v5.

## Entrada e saída

A entrada BGR uint8 do projeto é convertida para RGB. O RMI contém os canais
R, max(G,B) e luminância PIL, calculados antes do resize bicúbico para 480 × 640.
A entrada é float32 dividida por 255. A saída inversa é redimensionada com
interpolação bilinear (`align_corners=False`) antes da conversão.

A convenção fixada neste experimento trata a saída bruta como profundidade inversa
relativa, maior para mais perto. Por isso o adaptador usa `OutputKind.DISPARITY`
no avaliador comum. Esse nome da interface não implica disparidade estéreo em
pixels nem calibração métrica. A profundidade relativa exportada é
`1 / max(saida_inversa, 1e-6)`, maior para mais longe, sem normalização min–max.

| Artefato em `--save-predictions` | Conteúdo |
|---|---|
| `relativas/<sequência>/<frame_id>.npy` | Profundidade relativa float32, sem GT, sem teto de uint16; não são metros |
| `indice_relativas.csv` | Associação sequência/frame/mapa e SHA-256 |
| `alinhadas_ssi/<frame_id>.png` | Saída auxiliar alinhada ao GT pelo SSI, uint16 × 1000 |
| `formato_predicoes.json` | Convenções, unidades e uso de GT em cada formato |

Os PNGs são produtos da avaliação e não representam a previsão relativa original.
Os NPYs são os mapas a usar quando o grupo precisar da profundidade relativa.

## Resultados versionados

O arquivo [udepth_evaluation_flsea.csv](../artifacts/udepth_evaluation_flsea.csv)
contém 595 linhas: 119 imagens × cinco protocolos, incluindo `rescaled`.
O manifesto comum contém
120 pares em seis sequências. O frame `red_sea/cross_pyramid_loop/16315964096184783`
foi excluído pelo critério original de menos de 2% de GT válido.

| Protocolo | AbsRel ↓ | RMSE alinhado ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | SILog ↓ |
|---|---:|---:|---:|---:|---:|---:|
| ssi | 0.1291 | 0.9278 | 0.8466 | 0.9617 | 0.9870 | 17.5406 |
| median | 0.2295 | 1.6208 | 0.7243 | 0.8972 | 0.9543 | 25.5807 |
| rescaled | 0.4849 | 2.1572 | 0.3205 | 0.6258 | 0.8223 | 30.2915 |
| global-ssi | 0.3496 | 1.4755 | 0.3389 | 0.6447 | 0.8575 | 26.8653 |
| constant | 0.3230 | 1.6346 | 0.3997 | 0.7136 | 0.8770 | 44.4797 |

Valores são médias por imagem, com o mesmo peso por frame. δ1 é a fração de pixels
válidos cuja razão previsão/referência, em qualquer sentido, é menor que 1.25.
Não é uma porcentagem de imagens classificadas corretamente.

`ssi` e `median` usam o GT da própria imagem e medem estrutura após alinhamento.
`rescaled` ajusta somente escala; `global-ssi` ajusta escala e deslocamento.
Ambos usam a outra metade dos índices originais do manifesto, sem usar o GT do
próprio frame no ajuste. Continuam dependendo de calibração na distribuição de
teste. Em `rescaled`, AbsRel, RMSE e δ1 ficaram piores que o controle constante;
em `global-ssi`, AbsRel e δ1 ficaram piores. Isso indica limitação da calibração
entre imagens. Os 84,66% de δ1 do SSI e os 32,05% de `rescaled` pertencem a
protocolos diferentes. Os resultados não demonstram profundidade métrica
independente e não devem ser misturados com o piloto anterior de 150 imagens.

Os resultados atuais foram produzidos no Colab em 22/11/2026, em CPU, usando
diretamente o avaliador do commit `fa5f0c8badef0a22574802cad148d364c0da23f5`,
com UDepth integrado ao pacote e sem treinamento. As versões do ambiente, o hash
do CSV, os hashes das fontes e o resumo das métricas estão no
[registro da validação técnica](../artifacts/udepth_flsea_validation.json).
O registro confirma os 119 mapas relativos exportados e a verificação dos seus
hashes na execução. O [índice dos mapas](../artifacts/udepth_flsea_relative_index.csv)
preserva a associação entre sequência, frame, caminho relativo e SHA-256.

O [protocolo original](../artifacts/udepth_flsea_protocol.json) registra os 120 pares,
hashes dos RGBs/GTs e das fontes da avaliação anterior, no commit `da0166ce`.
Os hashes das fontes da execução atual estão no registro de validação técnica.
A
[verificação de comparabilidade](../artifacts/udepth_flsea_comparability.json)
confirma os mesmos IDs, frações válidas e métricas do controle constante dos três
CSVs do grupo. As [exclusões](../artifacts/udepth_flsea_excluded_frames.json) também
ficam registradas. Os resultados dos colegas não foram reexecutados nesta etapa.

## Pesos e arquivos grandes

Fontes oficiais: [uf-robopi/UDepth](https://github.com/uf-robopi/UDepth), commit
`baa8fcf7893fc4a485f115ff567a39481f9f24b6`, checkpoint `saved_model/model_RMI.pth`.
O download verifica SHA-256 das quatro fontes e do checkpoint antes de executar.
SHA-256 dos pesos:
`f97343d3356feca7b8cca1a8951cea38ce1f17eeb3a797a63d98f2bfae764c89`.
Não é aplicado o pós-processamento CPD/guided filter.

Os mapas desta execução estão no Drive de vma3, em
`MyDrive/projeto_profundidade/UDepth_FLSea_fa5f0c8b_rescaled_120/predicoes_udepth/`.
Os NPYs ficam em `relativas/` e os PNGs auxiliares em `alinhadas_ssi/`.
O [Colab com os resultados](https://colab.research.google.com/drive/1SyMx68Dzv1ywcLmA0tT4Vci1HimV8eqr)
registra o destino completo. O acesso depende das permissões da conta/pasta.

O ZIP da execução anterior, com os 119 mapas relativos e os PNGs auxiliares, está na
[pasta de resultados no Drive](https://drive.google.com/drive/folders/10tms9tiqb5yA4rWIOqZ9tEEsMNGv8O9p).
O acesso depende das permissões da pasta. O arquivo
`UDepth_FLSea_Grupo_119.zip` tem 282072384 bytes e SHA-256
`27cef9eba9424047dca232468281348603fc4c942deb94da4fa9c69e14d42fef`.
Datasets, pesos, predições grandes e credenciais ficam fora do Git.

## Validação

```bash
uv run --extra cu --extra flsea --extra udepth --group dev python -m pytest tests/test_udepth.py
```

Os testes verificam entrada BGR/RGB, cache e integridade, conversão relativa,
exportação NPY sem quantização e PNG SSI em pastas separadas, exclusão de GT
insuficiente e consistência dos artefatos com os três CSVs do grupo.
