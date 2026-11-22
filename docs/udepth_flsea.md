# UDepth-RMI no FLSea

Inferência do UDepth com os pesos oficiais pré-treinados e congelados, sem treino
ou ajuste no FLSea. A avaliação segue a main `1ad6133`: `ssi`, `median`, `control`
e SILog com λ = 1. Não há ajuste global no conjunto de teste.

## Executar e salvar para visualização

Na raiz do repositório, com Python >= 3.13:

```bash
uv run --extra cu --extra flsea --extra udepth python scripts/evaluate_flsea.py \
  --model udepth --subset arxiv-2507.02148 --device cpu \
  --out results/udepth_evaluation_flsea.csv \
  --save-predictions results/udepth_predictions
```

`--save-predictions` é opcional. Ao omiti-lo, o avaliador calcula e grava o CSV
sem exportar os PNGs/NPYs para a pasta de visualização. O cache interno do modelo
continua separado. A PR corrige as duas tentativas de usar a pasta quando o
argumento não foi informado. A exportação com uma pasta informada é mantida.

CPU é suficiente; GPU é opcional. O [notebook](../notebooks/Udepth_FLSea.ipynb)
tem `SALVAR_PREDICOES = True` para gerar as figuras da entrega e pode ser
executado com esse parâmetro desligado. Os testes cobrem ambos os casos.

Com o FLSea já disponível, acrescente `--local-root /caminho/flsea-vi/versions/5`.
O download comum obtém somente os arquivos do manifesto. Os resultados usam a
versão 5; o notebook fixa essa versão e confere os hashes de RGB e referência.
Se o Kaggle solicitar autenticação, configure-a fora do Git.

`UDEPTH_ASSETS` define a pasta de fontes/pesos oficiais e `UDEPTH_CACHE` define o
cache de predições. O cache só é reutilizado quando o protocolo, a imagem e os
hashes dos mapas conferem. Reaproveitar uma previsão não reaproveita métricas:
o avaliador calcula novamente as métricas com a fórmula atual.

## Entrada e formatos das predições

A entrada BGR uint8 é convertida para RGB e depois RMI: R, max(G,B) e luminância
PIL. O RMI é calculado antes do resize bicúbico para 480 × 640, e a entrada é
float32 dividida por 255. A saída inversa é redimensionada por interpolação
bilinear antes da conversão.

A convenção fixada trata a saída bruta como profundidade inversa relativa,
maior para mais perto. O adaptador usa `OutputKind.DISPARITY` na interface;
isso não significa disparidade estéreo calibrada. A profundidade relativa
exportada é `1/max(inversa,1e-6)`, maior para mais longe, sem normalização min–max.

| Arquivo em `--save-predictions` | Conteúdo e uso |
|---|---|
| `relativas/<sequência>/<frame_id>.npy` | Profundidade relativa float32, sem GT na geração; não são metros |
| `indice_relativas.csv` | Sequência, frame, caminho e SHA-256 do NPY |
| `alinhadas_ssi/<frame_id>.png` | PNG uint16 × 1000, alinhado ao GT por SSI; auxiliar para avaliação/visualização |
| `formato_predicoes.json` | Formatos, unidades e uso de referência |

O pedido de PNGs para visualização é atendido pela pasta `alinhadas_ssi`.
Os mapas relativos originais permanecem separados em `relativas`.

## Resultados atuais

O [CSV](../artifacts/udepth_evaluation_flsea.csv) contém **357 linhas**:
119 imagens × três protocolos. O manifesto original contém 120 pares em seis
sequências. O frame `red_sea/cross_pyramid_loop/16315964096184783` foi excluído
pelo critério original de cobertura de GT inferior a 2%.

| Protocolo | AbsRel ↓ | RMSE alinhado ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | SILog λ = 1 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| ssi | 0.1291 | 0.9278 | 0.8466 | 0.9617 | 0.9870 | 17.5301 |
| median | 0.2295 | 1.6208 | 0.7243 | 0.8972 | 0.9543 | 25.3564 |
| control | 0.3230 | 1.6346 | 0.3997 | 0.7136 | 0.8770 | 44.3266 |

Os valores são médias por imagem. δ1 é a fração de pixels válidos cuja razão
previsão/referência, em qualquer sentido, é menor que 1,25; não é uma porcentagem
de imagens classificadas corretamente. SSI e mediana usam o GT da própria
imagem e avaliam estrutura após alinhamento. Os resultados não demonstram
profundidade métrica independente.

A execução usa a main `1ad6133260c07cbcef754777b3f7390ade8e1a43` com apenas a
correção de exportação opcional da PR #2 no CLI. As fontes de `evaluation.py` e
`metrics.py` são as da main, sem alterações. O [registro de validação
técnica](../artifacts/udepth_flsea_validation.json) contém os hashes do código
efetivamente executado, do CSV, os formatos e hashes das predições e as versões
do ambiente. O [índice dos mapas](../artifacts/udepth_flsea_relative_index.csv)
associa os 119 NPYs aos respectivos frames.

## CSVs antigos dos colegas

Os CSVs dos demais modelos ainda precisam de atualização para o novo protocolo.
O SILog anterior usa λ = 0,85 e não pode ser comparado com o SILog atual, λ = 1.
Excluir protocolos ou renomear `constant` para `control` não recalcula o SILog.

A [checagem de comparabilidade](../artifacts/udepth_flsea_comparability.json)
verifica os mesmos IDs, as frações válidas e as métricas não alteradas do mapa
plano de controle. Ela não apresenta o SILog dos CSVs antigos como comparável
nem substitui a reavaliação dos colegas.

O [protocolo original](../artifacts/udepth_flsea_protocol.json) é preservado para
fixar os 120 pares e seus hashes de entrada. Os hashes de avaliação e λ = 0,85
contidos nele pertencem à execução histórica; os dados da execução atual estão
no registro de validação técnica. As [exclusões](../artifacts/udepth_flsea_excluded_frames.json)
continuam registradas.

## Pesos e arquivos grandes

Fontes oficiais: [uf-robopi/UDepth](https://github.com/uf-robopi/UDepth), commit
`baa8fcf7893fc4a485f115ff567a39481f9f24b6`, checkpoint `saved_model/model_RMI.pth`.
O download verifica os hashes das fontes e dos pesos. SHA-256 do checkpoint:
`f97343d3356feca7b8cca1a8951cea38ce1f17eeb3a797a63d98f2bfae764c89`.
Não é aplicado o pós-processamento CPD/guided filter.

Os resultados desta execução ficam no Drive de vma3, em
`MyDrive/projeto_profundidade/UDepth_FLSea_1ad6133_silog1_120/`.
Os 119 NPYs e 119 PNGs estão em `predicoes_udepth/`.
O [Colab](https://colab.research.google.com/drive/1SyMx68Dzv1ywcLmA0tT4Vci1HimV8eqr)
registra o destino completo. O acesso depende das permissões da conta/pasta.
Datasets, pesos, predições grandes e credenciais ficam fora do Git.

## Testes

```bash
uv run --extra cu --extra flsea --extra udepth --group dev python -m pytest tests/test_udepth.py
```

Os testes verificam execução com e sem exportação, separação entre NPY relativo
e PNG SSI, integridade do cache, exclusão de GT insuficiente e consistência do
CSV/índice/registro atual. A comparação com os CSVs antigos exclui explicitamente
o SILog, cuja fórmula mudou.
