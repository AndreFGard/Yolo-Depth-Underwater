// Seção "Análise Exploratória dos Dados". Tabelas em ../tables/*.csv (separador ";"),
// figuras em ../figures/eda/*.png (geradas pelo notebook notebooks/eda_usod10k_flsea.ipynb).

#let csv-table(path, columns, aligns, bordered: false) = {
  let rows = csv(path, delimiter: ";")
  set text(size: 7pt)
  table(
    columns: columns,
    align: aligns,
    stroke: if bordered { none } else { (x, y) => if y == 0 { (bottom: 0.6pt) } else { none } },
    ..(if bordered { (inset: (x: 5pt, y: 4pt)) } else { (:) }),
    if bordered { table.hline(stroke: 1pt) },
    table.header(..rows.first().map(c => strong(c))),
    if bordered { table.hline(stroke: 0.5pt) },
    ..rows.slice(1).flatten(),
    if bordered { table.hline(stroke: 1pt) },
  )
}

#let eda = [
  = Análise Exploratória dos Dados <sec:eda>

  Esta seção descreve os dois conjuntos de dados usados no projeto: o USOD10K, usado para o ajuste fino e a avaliação interna,
  e o FLSea, usado como teste fora da distribuição de treino. Todas as estatísticas foram obtidas no notebook de EDA do repositório.
  No USOD10K, as estatísticas de cor e de labels usam 400 imagens sorteadas por split, e aqui o FLSea usa os 240 amostrados uniformemente cronologicamente.

  == USOD10K

  O USOD10K @usod10k reúne 10.255 imagens subaquáticas, todas com 640×480 pixels, divididas em
  7.178 (treino), 2.051 (validação) e 1.026 (teste), aproximadamente 70/20/10%. Os mapas de profundidade que acompanham
  as imagens são estimativas de um modelo monocular, e não medições, tendo sido usado um modelo DPT @ranftl2021dpt para obtê-las. Na conversão usada neste projeto, a disparidade `d` de cada imagem é normalizada por mín-máx
  e transformada em `profundidade_rel = ε / (d + ε)` com ε = 0,25, gravada em 16 bits (valores de 13.107 a 65.535, maior = mais longe).
  Portanto, *os labels não são métricos e seus valores não são comparáveis entre imagens*.

  O USOD10K foi escolhido pela sua muito alta variedade de imagens e de condições de água, combinada ao tamanho grande para o domínio submarino, especialmente considerando que outros datasets grandes costumam usar vídeos, em que imagens são altamente correlacionadas, além de já ter sido usado em experimentos de MDE@udepth.

  === Cor

  A @tab:usod-color e a @fig:usod-color mostram as estatísticas de cor por split. A razão R/G fica entre 0,69 e 0,74:
  o canal vermelho tem em média 26% a 31% menos intensidade que o verde, enquanto azul e verde são próximos (B/G entre 1,01 e 1,04).
  Isso é o esperado da absorção da água, que atenua o vermelho primeiro @seathru. Os três splits têm luminância (≈ 102) e contraste
  (≈ 33) muito próximos, e a maior diferença, o R de validação, equivale a cerca de 1,7 erros padrão.
  O desvio entre imagens é grande (R ≈ 41), o que indica condições de água muito variadas.

  #figure(
    csv-table(
      "../tables/eda_usod_color.csv",
      (1fr,) + (1.6fr,) * 5 + (1fr,) * 2,
      (left,) + (center,) * 7,
      bordered: true,
    ),
    placement: bottom,
    scope: "parent",
    caption: [Estatísticas de cor do USOD10K (média ± desvio entre imagens, escala 0–255; 400 imagens por split).],
  ) <tab:usod-color>

  #figure(
    image("../figures/eda/eda_usod_color.png", width: 100%),
    caption: [Distribuição das médias de R, G e B por imagem em cada split do USOD10K.],
  ) <fig:usod-color>

  === Imagens extremas

  Entre 1.200 imagens amostradas, 51 (4,3%) têm canal vermelho praticamente zerado (média de R abaixo de 5) e aparência de imagem
  processada, 6 (0,5%) são muito escuras (luminância abaixo de 20) e nenhuma está estourada (luminância acima de 235)
  (@fig:usod-extremes). No outro extremo, as maiores razões R/G vêm de cenas com luz artificial ou objetos de cor própria
  (águas-vivas, peixes vermelhos). Assim, o dataset mistura águas verdes, azuis, escuras e iluminadas, e a média R/G ≈ 0,7
  resume uma distribuição heterogênea nesse sentido.

  #figure(
    image("../figures/eda/eda_usod_extremes.png", width: 100%),
    placement: top,
    scope: "parent",
    caption: [Imagens extremas do USOD10K: menor R médio (linha 1), mais escuras (linha 2) e maior razão R/G (linha 3).],
  ) <fig:usod-extremes>

  === Labels de profundidade

  Em todas as imagens amostradas dos três splits, os labels usam a faixa inteira (13.107 a 65.535), consequência direta da normalização
  mín-máx por imagem (@tab:usod-labels). A mediana por imagem é parecida entre os splits (≈ 27.000 a 28.000), com distribuição
  assimétrica e massa concentrada na parte "perto" da faixa, mais um pico isolado no extremo "longe" (@fig:usod-labels).
  Os pares imagem/label (@fig:usod-pairs) mostram mapas suaves, em forma de manchas, em que os contornos dos objetos se tornam ainda menos nítidos conforme se afastam da câmera.

  #figure(
    csv-table(
      "../tables/eda_usod_labels.csv",
      (1fr, 1fr, 1fr, 2fr, 1fr, 1fr, 1.3fr),
      (left,) + (center,) * 6,
      bordered: true,
    ),
    placement: bottom,
    scope: "parent",
    caption: [Estatísticas dos labels do USOD10K (faixa cheia = mín ≤ 13.200 e máx ≥ 65.400; 400 imagens por split).],
  ) <tab:usod-labels>

  #figure(
    image("../figures/eda/eda_usod_labels.png", width: 100%),
    caption: [Mediana do label por imagem em cada split (esquerda) e histograma de pixels amostrados (direita).],
  ) <fig:usod-labels>

  #figure(
    image("../figures/eda/eda_usod_pairs.png", width: 100%),
    placement: top,
    scope: "parent",
    caption: [Pares imagem (acima) e label de profundidade relativa (abaixo, mais claro = mais perto) do split de teste.],
  ) <fig:usod-pairs>

  == FLSea

  O FLSea @flsea é um conjunto de imagens e dados inerciais, gravado em cânions do
  Mediterrâneo (4–7 m de profundidade) e no Mar Vermelho (3–8 m). Usamos a parte visual-inercial (12 sequências). A referência de
  profundidade vem de fotogrametria (SfM) com escala dada por objetos de tamanho conhecido, em metros, e é esparsa, de forma que
  pixels sem estimativa são ditos inválidos. Para a avaliação, usamos 120 quadros amostrados uniformemente, 20 de cada cena, usando a seleção de cena de @synthetic_bench_zijie, mas na presente análise utilizamos 240 quadros, obtidos ao incluir também as cenas restantes dos dataset, para analisá-lo de forma mais abrangente. Todas as imagens têm 608×968 pixels.

  O FLSea é um dos datasets mais usados no campo de estimação de profundidade submarina@maudepth@synthetic_bench_zijie@watermono, além de ser relativamente grande, apesar das suas imagens serem obtidas de vídeos, motivando a sua escolha como conjunto de testes deste trabalho.

  A profundidade de referência é de curto alcance (@fig:flsea): mediana de 2,26 m e p95 de 6,34 m, e apenas 1,2% dos pixels passam
  de 10 m e praticamente nenhum passa de 20 m, o corte de profundidade máxima empregado. Em média,
  de 63% a 90% dos pixels de cada sequência têm referência válida (@tab:flsea-seq),e somente dois quadros têm menos de 2% e são descartados dos 240 avaliados. Como o erro é calculado apenas onde há referência, regiões sem textura tendem a ser excluídas.
  A razão R/G fica entre 0,30 e 0,40, bem abaixo da do USOD10K (≈ 0,7): o FLSea é, portanto, um deslocamento de domínio real
  em relação aos dados de treino#footnote[As razões foram calculadas de forma ligeiramente diferente nos dois datasets, razão das médias
    no USOD10K e média das razões por quadro no FLSea.].

  #figure(
    image("../figures/eda/eda_flsea_depth.png", width: 100%),
    placement: top,
    scope: "parent",
    caption: [FLSea (240 quadros): histograma da profundidade de referência com marcas em 10 m e 20 m (esquerda), fração de pixels válidos
      por sequência (centro) e p95 da profundidade por sequência (direita).],
  ) <fig:flsea>

  #figure(
    csv-table("../tables/eda_flsea_sequences.csv", (2.4fr,) + (1fr,) * 6, (left,) + (center,) * 6, bordered: true),
    placement: top,
    scope: "parent",
    caption: [Estatísticas por sequência do FLSea. "Válido" = fração de pixels com referência; mediana, p95 e máximo da profundidade em metros.],
  ) <tab:flsea-seq>

  == Implicações para o projeto

  - *Avaliação alinhada.* Os labels do USOD10K não têm escala, portanto só se avalia com alinhamento por imagem (escala e deslocamento),
    e o resultado mede concordância com o rótulo sintético, apenas.
  - *Teste fora da distribuição.* O FLSea tem profundidade métrica e real, alcance curto e cor bem diferente, servindo como teste fora de domínio para os modelos.
  - *Aumento de dados de cor.* A forte atenuação do vermelho e a presença de imagens com canal zerado motivam perturbações de cor
    por canal como experimento.
]
