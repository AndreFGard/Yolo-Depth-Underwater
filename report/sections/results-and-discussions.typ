#let best-number(values, minimize) = {
  let best = none
  for value in values {
    if value != "—" and value != "" {
      let x = float(value)
      if best == none or (if minimize { x < best } else { x > best }) {
        best = x
      }
    }
  }
  best
}

#let emphasize-best(cell, best) = {
  if best != none and cell != "—" and cell != "" and float(cell) == best {
    strong(cell)
  } else {
    cell
  }
}

#let results-and-discussions = [
  = Análise dos resultados
  Após a execução dos experimentos e obtenção dos resultados, comparamos, a princípio, os modelos baseline no FLSea, e depois analisamos os resultados após o finetune de cada modelo derivado do YOLO.

  == Comparação dos modelos no FLSea

  Primeiro, comparamos o YOLO26n-Depth, sem ajuste, com seus professores e com o UDepth-RMI. As tabelas resumem a média aritmética das métricas nos 119 quadros válidos de seis sequências do subconjunto de @synthetic_bench_zijie.

  #let flsea-models = csv("../tables/models_on_flsea.csv")
  #figure(
    {
      set text(size: 8pt, hyphenate: false)
      table(
        columns: (2.1fr, auto, auto, auto, auto),
        align: (left, center, center, center, center),
        inset: (x: 5pt, y: 4pt),
        table.hline(stroke: 1pt),
        table.header(..flsea-models.first().map(c => strong(c))),
        table.hline(stroke: 0.5pt),
        ..flsea-models
          .slice(1)
          .map(row => row
            .enumerate()
            .map(pair => {
              let (j, cell) = pair
              if j == 0 {
                cell
              } else {
                let minimize = flsea-models.first().at(j).contains("↓")
                let best = best-number(
                  flsea-models.slice(1).map(r => r.at(j)),
                  minimize,
                )
                emphasize-best(cell, best)
              }
            }))
          .flatten(),
        table.hline(stroke: 1pt),
      )
    },
    placement: none,
    caption: [Comparação zero-shot no FLSea-VI: médias por frame de AbsRel e $delta^1$],
  ) <tab:flsea-models>

  Entre as saídas com escala métrica, o UniDepth apresenta os menores erros e a maior $delta^1$ tanto em `raw` quanto em SSI, conforme reproduzido na literatura @synthetic_bench_zijie. Como o alinhamento `raw` avalia a saída sem qualquer ajuste de escala, este só é aplicável a modelos que produzem profundidade métrica, ou seja, o YOLO baseline e o UniDepth.

  == Efeito do ajuste fino

  Em seguida, comparamos o YOLO26n-Depth sem ajuste com os modelos ajustados. Para cada execução, usamos `best.pt`, selecionado pela validação do conjunto de treinamento, e avaliamos os mesmos 119 quadros do FLSea.

  #let finetuned-models = csv("../tables/finetuned_models_on_flsea.csv")
  #place(
    top + center,
    float: true,
    scope: "parent",
    clearance: 1.2em,
    figure(
      {
        set text(size: 8pt, hyphenate: false)
        table(
          columns: (1.4fr, 1fr, 1fr, 1fr, 1fr),
          align: (left, center, center, center, center),
          inset: (x: 5pt, y: 4pt),
          table.hline(stroke: 1pt),
          table.header(..finetuned-models.first().map(c => strong(c))),
          table.hline(stroke: 0.5pt),
          ..finetuned-models
            .slice(1)
            .map(row => {
              let minimize = row.first().contains("AbsRel")
              let best = best-number(row.slice(1), minimize)
              row
                .enumerate()
                .map(pair => {
                  let (j, cell) = pair
                  if j == 0 {
                    cell
                  } else {
                    emphasize-best(cell, best)
                  }
                })
            })
            .flatten(),
          table.hline(stroke: 1pt),
        )
      },
      caption: [Comparação do YOLO Baseline e dos modelos após finetune],
    ),
  ) <tab:flsea-finetuned>


  Observa-se que, em geral, nenhum dos modelos finetunados obteve resultados melhores que o baseline consistentemente. O YOLO-DA supera em apenas 0,01 o baseline no AbsRel quando com o alinhamento SSI e empata na $delta^1$, enquanto o YOLO-USOD fica próximo em AbsRel, mas tem $delta^1$ menor. Por outro lado, o YOLO-UniDepth apresenta a maior diferença para o baseline no $delta^1$ raw, subindo de 0,39 para 0,50, apesar de um aumento no erro absoluto relativo, indicando uma melhora na escala métrica das predições do modelo #footnote[Essa melhora é atribuída à qualidade da calibração feita com os mapas do UniDepth, não ao modelo em si. Nos casos do YOLO-DA e do YOLO-USOD a mesma calibração é debilitante, visto que os seus professores não produzem mapas em escala realista.]. Apesar disso, o YOLO-UniDepth apresentou uma perda considerável com o modo de alinhamento SSI, quando a escala é ajustada por uma transformação afim, mostrando que na verdade houve piora na estrutura das predições do modelo.

  //TODO botar uma ilustração sobre essa diferença do SSI e da median? com dado sintético ou só caçando mesmo uma imagem no FLSea que demonstre isso? Acho que só seria visível num dado sintético.
  Usando o alinhamento por mediana, que corrige apenas a escala de cada imagem, ignorando menos distorções que a transformação afim obtida pelo SSI, os modelos finetunados ficam abaixo do baseline (0,81), incluindo o YOLO-DA (0,76) e o YOLO-UniDepth (0,73). Com essa comparação, podemos concluir que essa piora é relativa à forma da distribuição de profundidade, de maneira que o YOLO-DA melhora a geometria sob o alinhamento SSI mas distorce a distribuição sob o alinhamento por mediana no mesmo modelo, e essa piora é comum aos três professores.



  == Estudo do efeito do finetune ao longo do treinamento
  Considerando a avaliação supracitada dos modelos no FLSea, se mostrou importante analisar mais profundamente o efeito do finetuning neles, visando o entendimento do que levou o finetuning a ser, em última instância, infrutífero. De forma posterior ao treino, avaliamos no FLSea 25 pontos ao longo das 80 épocas de treino de cada finetune, ou seja, o modelo obtido após terminar cada época.

  //TODO create a top level figure with displacement=none which makes these two appear side-by-side, taking up two columns.


  Como a loss do finetune é invariante à escala, não ajustando pesos nesse sentido, a escala do modelo muda livremente ao longo do treinamento, e como os _checkpoints_ intermediários preservam a calibração original do modelo, o $delta^1$ _raw_ mediria o deslocamento dessa escala, e portanto não é produtivo para esta análise, restando os alinhamentos SSI e por mediana.

  No entanto, há de se destacar que a calibração automaticamente aplicada pela biblioteca na última camada e no modelo escolhido, independente do aprendizado do modelo, corrigiria esse aspecto inteiramente, caso os mapas de profundidade em si estivessem na escala desejada, o que é falso no caso dos mapas do USOD10K e do Depth Anything. Conforme esperado com base no desempenho do Unidepth, a calibração baseada nos mapas dele, aplicada ao fim do treino e carregada pelos _checkpoints_ finais, melhorou significativamente essa métrica no YOLO-UniDepth#footnote[Verificamos que esse aumento se deve inteiramente à calibração, por meio da sua reversão.].. Associado às evidências supracitadas, isso indica que a escala métrica do UniDepth é de alta qualidade, conforme esperado considerando a avaliação deste, mas a transferência de conhecimento através do treinamento não foi positiva.

  #figure(
    image("../figures/D1_per_epoch_ssi.png"),
    placement: none,
    caption: [
      Qualidade estrutural no FLSea: $delta^1$ SSI por época.
    ],
  ) <fig:d1_ssi_per_epoch>

  Em @fig:d1_ssi_per_epoch, o alinhamento SSI remove escala e deslocamento por imagem e representa a qualidade estrutural, permitindo uma comparação direta dos efeitos do finetuning. O YOLO-UniDepth sai de 0,90 para 0,75 na época 2, a maior queda entre eles e termina em 0,84, abaixo do baseline e dos outros modelos. O YOLO-USOD cai para 0,82 e se recupera até 0,88, e o YOLO-DA também cai para 0,82, mas se recupera até 0,90, empatando com o baseline.

  #figure(
    image("../figures/D1_per_epoch_val.png"),
    placement: none,
    caption: [$delta^1$ alinhado por mediana, no conjunto de validação do treinamento, por época],
  ) <fig:d1_per_epoch_val>
  Apesar dos resultados negativos no FLSea, o $delta^1$ de validação (com os mapas de profundidade do respectivo modelo) durante o treino (@fig:d1_per_epoch_val), computado após o alinhamento por mediana de cada imagem e portanto imune à deriva de escala, sobe consideravelmente em todos os modelos #footnote[Apesar dos requisitos computacionais impedirem maiores comparações, estudos preliminares com taxas de aprendizagem menores indicaram que estas superam a taxa padrão apenas nas primeiras épocas, em que a instabilidade do aprendizado é mais acentuada].


  A comparação entre professor e aluno reforça esse ponto. No FLSea o UniDepth V2 é o melhor professor ($delta^1$ SSI de 0,98) e gerou o pior aluno (0,84), enquanto o Depth Anything V2, ligeiramente pior como professor (0,97), gerou o melhor aluno (0,90). A ordem se inverte, o que indica que a qualidade do professor no domínio de avaliação não é o fator determinante.

  = Conclusões e discussões

]
