#import "results-and-discussions.typ": best-number, emphasize-best

#let apendice = [
  = Apêndice: efeito dos rótulos sintéticos no finetune <sec:apendice>
  Conforme elaborado, a ausência de rótulos _ground truth_ no USOD10K dificulta o isolamento de fatores importantes para explicar o desempenho dos modelos _out of domain_, particularmente quanto à comparação entre professores. Em caráter adicional, fizemos a tunagem do YOLO usando as 4 cenas do cenário _red sea_ do FLSea não utilizadas no teste dos modelos regulares, usando rótulos produzidos pelo Unidepth e Depth Anything V2 e os _ground truth_, permitindo que comparemos diretamente o efeito dos rótulos sintéticos comparado aos rótulos verdadeiros.

  Usamos a mesma configuração de treino, mas com apenas 60 épocas, e obtivemos, cronologicamente em cada uma das cenas (`dice_path`, `landward_path`, `northeast_path` e `pier_path`), 50 imagens de teste, 167 de validação e o restante para treinamento, resultando em 4.859 quadros de treino, mas para os resultados finais usamos os mesmos dados de teste usados anteriormente. Assim, treino e teste permanecem cenas disjuntas, mas pertencentes ao mesmo domínio, e portanto este experimento é fundamentalmente distinto dos principais do estudo.

  Assim, avaliamos as três variantes obtidas com rótulos _ground truth_, Unidepth e Depth Anything V2 em @tab:flsea-arms.

  #let flsea-arms = csv("../tables/flsea_teacher_arms.csv")
  #let arms-best = flsea-arms.slice(1)
  #figure(
    placement: auto,
    scope: "parent",
    {
      set text(size: 8pt, hyphenate: false)
      table(
        columns: (1.5fr, 1fr, 1fr, 1fr, 1fr),
        align: (left, center, center, center, center),
        stroke: none,
        inset: (x: 5pt, y: 4pt),
        table.hline(stroke: 1pt),
        table.header(..flsea-arms.first().map(c => strong(c))),
        table.hline(stroke: 0.5pt),
        ..arms-best
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
    caption: [
      Comparação dos modelos com finetune no FLSea
    ],
  ) <tab:flsea-arms>

  #figure(
    image("../figures/D1_per_epoch_median_flsea.png"),
    caption: [
      $delta^1$ sob alinhamento por mediana, conjunto de testes original
    ],
  ) <fig:flsea-arms-epoch>

  Os três ajustes superam o baseline, e o aluno treinado com os rótulos do UniDepth é o melhor dos três: 0,869 de $delta^1$ sob mediana, contra 0,811 do baseline, mas desempenho quase idêntico ao modelo ajustado com _ground truth_ e muito próximo ao YOLO-F-DA.

  A @fig:flsea-arms-epoch mostra que as três curvas partem do valor do baseline, caem
  nas primeiras épocas e se recuperam até um patamar estável por volta da 20ª à 36ª
  época, mas dois dos modelos apresentam uma leve piora no restante do treinamento.

  Portanto, a comparação desses modelos com o YOLO-F-GT permite concluir que os professores são capazes de produzir rótulos aproximadamente tão eficientes quanto os rótulos _ground truth_, ao menos nesse domínio, além do ajuste ter promovido uma melhora considerável sobre o baseline, enfraquecendo a associação da qualidade dos rótulos sintéticos ao resultado ruim dos modelos treinados no USOD10K.
]
