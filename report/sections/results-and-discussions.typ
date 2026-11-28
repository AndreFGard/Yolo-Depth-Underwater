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

  Primeiro, comparamos o YOLO26n-Depth, sem ajuste, com seus professores e com o UDepth-RMI. A tabela resume a média aritmética das métricas nos 119 quadros válidos de seis sequências do subconjunto de @synthetic_bench_zijie.

  #let flsea-models = csv("../tables/models_on_flsea.csv")
  #figure(
    {
      set text(size: 8pt, hyphenate: false)
      table(
        columns: (2.1fr, auto, auto, auto, auto),
        align: (left, center, center, center, center),
        stroke: none,
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
  #figure(
    placement: top,
    scope: "parent",
    {
      set text(size: 8pt, hyphenate: false)
      table(
        columns: (1.4fr, 1fr, 1fr, 1fr, 1fr),
        align: (left, center, center, center, center),
        stroke: none,
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
  ) <tab:flsea-finetuned>


  No conjunto, nenhum dos modelos ajustados supera o baseline de forma consistente. O melhor caso é a melhora marginal do YOLO-DA de 0,01 no AbsRel sob alinhamento SSI, com empate no $delta^1$, enquanto o YOLO-USOD empata no AbsRel, mas perde na $delta^1$, ligeiramente. O caso mais instrutivo é o YOLO-UniDepth, que apresenta a maior diferença para o baseline no $delta^1$ _raw_, subindo de 0,39 para 0,50, enquanto o AbsRel piora, de 0,27 para 0,37, e sob SSI o modelo perde de forma clara com $0,84 delta^1$, indicando piora estrutural nos mapas. Essa melhora ficou restrita à escala métrica, que a calibração herdou do professor, e veio acompanhada de uma piora estrutural #footnote[Essa melhora é atribuída à qualidade da calibração feita com os mapas do UniDepth, não ao modelo em si. Nos casos do YOLO-DA e do YOLO-USOD a mesma calibração é debilitante, visto que os seus professores não produzem mapas em escala realista.].

  O alinhamento por mediana também reforça as evidências acima. Como ele corrige apenas uma escala por imagem, enquanto o SSI ajusta uma transformação afim completa no espaço da disparidade, sobra nele todo erro que não seja de escala pura, e é esse o tipo de erro que se destaca no finetune, visto que os três modelos ficam abaixo do baseline de 0,81, com o YOLO-DA em 0,76 e o YOLO-UniDepth em 0,73. A leitura é que o finetune distorce a forma da distribuição de profundidade, um defeito que a transformação afim do SSI absorve em parte. A distorção é comum aos três professores, e a melhora marginal de geometria do YOLO-DA só aparece sob a régua mais permissiva.



  == Estudo do efeito do finetune ao longo do treinamento
  Considerando a avaliação supracitada dos modelos no FLSea, se mostrou importante analisar mais profundamente o efeito do finetuning neles, visando o entendimento do que levou o finetuning a ser, em última instância, infrutífero. De forma posterior ao treino, avaliamos no FLSea 25 pontos ao longo das 80 épocas de treino de cada finetune, ou seja, o modelo obtido após terminar cada época.


  Como a loss do finetune é invariante à escala, ela não ajusta pesos nesse sentido, a escala do modelo muda livremente ao longo do treinamento. Como os _checkpoints_ intermediários preservam a calibração original, o $delta^1$ _raw_ mediria principalmente esse deslocamento, e portanto não é produtivo para a análise, restando os alinhamentos SSI e por mediana.

  No entanto, há de se destacar que a calibração automaticamente aplicada pela biblioteca na última camada e no modelo escolhido, independente do aprendizado do modelo, corrigiria esse aspecto inteiramente, caso os mapas de profundidade em si estivessem na escala desejada, o que é falso no caso dos mapas do USOD10K e do Depth Anything. Conforme esperado com base no desempenho do UniDepth, a calibração baseada nos mapas dele, aplicada ao fim do treino e carregada pelos _checkpoints_ finais, melhorou significativamente essa métrica no YOLO-UniDepth#footnote[Verificamos que esse aumento se deve inteiramente à calibração, por meio da sua reversão.]. Somado às evidências acima, isso indica que a escala métrica do UniDepth é de alta qualidade, conforme esperado considerando a avaliação deste, mas a transferência do conhecimento dele através do treinamento não foi positiva.

  #figure(
    grid(
      columns: (1fr, 1fr),
      column-gutter: 8pt,
      align(center)[
        #image("../figures/D1_per_epoch_ssi.png", width: 100%) \
        #text(size: 8pt)[$(a)$ $delta^1$ SSI por época, no FLSea.]
      ],
      align(center)[
        #image("../figures/D1_per_epoch_val.png", width: 100%) \
        #text(size: 8pt)[$(b)$ $delta^1$ de validação por época, alinhado por mediana.]
      ],
    ),
    placement: top,
    scope: "parent",
    caption: [
      Transferência e especialização ao longo do finetune no FLsea (SSI)
    ],
  ) <fig:d1_per_epoch>

  No gráfico @fig:d1_per_epoch, o alinhamento SSI remove escala e deslocamento por imagem e representa a qualidade estrutural, permitindo uma comparação direta dos efeitos do finetuning. O YOLO-UniDepth sai de 0,90 para 0,75 na época 2, a maior queda entre eles, e termina em 0,84, abaixo do baseline e dos outros modelos. O YOLO-USOD cai para 0,82 e se recupera até 0,88, e o YOLO-DA também cai para 0,82, mas chega ao fim em 0,90, empatando com o baseline.

  Apesar dos resultados negativos no FLSea, o $delta^1$ de validação durante o treino, computado contra os rótulos do respectivo conjunto de treino, após o alinhamento por mediana de cada imagem e portanto imune à deriva da escala, sobe consideravelmente em todos os modelos #footnote[Apesar dos requisitos computacionais impedirem maiores comparações, estudos preliminares com taxas de aprendizagem menores indicaram que estas superam a taxa padrão apenas nas primeiras épocas, em que a instabilidade do aprendizado é mais acentuada].


  = Conclusões e discussões

  A hipótese que motivou este trabalho, de que rótulos sintéticos produzidos por professores maiores bastariam para adaptar o YOLO26n-Depth ao domínio subaquático, não se confirmou. Nenhum dos três alunos supera o baseline de forma consistente (@tab:flsea-finetuned). O melhor deles, o YOLO-DA, apenas o iguala sob SSI ($delta^1$ de 0,90; AbsRel de 0,10 contra 0,11), e os três ficam abaixo dele sob alinhamento por mediana (0,76, 0,75 e 0,73 contra 0,81). Assim, a queda foi presente em todos os modelos, a torna o sinal mais consistente do experimento. Como é comum a professores muito diferentes, entendemos que sua origem esteja provavelmente associada a algo compartilhado pelas três execuções, em particular ao domínio do USOD10K, o tamanho do aluno ou a configuração de treino, ao invés de propriedade de cada professor.

  O único ganho expressivo, o $delta^1$ _raw_ do YOLO-UniDepth (de 0,39 para 0,50), não pode ser entendido como aprendizado. Ele desaparece ao revertermos a calibração incorporada ao _checkpoint_, e no mesmo modelo o AbsRel _raw_ piora (de 0,27 para 0,37) e a qualidade estrutural cai (SSI de 0,90 para 0,84). A calibração feita com os mapas métricos do UniDepth corrigiu a escala do YOLO, o que é coerente com a qualidade do professor, mas o treinamento em si foi prejudicial.

  O $delta^1$ de validação sobe de forma contínua nos três modelos (@fig:d1_per_epoch), mas esse número mede a concordância com os rótulos do próprio conjunto de treino e não é comparável entre alunos, já que cada um é medido contra rótulos diferentes, e mostra apenas a adaptação a eles. Essas evidências indicam que o ajuste fino especializa o modelo no USOD10K, cuja razão R/G é de cerca de 0,7, enquanto no FLSea ela fica entre 0,30 e 0,40. O que se aprende nesse domínio não é transferido, e desloca o modelo da solução que o pré-treino havia encontrado. A queda nas primeiras épocas (de 0,90 para algo entre 0,75 e 0,82 sob SSI), indica que as primeiras atualizações de pescoço e cabeça já bastam para afastá-lo dessa solução, e a recuperação posterior é insuficiente em dois dos três casos. Em testes preliminares, reduzir a taxa de aprendizagem atenuou essa queda inicial, mas não afetou o resultado final.

  O resultado de mais difícil interpretação é que o UniDepth, o professor melhor avaliado e que produz rótulos métricos, gerou o pior aluno. Para esclarecer este ponto, elaboramos um experimento adicional inicial, conforme elaborado em @sec:apendice, em que fizemos finetune do YOLO em cenas do FLSea, usando rótulos _ground truth_, como referência de rótulos de qualidade, e rótulos produzidos pelo Unidepth e Depth Anything V2. Neste experimento, o modelo treinado com os rótulos do Unidepth foi o melhor dos três, ainda que sem diferença significativa entre eles, mas com todos vencendo o baseline, o que indica que o Unidepth não é fundamentalmente um professor ruim para essa aplicação, além de apontar para a qualidade dos seus rótulos ser satisfatória. Assim, continua a necessidade de investigar como a interação dos rótulos do Unidepth com os dados do USOD provoca um desempenho _out of domain_ tão negativo.

  == Limitações e trabalhos futuros

  Apesar da identificação da inefetividade do finetuning, não foi possível isolar experimentalmente mais váriaveis relacionadas ao treinamento para elucidar as questões remanescentes. Particularmente, não foi possível testar hiperparâmetros de treino e nem estudos quanto ao efeito do tamanho do dataset no finetuning. Além disso, a indisponibilidade de rótulos _ground truth_ para ele dificulta a avaliação objetiva de modelos e professores nele mesmo, de maneira que esses só podem ser avaliados _out of domain_, com rótulos _ground truth_, o que dificulta a identificação da causa de alguns resultados, exigindo experimentos auxiliares.

  Ademais, adaptações da estrutura do modelo ao domínio também demandariam mais recursos e dados, mas representam um caminho promissor e bastante abordado na literatura @udepth@maudepth@watermono. 

]
