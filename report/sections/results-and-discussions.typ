#let results-and-discussions = [
  = Resultados e discussão

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
        ..flsea-models.slice(1).flatten(),
        table.hline(stroke: 1pt),
      )
    },
    placement: none,
    caption: [Comparação zero-shot no FLSea-VI: médias por frame de AbsRel e $delta^1$],
  ) <tab:flsea-models>

  Entre as saídas com escala métrica, o UniDepth apresenta os menores erros e a maior $delta^1$ tanto em `raw` quanto em SSI, conforme reproduzido na literatura @synthetic_bench_zijie. Como o alinhamento `raw` avalia a saída sem qualquer ajuste de escala, este só é aplicável a modelos que produzem profundidade métrica, ou seja, o YOLO e o Unidepth.

  == Efeito do ajuste fino

  Em seguida, comparamos o YOLO26n-Depth sem ajuste com os três modelos ajustados. Para cada execução, usamos `best.pt`, selecionado pela validação do conjunto de treinamento, e avaliamos os mesmos 119 quadros do FLSea.

  #let finetuned-models = csv("../tables/finetuned_models_on_flsea.csv")
  #figure(
    {
      set text(size: 8pt, hyphenate: false)
      table(
        columns: (2.1fr, auto, auto, auto, auto),
        align: (left, center, center, center, center),
        inset: (x: 5pt, y: 4pt),
        table.hline(stroke: 1pt),
        table.header(..finetuned-models.first().map(c => strong(c))),
        table.hline(stroke: 0.5pt),
        ..finetuned-models.slice(1).flatten(),
        table.hline(stroke: 1pt),
      )
    },
    placement: none,
    caption: [Comparação do YOLO Baseline e dos modelos após finetune],
  ) <tab:flsea-finetuned>


  //TODO acho que fez falta uma comparação com o alinhamento mediana.
  Observa-se que, em geral, nenhum dos modelos finetunados obteve resultados melhores que o baseline consistentemente. O YOLO-DA supera em apenas 0,01 o baseline em ambas as métricas quando com o alinhamento SSI, enquanto o YOLO-USOD fica próximo em AbsRel, mas tem $delta^1$ menor. Por outro lado, o YOLO-UniDepth apresenta a maior diferença para o baseline, com uma vantagem de 11% no $delta^1$ raw, apesar de um aumento no erro absoluto relativo, indicando uma melhora na escala métrica das predições do modelo #footnote[Conforme elaborado na seção de efeito do finetune, essa melhora é associada à qualidade da calibração feita com os mapas do Unidepth, não ao modelo em si]. Apesar disso, o YOLO-Unidepth observou uma perda considerável com o modo de alinhamento SSI, quando a escala é ajustada por uma transformação afim, mostrando que na verdade houve piora na estrutura das predições do modelo.

  

  == Estudo do efeito do finetune ao longo do treinamento
  Considerando a avaliação supracitada dos modelos no FLSea, se mostrou importante analisar mais profundamente o efeito do finetuning neles , visando o entendimento do que levou o finetuning a ser, em última instância, infrutífero. de forma posterior ao treino, avaliamos aproximadamente um a cada quatro _checkpoints_, ou seja, o modelo obtido após terminar uma dada época, no FLSea.

  //TODO create a top level figure with displacement=none which makes these two appear side-by-side, taking up two columns.
  #figure(
    image("../figures/D1_per_epoch_raw.png"),
    placement: none,
    caption: [$delta^1$ por época no FLSea (raw)],
  ) <fig:d1_per_epoch_raw>
  #figure(
    image("../figures/D1_per_epoch_ssi.png"),
    placement: none,
    caption: [Qualidade estrutural no FLSea: $delta^1$ SSI por época],
  ) <fig:d1_ssi_per_epoch_raw>


  Conforme vê-se em @fig:d1_per_epoch_raw, todos os modelos observaram mudanças grandes no $delta^1$ (_raw_) nas primeiras épocas, representando uma adaptação inicial intensificada pelo warmup do otimizador Adam. Além disso,  a tendência do YOLO-Unidepth é bastante distante dos demais, pos este alcançou um máximo com melhora considerável de quase 20% na época 16, ao contrário da queda brusca no caso dos outros modelos. Então, ele começa a perder desempenho e seu $delta^1$ varia bastante até se tornar inferior ao baseline e atingir um platô, também observado na $delta^1$ de validação durante o treino.

  Em contraste, o YOLO-DA e YOLO-USOD caíram bruscamente, mas se recuperaram lentamente. No entanto, o YOLO-USOD não recuperou o seu baseline e continuou bastante abaixo deste, enquanto o YOLO-DA obteve uma boa recuperação, alcançando um platô similar ao do YOLO-Unidepth, mas ainda abaixo do baseline.

  Já em @fig:d1_ssi_per_epoch_raw, é possível analisar, através do $delta^1$ com alinhamento SSI, a qualidade estrutural dos mapas dos modelos. Vemos que todos sofrem uma queda considerável nas primeiras épocas, em especial para o YOLO-Unidepth. Esse histórico reitera que o Depth Anything V2 foi o único professor a de fato melhorar a qualidade estrutural e relativa da estimação do YOLO, e elucida que os outros modelos não recuperaram inteiramente a capacidade baseline do YOLO nesse aspecto.

  A queda brusca na última época e a perda de desempenho no $delta^1$ _raw_ do modelo escolhido do YOLO-DA e YOLO-USOD, conforme esperado, se deve ao processo de calibração ajustada ao conjunto de validação, produzidos pelos seus professores, que é aplicado automaticamente pela biblioteca utilizada. Conforme discutido, estes modelos exigem métricas robustas à falta de escala inerente aos mapas dos seus datasets. Por outro lado, essa mesma calibração quando baseada na escala observada no dataset produzido pelo Unidepth, melhorou significativamente o YOLO-Unidepth. Associado às evidências do $delta^1$, isso indica que a escala métrica do Unidepth é de alta qualidade, conforme esperado considerando a avaliação deste, mas a transferência de conhecimento através do treinamento não foi positiva.

  = Discussion


]
