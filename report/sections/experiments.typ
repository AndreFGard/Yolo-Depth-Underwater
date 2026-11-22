#let experiments = [
  = Modelo


  //TODO adicionar referencia NYU v2
  Considerando a sua eficiência computacional e ambiente de desenvolvimento e implantação maduros, o modelo escolhido foi o YOLO26@yolodepth, na variante variante nano, projetada para inferência em tempo real. Este modelo foi treinado em um conjunto variado de ~2,19M imagens provenientes de múltiplos datasets, todos terrestres, e alcança desempenho competente em conjuntos como o NYU V2 REF_NYU, apesar de possuir apenas 6,3M parâmetros.

  //TODO detalhar um pouco mais a arquitetura dele e botar um diagrama da arquitetura.

  Além dele, dois modelos de estado da arte de MDE terrestre atuam como professores, o UniDepth V2 @unidepthv2 e o Depth Anything V2 @depthanythingv2, que possuem altíssimo desempenho no FLSea mesmo uma avaliação _zeroshot_@synthetic_bench_zijie. Ademais, a disponibilidade dos pesos destes e a existência de um ecossistema ao redor deles são cruciais para a fácil reproducibilidade do seu uso. No entanto, ambos produzem resultados em formatos diferentes, profundidade métrica para o Unidepth, e disparidade relativa para o Depth Anything V2, o que representa um ponto de atenção durante a  comparação das suas métricas.

  //TODO usod10k reference
  A título informativo, o UDepth@udepth,  considerando que este foi treinado nativamente no USOD10k REF_USOD10k e representa um modelo do domínio subaquático, também será usado como referência de desempenho.

  = Experimentos


  == Métricas

  Adotamos principalmente o conjunto de métricas consolidado por @eigen, que permanece uma referência @maudepth@synthetic_bench_zijie e permite a comparação com trabalhos relacionados. Assim, reportaremos  AbsRel, RMSE, $delta^1$ e Silog.

  === AbsRel

  === RMSE

  === $delta^1$ (threshold accuracy)

  === Silog

  == Protocolo de avaliação

  Devido à ambiguidade intrínseca da escala da profundidade associada ao problema abordado e à variedade de modelos utilizados neste trabalho, bem como para possibilitar comparações entre trabalhos, optamos por adotar um sistema flexível de avaliação. Assim, algumas modalidades diferentes de alinhamento são aplicadas aos mapas de profundidade antes da computação das métricas. A modalidade mais simples (raw) consiste em utilizar os mapas diretamente#footnote[No caso do YOLO, a Ultralytics automaticamente incorpora parâmetros simples de calibração no modelo com base no conjunto de validação do treinamento, tendo em vista a pretensão dos seus mapas a serem usado como métricos.] para a computação das métricas, o que é particularmente relevante para métodos que produzem profundidade métrica. Ademais, adotamos também um alinhamento invariante a escala e a deslocamento (SSI) a nível de imagem, bem como um global.

  === SSI a nível de imagem
  === SSI global
  Uma alternativa ao SSI a nível de imagem que garante mais consistência na avaliação de várias imagens é um SSI resolvido para todo o conjunto de imagens. Para evitar inteiramente que um mapa seja ajustado com base no seu próprio _ground truth_, computamos um SSI para todos os frames ímpares usados na avaliação e outro para os pares, e utilizamos o SSI de paridade a cada imagem avaliada.


  Apesar dessa flexibilidade possibilitar comparações em algum grau entre modelos com premissas diferentes, é preciso considerar que não é possível comparar todos os modelos de forma justa diretamente.

  == Estudo comparativo das baselines em zero-shot

  == Finetune com professores variados


]
