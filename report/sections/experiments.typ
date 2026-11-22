#let experiments = [
  = Experimentos
  Nossos experimentos consistem em três finetunings baseados no YOLO26 em três conjuntos de dados com rótulos sintéticos diferentes, que são os já imbutidos no USOD10K, os produzidos pelo Unidepth e os produzidos pelo Depth Anything V2. Além disso, fazemos avaliações comparativas entre os mapas de profundidade e desempenho de todos os modelos envolvidos.

  == Métricas

  Adotamos principalmente o conjunto de métricas consolidado por @eigen, que permanece uma referência @maudepth@synthetic_bench_zijie e permite a comparação com trabalhos relacionados. Assim, reportaremos as métricas AbsRel, RMSE, $delta^1$ e Silog.

  Conforme elaborado em @eigen, a escala, que é fundamentalmente ambígua nas cenas, é sozinha responsavél por uma grande parcela do erro das métricas, fazendo-se necessário o uso de métricas específicas para avaliar a qualidade estrutural dos mapas de profundidade, disassociada da escala.


  //TODO setas pra cima e pra baixo
  === AbsRel
  A diferença absoluta relativa (*AbsRel*) quantifica o erro em razão da magnitude do ground truth, evitando que erros pequenos comparados à profundidade de um ponto distorçam a métrica @eigen.
  //TODO Formula

  === RMSE
  A *RMSE* quantifica o erro na unidade de profundidade, o que a faz ser de interpretação complexa no contexto do estudo e a ser dominada por erros em profundidades maiores, independente da qualidade da predição fina em objetos mais próximos.

  //TODO Formula
  === $delta^1$
  _*Threshold accuracy*_ (*$delta^1$*) é uma das métricas mais amplamente utilizadas no campo @eigen @synthetic_bench_zijie e mede a porcentagem de pontos que distam do ground truth dentro de uma tolerância.
  //TODO Formula

  === Silog
  A *Silog* ou _scale-invariant mean squared error_, é derivada da RMSE, mas atua no espaço logarítimico e de forma invariante a escala, capturando a qualidade estrutural do mapa de profundidade, em que o escalar $alpha(y, ŷ)$ é a escala que melhor alinha o mapa predito ao ground truth.
  //TODO formula do eigen.

  == Protocolo de avaliação

  Devido à ambiguidade intrínseca da escala da profundidade associada ao problema abordado e à variedade de modelos utilizados neste trabalho, bem como para possibilitar comparações entre trabalhos, optamos por adotar um sistema flexível de avaliação. Notóriamente, os mapas de profundidade do USOD10K e do Depth Anything V2 precisam ser convertidos de disparidade para profundidade relativa, e portanto necessariamente exijem ajuste de escala para interpretação métrica.

  Portanto, algumas modalidades diferentes de alinhamento são aplicadas aos mapas de profundidade antes da computação das métricas. A modalidade mais simples (raw) consiste em utilizar os mapas diretamente#footnote[No caso do YOLO, a Ultralytics automaticamente incorpora parâmetros simples de calibração no modelo com base no conjunto de validação do treinamento, tendo em vista a pretensão dos seus mapas a serem usado como métricos.] para a computação das métricas, o que é particularmente relevante para métodos que produzem profundidade métrica. Além disso, adotamos o alinhamento por imagem invariante a escala e a deslocamento (SSI) REF_MIDAS.

  Apesar dessa flexibilidade possibilitar comparações em algum grau entre modelos com premissas diferentes, é preciso considerar que é difícil comparar todos os modelos de forma justa diretamente, pois os métodos de alinhamento possuem requisitos de oráculo muito diferentes, e, logo, nossas conclusões serão limitadas a isso.

  == Finetune do YOLO com rótulos sintéticos
  //TODO lembrar de escrever sobre eles na base de dados.
  Fizemos um finetuning do YOLO26 para cada dataset de rótulos sintéticos obtidos do USOD10K e dos dois professores, usando o split do USOD10K entre treino, validação e teste, usando as imagens base inalteradas do USOD10K e os mapas de profundidade devidamente convertidos, conforme descrito em REF_BASE_DE_DADOS.


  //TODO adicionar footnote com os estudos limitados sobre Learning rate e não-congelamento de backbone.
  Cada modelo foi treinado com imagens em resolução completa, 640x480, por 80 épocas, com o otimizador AdamW padrão, com taxa de aprendizado inicial 0.01 e decaimento linear, além do congelamento das 11 camadas do _backbone_ do modelo. Devido a restrições computacionais, não fizemos tunagem de parâmetros de treinamento.

  == Estudo comparativo das baselines em zero-shot



]
