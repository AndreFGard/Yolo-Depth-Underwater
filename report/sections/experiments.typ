#let experiments = [
  = Experimentos
  Nossos experimentos incluem ajustes do YOLO26 com três fontes de rótulos: os mapas fornecidos com o USOD10K e os pseudo-rótulos produzidos pelo UniDepth e pelo Depth Anything V2. Para avaliar o efeito da taxa de aprendizado, também repetimos o ajuste com o Depth Anything V2 usando uma taxa menor. Além disso, comparamos as predições dos modelos no FLSea.

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

  == Finetune do YOLO com rótulos sintéticos
  //TODO lembrar de escrever sobre eles na base de dados.
  Fizemos um finetuning do YOLO26 para cada dataset de rótulos sintéticos obtidos do USOD10K e dos dois professores, usando o split do USOD10K entre treino, validação e teste, usando as imagens base inalteradas do USOD10K e os mapas de profundidade devidamente convertidos, conforme descrito em REF_BASE_DE_DADOS.

  Devido a restrições computacionais, não fizemos uma busca ampla de hiperparâmetros. Os ajustes duraram 80 épocas, usaram AdamW com decaimento linear e mantiveram congelados os 11 primeiros módulos do modelo. A taxa de aprendizado inicial variou entre as execuções: 0,01 para USOD10K e para um dos ajustes com Depth Anything V2; 0,002 para UniDepth; e 0,0005 para a repetição com Depth Anything V2. A avaliação usa o `best.pt` escolhido pela validação do respectivo conjunto de treinamento, nunca pelo FLSea, com a calibração obtida nessa mesma validação, conforme o protocolo da Ultralytics.

  == Protocolo de avaliação

  Devido à ambiguidade intrínseca da escala da profundidade associada ao problema abordado e à variedade de modelos utilizados neste trabalho, bem como para possibilitar comparações entre trabalhos, optamos por adotar um sistema flexível de avaliação. Notóriamente, os mapas de profundidade do USOD10K e do Depth Anything V2 precisam ser convertidos de disparidade para profundidade relativa, e portanto necessariamente exijem ajuste de escala para interpretação métrica.

  Tendo em vista as características desses modelos e o passo de calibração aplicado após o treinamento, algumas modalidades diferentes de alinhamento são aplicadas aos mapas de profundidade antes da computação das métricas. A modalidade mais simples (raw) consiste em utilizar os mapas diretamente para a computação das métricas, o que é relevante para métodos que produzem profundidade métrica e permite comparações com trabalhos relacionados. Além disso, adotamos o alinhamento por imagem invariante a escala e a deslocamento (SSI) REF_MIDAS.

  Apesar da variedade das métricas possibilitar comparações em algum grau entre modelos com premissas diferentes, é preciso considerar que é difícil comparar todos os modelos de forma justa diretamente, pois os métodos de alinhamento possuem requisitos de oráculo muito diferentes, e, logo, nossas conclusões terão escopo adequado a essas condições.

  == Estudo comparativo das baselines em zero-shot
  //TODO incluir aqui as informações sobre a conversão dos datasets e outputs de profundidade  diferentes, para todos serem profundiade relativa ou métrica. 

  == Estudo do efeito do finetuning época a época


]
