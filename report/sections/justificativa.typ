#let justificativa = [
  = Justificativa

  A estimativa de profundidade a partir de uma única câmera RGB pode apoiar a navegação de robôs subaquáticos, a inspeção de estruturas e o mapeamento de ambientes. Uma abordagem monocular reduz os requisitos de sensoriamento e fornece um mapa denso de profundidade a partir de cada imagem @udepth. No ambiente subaquático, a absorção seletiva e o espalhamento da luz alteram cores, contraste e visibilidade. Essas alterações dificultam a aplicação direta de modelos treinados em cenas terrestres. A escassez de imagens subaquáticas com profundidade métrica confiável também limita a supervisão disponível para adaptar esses modelos @synthetic_bench_zijie.

  O trabalho investiga se um modelo compacto, o YOLO26n-Depth, pode melhorar suas estimativas subaquáticas por meio de ajuste fino com pseudorrótulos de profundidade. Os experimentos mantêm as mesmas imagens reais do USOD10K @usod10k e variam a fonte de supervisão entre os mapas distribuídos com o conjunto e as previsões de dois modelos pré-treinados, UniDepth V2 e Depth Anything V2. Esse desenho permite estudar diferentes formas de supervisão preservando as condições visuais das imagens subaquáticas.

  A avaliação usa o FLSea-VI, um conjunto independente com mapas de referência em escala métrica obtidos por fotogrametria @flsea. A comparação com o YOLO original permite verificar se o desempenho melhora nesse conjunto após o ajuste fino. A avaliação direta dos dois professores e do UDepth, um modelo especializado no domínio subaquático, contextualiza os resultados em relação a outras abordagens de estimação monocular.
]
