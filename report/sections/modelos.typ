#let modelos = [
  = Modelos de Aprendizagem Profunda

  == YOLO26n-Depth

  O YOLO26n-Depth é a variante nano da família de estimação monocular da Ultralytics. Com 6,3 milhões de parâmetros, combina um extrator de características convolucional, uma pirâmide de características e um decodificador do tipo _Dense Prediction Transformer_ (DPT) para produzir um mapa denso de profundidade. A calibração do checkpoint determina a escala métrica da saída @yolodepth.

  O checkpoint público, pré-treinado em cerca de 2,19 milhões de imagens de cenas internas e externas, serve como referência _zero-shot_ e como ponto de partida para o ajuste fino @yolodepth. O trabalho avalia três versões ajustadas sobre as imagens do USOD10K, supervisionadas pelos mapas distribuídos com o conjunto ou pelos produzidos por cada um dos dois professores.

  == UniDepth V2

  O UniDepth V2 emprega, na variante ViT-L/14 usada no trabalho, um codificador visual do tipo Vision Transformer e um módulo que estima uma representação da câmera a partir da imagem para condicionar a predição. O modelo fornece profundidade métrica e atua como professor ao produzir pseudorrótulos densos para o YOLO. Também é avaliado diretamente no FLSea-VI com seus pesos pré-treinados @unidepthv2.

  == Depth Anything V2

  O Depth Anything V2 combina um codificador DINOv2 com um decodificador DPT para estimar profundidade densa @depthanythingv2. Na configuração relativa usada no trabalho, a saída é uma profundidade inversa relativa, em que valores maiores correspondem a pontos mais próximos. O modelo atua como segundo professor, oferecendo outra fonte de pseudorrótulos. A comparação direta no FLSea-VI inclui a variante Large.

  == UDepth-RMI

  O UDepth-RMI é uma referência especializada em imagens subaquáticas. A representação RMI combina o canal vermelho, o máximo entre os canais verde e azul e a luminância. O modelo usa um extrator MobileNetV2 e um Transformer compacto para estimar o mapa de profundidade @udepth. Neste trabalho, o checkpoint oficial é usado diretamente no FLSea-VI e sua saída é tratada como profundidade inversa relativa.

  == Comparação das saídas

  Para comparar as predições relativas e métricas no FLSea-VI, o protocolo ajusta escala e deslocamento no domínio da profundidade inversa usando a referência de cada imagem. Essa comparação avalia a estrutura dos mapas após o alinhamento. A avaliação sem alinhamento permite examinar a escala absoluta quando a saída é expressa em profundidade métrica.
]
