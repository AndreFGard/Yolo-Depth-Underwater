#let modelos = [
  = Modelos de Aprendizagem Profunda

  Além do YOLO-Depth, dois modelos foram selecionados para atuar como professores e produzir os rótulos sintéticos. Considerando o seu ótimo desempenho _zero-shot_ no FLSea-VI @synthetic_bench_zijie, superando todos os modelos específicos do domínio submarino avaliados, foram escolhidos para este fim o unidepthV2@unidepthv2 e o Depth-Anything V2 @depthanythingv2. Considerando a base de dados utilizada e para fim de comparações com um modelo treinado inteiramente no domínio submarino, também incluimos nas avalições o UDepth-RMI@udepth, uma referência no uso do USOD10K e dos seus labels sintéticos para MDE.

  == YOLO26n-Depth

  O YOLO26n-Depth é a variante nano da família de estimação monocular da Ultralytics. Com 6,3 milhões de parâmetros, combina um extrator de características convolucional, uma pirâmide de características e um decodificador do tipo _Dense Prediction Transformer_ (DPT) para produzir um mapa denso de profundidade. A calibração do checkpoint determina a escala métrica da saída @yolodepth.

  O checkpoint público, pré-treinado em cerca de 2,19 milhões de imagens de cenas internas e externas, serve como referência _zero-shot_ e como ponto de partida para o ajuste fino @yolodepth. Assim, esse checkpoint será utilizado como base para o finetuning e testes deste trabalho
  
  === Função de perda de profundidade

  No ajuste fino, o YOLO26n-Depth usa o critério `DepthLoss26` da Ultralytics @yolodepth. A função combina o erro logarítmico invariante à escala (SILog) @eigen com a correspondência entre gradientes espaciais. Com os pesos padrão `dlog = 1,0` e `dgrad = 0,5`, a perda para um lote de $B$ imagens é
  $ L = B (L_"SILog" + 0.5 L_"grad"). $

  Quando necessário, a predição é redimensionada por interpolação bilinear para a resolução do rótulo. São válidos os pixels com profundidade de referência maior que 0,001. A predição recebe um piso de 0,001 antes do logaritmo. Lotes com menos de dez pixels válidos têm perda zero.

  *SILog.* Para cada pixel válido $i$, o resíduo $e_i = ln(hat(z)_i) - ln(z_i)$ é a diferença entre os logaritmos da profundidade predita e do rótulo. Com a média $m$ e a variância $op("Var")(e)$ calculadas sobre todos os pixels válidos do lote, o termo é
  $ L_"SILog" = sqrt(op("Var")(e) + (1 - lambda) m^2 + 10^(-6)). $
  Com $lambda = 1$ (padrão de `dlam`), resta a variância dos resíduos sob a raiz, mais a constante de estabilidade numérica. Acima do piso usado no logaritmo, multiplicar todas as predições do lote por um mesmo fator $c > 0$ soma $ln(c)$ a cada resíduo. A centralização pela média cancela essa constante. Fatores distintos por imagem podem alterar a variância do lote.

  *Gradientes.* O termo $L_"grad"$ usa diferenças de log-profundidade entre pixels vizinhos nas direções horizontal e vertical. Para cada direção, calculamos o erro absoluto médio entre as diferenças da predição e as do rótulo. Pares com um pixel inválido contribuem com zero. Somamos esses erros em até quatro resoluções, reduzindo os mapas em blocos de $2 times 2$ pela média dos pixels válidos. Acima do piso numérico, um fator de escala uniforme por imagem gera um deslocamento nos logaritmos que se cancela nas diferenças. Assim, o termo favorece a preservação dos contornos e das variações locais de profundidade.


  === Calibração

  A cabeça do modelo YOLO incorpora dois valores, $a$ e $b$, aplicados apenas no modo de avaliação: a predição bruta $d$ sai como $d^a dot e^b$. Depois do treinamento, um ajuste em forma fechada, por mínimos quadrados no espaço logarítmico, escolhe esses valores sobre um conjunto rotulado. Por padrão, $a$ é fixo e apenas a escala é ajustada. Esse mecanismo permite que o YOLO produza profundidade métrica, com a calibração padrão tendo sido obtida durante nas grandes bases de dados usadas no seu treinamento.
  Essa calibração é ajustada no modelo automaticamente pela biblioteca com base nos dados do treinamento, inclusive no caso do finetune deste trabalho.

  == UniDepth V2

  O UniDepth V2 emprega, na variante ViT-L/14 usada no trabalho, um codificador visual do tipo Vision Transformer e um módulo que estima uma representação da câmera a partir da imagem para condicionar a predição. O modelo fornece profundidade métrica e atua como professor ao produzir pseudorrótulos densos para o YOLO. Também é avaliado diretamente no FLSea-VI com seus pesos pré-treinados @unidepthv2.

  == Depth Anything V2
  O Depth Anything V2 combina um codificador DINOv2 com um decodificador DPT para estimar profundidade densa @depthanythingv2. Na configuração relativa usada no trabalho, a saída é uma profundidade inversa relativa, em que valores maiores correspondem a pontos mais próximos. O modelo atua como segundo professor, oferecendo outra fonte de pseudorrótulos. A comparação direta no FLSea-VI inclui a variante Large.
  == UDepth-RMI

  O UDepth-RMI é a referência especializada em imagens subaquáticas @udepth. A entrada é no espaço RMI, que combina o canal vermelho, o máximo entre os canais verde e azul e a luminância. A luz vermelha é a primeira a se atenuar debaixo d'água, então a diferença entre o canal vermelho e o máximo de verde e azul carrega informação de profundidade, enquanto a luminância preserva o conteúdo estrutural @udepth.

  O modelo foi treinado de forma supervisionada nos pares RGB-D do USOD10K @udepth, sua saída usa o mesmo formato que os rótulos do conjunto.
]
