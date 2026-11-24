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
  Ajustamos o YOLO26n-Depth com as imagens RGB e as divisões de treino, validação e teste do USOD10K. Em cada conjunto de treino, usamos um mapa de profundidade por imagem, produzido a partir dos rótulos originais do USOD10K ou das predições de um dos dois professores.

  Devido a restrições computacionais, não fizemos uma busca ampla de hiperparâmetros. Os ajustes duraram 80 épocas, usaram AdamW com decaimento linear e mantiveram congelados os 11 primeiros módulos do modelo. A taxa de aprendizado inicial variou entre as execuções: 0,01 para USOD10K e para um dos ajustes com Depth Anything V2; 0,002 para UniDepth; e 0,0005 para a repetição com Depth Anything V2.

  === Escala e conversão dos mapas de treino

  A disparidade relativa tem valores maiores nos pontos mais próximos. Para obter mapas em que valores maiores indicam pontos mais distantes, convertemos a disparidade normalizada $d$ em profundidade relativa $z_"rel"$ pela relação
  $ z_"rel" = frac(epsilon, d + epsilon), quad epsilon = 0.25. $
  O termo $epsilon$ evita a divisão por zero. Para $d$ entre 0 e 1, $z_"rel"$ varia de 1 a 0,2 na direção do ponto mais distante para o mais próximo. Gravamos o valor arredondado de $65535 z_"rel"$ como inteiro de 16 bits no PNG.

  *USOD10K (DPT).* Os mapas originais já armazenam disparidade relativa normalizada por imagem @usod10k @ranftl2021dpt. O conversor divide cada valor de 16 bits por 65.535 para obter $d$ e aplica a inversão acima.#footnote[Implementação no #link("https://github.com/AndreFGard/Yolo-Depth-Underwater/blob/main/scripts/convert_usd10k_to_coco.py")[conversor do USOD10K].]

  *Depth Anything V2 Small (ViT-S).* Para cada imagem, subtraímos o mínimo da disparidade predita e dividimos pela diferença entre o máximo e o mínimo, com piso de $10^(-9)$ no denominador. A mesma inversão gera pseudo-rótulos com a faixa de 13.107 a 65.535 usada nos PNGs convertidos do USOD10K. A geração com o Small foi registrada para as três divisões do dataset @depthanythingv2.#footnote[Código e execução no #link("https://github.com/AndreFGard/Yolo-Depth-Underwater/blob/main/notebooks/eda_usod10k_flsea.ipynb")[notebook de geração dos mapas].]

  *UniDepth V2 (ViT-L).* Usamos diretamente a saída de profundidade $z$ em metros @unidepthv2. Para gravar o PNG de 16 bits, multiplicamos $z$ por 1.000, arredondamos e limitamos os valores válidos à faixa de 1 a 65.535, com teto de 65,535 m. O valor zero representa pixels inválidos.

  O carregador divide os valores dos PNGs por 1.000, definido como `depth_scale` no YAML. Essa divisão recupera os metros com a quantização do PNG para os rótulos do UniDepth. Para USOD10K e Depth Anything V2, os valores carregados continuam em unidades relativas a cada imagem.

  === Calibração dos checkpoints do YOLO

  A cabeça do YOLO aplica à profundidade $z$ a calibração do checkpoint, $z_"cal" = exp(a log(z) + b)$, com os coeficientes globais `cal_a` e `cal_b` @yolodepth. Nos checkpoints `best.pt` registrados no estudo por época, $a = 1$, de modo que a calibração multiplica o mapa por $exp(b)$.

  Para cada ajuste fino, usamos o `best.pt` selecionado pela validação do respectivo conjunto de treino. Os rótulos dessa mesma validação servem para escolher a calibração. O YOLO-UniDepth é calibrado com rótulos em metros. O YOLO-DA e o YOLO-USOD são calibrados com rótulos relativos. O FLSea é reservado à avaliação externa.

  == Protocolo de avaliação

  A referência do FLSea é dada em metros @flsea. Redimensionamos a predição para a grade da referência com interpolação bilinear quando necessário. Calculamos as métricas nos pixels com referência finita e positiva.

  *Sem alinhamento (_raw_).* Avaliamos a profundidade fornecida pelo modelo, limitada à faixa de 0,01 a 20 m, sem ajustar a escala ao quadro do FLSea. Nas tabelas comparativas, reportamos _raw_ para o UniDepth V2, o YOLO26n-Depth pré-treinado e o YOLO-UniDepth, que fornecem profundidade em escala métrica.

  *Alinhamento por imagem (_SSI_).* Para YOLO e UniDepth, obtemos a profundidade inversa $p = 1 / max(z, 0.01)$. Para Depth Anything V2 e UDepth-RMI, partimos da saída relativa em profundidade inversa. Em cada quadro, ajustamos a escala $s$ e o deslocamento $t$ por mínimos quadrados entre $s p + t$ e a referência inversa $1 / z_"ref"$, usando os pixels válidos. Limitamos a profundidade a 20 m ao aplicar um piso de $1/20$ à disparidade alinhada e inverter o resultado para obter
  $ z_"SSI" = frac(1, max(s p + t, 1 / 20)). $
  As métricas SSI medem a estrutura recuperada após o ajuste à referência do próprio quadro. O avaliador também registra o alinhamento pela mediana, que usa a razão entre as medianas das profundidades inversas da referência e da predição como $s$, com $t = 0$.#footnote[Implementação em #link("https://github.com/AndreFGard/Yolo-Depth-Underwater/blob/main/src/yolo_depth_underwater/metrics.py")[metrics.py].]

  == Estudo comparativo das baselines em zero-shot

  Comparamos o YOLO26n-Depth pré-treinado, o UniDepth V2 (ViT-L), o Depth Anything V2 Large (ViT-L) e o UDepth-RMI nos mesmos quadros do FLSea. A disparidade predita pelo Large entra diretamente no alinhamento SSI.

  No protocolo do UDepth-RMI @udepth, tratamos a saída $u$, redimensionada por interpolação bilinear, como profundidade inversa relativa. Aplicamos o piso $p = max(u, 10^(-6))$ antes do SSI. Para exportar o mapa relativo, gravamos $1 / p$ em NPY de 32 bits em ponto flutuante, sem normalização por imagem. A exportação de avaliação em PNG para UDepth-RMI e Depth Anything V2 usa o mapa obtido por SSI, que está em metros por ter sido alinhado à referência do FLSea.

  == Estudo do efeito do finetuning época a época


]
