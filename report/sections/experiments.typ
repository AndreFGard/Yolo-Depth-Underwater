#let experiments = [
  = Experimentos
  Nossos experimentos incluem ajustes do YOLO26 com três fontes de rótulos: os mapas fornecidos com o USOD10K e os pseudo-rótulos produzidos pelo UniDepth e pelo Depth Anything V2. Para avaliar o efeito da taxa de aprendizado, também repetimos o ajuste com o Depth Anything V2 usando uma taxa menor. Além disso, comparamos as predições dos modelos no FLSea.

  == Métricas

  Adotamos principalmente o conjunto de métricas consolidado por @eigen, que permanece uma referência @maudepth@synthetic_bench_zijie e permite a comparação com trabalhos relacionados. Assim, reportaremos as métricas AbsRel, RMSE, $delta^1$ e Silog. Em todas as métricas, $z_i$ é a referência, $hat(z)_i$ a predição e $N$ o número de pixels válidos do quadro.

  Conforme elaborado em @eigen, a escala, que é fundamentalmente ambígua nas cenas, é sozinha responsável por uma grande parcela do erro das métricas, o que pede também métricas que meçam a qualidade estrutural dos mapas, dissociada da escala.

  === AbsRel
  A diferença absoluta relativa (*AbsRel*) quantifica o erro em razão da magnitude do ground truth, evitando que erros pequenos comparados à profundidade de um ponto distorçam a métrica @eigen.

  $ "AbsRel" = frac(1, N) sum_i frac(|hat(z)_i - z_i|, z_i) $

  === RMSE
  A *RMSE* quantifica o erro na unidade de profundidade, o que a faz de interpretação complexa no contexto do estudo e a ser dominada por erros em profundidades maiores, independente da qualidade da predição fina em objetos mais próximos.

  $ "RMSE" = sqrt(frac(1, N) sum_i (hat(z)_i - z_i)^2) $

  === $delta^1$
  _*Threshold accuracy*_ (*$delta^1$*) é uma das métricas mais amplamente utilizadas no campo @eigen@synthetic_bench_zijie e mede a porcentagem de pontos que distam do ground truth dentro de uma tolerância: a porcentagem de pixels que satisfazem $max(z_i, hat(z)_i) < 1.25 min(z_i, hat(z)_i)$.

  === Silog
  A *Silog* é a RMSE trasladada para o espaço logarítmico e corrigida da escala. Com $g_i = ln hat(z)_i - ln z_i$ e $m$ a média dos resíduos,

  $ "SILog" = 100 sqrt(frac(1, N) sum_i (g_i - m)^2). $

  Subtrair $m$ equivale, na forma original de @eigen, a multiplicar a predição pela escala $alpha = exp(-m)$ que melhor a alinha à referência. Um erro global de escala desloca todos os $g_i$ igualmente e some da variância: o que sobra mede a qualidade estrutural do mapa. É a mesma Silog que treina o YOLO (seção de modelos), aqui no papel de métrica.

  == Ajuste fino do YOLO com rótulos sintéticos

  Fizemos três _finetunes_ do YOLO26n-Depth com as imagens do USOD10K e o mesmo split original do dele, com 7.178 imagens de treino, 2.051 de validação e 1.026 de teste. Fizemos dois datasets com as imagens do USOD10K, mas com os mapas de profundidade produzidos por um modelo professor cada, o Depth Anything V2 (ViT-L)@depthanythingv2 e o Unidepth V2@unidepthv2, com o terceiro finetuning sendo feito inteiramente com o USOD10K original. Assim, referimo-nos a esses modelos como YOLO-DA, YOLO-Unidepth e YOLO-USOD, respectivamente.

  Só os mapas mudaram entre as execuções, e garantimos que todos codificassem profundidade relativa ou métrica, para possibilitar o treinamento. Devido a restrições computacionais, não fizemos uma tunagem de hiperparâmetros, respeitando os padrões estabelecidos pela biblioteca dos desenvolvedored do YOLO. Foram usadas 80 épocas, imagens em 640 px, _mini-batch_ de 16, backbone congelado (os 11 primeiros módulos@yolodepth), warmup de 3 épocas e decaimento linear. Também usamos o AdamW, com taxa de aprendizagem inicial de 0,002, o padrão.

  === Rótulos e conversões
  Para adequar os mapas de profundidade à profundidade relativa ou métrica exigida pela modelagem do YOLO, foi necessário adotar um método de conversão para os modelos que não produziam mapas nativamente neste formato.

  *USOD10K (DPT)*: Os mapas que acompanham o USOD10K são estimativas de um DPT, armazenadas como disparidade relativa normalizada por imagem, com valores maiores significando pontos mais próximos @usod10k @ranftl2021dpt (detalhes na análise exploratória, @sec:eda). Para obter a profundidade relativa, aplicamos a relação:

  $ z = frac(epsilon, d + epsilon), quad epsilon = 0.25, $

  O termo $epsilon$ evita a divisão por zero. Para $d$ entre 0 e 1, temos que $z in [1.0, 0.2]$, na direção do ponto mais distante para o mais próximo.


  *Depth Anything V2 Large (ViT-L)*: Os mapas de  são disparidade não normalizada, então os normalizamos e os convertemos para profundidade relativa @depthanythingv2.

  *UniDepth V2 (ViT-L).* Usamos diretamente a saída métrica em metros @unidepthv2, gravada em milímetros.

  Como a perda de treino é invariante a um fator global de escala (seção de modelos), a escala escolhida para cada conjunto não afeta o aprendizado dos modelos, apenas a geometria e estrutura da profundidade.

  === Calibração dos checkpoints do YOLO

  A cabeça do YOLO aplica, apenas em avaliação, a calibração do checkpoint, $z_"cal" = exp(a ln z + b)$, com os coeficientes globais `cal_a` e `cal_b` (seção de modelos). Nos checkpoints deste estudo $a = 1$, de modo que a calibração se reduz a multiplicar o mapa por $exp(b)$.

  Para cada ajuste fino, usamos o melhor modelo selecionado pela validação do respectivo conjunto de validação. Os rótulos dessa mesma validação servem para escolher a calibração, feita automaticamente pela biblioteca do YOLO e incorporada ao modelo, o que faz sentido no caso do Unidepth, mas, devido à natureza dos rótulos, não deve ser considerado no caso dos outros dois datasets.


  == Protocolo de avaliação

  Devido à ambiguidade intrínseca da escala da profundidade e à variedade de modelos utilizados, optamos por um sistema de avaliação flexível, em que cada predição passa por três alinhamentos alternativos, e as métricas são calculadas sobre cada uma. No protocolo raw, avaliamos apenas modelos com saída em escala métrica, sem alinhar as predições à referência de cada quadro do FLSea-VI. O YOLO Baseline usa a calibração do checkpoint público, enquanto o YOLO-UniDepth usa a calibração ajustada no conjunto de validação do ajuste fino e armazenada no checkpoint selecionado.

  A avaliação externa usa o FLSea-VI, conforme explicado. Avaliamos as 6 sequências usadas por @synthetic_bench_zijie, resultando em 120 quadros. Dessa maneira, todos os modelos são avaliados no FLSea de forma _zero-shot_, enquanto os efeitos do finetuning do YOLO com os rótulos sintéticos podem ser avaliados com dados reais e de alta qualidade quanto ao ganho de capacidade fora do seu domínio de treino.

  - *Sem alinhamento (`raw`).* A profundidade do modelo é usada diretamente, limitada ao intervalo de 0,01 a 20 m, seguindo o padrão da Ultralytics. Aplica-se apenas a modelos que produzem profundidade métrica e é a linha mais facilmente comparável com a literatura.

  - *Alinhamento por imagem (`ssi`)*: Em cada quadro ajustamos a escala $s$ e o deslocamento $t$ por mínimos quadrados entre $s p + t$ e a referência inversa $1/z_"ref"$, nos pixels válidos @midas, e limitamos a profundidade a 20 m ao inverter:
    $ z_"SSI" = frac(1, max(s p + t, 1/20)). $

  - *Alinhamento por mediana (`median`)*: Só é ajustada a escala, pela razão entre as medianas da referência e da predição. Os dois alinhamentos por imagem são oráculos que usam a referência do próprio quadro, então medem a estrutura, apenas. Diferem, porém, no escopo de defeitos que podem ser detectados, pois a transformação afim do SSI de fato pode se ajustar a erros que vão além da premissa de robustez à escala.

  Assim, pesar de as métricas permitirem comparações em algum grau entre modelos com premissas diferentes, os alinhamentos têm requisitos de oráculo distintos, e nossas conclusões têm escopo adequado a essas condições. Reportamos a média aritmética das métricas por quadro.

  == Estudo comparativo das baselines em zero-shot

  Com esse protocolo, avaliamos quatro modelos sem nenhum treino ou ajuste no FLSea, com seus pesos oficiais: o UniDepth V2 (ViT-L), o Depth Anything V2 Large (ViT-L) usados em @synthetic_bench_zijie, o YOLO26n-Depth baseline e o UDepth-RMI.

  == Estudo do efeito do finetune o época a época

  Para entender o efeito do finetune ao longo do treinamento, fazemos, posteriormente ao seu treino, avaliações dos seus passos intermediários. Os pesos são salvos a cada duas épocas, e pontuamos no FLSea 25 destes pontos, aproximadamente a cada duas épocas, e comparamos cada ponto com o baseline. A leitura principal usa o SSI e a mediana, que removem a escala, e incluímos também a $delta^1$ de validação registrada durante o treino, que mostra a especialização ao domínio de treinamento.

]
