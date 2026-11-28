#let objectives = [
  = Objetivos
  A estimação de profundidade submarina é de extrema importância para operações industriais e de veículos autônomos não tripulados, e a estimação monocular de profundidade (MDE) é interessante em aplicações com recursos limitados que dificultam outras técnicas, como visão estéreo, além de fornecer mapas densos, ao contrário de técnicas baseadas em _feature matching_@udepth. No entanto, um desafio frequente é a baixa variedade de conjuntos de dados com profundidade confiável e de tamanho suficiente para o treinamento destes modelos nesse cenário, sendo frequente o uso de dados sintéticos para contornar esse problema. Além disso, em aplicações reais, em que dados com profundidade reais são raros, rótulos sintéticos de profundidade possibilitam a adaptação de imagens verdadeiras do domínio da aplicação para o possível ajuste de modelos.

  Em contraste, modelos fundacionais terrestes obtiveram ótimo desempenho através do treinamento em datasets muito grandes e diversos @unidepthv2@depthanythingv2@depthanythingv3, como o NYU Depth v2 @nyuv2 e o KITTI @kitti, de tal maneira que conseguem superar até modelos  específicos de forma _zeroshot_ em datasets submarinos @synthetic_bench_zijie. Nesse sentido, o lançamento da família YOLO26-Depth @yolodepth, também beneficiada por essa disponibilidade de dados, tornou o campo de MDE mais acessível e, particularmente, de mais fácil implantação, como resultado da alta eficiência e ecossistema de desenvolvimento maduro ao redor desses modelos que, nas suas modalidades de segmentação e detecção de objetos, são notórios pela facilidade de finetuning para domínios diferentes.

  Dessa maneira, esse trabalho procura avaliar se é possível melhorar o desempenho de um modelo computacionalmente eficiente da família YOLO26 para o cenário de MDE submarina, com imagens reais, e rótulos sintéticos produzidos por modelos mais capazes, adotando-os como professores.


  // Image block
  // #figure(
  //   image("imagens/sample underwater.png", width: 80%),
  //   caption: [Example underwater scene from the dataset.],
  // ) <fig:sample>
]
