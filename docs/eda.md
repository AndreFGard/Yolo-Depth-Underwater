# EDA: USOD10K e FLSea

Resumo numérico da análise exploratória (notebooks em `notebooks/`). Amostras: 400 imagens por split no USOD10K
(sorteio com semente fixa); os 240 pares do manifesto `configs/flsea_eval_frames.csv` no FLSea.

## USOD10K (conversão COCO-depth do repositório)

* 10 255 imagens, todas 640×480; splits train/val/test = 7178 / 2051 / 1026 (70/20/10%).
* Labels: pseudo-profundidade (DPT) normalizada por imagem (min-max da disparidade `d`),
  `depth_rel = eps / (d + eps)`, `eps = 0.25`, gravada como uint16 (`× 65535`): faixa 13107–65535, maior = mais longe.
  **Não é métrica e não há escala comparável entre imagens.**

### Cor (média ± desvio entre imagens, 0–255)

| split | R | G | B | luminância | contraste | R/G | B/G |
|---|---|---|---|---|---|---|---|
| train | 74,7±41,2 | 113,0±38,0 | 113,2±47,7 | 101,6±32,7 | 32,8±14,7 | 0,69 | 1,03 |
| val | 79,8±42,1 | 111,7±35,4 | 109,8±45,5 | 101,9±31,4 | 33,8±15,0 | 0,74 | 1,01 |
| test | 76,8±42,5 | 113,1±36,8 | 114,0±46,6 | 102,4±32,3 | 33,0±14,9 | 0,70 | 1,04 |

Vermelho ≈ 26–31% menor que o verde; azul ≈ verde; splits comparáveis (a maior diferença, R de val, ≈ 1,7 erros padrão).

### Imagens extremas (1200 amostradas)

* R médio < 5 (canal vermelho praticamente zerado, aparência de imagem processada): 51 (4,3%).
* Luminância < 20: 6 (0,5%). Luminância > 235: 0.
* R/G alto: cenas com luz artificial ou objetos de cor própria (águas-vivas, peixes vermelhos).

### Labels

| split | min | max | mediana/img | p5/img | p95/img | faixa cheia |
|---|---|---|---|---|---|---|
| train | 13107 | 65535 | 27160±9237 | 15528 | 51071 | 100% |
| val | 13107 | 65535 | 27800±9457 | 15420 | 52443 | 100% |
| test | 13107 | 65535 | 27937±9554 | 15533 | 51503 | 100% |

Todas as imagens usam a faixa inteira (consequência do min-max por imagem). Distribuição assimétrica (massa na parte "perto"),
com pico isolado no extremo "longe" (65535). Consequência: avaliar apenas com alinhamento por imagem (escala e deslocamento);
o resultado mede concordância com o pseudo-label do DPT, não exatidão em metros.

## FLSea-VI (240 pares: 12 sequências × 20 quadros equiespaçados)

* Todas as imagens 608×968. Referência de profundidade em metros (SfM), esparsa.
* Pixels amostrados (2000 por quadro): mediana 2,26 m, p95 6,34 m, > 10 m: 1,2%, > 20 m: ~0,0%
  (o corte de 20 m usado na avaliação não afeta praticamente nenhum pixel).
* 2 quadros com < 2% de referência válida (descartados pelo avaliador: 238 avaliados).

| sequência | n | válido | prof. mediana (m) | p95 (m) | máx. (m) | > 10 m | > 20 m | R/G |
|---|---|---|---|---|---|---|---|---|
| canyons/flatiron | 20 | 0,855 | 2,018 | 5,108 | 20,408 | 0,011 | 0,000 | 0,345 |
| canyons/horse_canyon | 20 | 0,797 | 1,897 | 3,534 | 15,130 | 0,006 | 0,000 | 0,377 |
| canyons/tiny_canyon | 20 | 0,737 | 1,553 | 2,492 | 6,414 | 0,000 | 0,000 | 0,393 |
| canyons/u_canyon | 20 | 0,901 | 2,416 | 5,798 | 16,390 | 0,008 | 0,000 | 0,350 |
| red_sea/big_dice_loop | 20 | 0,687 | 2,774 | 7,554 | 36,646 | 0,020 | 0,003 | 0,330 |
| red_sea/coral_table_loop | 20 | 0,631 | 2,342 | 6,875 | 18,601 | 0,012 | 0,000 | 0,303 |
| red_sea/cross_pyramid_loop | 20 | 0,721 | 1,854 | 4,700 | 13,871 | 0,001 | 0,000 | 0,323 |
| red_sea/dice_path | 20 | 0,765 | 2,373 | 6,080 | 17,434 | 0,014 | 0,000 | 0,358 |
| red_sea/landward_path | 20 | 0,673 | 2,080 | 4,910 | 10,775 | 0,000 | 0,000 | 0,337 |
| red_sea/northeast_path | 20 | 0,753 | 3,293 | 9,171 | 29,606 | 0,050 | 0,002 | 0,354 |
| red_sea/pier_path | 20 | 0,829 | 2,467 | 6,050 | 24,779 | 0,015 | 0,000 | 0,398 |
| red_sea/sub_pier | 20 | 0,696 | 1,809 | 5,645 | 20,145 | 0,008 | 0,000 | 0,389 |

(`válido` = fração de pixels com referência; mediana e p95 são medianas por sequência dos valores por quadro; `R/G` = média das razões por quadro.)

Observações: alcance curto (mediana ~2 m); cerca de um quarto de cada imagem sem referência; R/G ≈ 0,30–0,40,
bem abaixo do USOD10K (≈ 0,7), o que caracteriza um deslocamento de domínio. As razões R/G dos dois datasets foram calculadas
de formas ligeiramente diferentes (razão das médias no USOD, média das razões no FLSea), mas a diferença é grande demais para ser artefato disso.
