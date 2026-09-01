# Etapa 0 — Construção do `DataSet_tratado` e do *ground truth*

> Código: `Codigo/haxr_io.py`, `Codigo/etapa0_build.py`, `Codigo/tratado_io.py`,
> `Codigo/plots_eval.py`, `Codigo/etapa0_gt.py`, `Codigo/etapa0_figs.py`
> Saídas: `DataSet_tratado/`, `Tabelas/etapa0_*.csv`, `Figuras/etapa0_*.png`

---

## 1. Objetivo

O dataset original (HAXR — *Hamburg X-band Radar*, DOI
<https://zenodo.org/records/19824555>) contém **vídeo radar bruto** de 13 estações
de vigilância portuária do porto de Hamburgo e as **mensagens AIS** correspondentes.
A cena é densamente povoada por *clutter* fixo — cais, guindastes, margens do Elba,
embarcações atracadas com AIS desligado, boias — que produziria falsos positivos
espúrios se todo o campo de visão fosse avaliado contra um *ground truth* baseado
apenas em AIS.

A Etapa 0 resolve isso em duas frentes:

1. **Recorte** — preservar apenas as regiões de interesse (ROI) em torno de alvos
   AIS efetivamente visíveis, mantendo o *clutter* local dentro dessas regiões
   (indispensável para que as etapas seguintes tenham ruído real a filtrar);
2. **Ground truth** — determinar, para cada varredura, quais alvos AIS devem ser
   detectados e a qual *possível plot* cada AIS corresponde, resolvendo as
   ambiguidades com o **algoritmo húngaro** e confirmando o resultado com a
   **fusão entre estações de campo de visão sobreposto**.

---

## 2. Caracterização do dado bruto

| Grandeza | Valor |
|---|---|
| Estações | 13 (`stations.csv`, coordenadas cartesianas locais em metros) |
| Horas | 08, 09 e 11 UTC (3 arquivos por estação → 39 pares radar+AIS) |
| Formato do radar | HDF5 esparso: `az1, az2, r1, r2, amp, tod` por célula |
| Resolução em azimute | 0,0879° (4096 células por volta) |
| Resolução em distância | ≈ 3,0 m |
| Período de rotação | 2,9 s (≈ 20,6 rpm), medido pelos saltos de azimute |
| Amplitude | inteiro 8 bits, 1–255 (o vídeo já sai limiarizado do sensor) |
| AIS | `tod, uid, range (meters), azimuth (degrees)`, reamostrado a 1 Hz |

Duas observações determinam todo o desenho do pipeline:

* **O vídeo é esparso.** Só são armazenadas células com retorno; uma célula ausente
  tem amplitude 0. Dentro de uma ROI típica apenas ≈ 3 % das células têm retorno.
  Logo, a unidade natural de classificação é a **célula com retorno** ("*hit* de
  vídeo"), e não toda a grade polar.
* **O grupo `cycle` é uma janela deslizante**, com uma entrada a cada ≈ 0,15 s.
  Uma varredura completa corresponde a `[first[i], last[i]]`, mas ciclos
  consecutivos se sobrepõem quase inteiramente. `RadarFile.independent_cycles()`
  seleciona gulosamente ciclos separados por, no mínimo, um período de rotação,
  produzindo **uma varredura independente por volta de antena**.

### 2.1 Referencial cartesiano comum (validação)

O CSV de AIS já vem reprojetado no referencial **polar de cada estação**. Adotamos

$$X = x_{\text{estação}} + r\,\sin\theta, \qquad Y = y_{\text{estação}} + r\,\cos\theta$$

com θ em graus a partir do Norte, sentido horário. A convenção foi **validada
empiricamente**: para os 47 950 pares (`uid`, `tod`) observados simultaneamente por
duas ou mais estações na hora 08, a dispersão entre as posições reprojetadas é de
**10,2 m em X e 7,1 m em Y (mediana)** — compatível com a quantização do CSV
(1 m em distância, 0,01° em azimute). As 13 estações compartilham, portanto, um
único referencial métrico, o que viabiliza a fusão da Seção 5.

---

## 3. Envelope de cobertura e máscara de visibilidade

Nem todo alvo AIS listado no CSV é fisicamente visível pela estação: há setores
cegos, sombreamento por cais e edificações, e alvos além do alcance efetivo.
Tratá-los como *ground truth* geraria falsos negativos que **nenhum** detector
poderia evitar.

Explorando a esparsidade do vídeo, estimamos por estação o **envelope de alcance
por setor de azimute**:

```
r_max(θ) = max sobre 150 varreduras da maior distância com retorno no setor θ
           (720 setores de 0,5°, seguido de fechamento morfológico de ±1,5°)
```

Chamamos a curva $r_{\max}(\theta)$ de **envelope de cobertura** da estação — ela
é o horizonte de sombreamento imposto pela costa, pelos cais e pelas edificações
— e o teste $r \le r_{\max}(\theta)$ de **máscara de visibilidade**. Um alvo AIS
é declarado **visível** quando satisfaz esse teste. Os dois nomes correspondem às
funções `coverage_profile()` e `visible_mask()` de `Codigo/haxr_io.py`.

**Validação do critério** (amostra de 221 alvos-varredura em 5 estações):

| Critério | n | frac. com eco radar a menos de 50 m |
|---|---|---|
| visível ($r \le r_{\max}$) | 71 | **95,8 %** |
| não visível | 150 | **0,7 %** |

A separação é praticamente binária, o que confirma que $r_{\max}(\theta)$ captura a
geometria real de visibilidade. A Figura `etapa0_fig1_cobertura.png` mostra os 13
envelopes no referencial comum, evidenciando as amplas regiões de sobreposição.

---

## 4. Recorte: geração do `DataSet_tratado`

`Codigo/etapa0_build.py`. Para cada par (estação, hora):

1. estima-se o período de rotação e o envelope $r_{\max}(\theta)$;
2. selecionam-se **2 janelas de 120 s** uniformemente distribuídas na hora
   (a duração de 2 min é a exigida pela extração de cinemática das Etapas 4 e 6);
3. dentro de cada janela tomam-se as varreduras independentes (≈ 41 por janela);
4. **correção do instante de iluminação**: a antena leva 2,9 s por volta, e um navio
   a 10 kn percorre ≈ 15 m nesse intervalo. Para cada alvo, resolve-se por iteração
   de ponto fixo o instante $t_{\text{feixe}} = t_0 + \frac{(\theta-\theta_0)\bmod 360}{360}T_{\text{rot}}$
   em que o feixe cruza o azimute do alvo, e a posição AIS é interpolada
   (e extrapolada pela velocidade local) para esse instante;
5. mantêm-se **apenas as células a menos de $R_{\text{ROI}} = 200$ m de algum alvo
   visível**; todo o restante é descartado;
6. grava-se `DataSet_tratado/<estação>_<hora>-UTC_tratado.hdf5`.

**Escolha de $R_{\text{ROI}} = 200$ m.** O disco de 200 m acomoda a maior embarcação
que trafega em Hamburgo (≈ 400 m de comprimento) em torno da antena AIS e ainda
preserva uma coroa de *clutter* e ruído em volta do alvo — sem essa coroa não
haveria falsos positivos a medir. A ROI é ≈ 30× maior em área que o eco típico de
um navio.

### Resultado do recorte

| | |
|---|---|
| Varreduras preservadas | **3 138** |
| Células de vídeo preservadas | **57 695 013** |
| Pares (alvo AIS, varredura) | **37 557** |
| Embarcações distintas (`uid`) | **340** |
| Tamanho em disco | 78 MB (contra 5,1 GB do bruto) |

Detalhamento por estação: `Tabelas/etapa0_dataset_tratado.csv`.
A Figura `etapa0_fig2_recorte.png` compara a varredura completa com o recorte.

### Formato do arquivo tratado

```
attrs : station, hour, x0, y0, roi_radius_m, rot_period_s, r0, dr, n_r, n_az, az_step
scan/ : cycle, t0, t1, t, window, first, last      (uma linha por varredura)
cell/ : az_idx (uint16), r_idx (uint16), amp (uint8), tod  (células preservadas)
tgt/  : scan, uid, X, Y, r, az, vx, vy             (alvos AIS visíveis por varredura)
coverage_prof : r_max(θ), 720 setores
```

`Codigo/tratado_io.py` fornece a classe `Tratado`, com acesso às células, aos alvos,
à rasterização polar densa (`raster`) e à máscara de ROI (`roi_mask`).

---

## 5. Ground truth

`Codigo/etapa0_gt.py`.

### 5.1 Possíveis plots

De cada varredura extraem-se os **possíveis plots** diretamente do vídeo recortado,
sem qualquer filtragem: as células são rasterizadas em uma grade métrica de 5 m e
rotuladas por **componentes conexas** (conectividade 8); cada componente com ≥ 3
células vira um plot, com centroide ponderado pela amplitude
(`plots_eval.extract_plots`). É a hipótese mais permissiva possível — e é
exatamente o extrator de plots usado por **todas as seis propostas**, de modo que a
comparação entre elas isole apenas o efeito do filtro e do clusterizador.

O resultado quantifica o problema a ser resolvido:

| | mediana | média | máximo |
|---|---|---|---|
| alvos AIS visíveis por varredura | 10 | 12,0 | 39 |
| possíveis plots por varredura | 71 | 68,4 | 189 |
| razão possíveis-plots / alvos | **6,0** | | |

95 % das varreduras contêm mais de um alvo AIS e **100 % contêm mais de um possível
plot** — a atribuição é, portanto, sempre ambígua e exige solução global.

### 5.2 Atribuição pelo algoritmo húngaro

Em cada varredura monta-se a matriz de custo $C_{ij} = \lVert p_i - a_j\rVert$ entre
possíveis plots $p_i$ e alvos AIS visíveis $a_j$, com porta de 100 m (pares acima
da porta recebem custo proibitivo). A atribuição ótima é obtida por
`scipy.optimize.linear_sum_assignment` (algoritmo húngaro / Kuhn–Munkres).

O uso da atribuição **global** — em vez do vizinho mais próximo — é essencial: com
6 possíveis plots por alvo em média, dois AIS vizinhos disputariam o mesmo eco e a
escolha gulosa produziria associações inconsistentes. A Figura
`etapa0_fig3_hungaro.png` mostra uma cena de `koehlbrandhoeft` com 102 possíveis
plots, 32 alvos AIS e as 27 atribuições resolvidas pelo algoritmo.

### 5.3 Confirmação por fusão multiestação

As 13 estações se sobrepõem: **74,3 %** dos alvos-varredura são vistos
simultaneamente por 2 ou mais estações (até 8). Para cada `uid`, os ecos atribuídos
por estações distintas em janelas de ±1,6 s são levados ao referencial comum e
comparados. Se dois ou mais ecos independentes concordam dentro de 60 m, a
atribuição é declarada **confirmada por fusão** — uma verificação que não depende de
nenhuma estação isolada.

| Status | Alvos-varredura | % |
|---|---|---|
| `confirmado_fusao` — eco atribuído e corroborado por ≥ 2 estações | 21 021 | 56,0 % |
| `confirmado_local` — eco atribuído por uma única estação | 13 865 | 36,9 % |
| `nao_confirmado` — nenhum eco a menos de 100 m | 2 671 | **7,1 %** |

**Concordâncias medidas** (Figura `etapa0_fig4_offsets.png`):

* deslocamento AIS → eco atribuído: **mediana 28,5 m**, p90 **69,7 m**;
* discordância entre estações sobrepostas para o mesmo alvo: **mediana 26,2 m**.

A Figura `etapa0_fig6_fusao.png` mostra um alvo observado por 5 estações
simultaneamente, com dispersão mediana de 50 m entre os ecos atribuídos.

### 5.4 Definição adotada e porta de avaliação

* **Posição de *ground truth*** = posição AIS interpolada para o instante de
  iluminação do feixe. Deliberadamente **não** se usa o centroide do eco atribuído:
  isso tornaria o *ground truth* dependente do mesmo extrator de plots que alimenta
  os detectores avaliados, introduzindo circularidade.
* **Conjunto de *ground truth*** = todos os alvos AIS **visíveis** da varredura
  (Seção 3), independentemente de terem ou não eco. Os 7,1 % `nao_confirmado`
  permanecem no *ground truth* e constituem um **piso irredutível de falsos
  negativos**: o *recall* máximo alcançável por qualquer método é ≈ **0,929**.
* **Porta de associação** para avaliação: **100 m**, justificada pelo p90 = 69,7 m
  do deslocamento AIS → eco (a antena AIS não coincide com o centro de gravidade do
  eco radar, sobretudo em navios longos).

A atribuição húngara e a fusão multiestação são registradas em
`DataSet_tratado/ground_truth.csv` (37 557 linhas) e usadas para **caracterizar e
validar** o *ground truth*, além de fornecer a análise de sensibilidade da Etapa 7
restrita aos alvos confirmados.

### 5.5 Partição treino / teste

Para as etapas supervisionadas (ClusWiSARD) adota-se uma **partição temporal
estrita**, evitando qualquer vazamento entre varreduras vizinhas:

| Partição | Horas | Alvos-varredura | Varreduras |
|---|---|---|---|
| Treino | 08 UTC | 11 693 | 1 050 |
| Teste (métricas reportadas) | 09 e 11 UTC | 25 864 | 2 088 |

Os métodos não supervisionados (OS-CFAR, DBSCAN) são aplicados **exatamente ao mesmo
conjunto de teste**, de forma que todas as seis propostas sejam comparadas sobre os
mesmos dados.

---

## 6. Referência das figuras e tabelas

| Arquivo | Conteúdo |
|---|---|
| `Figuras/etapa0_fig1_cobertura.png` | envelopes $r_{\max}(\theta)$ das 13 estações e sobreposição |
| `Figuras/etapa0_fig2_recorte.png` | vídeo bruto completo × `DataSet_tratado` |
| `Figuras/etapa0_fig3_hungaro.png` | atribuição húngara possíveis-plots ↔ AIS |
| `Figuras/etapa0_fig4_offsets.png` | deslocamento AIS→eco e discordância entre estações |
| `Figuras/etapa0_fig5_confirmacao.png` | status de confirmação do GT por estação |
| `Figuras/etapa0_fig6_fusao.png` | fusão multiestação de um mesmo alvo |
| `Tabelas/etapa0_dataset_tratado.csv` | varreduras/células/alvos por (estação, hora) |
| `Tabelas/etapa0_gt_por_estacao.csv` | estatísticas do GT por estação |
| `Tabelas/etapa0_gt_global.csv` | resumo global do GT |
| `Tabelas/etapa0_por_varredura.csv` | nº de alvos e de possíveis plots por varredura |
| `DataSet_tratado/ground_truth.csv` | *ground truth* completo, com atribuição e status |
