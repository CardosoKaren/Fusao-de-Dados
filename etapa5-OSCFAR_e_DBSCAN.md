# Etapa 5 — Proposta 5: filtro OS-CFAR + clusterizador DBSCAN

> Código: `Codigo/etapa5_oscfar_dbscan.py` (usa `Codigo/cluster_common.py`)
> Entrada: `Resultados/etapa2_oscfar_plots.csv.gz` (159 728 plots)
> Saídas: `Resultados/etapa5_*`, `Tabelas/etapa5_*`, `Figuras/etapa5_*`

---

## 1. Objetivo

Contraparte clássica da Etapa 3: agrupar os plots primitivos aprovados pelo
OS-CFAR com o **DBSCAN** (Ester et al., 1996), de modo que uma embarcação longa
— que produz várias manchas de eco — resulte em um único plot.

## 2. Método

Para cada varredura, DBSCAN sobre as coordenadas $(x, y)$ dos plots primitivos,
com `min_samples = 1`: todo plot pertence a algum agrupamento, pois a rejeição de
ruído já foi feita pelo filtro. Cada agrupamento é substituído por um único plot
no centroide ponderado pela amplitude — **exatamente a mesma função de fusão**
usada na Etapa 3 (`cluster_common.merge_by_label`), de modo que a comparação
isole o critério de vizinhança.

O DBSCAN é a escolha canônica aqui: não exige o número de grupos, trata grupos de
formato arbitrário (o eco de um navio é alongado) e sua vizinhança é
explicitamente métrica — o mesmo papel que `min_score` desempenha no
clusterizador ClusWiSARD, mas com raio euclidiano em vez de $L_1$.

## 3. Ajuste

$\varepsilon$ varrido em $\{15, 25, 35, 45, 60, 80, 100, 130\}$ m na **janela de
validação** (janela 1 da hora 08), maximizando o F1 de plots — mesmo critério e
orçamento do `min_score` da Etapa 3 (`Tabelas/etapa5_ajuste_dbscan.csv`,
Figura `etapa5_fig2_ajuste.png`).

| $\varepsilon$ | precisão | recall | F1 | plots/alvo |
|---|---|---|---|---|
| 15 m | 0,190 | 0,773 | 0,306 | 4,06 |
| 25 m | 0,209 | 0,768 | 0,329 | 3,67 |
| 35 m | 0,234 | 0,755 | 0,358 | 3,22 |
| 45 m | 0,253 | 0,733 | 0,376 | 2,90 |
| 60 m | 0,277 | 0,700 | 0,396 | 2,53 |
| **80 m** | 0,299 | 0,643 | **0,408** | 2,15 |
| 100 m | 0,319 | 0,546 | 0,403 | 1,71 |
| 130 m | 0,372 | 0,422 | 0,395 | 1,13 |

**Escolhido: $\varepsilon = 80$ m** — ótimo interior da grade estendida.

É um raio **muito maior** que o do clusterizador ClusWiSARD (≈ 40 m). A razão é
que o OS-CFAR entrega 4,3 plots por alvo, contra 1,3 da ClusWiSARD: com muito
mais fragmentos por alvo, é preciso um raio maior para reuni-los.

## 4. Resultados

Clusterização completa: **159 728 → 83 220 plots** (redução de 48 %).

**Partição de teste — horas 09 e 11 UTC:**

| Métrica | Etapa 2 (só filtro) | **Etapa 5 (+ DBSCAN)** |
|---|---|---|
| VP | 20 868 | **18 211** |
| FP | 90 178 | **39 870** |
| FN | 4 996 | **7 653** |
| Precisão | 0,188 | **0,314** |
| Recall | 0,807 | **0,704** |
| F1 | 0,305 | **0,434** |
| Plots por alvo | 4,29 | 2,25 |

Janela de validação: F1 = 0,408. Restrita aos alvos com eco confirmado:
F1 = 0,439, recall = 0,748. Por estação: `Tabelas/etapa5_por_estacao.csv`.

## 5. Discussão

O DBSCAN produz o **maior ganho relativo de todas as etapas de agrupamento**:
o F1 salta de 0,305 para 0,434 (+42 %), eliminando 50 308 falsos positivos
(−56 %) e quase dobrando a precisão. O ganho é grande justamente porque a
entrada era muito fragmentada.

Ainda assim, o resultado final (0,434) fica **abaixo** do ramo ClusWiSARD
(0,479), e a razão está na precisão: mesmo depois de agrupar, restam 2,25 plots
por alvo, contra 0,61 do ramo ClusWiSARD. O agrupamento consegue reunir
fragmentos do mesmo objeto, mas **não consegue descartar objetos inteiros** —
cais, guindastes e embarcações atracadas sem AIS continuam produzindo plots
compactos e persistentes que nenhum critério de vizinhança elimina. Essa é uma
limitação estrutural: o que o OS-CFAR deixa passar, o DBSCAN não pode consertar.

O caráter complementar dos dois ramos é evidente no plano precisão–recall
(Figura `etapa7_fig2_precisao_recall.png`): P5 opera a recall 0,70 / precisão
0,31, e P3 a recall 0,39 / precisão 0,63.

## 6. Figuras e tabelas

| Arquivo | Conteúdo |
|---|---|
| `Figuras/etapa5_fig1_cena.png` | plots antes e depois do DBSCAN, em uma cena |
| `Figuras/etapa5_fig2_ajuste.png` | F1, recall e precisão × $\varepsilon$ (validação) |
| `Tabelas/etapa5_ajuste_dbscan.csv` | varredura completa de $\varepsilon$ |
| `Tabelas/etapa5_por_estacao.csv` | métricas por estação |
| `Resultados/etapa5_oscfar_dbscan_plots.csv.gz` | plots agrupados |
