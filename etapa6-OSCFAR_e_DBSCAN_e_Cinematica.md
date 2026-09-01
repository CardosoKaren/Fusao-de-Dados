# Etapa 6 — Proposta 6: filtro OS-CFAR + DBSCAN + cinemática

> Código: `Codigo/etapa6_oscfar_dbscan_cinematica.py`
> Entrada: `Resultados/etapa2_oscfar_plots.csv.gz`
> Saídas: `Resultados/etapa6_*`, `Tabelas/etapa6_*`

---

## 1. Objetivo

Contraparte clássica da Etapa 4: acrescentar a cinemática ao espaço de
agrupamento do DBSCAN, para que embarcações próximas com rumos ou velocidades
distintas não sejam fundidas.

## 2. Método

A velocidade de cada plot primitivo é estimada exatamente como na Etapa 4 — pelo
rastreador com predição e *coasting* sobre as rotações de antena da janela de
120 s (`cluster_common.estimate_kinematics`); o código é literalmente o mesmo,
de modo que a única diferença entre os ramos seja o clusterizador.

O DBSCAN passa a operar em **quatro dimensões**,
$(x,\ y,\ k\,v_x,\ k\,v_y)$, em que o fator $k$ [m / (m/s)] converte diferença de
velocidade em distância equivalente — assim $\varepsilon$ mantém unidade de
metro. $k$ e $\varepsilon$ são ajustados em conjunto na janela de validação.

## 3. Ajuste

Grade de $3 \times 8 = 24$ combinações, $k \in \{5, 15, 40\}$,
$\varepsilon \in \{15, \dots, 130\}$ m (`Tabelas/etapa6_ajuste_dbscan.csv`).

| $k$ | $\varepsilon$ | precisão | recall | F1 |
|---|---|---|---|---|
| **5** | **80 m** | 0,299 | 0,644 | **0,4079** |
| 15 | 80 m | 0,298 | 0,644 | 0,4078 |
| 40 | 80 m | 0,298 | 0,645 | 0,4077 |
| — (Etapa 5, sem cinemática) | 80 m | 0,299 | 0,643 | 0,4076 |

**Escolhidos: $\varepsilon = 80$ m, $k = 5$.**

O ajuste escolheu o **menor** fator de escala da grade, isto é, o menor peso
possível para a cinemática, e o F1 na validação (0,4079) é indistinguível do
obtido sem cinemática (0,4076). O procedimento de ajuste, portanto, "desligou"
sozinho o atributo cinemático.

## 4. Resultados

**Partição de teste — horas 09 e 11 UTC:**

| Métrica | Etapa 5 (posicional) | **Etapa 6 (+ cinemática)** |
|---|---|---|
| VP | 18 211 | 18 239 |
| FP | 39 870 | 40 026 |
| FN | 7 653 | 7 625 |
| Precisão | 0,3135 | 0,3130 |
| Recall | 0,7041 | 0,7052 |
| F1 | **0,4339** | 0,4336 |
| Plots por alvo | 2,25 | 2,25 |

Janela de validação: F1 = 0,408. Restrita aos alvos com eco confirmado:
F1 = 0,439, recall = 0,749.

## 5. Discussão

**A cinemática não altera o resultado** — a diferença de 0,0003 em F1 é ruído.
O achado é idêntico ao da Etapa 4, obtido por um mecanismo de agrupamento
completamente diferente, o que confirma que a limitação está no **dado**, e não
no clusterizador:

* a mediana de $|v|$ dos alvos AIS no `DataSet_tratado` é **0,05 m/s**, e apenas
  **29 %** dos alvos-varredura têm $|v| > 1$ m/s — em um porto a maioria das
  embarcações com AIS ligado está atracada;
* o *clutter* (cais, guindastes, embarcações atracadas) também está parado, de
  modo que a velocidade não separa nem alvo de alvo, nem alvo de *clutter*;
* a mediana de $|v|$ estimada para os 159 728 plots do OS-CFAR é de apenas
  **0,04 m/s**, com p90 de 1,64 m/s — coerente com uma cena majoritariamente
  estática.

Vale registrar que a cinemática **é corretamente extraída**: a versão final do
rastreador produz trilhas de 8 rotações (mediana) e p90 de velocidade de
2,42 m/s nos plots da Etapa 1, compatível com o AIS. O atributo simplesmente não
carrega informação discriminante *neste cenário*. Em vigilância de águas
abertas, com fração alta de alvos em movimento, a conclusão seria provavelmente
diferente — e essa é uma direção clara de trabalho futuro.

## 6. Figuras e tabelas

| Arquivo | Conteúdo |
|---|---|
| `Tabelas/etapa6_ajuste_dbscan.csv` | grade de 24 combinações ($k$, $\varepsilon$) |
| `Tabelas/etapa6_por_estacao.csv` | métricas por estação |
| `Resultados/etapa6_oscfar_dbscan_cinematica_plots.csv.gz` | plots agrupados |
| `Figuras/etapa4_fig1_cinematica.png` | distribuição da velocidade estimada (comum às Etapas 4 e 6) |
