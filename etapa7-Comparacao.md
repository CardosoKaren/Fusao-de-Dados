# Etapa 7 — Comparação das seis propostas

> Código: `Codigo/etapa7_comparacao.py`, `Codigo/figuras_resultados.py`
> Saídas: `Tabelas/etapa7_*.csv`, `Tabelas/etapa7_*.tex`, `Figuras/etapa7_*.png`

---

## 1. Protocolo comum

Todas as seis propostas foram avaliadas **exatamente sobre os mesmos dados, com
os mesmos blocos auxiliares e o mesmo orçamento de ajuste**:

| Elemento | Valor |
|---|---|
| Dados de teste | `DataSet_tratado`, horas **09 e 11 UTC**, 13 estações |
| Varreduras | 2 088 |
| Alvos-varredura (*ground truth*) | 25 864 |
| Extrator de plots | componentes conexas, grade métrica de 5 m, mín. 3 células |
| Associação plot ↔ *ground truth* | algoritmo húngaro, porta de 100 m |
| Ajuste de hiperparâmetros | janela de validação de 120 s da hora 08 UTC |
| Teto de *recall* (alvos sem eco) | 0,929 |

Inclui-se a linha de base **B0**: os plots extraídos do vídeo bruto recortado,
sem filtro algum.

## 2. Resultado principal

**Partição de teste, *ground truth* completo** (`Tabelas/etapa7_comparacao.csv`,
Figura `etapa7_fig1_comparacao.png`):

| Proposta | VP | FP | FN | Precisão | Recall | **F1** | plots/alvo |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 Vídeo bruto (sem filtro) | 24 110 | 122 584 | 1 754 | 0,164 | **0,932** | 0,279 | 5,67 |
| P1 Filtro ClusWiSARD | 12 337 | 20 591 | 13 527 | 0,375 | 0,477 | 0,420 | 1,27 |
| P2 Filtro OS-CFAR | 20 868 | 90 178 | 4 996 | 0,188 | 0,807 | 0,305 | 4,29 |
| **P3 ClusWiSARD + clust. ClusWiSARD** | 9 993 | **5 841** | 15 871 | **0,631** | 0,386 | **0,479** | 0,61 |
| P4 P3 + cinemática | 11 008 | 9 272 | 14 856 | 0,543 | 0,426 | 0,477 | 0,78 |
| P5 OS-CFAR + DBSCAN | 18 211 | 39 870 | 7 653 | 0,314 | 0,704 | 0,434 | 2,25 |
| P6 P5 + cinemática | 18 239 | 40 026 | 7 625 | 0,313 | 0,705 | 0,434 | 2,25 |

## 3. Análise de sensibilidade

Restringindo o *ground truth* aos alvos AIS com eco radar confirmado (Etapa 0),
o que remove o piso irredutível de falsos negativos
(`Tabelas/etapa7_comparacao_confirmados.csv`):

| Proposta | Precisão | Recall | F1 |
|---|---:|---:|---:|
| B0 Vídeo bruto | 0,164 | **1,000** | 0,282 |
| P1 Filtro ClusWiSARD | 0,361 | 0,493 | 0,417 |
| P2 Filtro OS-CFAR | 0,185 | 0,852 | 0,304 |
| **P3 ClusWiSARD + clust. ClusWiSARD** | **0,616** | 0,405 | **0,488** |
| P4 P3 + cinemática | 0,525 | 0,442 | 0,480 |
| P5 OS-CFAR + DBSCAN | 0,311 | 0,748 | 0,439 |
| P6 P5 + cinemática | 0,310 | 0,749 | 0,439 |

O recall de 1,000 da linha de base confirma a consistência do *ground truth*: por
construção, todo alvo confirmado tem um plot no vídeo bruto dentro da porta. A
ordenação das propostas é **idêntica** à da tabela principal — o resultado não
depende do tratamento dado aos alvos sem eco.

## 4. Desempenho por estação

`Tabelas/etapa7_f1_por_estacao.csv`, Figura `etapa7_fig3_por_estacao.png`.

| Estação | P1 | P2 | **P3** | P4 | P5 | P6 |
|---|---|---|---|---|---|---|
| altona | 0,466 | 0,326 | 0,501 | 0,518 | 0,464 | 0,463 |
| amerikahoeft | 0,271 | 0,225 | 0,315 | 0,341 | **0,479** | 0,478 |
| ellerholzhoeft | 0,461 | 0,210 | **0,591** | 0,554 | 0,297 | 0,297 |
| hohe_schaar | 0,381 | 0,248 | **0,455** | 0,419 | 0,386 | 0,384 |
| kattwyk | 0,413 | 0,258 | **0,465** | 0,459 | 0,350 | 0,350 |
| koehlbrandhoeft | 0,460 | 0,371 | 0,475 | **0,490** | 0,477 | 0,476 |
| krusenbusch | 0,418 | 0,274 | **0,516** | 0,492 | 0,369 | 0,369 |
| landungsbruecken | 0,432 | 0,415 | 0,386 | 0,436 | 0,509 | **0,510** |
| nesssand | 0,332 | 0,363 | 0,393 | 0,356 | **0,419** | 0,418 |
| parkhafen | 0,374 | 0,255 | **0,400** | 0,392 | 0,352 | 0,352 |
| reiherstieg | 0,506 | 0,214 | **0,657** | 0,632 | 0,346 | 0,346 |
| sandauhafen | 0,455 | 0,300 | **0,572** | 0,539 | 0,464 | 0,463 |
| seemannshoeft | 0,431 | 0,337 | **0,552** | 0,499 | 0,434 | 0,434 |

O ramo ClusWiSARD (P3/P4) vence em **10 das 13 estações**. As três exceções —
*amerikahoeft*, *landungsbruecken* e *nesssand* — têm em comum uma fração maior
de água aberta no campo de visão, cenário em que o pressuposto do CFAR (fundo
limitado por ruído) volta a valer.

## 5. Discussão

### 5.1 O filtro é o que decide

Comparando os pares (filtro sozinho → filtro + agrupamento):

| Ramo | filtro | + agrupamento | ganho |
|---|---|---|---|
| ClusWiSARD | 0,420 | 0,479 | +14 % |
| OS-CFAR | 0,305 | 0,434 | +42 % |

O DBSCAN ganha mais porque parte de uma entrada muito mais fragmentada (4,29
plots/alvo contra 1,27). Mas o agrupamento **não recupera a diferença de
qualidade do filtro**: a melhor proposta clássica (0,434) permanece abaixo da
melhor proposta com rede sem peso (0,479). Agrupar reúne fragmentos do mesmo
objeto; não descarta objetos inteiros.

### 5.1.1 Por que o filtro sem peso vence

Medida sobre 1 324 513 células da partição de teste (alvo: < 50 m de um AIS;
fundo: 90–150 m, evitando a borda da ROI):

| Variável de decisão | AUC |
|---|---|
| razão CFAR $a/(Z_{(0{,}98)}+1)$ — a do OS-CFAR | 0,546 |
| amplitude bruta | 0,540 |
| densidade local de retornos (escala grossa) | 0,533 |
| densidade local de retornos (escala fina) | 0,509 |
| **retina $n$-tupla completa (ClusWiSARD)** | **0,777** |

Nenhum atributo escalar isolado separa as classes. O OS-CFAR é, por construção,
um limiar sobre um escalar de AUC 0,546 — seu teto de desempenho está fixado
antes de qualquer ajuste. A ClusWiSARD lê a **configuração espacial conjunta**
dos retornos, informação que nenhuma estatística marginal carrega.

### 5.2 Regimes complementares

No plano precisão–recall (Figura `etapa7_fig2_precisao_recall.png`) as seis
propostas se organizam em dois regimes claros:

* **ramo clássico** (P2, P5, P6) — alto recall (0,70–0,81), baixa precisão
  (0,19–0,31), próximo da linha de base;
* **ramo ClusWiSARD** (P1, P3, P4) — precisão alta (0,38–0,63), recall moderado
  (0,39–0,48).

A escolha entre eles é uma decisão de aplicação. Para alimentar um rastreador
que consegue rejeitar falsos alarmes por consistência temporal, o alto recall de
P5 pode ser preferível. Para exibir plots a um operador, a precisão de P3 é o que
importa. Como as curvas de ajuste de ambos os ramos são suaves e monotônicas
(Etapas 3 e 5), qualquer ponto intermediário é acessível apenas mudando um
parâmetro.

### 5.3 A cinemática não contribuiu

P4 ≈ P3 e P6 ≈ P5, com diferenças abaixo de 0,003 em F1. O achado se repete em
dois mecanismos de agrupamento independentes, e a causa está medida no dado: a
mediana de $|v|$ dos alvos AIS é 0,05 m/s e apenas 29 % têm $|v| > 1$ m/s — a
cena é majoritariamente estática. Em vigilância de águas abertas o resultado
seria provavelmente outro.

### 5.4 Limites

* **Teto de recall de 0,929**, imposto pelos 7,1 % de alvos AIS visíveis sem
  eco radar detectável.
* **Generalização temporal.** A ClusWiSARD perde desempenho da janela de
  validação (F1 = 0,486 em P1) para o teste três horas depois (0,420) — efeito
  da memorização característica das redes sem peso. O OS-CFAR, sem parâmetros
  aprendidos, não sofre esse efeito (0,297 → 0,305).
* **Efeito de borda da ROI.** Células a menos de 84 m da borda do recorte veem
  zeros artificiais na sua vizinhança. O efeito atinge apenas células a mais de
  116 m do alvo e é idêntico para os dois filtros, mas existe.

## 6. Figuras e tabelas

| Arquivo | Conteúdo |
|---|---|
| `Figuras/etapa7_fig1_comparacao.png` | F1, recall e precisão das seis propostas |
| `Figuras/etapa7_fig2_precisao_recall.png` | plano precisão–recall com isolinhas de F1 |
| `Figuras/etapa7_fig3_por_estacao.png` | F1 por estação para as seis propostas |
| `Tabelas/etapa7_comparacao.csv` / `.tex` | tabela principal |
| `Tabelas/etapa7_comparacao_confirmados.csv` / `.tex` | análise de sensibilidade |
| `Tabelas/etapa7_f1_por_estacao.csv` | F1 por estação |
| `Tabelas/etapa7_auc_atributos.csv` | AUC de cada atributo escalar isolado |
| `Tabelas/etapa7_estatisticas_celula.csv` | densidade e amplitude por faixa de distância ao alvo |
| `Tabelas/etapa7_baseline_por_estacao.csv` | linha de base por estação |
