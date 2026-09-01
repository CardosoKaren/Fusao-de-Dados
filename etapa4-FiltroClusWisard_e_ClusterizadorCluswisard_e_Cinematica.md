# Etapa 4 — Proposta 4: filtro ClusWiSARD + clusterizador ClusWiSARD + cinemática

> Código: `Codigo/cluster_common.py` (rastreador e cinemática),
> `Codigo/etapa4_cluswisard_cluster_cinematica.py`
> Entrada: `Resultados/etapa1_cluswisard_plots.csv.gz`
> Saídas: `Resultados/etapa4_*`, `Tabelas/etapa4_*`, `Figuras/etapa4_*`

---

## 1. Motivação

Duas manchas de eco vizinhas que se deslocam com **rumos ou velocidades
distintas** não podem pertencer ao mesmo casco. A Etapa 3 agrupa apenas por
posição e, por isso, funde indevidamente embarcações que se cruzam ou que passam
próximas de outra atracada. A cinemática de cada plot deveria resolver esse caso.

## 2. Extração da cinemática pelas rotações de antena

A velocidade de cada plot primitivo é estimada pela **observação dos plots ao
longo das rotações de antena** dentro da janela de 120 s preservada no
`DataSet_tratado` (≈ 41 rotações de 2,9 s).

### 2.1 Rastreador com predição e tolerância a falhas

Uma primeira implementação encadeava simplesmente as associações entre
varreduras consecutivas. O resultado foi inadequado: apenas 73 % dos plots
tinham antecessor, e a trilha mediana tinha **3 pontos** — insuficiente para uma
regressão de velocidade, de modo que 54 % dos plots recebiam velocidade nula.

A versão final (`cluster_common.estimate_kinematics`) é um rastreador de vizinho
mais próximo com predição e *coasting*:

1. cada trilha viva é propagada para o instante da varredura pela sua velocidade
   corrente (predição);
2. resolve-se a associação ótima (**algoritmo húngaro**) entre as predições e os
   plots da varredura, sob porta de 45 m — compatível com o deslocamento máximo
   de uma embarcação em um período de antena (2,9 s a 15 m/s = 44 m);
3. plots não associados iniciam novas trilhas; trilhas não associadas seguem em
   voo cego por até 2 rotações antes de serem descartadas.

A velocidade atribuída a cada plot é o coeficiente angular da regressão linear de
$x$ e $y$ contra o tempo sobre as últimas 12 posições da trilha (≈ 35 s).

**Efeito da correção:** trilha mediana de 3 → **8 pontos**; fração de plots com
ao menos 4 pontos de 46 % → **70 %**; p90 de $|v|$ de 0,47 → **2,42 m/s**.

### 2.2 Codificação da velocidade

Dois eixos de velocidade são acrescentados à retina do clusterizador, em
termômetro sobre $\pm 15$ m/s. O **número de bits** dedicado a cada eixo controla
o peso relativo da cinemática na resposta da rede (mais bits ⇒ uma dada diferença
de velocidade divide uma fração maior da retina) e foi varrido em
$\{72, 144, 288\}$ junto com `min_score`.

## 3. Ajuste

Grade de $3 \times 7 = 21$ combinações, avaliada na **janela de validação**
(`Tabelas/etapa4_ajuste_clusterizador.csv`, Figura `etapa4_fig1_cinematica.png`b).

| bits de velocidade | `min_score` | precisão | recall | F1 |
|---|---|---|---|---|
| **144** | **0,94** | 0,610 | 0,506 | **0,5533** |
| 288 | 0,91 | 0,657 | 0,469 | 0,5472 |
| 288 | 0,94 | 0,561 | 0,526 | 0,5431 |
| 144 | 0,96 | 0,543 | 0,542 | 0,5426 |
| 72 | 0,94 | 0,619 | 0,482 | 0,5422 |
| — (Etapa 3, sem cinemática) | 0,94 | 0,675 | 0,466 | 0,5514 |

**Escolhidos: 144 bits por eixo de velocidade, `min_score` = 0,94.**

O melhor resultado com cinemática (0,5533) supera o melhor sem cinemática
(0,5514) por **0,002** — dentro do ruído estatístico da validação.

## 4. Resultados

Clusterização completa: **52 682 → 31 416 plots**.

**Partição de teste — horas 09 e 11 UTC:**

| Métrica | Etapa 3 (posicional) | **Etapa 4 (+ cinemática)** |
|---|---|---|
| VP | 9 993 | **11 008** |
| FP | 5 841 | **9 272** |
| FN | 15 871 | **14 856** |
| Precisão | 0,631 | **0,543** |
| Recall | 0,386 | **0,426** |
| F1 | **0,479** | 0,477 |
| Plots por alvo | 0,61 | 0,78 |

Janela de validação: F1 = 0,553. Restrita aos alvos com eco confirmado:
F1 = 0,480, recall = 0,442.

## 5. Discussão

**A cinemática não trouxe ganho** (F1 de 0,477 contra 0,479). O que ela fez foi
deslocar o ponto de operação: ao acrescentar dois eixos à retina, dois plots
precisam coincidir também em velocidade para serem fundidos, o que **reduz** a
fusão — daí mais VP (+1 015), mais FP (+3 431) e mais plots por alvo (0,61 →
0,78). O efeito líquido sobre o F1 é nulo.

A explicação é do próprio cenário, e está medida:

* **A maior parte dos alvos está parada.** Nos dados AIS do `DataSet_tratado`, a
  mediana de $|v|$ é **0,05 m/s** e apenas **29 %** dos alvos-varredura têm
  $|v| > 1$ m/s. Em um porto, a maioria das embarcações com AIS ligado está
  atracada ou fundeada. Para alvos parados a velocidade não distingue nada, e a
  cinemática degenera para um eixo constante.
* **O *clutter* também está parado**, de modo que a velocidade tampouco ajuda a
  separar alvo de *clutter* nesse estágio.
* **A velocidade estimada é ruidosa.** A distribuição de $|v|$ dos plots
  (Figura `etapa4_fig1_cinematica.png`a) tem p90 de 2,42 m/s contra 4,16 m/s do
  AIS: o rastreador subestima o movimento, porque plots de embarcações em
  movimento são exatamente os menos estáveis entre varreduras.

A conclusão é específica do cenário e **não** deve ser generalizada: em vigilância
de águas abertas, onde a fração de alvos em movimento é alta, o mesmo atributo
teria peso muito maior. A Etapa 6 reproduz exatamente o mesmo achado no ramo
clássico (DBSCAN), o que reforça que a limitação é do dado, não do método de
agrupamento.

## 6. Figuras e tabelas

| Arquivo | Conteúdo |
|---|---|
| `Figuras/etapa4_fig1_cinematica.png` | (a) velocidade dos plots × velocidade AIS; (b) F1 × `min_score` com e sem cinemática |
| `Tabelas/etapa4_ajuste_clusterizador.csv` | grade de 21 combinações |
| `Tabelas/etapa4_cinematica.csv` | estatísticas da velocidade estimada |
| `Tabelas/etapa4_por_estacao.csv` | métricas por estação |
| `Resultados/etapa4_cluswisard_cluster_cinematica_plots.csv.gz` | plots agrupados |
