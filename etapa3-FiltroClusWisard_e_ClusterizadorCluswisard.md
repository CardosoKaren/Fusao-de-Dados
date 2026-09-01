# Etapa 3 — Proposta 3: filtro ClusWiSARD + clusterizador ClusWiSARD

> Código: `Codigo/cluster_common.py` (clusterizador e fusão de plots),
> `Codigo/etapa3_cluswisard_cluster.py`
> Entrada: `Resultados/etapa1_cluswisard_plots.csv.gz` (52 682 plots)
> Saídas: `Resultados/etapa3_*`, `Tabelas/etapa3_*`, `Figuras/etapa3_*`

---

## 1. Motivação

Um navio longo **não** produz um único eco compacto. O feixe varre casco,
superestrutura e guindastes de bordo em azimutes diferentes, e o filtro aprova
células em manchas separadas. As componentes conexas do extrator de plots
resultam então em **vários plots para uma única embarcação** — falsos positivos
que não correspondem a alvo algum.

O efeito é mensurável na saída da Etapa 1: **1,27 plots por alvo** na partição de
teste. O estágio de clusterização agrupa, dentro de cada varredura, os plots
primitivos pertencentes à mesma embarcação e os substitui por um único plot no
centroide ponderado pela amplitude.

## 2. A ClusWiSARD como clusterizador

Aqui a mesma rede sem peso é usada **sem rótulos**: cada discriminador criado
dinamicamente representa um agrupamento (uma embarcação). Um plot é absorvido
pelo discriminador de maior resposta se esta atinge `min_score`; caso contrário,
inicia um novo agrupamento. Uma instância é criada **por varredura**, e há uma
segunda passada de consolidação que reatribui os plots aos discriminadores já
formados, estabilizando o resultado quanto à ordem de apresentação.

### 2.1 Codificação e a escala métrica do agrupamento

Cada plot é codificado por **termômetro** em $X$ e $Y$ — 2048 bits por eixo sobre
uma extensão $S = 12$ km do referencial comum, isto é ≈ 5,9 m por bit.

A escolha do termômetro (e não do campo receptivo) é deliberada: ele é **denso**
(≈ 50 % de bits ativos), o que evita a degeneração das tuplas para o endereço
nulo, e faz a resposta decair de forma **controlada e previsível** com a
distância. Duas posições separadas de $d$ diferem em $d/S$ dos bits; a
probabilidade de uma tupla de $n$ bits não conter nenhum bit divergente é
$(1 - d/S)^n$, que é justamente a resposta esperada. Logo `min_score` equivale a
um **raio métrico de agrupamento**

$$d_{\text{cluster}} \;\propto\; S\left(1 - \texttt{min\_score}^{1/n}\right)$$

Como os dois eixos ocupam metades iguais da retina, um deslocamento **puramente
em $X$** altera apenas $d/(2S)$ do total de bits, o que dobra a constante da
expressão acima. A relação foi calibrada empiricamente com $n = 24$ e $N = 128$
RAMs, afastando dois plots isolados até que o clusterizador os separe
(`Tabelas/etapa3_calibracao_raio.csv`):

| `min_score` | 0,80 | 0,85 | 0,88 | 0,91 | **0,94** | 0,96 | 0,98 |
|---|---|---|---|---|---|---|---|
| $S(1-\texttt{min\_score}^{1/n})$ | 111 | 81 | 64 | 47 | **31** | 20 | 10 |
| $2S(1-\texttt{min\_score}^{1/n})$ | 222 | 162 | 128 | 94 | **62** | 41 | 20 |
| raio medido, ao longo de um eixo | 200 | 160 | 118 | 72 | **42** | 30 | 12 |
| raio medido, na diagonal | 150 | 100 | 68 | 50 | **34** | 26 | 10 |

(valores em metros). A expressão de dois eixos reproduz a ordem de grandeza e o
comportamento; a constante exata depende ainda da granularidade da resposta
($1/N$) e da passada de consolidação, e por isso o parâmetro é ajustado
empiricamente e não calculado.

O raio medido ao longo de um eixo é maior que na diagonal: como as tuplas
misturam bits de $X$ e de $Y$, a vizinhança induzida é do tipo **$L_1$**
(losangular), e não circular — uma diferença de forma em relação ao DBSCAN da
Etapa 5, cujo $\varepsilon$ é euclidiano.

## 3. Ajuste

`min_score` foi varrido em $\{0{,}80;\ 0{,}85;\ 0{,}88;\ 0{,}91;\ 0{,}94;\ 0{,}96;\ 0{,}98\}$
na **janela de validação** (janela 1 da hora 08), maximizando o F1 de plots —
mesmo critério e orçamento do $\varepsilon$ do DBSCAN na Etapa 5
(`Tabelas/etapa3_ajuste_clusterizador.csv`, Figura `etapa3_fig2_ajuste.png`).

| `min_score` | raio equiv. | precisão | recall | F1 |
|---|---|---|---|---|
| 0,80 | 111 m | 0,833 | 0,343 | 0,486 |
| 0,85 | 81 m | 0,805 | 0,366 | 0,503 |
| 0,88 | 64 m | 0,780 | 0,386 | 0,517 |
| 0,91 | 47 m | 0,747 | 0,409 | 0,529 |
| **0,94** | **31 m** | 0,675 | 0,466 | **0,551** |
| 0,96 | 20 m | 0,608 | 0,500 | 0,548 |
| 0,98 | 10 m | 0,483 | 0,561 | 0,519 |

**Escolhido: `min_score` = 0,94** — raio de agrupamento medido de ≈ 42 m ao
longo de um eixo e ≈ 34 m na diagonal. O ótimo é interior
e a curva é suave — sinal de que o parâmetro está bem condicionado.

## 4. Resultados

Clusterização completa: **52 682 → 25 042 plots** (redução de 52 %).

**Partição de teste — horas 09 e 11 UTC:**

| Métrica | Etapa 1 (só filtro) | **Etapa 3 (+ clusterizador)** |
|---|---|---|
| VP | 12 337 | **9 993** |
| FP | 20 591 | **5 841** |
| FN | 13 527 | **15 871** |
| Precisão | 0,375 | **0,631** |
| Recall | 0,477 | **0,386** |
| F1 | 0,420 | **0,479** |
| Plots por alvo | 1,27 | **0,61** |

Janela de validação: F1 = 0,551. Restrita aos alvos com eco confirmado:
F1 = 0,488, recall = 0,405. Por estação: `Tabelas/etapa3_por_estacao.csv`
(de 0,315 em *amerikahoeft* a 0,657 em *reiherstieg*).

## 5. Discussão

O clusterizador **funciona como previsto**: elimina 14 750 falsos positivos
(−72 %) e eleva a precisão de 0,375 para 0,631. O F1 sobe de 0,420 para 0,479 —
o **melhor resultado entre as seis propostas**.

O custo é uma queda de recall de 0,477 para 0,386, com duas causas distintas:

1. **Fusão indevida de alvos vizinhos.** Em berços de atracação as embarcações
   ficam a poucas dezenas de metros umas das outras; um raio de ≈ 40 m já basta
   para colapsar duas em um plot só, transformando um VP em um FN. A relação
   plots/alvo cai a 0,61, ou seja, o sistema entrega **menos plots do que alvos**
   — o regime está deslocado para o lado conservador.
2. **Deslocamento do centroide.** Ao fundir várias manchas, o centroide
   resultante pode sair da porta de 100 m em relação à posição AIS, sobretudo em
   navios longos, cuja antena AIS não coincide com o centro de massa do eco.

A curva de ajuste (Seção 3) mostra que o compromisso é contínuo e controlável:
raios menores privilegiam recall, raios maiores privilegiam precisão. Uma
aplicação operacional que priorize não perder alvo escolheria `min_score` = 0,98
(recall 0,561 na validação) em vez do ótimo de F1.

## 6. Figuras e tabelas

| Arquivo | Conteúdo |
|---|---|
| `Figuras/etapa3_fig1_cena.png` | plots antes e depois do agrupamento, em uma cena |
| `Figuras/etapa3_fig2_ajuste.png` | F1, recall e precisão × `min_score` (validação) |
| `Tabelas/etapa3_ajuste_clusterizador.csv` | varredura completa de `min_score` |
| `Tabelas/etapa3_calibracao_raio.csv` | calibração do raio métrico de agrupamento |
| `Tabelas/etapa3_por_estacao.csv` | métricas por estação |
| `Resultados/etapa3_cluswisard_cluster_plots.csv.gz` | plots agrupados |
