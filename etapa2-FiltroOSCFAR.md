# Etapa 2 — Proposta 2: filtro OS-CFAR

> Código: `Codigo/etapa2_filtro_oscfar.py` (usa `Codigo/features.py`,
> `Codigo/plots_eval.py`, `Codigo/pipeline.py`)
> Saídas: `Resultados/etapa2_oscfar_*`, `Tabelas/etapa2_*`, `Figuras/etapa2_*`

---

## 1. Objetivo

Estabelecer a referência clássica de detecção contra a qual a rede neural sem
peso será comparada. O OS-CFAR (*Ordered Statistic Constant False Alarm Rate*,
Rohling, 1983) é o detector padrão em radares de vigilância marítima.

## 2. Formulação

Para cada célula sob teste (CUT) com amplitude $a$, estima-se o nível de fundo
local pelo $k$-ésimo menor valor $Z_{(k)}$ entre as $N$ células de treinamento
vizinhas, e declara-se detecção quando

$$a > \alpha \, Z_{(k)}$$

A estatística de ordem — em vez da média do CA-CFAR — é a escolha correta em
ambiente portuário: ela é robusta à presença de **alvos interferentes** dentro
da janela de treinamento, situação permanente em um porto, onde várias
embarcações e estruturas de cais coexistem a poucas dezenas de metros.

### Janela métrica

A janela de treinamento é definida em **metros**, não em células:

* amostragem em uma grade de $23\times23$ pontos espaçados de 8 m
  (janela de $\pm 88$ m), no referencial local **distância × travessia** do
  radar;
* células de guarda: todas as amostras a menos de $R_g$ do centro;
* células de treinamento: as amostras na coroa $R_g < d \le 88$ m.

Duas razões justificam essa escolha. Primeira, a célula de resolução em
travessia cresce com a distância (0,0879° equivalem a 1,5 m a 1 km e a 6 m a
4 km); uma janela definida em células teria extensão física variável ao longo
da cobertura. Segunda, ela iguala a extensão do contexto usado pela retina da
ClusWiSARD (Etapa 1, $\pm 84$ m), de modo que os dois filtros enxerguem a
**mesma vizinhança física** e a comparação isole o mecanismo de decisão.

### Raio de guarda

O raio de guarda também é ajustado. Um navio de 200–300 m ocupa dezenas de
células de treinamento; com guarda insuficiente ele **mascara a si mesmo**
(*self-masking*), elevando $Z_{(k)}$ e suprimindo a própria detecção.

## 3. Ajuste dos parâmetros

Todos os hiperparâmetros ($R_g$, ordem $k$, fator $\alpha$) foram escolhidos na
**partição de treino (08 UTC), janela de validação de 120 s**, maximizando o F1
em nível de plot — exatamente o mesmo critério, protocolo e orçamento de busca
usados na Etapa 1. Grade explorada: $R_g \in \{24, 40, 56\}$ m,
$k \in \{0{,}60;\ 0{,}75;\ 0{,}90;\ 0{,}98\}$,
$\alpha \in \{1{,}0;\ 1{,}3;\ 1{,}6;\ 2{,}0;\ 2{,}5;\ 3{,}0;\ 4{,}0;\ 6{,}0\}$
— 96 combinações (`Tabelas/etapa2_ajuste_cfar.csv`, Figura
`etapa2_fig1_ajuste.png`).

**Configuração escolhida:** $R_g = 24$ m, $k = 0{,}98$, $\alpha = 1{,}6$.

Observações do ajuste:

* o raio de guarda é praticamente irrelevante (melhor F1 de 0,297 / 0,293 /
  0,290 para 24 / 40 / 56 m) — o *self-masking* não é o fator limitante;
* a ordem precisa ser **muito alta** ($k = 0{,}98$). Como o vídeo do HAXR já sai
  limiarizado do sensor e apenas ≈ 3 % das células da ROI têm retorno, as
  ordens usuais ($k \approx 0{,}75N$) devolvem $Z_{(k)} = 0$ e o detector aprova
  praticamente todas as células com retorno;
* o F1 varia pouco em toda a grade (0,24 – 0,30): **nenhuma escolha de
  parâmetros resolve o problema**.

## 4. Extração de plots e avaliação

As células aprovadas são aglutinadas em plots pelo extrator canônico
(componentes conexas em grade métrica de 5 m, mínimo de 3 células), o mesmo de
todas as demais propostas. Os plots são associados ao *ground truth* da Etapa 0
pelo algoritmo húngaro com porta de 100 m; plots não associados são falsos
positivos, alvos AIS não associados são falsos negativos.

## 5. Resultados

**Partição de teste — horas 09 e 11 UTC, 13 estações, 2 088 varreduras,
25 864 alvos-varredura:**

| Métrica | Valor |
|---|---|
| Verdadeiros positivos (VP) | **20 868** |
| Falsos positivos (FP) | **90 178** |
| Falsos negativos (FN) | **4 996** |
| Precisão | **0,188** |
| Recall | **0,807** |
| F1 | **0,305** |
| Plots por alvo | 4,29 |

Para referência: na partição de treino o F1 é 0,297 — o OS-CFAR **transfere sem
perda** entre horas, como esperado de um detector sem parâmetros aprendidos.
Restrita aos alvos com eco confirmado (Etapa 0), a métrica praticamente não
muda (F1 = 0,304, recall = 0,852).

Desempenho por estação: `Tabelas/etapa2_por_estacao.csv`.

## 6. Discussão

O comportamento do OS-CFAR neste cenário é claro e consistente:

* **recall alto (0,81)** — o detector encontra quase todos os alvos AIS
  visíveis, o que confirma que a energia radar está presente;
* **precisão muito baixa (0,19)** — para cada alvo verdadeiro o filtro entrega
  mais de quatro plots, quase todos correspondentes a *clutter* portuário.

A razão é estrutural, e foi medida diretamente. Sobre 1 324 513 células da
partição de teste (classe *alvo*: a menos de 50 m de um AIS visível; classe
*fundo*: entre 90 e 150 m, faixa que evita o efeito de borda da ROI), a
**variável de decisão do OS-CFAR é quase desprovida de informação**:

| Atributo escalar | AUC |
|---|---|
| razão CFAR $a/(Z_{(0{,}98)}+1)$ | **0,546** |
| amplitude bruta $a$ | 0,540 |
| densidade local de retornos (escala fina) | 0,509 |
| densidade local de retornos (escala grossa) | 0,533 |

Medianas: razão CFAR de **0,62 no alvo** contra **0,54 no fundo**; amplitude de
**40** contra **36**. Nenhum atributo escalar isolado separa as classes — todos
ficam entre 0,51 e 0,55 de AUC, praticamente o acaso.

Isso explica o teto de desempenho do OS-CFAR de forma exata: seu detector é um
**limiar sobre um único escalar cuja AUC é 0,546**. Nenhuma escolha de $\alpha$,
de ordem ou de raio de guarda pode extrair mais informação do que a variável de
decisão contém — e é por isso que o F1 permanece entre 0,24 e 0,30 em toda a
grade de 96 combinações.

Para efeito de contraste, a ClusWiSARD da Etapa 1, que usa a **configuração
espacial conjunta** dos retornos em vez de qualquer estatística marginal, atinge
AUC de **0,777** em varreduras nunca vistas sobre a mesma tarefa.

O OS-CFAR foi projetado para detecção limitada por **ruído**, sob a hipótese de
que o alvo se distingue do fundo pela amplitude. No porto o fundo é *clutter
estruturado*, com amplitude comparável ou superior à do alvo, e a hipótese
falha. O detector mantém a taxa constante de alarmes falsos — mas os alarmes
falsos são ecos reais de objetos que não interessam.

Vale notar que a estrutura espacial *existe* e é forte: a densidade areal de
células com retorno a menos de 25 m de um alvo é **6,75 vezes** a densidade
média da ROI (`Tabelas/etapa7_estatisticas_celula.csv`). Essa informação está na
*configuração* dos retornos, não em nenhuma estatística marginal por célula — e é
exatamente o que uma retina $n$-tupla captura e um limiar escalar não.

Esse é precisamente o vazio que a Etapa 1 tenta preencher: discriminar pela
**textura** do retorno (forma, densidade e extensão da mancha de ecos), e não
pelo nível.

## 7. Figuras e tabelas

| Arquivo | Conteúdo |
|---|---|
| `Figuras/etapa2_fig1_ajuste.png` | F1 × $\alpha$ para cada ordem $k$ (validação) |
| `Figuras/etapa2_fig2_cena.png` | células aprovadas × rejeitadas em uma cena |
| `Tabelas/etapa2_ajuste_cfar.csv` | grade completa de 96 combinações |
| `Tabelas/etapa2_por_estacao.csv` | VP, FP, FN, precisão, recall e F1 por estação |
| `Tabelas/etapa7_auc_atributos.csv` | AUC de cada atributo escalar isolado |
| `Tabelas/etapa7_estatisticas_celula.csv` | densidade e amplitude por faixa de distância ao alvo |
| `Resultados/etapa2_oscfar_plots.csv.gz` | plots detectados (entrada da Etapa 5) |
| `Resultados/etapa2_oscfar_por_varredura.csv` | contagens por varredura |
