# Etapa 1 — Proposta 1: filtro ClusWiSARD

> Código: `Codigo/cluswisard.py` (rede), `Codigo/features.py` (retina),
> `Codigo/etapa1_filtro_cluswisard.py` (treino e aplicação)
> Saídas: `Resultados/etapa1_cluswisard_*`, `Tabelas/etapa1_*`, `Figuras/etapa1_*`

---

## 1. Objetivo

Usar a rede neural sem peso **ClusWiSARD** como filtro de detecção: decidir,
célula a célula do vídeo radar recortado, se o retorno pertence a um alvo ou a
ruído/*clutter*. As células aprovadas são aglutinadas em plots pelo extrator
canônico e confrontadas com o *ground truth* da Etapa 0.

## 2. A rede

Implementação própria em `Codigo/cluswisard.py`, vetorizada com RAMs densas.

### 2.1 WiSARD

Um classificador $n$-tupla. A entrada é uma retina binária de $B$ bits,
particionada pseudoaleatoriamente em $N$ tuplas de $n$ bits; cada tupla endereça
uma RAM. **Treinar** é escrever nas $N$ posições endereçadas; a **resposta** de
um discriminador é a fração de RAMs cujo conteúdo no endereço visitado atinge o
limiar de leitura (*bleaching*). Não há pesos nem gradiente.

### 2.2 RAMs endereçadas por espalhamento

Tuplas grandes mostraram-se necessárias (Seção 5.1), e $2^n$ excede rapidamente
qualquer memória viável. Quando $n >$ `table_bits`, o endereço de $n$ bits é
reduzido por uma função de espalhamento multiplicativa, com semente distinta por
RAM. Isso permite $n = 48$ com RAMs de $2^{18}$ posições, ao custo de colisões
raras e uniformemente distribuídas.

### 2.3 ClusWiSARD e a correção de saturação

A ClusWiSARD admite até $L$ discriminadores por classe: ao treinar, o exemplo vai
para o discriminador de maior resposta; se nenhum atinge `min_score` e o limite
não foi alcançado, cria-se um novo.

**Um cuidado foi necessário.** Com RAMs endereçadas por espalhamento, o
discriminador mais treinado responde mais alto a *qualquer* exemplo e absorve
todos os demais — uma dinâmica de *rich-get-richer*. Sem correção, a ClusWiSARD
degenera em WiSARD: os discriminadores extras recebem 1 ou 2 exemplos cada.
A atribuição passou então a usar o **ganho sobre o acaso**,
$r_j(x) - o_j$, em que $o_j$ é a fração de posições de RAM já escritas no
discriminador $j$. Com essa correção os agrupamentos efetivamente se formam.

| Modelo | Tamanho dos discriminadores (classe "alvo") |
|---|---|
| ClusWiSARD com correção | 45 635 / 3 177 / 3 142 / 3 267 |
| ClusWiSARD sem correção | 55 216 / 2 / 2 / 1 |

## 3. A retina

A unidade de decisão é a **célula com retorno**: o vídeo do HAXR já sai
limiarizado do sensor e apenas ≈ 3 % das células da ROI têm retorno.

### 3.1 Amostragem em referencial distância × travessia

Para cada célula amostra-se a vizinhança em duas escalas, no referencial local
**distância (radial) × travessia (tangencial)** do radar:

| Escala | Amostras | Passo | Extensão |
|---|---|---|---|
| fina | 9 × 9 | 7 m | ± 28 m |
| grossa | 7 × 7 | 24 m | ± 84 m |

Nesse referencial a conversão para índices da grade polar é exata e barata: o
deslocamento em índice de distância é constante, e o de azimute vale
$k_t\,p / (r\,\Delta\theta)$. Assim a **extensão métrica** da retina é a mesma em
toda a cobertura, apesar de a célula de resolução em travessia crescer com a
distância (0,0879° equivalem a 1,5 m a 1 km e a 6 m a 4 km).

Cada amostra é binarizada em dois níveis ($>0$ e $>64$): **260 bits**.

### 3.2 Atributos escalares (termômetro, 12 bits cada)

amplitude própria; distância; densidade de células com retorno e amplitude média
em cada escala; e cinco estatísticas do tipo CFAR calculadas na coroa além do
raio de guarda de 24 m: $a/(Z_{(0{,}98)}+1)$, $a/(Z_{(0{,}75)}+1)$,
$a/(\bar Z+1)$, $a - Z_{(0{,}98)}$ e o posto de $a$ entre as amostras da coroa.

Total: **392 bits**.

### 3.3 Codificação complementar

A retina é concatenada com o seu complemento (**784 bits**). Sem isso, a
esparsidade do vídeo faz quase toda tupla endereçar a posição zero e as respostas
dos discriminadores saturam próximo de 1, anulando a discriminação. Medido na
configuração de sondagem inicial (tuplas de 14 bits, retina sem as estatísticas
CFAR): AUC de 0,660 sem complemento contra 0,711 com complemento.

## 4. Protocolo de treino

* **Rotulagem**: células a menos de 50 m de um alvo AIS visível → classe
  *alvo*; a mais de 90 m de qualquer alvo → classe *ruído*; a faixa intermediária
  é descartada (evita rótulos ambíguos na borda do eco).
* **Amostras**: 241 685 células (35,3 % da classe alvo), exclusivamente da
  **janela de ajuste** (janela 0 de 120 s) da hora 08 UTC, nas 13 estações.
* **Hiperparâmetros da rede**: $n = 48$ bits por tupla, $N = 128$ RAMs de
  $2^{18}$ posições, até 4 discriminadores por classe, `min_score` = 0,5,
  intervalo de crescimento 32, contadores de 8 bits. Modelo: 269 MB.
* **Bleaching**: escolhido em partição retida de células ($b = 1$).
* **Limiar de decisão**: a margem $m(x) = r_1(x) - r_0(x)$ é comparada a um
  limiar escolhido na **janela de validação** (janela 1 de 120 s da hora 08, meia
  hora depois, que a rede nunca viu), maximizando o F1 em nível de plot.
  Escolhido: $m > +0{,}02$ (Figura `etapa1_fig1_ajuste.png`).

**Por que a janela de validação é indispensável.** A WiSARD é um classificador
por memorização: sua resposta em dados já apresentados é sistematicamente mais
alta. Na primeira execução deste trabalho o limiar foi escolhido sobre a própria
partição de treino, e o F1 caiu de 0,49 (treino) para 0,37 (teste). Com a
separação em janelas, o limiar escolhido transfere muito melhor. A tabela abaixo
quantifica a memorização de forma direta:

| AUC em nível de célula | Valor |
|---|---|
| células retidas das **mesmas varreduras** de treino | **0,950** |
| células de varreduras **nunca vistas** (janela de validação) | **0,777** |

A diferença de 0,17 é o efeito de memorização; qualquer avaliação que não separe
varreduras o reporta como desempenho.

## 5. Ablações

### 5.1 Tamanho da tupla (AUC na janela de validação)

Varredura conduzida a $N = 40$ RAMs de $2^{18}$ posições e retina de 664 bits
(sem as estatísticas CFAR), variando apenas $n$:

| $n$ | 14 | 20 | 28 | 36 | 44 | 56 | 72 | 88 |
|---|---|---|---|---|---|---|---|---|
| AUC | 0,610 | 0,678 | 0,719 | 0,743 | **0,757** | 0,754 | 0,737 | 0,742 |

Tuplas grandes generalizam **melhor**, não pior — o oposto da intuição usual de
que mais capacidade significa mais sobreajuste. A explicação é que a retina é
espacialmente redundante: só uma tupla longa consegue amostrar um padrão
espacial extenso o bastante para caracterizar a textura de um eco de embarcação.

A partir de $n \approx 44$ o ganho vem do **número de RAMs**, não do tamanho da
tupla: passando de $N = 40$ para $N = 128$ RAMs e acrescentando as cinco
estatísticas do tipo CFAR à retina, a AUC na janela de validação sobe de 0,757
para **0,777** — a configuração final adotada ($n = 48$, $N = 128$).

### 5.2 ClusWiSARD × WiSARD (`Tabelas/etapa1_ablacao_discriminadores.csv`)

| Modelo | AUC (células retidas) | AUC (janela de validação) |
|---|---|---|
| ClusWiSARD, 4 discriminadores/classe, com correção | 0,9502 | 0,7767 |
| WiSARD, 1 discriminador/classe | 0,9531 | **0,7787** |
| ClusWiSARD sem correção de saturação | 0,9531 | 0,7787 |

**Resultado honesto: os discriminadores múltiplos não trazem ganho neste
problema** (0,7767 contra 0,7787 — diferença dentro do ruído). A classe "eco de
embarcação", tal como codificada por esta retina, não é suficientemente
multimodal para que a partição em agrupamentos internos ajude. Manteve-se a
ClusWiSARD como modelo principal, por ser o objeto de estudo, registrando-se a
equivalência.

## 6. Resultados

**Partição de teste — horas 09 e 11 UTC, 2 088 varreduras, 25 864
alvos-varredura:**

| Métrica | Valor |
|---|---|
| Verdadeiros positivos (VP) | **12 337** |
| Falsos positivos (FP) | **20 591** |
| Falsos negativos (FN) | **13 527** |
| Precisão | **0,375** |
| Recall | **0,477** |
| F1 | **0,420** |
| Plots por alvo | 1,27 |

Progressão entre partições, que mede a generalização temporal:

| Partição | F1 | Recall | Precisão |
|---|---|---|---|
| 08 UTC, janela de ajuste (vista no treino) | 0,524 | 0,767 | 0,398 |
| 08 UTC, janela de validação (não vista) | 0,486 | 0,596 | 0,410 |
| 09 e 11 UTC (teste) | **0,420** | 0,477 | 0,375 |

Restrita aos alvos com eco confirmado: F1 = 0,417, recall = 0,493.
Desempenho por estação: `Tabelas/etapa1_por_estacao.csv` — varia de 0,271
(*amerikahoeft*) a 0,506 (*reiherstieg*).

## 6.1 Por que a rede funciona: nenhum atributo isolado basta

Medida sobre 1 324 513 células da partição de teste (classe *alvo*: a menos de
50 m de um AIS visível; classe *fundo*: entre 90 e 150 m, faixa que evita o
efeito de borda da ROI):

| Variável de decisão | AUC |
|---|---|
| razão CFAR $a/(Z_{(0{,}98)}+1)$ — a do OS-CFAR | 0,546 |
| amplitude bruta | 0,540 |
| densidade local de retornos (escala grossa) | 0,533 |
| densidade local de retornos (escala fina) | 0,509 |
| **retina $n$-tupla completa (ClusWiSARD)** | **0,777** |

**Nenhum atributo escalar isolado separa as classes** — todos ficam entre 0,51 e
0,55, praticamente o acaso. A discriminação não está em nenhuma estatística
marginal por célula: está na **configuração espacial conjunta** dos retornos na
vizinhança, que só uma função de alta ordem sobre a vizinhança inteira consegue
ler. É exatamente o que uma tupla de 48 bits amostrando a retina binarizada faz,
e o que um limiar sobre um escalar, por construção, não pode fazer.

Que a estrutura espacial existe e é forte, mede-se de outra forma: a densidade
areal de células com retorno a menos de 25 m de um alvo é **6,75 vezes** a
densidade média da ROI, decaindo monotonicamente para 0,62 na coroa de
150–200 m, enquanto a amplitude mediana quase não varia com a distância ao alvo
(43 a menos de 25 m; 36 entre 100 e 150 m) —
`Tabelas/etapa7_estatisticas_celula.csv`.

## 7. Discussão

Comparado à linha de base sem filtro algum (vídeo bruto recortado: F1 = 0,279,
recall = 0,932, precisão = 0,164), o filtro ClusWiSARD **reduz os falsos
positivos de 122 584 para 20 591 — um fator de 6** — ao custo de recall.
O F1 sobe de 0,279 para 0,420 (+50 %).

Comparado ao OS-CFAR (Etapa 2, F1 = 0,305), a rede sem peso é claramente
superior em F1 e em precisão (0,375 contra 0,188), e claramente inferior em
recall (0,477 contra 0,807). Os dois detectores operam em regimes distintos do
plano precisão–recall: o OS-CFAR aproxima-se do vídeo bruto (aprova quase tudo),
enquanto a ClusWiSARD é conservadora.

A queda de recall entre a janela de validação (0,596) e o teste (0,477) é o
limite atual do método: a rede memoriza a assinatura das embarcações e das
estruturas presentes na hora de treino, e três horas depois a cena mudou. Um
treinamento incremental — natural em redes sem peso, cujo custo de escrita é
uma operação de memória — atenuaria esse efeito em operação contínua, e é o
trabalho futuro mais imediato.

## 8. Figuras e tabelas

| Arquivo | Conteúdo |
|---|---|
| `Figuras/etapa1_fig1_ajuste.png` | F1 × limiar de decisão nos dois modos de margem |
| `Figuras/etapa1_fig2_cena.png` | células aprovadas × rejeitadas em uma cena |
| `Tabelas/etapa1_ajuste_limiar.csv` | varredura completa do limiar (validação) |
| `Tabelas/etapa1_ajuste_bleaching.csv` | AUC por valor de *bleaching* |
| `Tabelas/etapa1_ablacao_discriminadores.csv` | ClusWiSARD × WiSARD |
| `Tabelas/etapa1_por_estacao.csv` | métricas por estação |
| `Tabelas/etapa7_auc_atributos.csv` | AUC de cada atributo escalar isolado |
| `Tabelas/etapa7_estatisticas_celula.csv` | densidade e amplitude por faixa de distância ao alvo |
| `Resultados/etapa1_cluswisard_plots.csv.gz` | plots detectados (entrada das Etapas 3 e 4) |
| `Resultados/etapa1_cluswisard_modelo.pkl` | rede treinada |
