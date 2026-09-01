# Detecção de plots radar com redes neurais sem peso, validada por AIS

Pipeline completo sobre o conjunto **HAXR** (*Hamburg X-band Radar*,
<https://zenodo.org/records/19824555>): 13 estações de vigilância portuária com
campos de visão sobrepostos, vídeo radar bruto e AIS associado.

Seis propostas de detecção são comparadas sob protocolo, dados, extrator de
plots e orçamento de ajuste idênticos:

| | Filtro | Clusterizador | F1 (teste) |
|---|---|---|---|
| B0 | — (vídeo bruto) | — | 0,279 |
| P1 | ClusWiSARD | — | 0,420 |
| P2 | OS-CFAR | — | 0,305 |
| **P3** | **ClusWiSARD** | **ClusWiSARD** | **0,479** |
| P4 | ClusWiSARD | ClusWiSARD + cinemática | 0,477 |
| P5 | OS-CFAR | DBSCAN | 0,434 |
| P6 | OS-CFAR | DBSCAN + cinemática | 0,434 |

---

## Estrutura

```
DataSet_HAXR/          dados originais (não versionados; 5,1 GB)
DataSet_tratado/       recorte gerado pela Etapa 0 (78 MB) + ground_truth.csv
Codigo/                todo o código
Figuras/               figuras (PNG)
Tabelas/               tabelas (CSV e LaTeX)
Resultados/            plots detectados e métricas de cada proposta
HISTORICO.md           histórico da sessão de trabalho (ler primeiro ao retomar)
Artigo/artigo.tex      relatório em formato de artigo científico (português)
artigo_English/        mesma versão em inglês: article.tex, Figures/, Tables/
etapa0.md … etapa7-Comparacao.md    documentação de cada etapa
```

## Documentação por etapa

| Arquivo | Conteúdo |
|---|---|
| [etapa0.md](etapa0.md) | recorte do dataset e construção do *ground truth* |
| [etapa1-FiltroClusWisard.md](etapa1-FiltroClusWisard.md) | P1 — filtro ClusWiSARD |
| [etapa2-FiltroOSCFAR.md](etapa2-FiltroOSCFAR.md) | P2 — filtro OS-CFAR |
| [etapa3-FiltroClusWisard_e_ClusterizadorCluswisard.md](etapa3-FiltroClusWisard_e_ClusterizadorCluswisard.md) | P3 — + clusterizador ClusWiSARD |
| [etapa4-FiltroClusWisard_e_ClusterizadorCluswisard_e_Cinematica.md](etapa4-FiltroClusWisard_e_ClusterizadorCluswisard_e_Cinematica.md) | P4 — + cinemática |
| [etapa5-OSCFAR_e_DBSCAN.md](etapa5-OSCFAR_e_DBSCAN.md) | P5 — OS-CFAR + DBSCAN |
| [etapa6-OSCFAR_e_DBSCAN_e_Cinematica.md](etapa6-OSCFAR_e_DBSCAN_e_Cinematica.md) | P6 — + cinemática |
| [etapa7-Comparacao.md](etapa7-Comparacao.md) | comparação das seis propostas |
| [HISTORICO.md](HISTORICO.md) | **histórico da sessão**: decisões e suas justificativas, caminhos abandonados, erros corrigidos, pendências |
| `Artigo/artigo.tex` | Etapa 8 — artigo científico completo |

## Código

**Módulos compartilhados**

| Arquivo | Responsabilidade |
|---|---|
| `Codigo/haxr_io.py` | leitura do HAXR, geometria, seleção de varreduras, máscara de visibilidade |
| `Codigo/tratado_io.py` | leitura do `DataSet_tratado`, rasterização polar, *ground truth* |
| `Codigo/features.py` | retina binária por célula e janelas métricas |
| `Codigo/cluswisard.py` | WiSARD e ClusWiSARD (classificador e clusterizador) |
| `Codigo/plots_eval.py` | extração de plots e métricas (algoritmo húngaro) |
| `Codigo/cluster_common.py` | rastreador/cinemática e fusão de plots agrupados |
| `Codigo/pipeline.py` | execução paralela, persistência e avaliação |
| `Codigo/viz.py` | paleta e estilo das figuras |
| `Codigo/i18n.py` | idioma das figuras (dicionário PT→EN) |

**Execução, na ordem**

```bash
python3 Codigo/etapa0_build.py --jobs 6      # gera o DataSet_tratado
python3 Codigo/etapa0_gt.py                  # gera o ground truth
python3 Codigo/etapa0_figs.py                # figuras da Etapa 0

python3 Codigo/etapa1_filtro_cluswisard.py   # P1  (~35 min)
python3 Codigo/etapa2_filtro_oscfar.py       # P2  (~15 min)
python3 Codigo/etapa3_cluswisard_cluster.py  # P3
python3 Codigo/etapa4_cluswisard_cluster_cinematica.py   # P4
python3 Codigo/etapa5_oscfar_dbscan.py       # P5
python3 Codigo/etapa6_oscfar_dbscan_cinematica.py        # P6

python3 Codigo/etapa7_comparacao.py          # tabelas e figuras comparativas
python3 Codigo/gerar_figuras_etapas.py       # figuras das Etapas 1 a 6
python3 Codigo/gerar_artigo_en.py            # figuras e tabelas em ingles
```

Dependências: `numpy`, `pandas`, `scipy`, `h5py`, `scikit-learn`, `matplotlib`.

Os dois artigos seguem o padrão **IEEE** (classe `IEEEtran`, formato de
*IEEE Transactions*). Para compilar (requer `pdflatex` com `IEEEtran` — pacote
`texlive-publishers` no TeX Live — e `babel`):

```bash
cd Artigo         && pdflatex artigo  && pdflatex artigo    # português
cd artigo_English && pdflatex article && pdflatex article   # inglês
```

A versão em inglês usa os **mesmos** scripts de figura: as cadeias visíveis
passam por `Codigo/i18n.py`, de modo que não há duplicação do código de
plotagem. Veja `artigo_English/README.md`.

## Protocolo de avaliação

* **Treino**: hora 08 UTC, janela 0 de 120 s (ajuste dos modelos).
* **Validação**: hora 08 UTC, janela 1 de 120 s (escolha de **todos** os
  hiperparâmetros de **todas** as propostas).
* **Teste**: horas 09 e 11 UTC — 2 088 varreduras, 25 864 alvos-varredura.
* **Associação** plot ↔ *ground truth*: algoritmo húngaro, porta de 100 m.
* **Teto de recall**: 0,929 (7,1 % dos alvos AIS visíveis não têm eco radar).

## Principais achados

1. A melhor proposta é a de rede sem peso: **P3, F1 = 0,479**, contra 0,434 da
   melhor clássica e 0,279 do vídeo bruto. Vence em 10 das 13 estações.
2. A vantagem está na **precisão** (0,631 contra 0,314); o ramo clássico mantém
   recall muito mais alto (0,704 contra 0,386). São regimes complementares.
3. **Nenhum atributo escalar isolado separa alvo de fundo** (AUC entre 0,509 e
   0,546 — inclusive a variável de decisão do OS-CFAR), enquanto a retina
   $n$-tupla completa atinge **0,777** em varreduras nunca vistas. É por isso que
   o CFAR falha em ambiente portuário: seu teto está fixado antes de qualquer
   ajuste.
4. Os **atributos cinemáticos não contribuíram** em nenhum dos ramos, porque a
   cena é majoritariamente estática (mediana de |v| dos alvos AIS = 0,05 m/s).
5. Avaliar redes sem peso exige **partição por varredura**: a AUC cai de 0,950
   para 0,777 quando varreduras deixam de ser compartilhadas entre treino e
   avaliação.
