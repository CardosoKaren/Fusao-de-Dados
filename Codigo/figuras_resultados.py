"""
figuras_resultados.py -- Figuras das Etapas 1 a 7.

Gera, a partir dos arquivos de Resultados/ e Tabelas/:
  etapa1_fig1  ajuste do limiar de decisao da ClusWiSARD (treino)
  etapa1_fig2  cena: celulas aprovadas x rejeitadas pelo filtro
  etapa2_fig1  ajuste de alpha e da ordem do OS-CFAR (treino)
  etapa2_fig2  cena: celulas aprovadas pelo OS-CFAR
  etapa35_fig  efeito do clusterizador sobre uma cena (antes/depois)
  etapa46_fig  cinematica estimada e seu efeito
  etapa7_fig1  comparacao das seis propostas (barras)
  etapa7_fig2  plano precisao-recall das seis propostas
  etapa7_fig3  F1 por estacao para as seis propostas
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import haxr_io as H
import i18n
import pipeline as PL
from i18n import T as TR
import tratado_io as T
import viz

viz.apply_style()
import matplotlib.pyplot as plt

FIG = H.FIG_DIR
os.makedirs(FIG, exist_ok=True)

_PROPOSTAS = [
    ("etapa1_cluswisard", "P1  Filtro ClusWiSARD"),
    ("etapa2_oscfar", "P2  Filtro OS-CFAR"),
    ("etapa3_cluswisard_cluster", "P3  ClusWiSARD + clust. ClusWiSARD"),
    ("etapa4_cluswisard_cluster_cinematica", "P4  P3 + cinematica"),
    ("etapa5_oscfar_dbscan", "P5  OS-CFAR + DBSCAN"),
    ("etapa6_oscfar_dbscan_cinematica", "P6  P5 + cinematica"),
]
PROPOSTAS = _PROPOSTAS   # rotulos em portugues (usados como chave de traducao)


def rotulos():
    """Pares (nome, rotulo) com o rotulo ja traduzido para o idioma corrente."""
    return [(n, TR(l)) for n, l in _PROPOSTAS]


def _load_metric(name):
    p = os.path.join(PL.RES_DIR, f"{name}_metricas.csv")
    return pd.read_csv(p).iloc[0] if os.path.exists(p) else None


# ------------------------------------------------------------------ etapa 1/2
def fig_ajuste(tab_csv, xcol, xlabel, out, group=None, glabel="", titulo=None):
    tab = pd.read_csv(os.path.join(H.TAB_DIR, tab_csv))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    if group is None:
        series = [("", tab)]
    else:
        series = [(f"{TR(glabel)}={g}", t) for g, t in tab.groupby(group)]
    for i, (lbl, t) in enumerate(series):
        t = t.sort_values(xcol)
        ax.plot(t[xcol], t["F1"], color=viz.SERIES[i % 3], marker="o", ms=4,
                label=("F1" if not lbl else TR("F1  {lbl}").format(lbl=lbl)))
    if group is None:
        ax.plot(tab.sort_values(xcol)[xcol], tab.sort_values(xcol)["recall"],
                color=viz.SERIES[1], marker="o", ms=4, label=TR("recall"))
        ax.plot(tab.sort_values(xcol)[xcol], tab.sort_values(xcol)["precisao"],
                color=viz.SERIES[2], marker="o", ms=4, label=TR("precisao"))
    b = tab.loc[tab["F1"].idxmax()]
    ax.scatter([b[xcol]], [b["F1"]], s=90, facecolor="none",
               edgecolor=viz.INK, lw=1.4, zorder=6)
    ax.annotate(TR("maximo  F1={v:.3f}").format(v=b["F1"]), (b[xcol], b["F1"]), xytext=(8, 10),
                textcoords="offset points", fontsize=8.5, color=viz.INK2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(TR("metrica (janela de validacao, 08 UTC)"))
    if titulo:
        ax.set_title(titulo)
    ax.legend(loc="best", ncol=2)
    viz.despine(ax)
    viz.save(fig, os.path.join(FIG, out))


# ------------------------------------------------------------------- cenas
def cena(station, hour, mask_fn, titulo, out, k=None):
    """Desenha uma varredura com as celulas aprovadas destacadas."""
    import features as Fe
    d = T.Tratado(station, hour)
    if k is None:
        k = int(np.argmax([len(d.targets(i)) for i in range(d.n_scan)]))
    az, ri, amp = d.cells(k)
    x, y = d.cell_xy(az, ri)
    keep = mask_fn(d, k)
    G = Fe.gt_xy(d, k)
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    ax.scatter(x[~keep], y[~keep], s=1.2, color=viz.INK3, alpha=0.28, lw=0,
               label=TR("rejeitadas ({n})").format(n=(~keep).sum()))
    ax.scatter(x[keep], y[keep], s=2.2, color=viz.SERIES[0], lw=0,
               label=TR("aprovadas ({n})").format(n=keep.sum()))
    ax.scatter(G[:, 0], G[:, 1], s=70, facecolor="none", edgecolor=viz.SERIES[1],
               lw=1.3, label=TR("ground truth AIS ({n})").format(n=len(G)))
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(titulo)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    viz.despine(ax)
    viz.save(fig, os.path.join(FIG, out))
    d.close()
    return k


# ------------------------------------------------------------------- etapa 7
def fig_comparacao():
    rows = []
    for n, lbl in rotulos():
        m = _load_metric(n)
        if m is None:
            continue
        rows.append(dict(proposta=lbl, F1=m["F1"], recall=m["recall"],
                         precisao=m["precisao"]))
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    yy = np.arange(len(t))
    h = 0.26
    for i, c in enumerate(["F1", "recall", "precisao"]):
        ax.barh(yy + (1 - i) * h, t[c], height=h * 0.92, color=viz.SERIES[i],
                label=TR(c), edgecolor=viz.SURFACE, lw=1.2)
        for j, v in enumerate(t[c]):
            ax.annotate(f"{v:.3f}", (v, yy[j] + (1 - i) * h), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=7.5,
                        color=viz.INK2)
    ax.set_yticks(yy)
    ax.set_yticklabels(t["proposta"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(TR("valor da metrica (particao de teste, 09 e 11 UTC)"))
    ax.set_title(TR("Comparacao das seis propostas de deteccao"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.grid(axis="y", visible=False)
    viz.despine(ax)
    viz.save(fig, os.path.join(FIG, "etapa7_fig1_comparacao.png"))
    return t


def fig_pr():
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    mk = ["o", "s", "^", "D", "v", "P"]
    off = {"P1": (9, 5), "P2": (9, -13), "P3": (9, 5), "P4": (9, -13),
           "P5": (9, 6), "P6": (9, -14)}
    b = _load_metric("etapa7_baseline")
    if b is None:
        import pandas as _pd
        bt = _pd.read_csv(os.path.join(H.TAB_DIR, "etapa7_comparacao.csv"))
        bt = bt[bt["proposta"].str.startswith("B0")]
        if len(bt):
            b = bt.iloc[0]
    if b is not None:
        ax.scatter(b["recall"], b["precisao"], s=95, color=viz.SERIES[2], marker="*",
                   edgecolor=viz.SURFACE, lw=1.0, zorder=4)
        ax.annotate("B0", (b["recall"], b["precisao"]), xytext=(9, 5),
                    textcoords="offset points", fontsize=8.5, color=viz.INK2)
        ax.scatter([], [], s=95, color=viz.SERIES[2], marker="*",
                   label=TR("video bruto, sem filtro (B0)"))
    for i, (n, lbl) in enumerate(rotulos()):
        m = _load_metric(n)
        if m is None:
            continue
        c = viz.SERIES[0] if n.startswith(("etapa1", "etapa3", "etapa4")) else viz.SERIES[1]
        p_ = lbl.split()[0]
        ax.scatter(m["recall"], m["precisao"], s=95, color=c, marker=mk[i],
                   edgecolor=viz.SURFACE, lw=1.2, zorder=4)
        ax.annotate(p_, (m["recall"], m["precisao"]), xytext=off.get(p_, (9, 5)),
                    textcoords="offset points", fontsize=8.5, color=viz.INK2)
    for f in (0.3, 0.4, 0.5, 0.6, 0.7):
        r = np.linspace(0.01, 1, 200)
        p = f * r / (2 * r - f)
        ok = (p > 0) & (p <= 1)
        ax.plot(r[ok], p[ok], color=viz.GRID, lw=0.9, zorder=1)
        if ok.any():
            ax.annotate(f"F1={f}", (r[ok][-1], p[ok][-1]), fontsize=7,
                        color=viz.INK3, xytext=(-22, 4), textcoords="offset points")
    ax.scatter([], [], s=95, color=viz.SERIES[0], label=TR("ramo ClusWiSARD (P1, P3, P4)"))
    ax.scatter([], [], s=95, color=viz.SERIES[1], label=TR("ramo classico (P2, P5, P6)"))
    ax.set_xlabel(TR("recall"))
    ax.set_ylabel(TR("precisao"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(TR("Plano precisao-recall (teste)"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=1)
    viz.despine(ax)
    viz.save(fig, os.path.join(FIG, "etapa7_fig2_precisao_recall.png"))


def fig_por_estacao():
    d = {}
    for n, lbl in rotulos():
        p = os.path.join(H.TAB_DIR, f"{n.split('_')[0]}_por_estacao.csv")
        if os.path.exists(p):
            t = pd.read_csv(p, index_col=0)
            d[lbl] = t["F1"]
    if not d:
        return
    T_ = pd.DataFrame(d)
    T_ = T_.loc[T_.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    yy = np.arange(len(T_))
    h = 0.8 / len(T_.columns)
    for i, c in enumerate(T_.columns):
        ax.barh(yy + (i - (len(T_.columns) - 1) / 2) * h, T_[c], height=h * 0.9,
                color=viz.SERIES[i], label=c, edgecolor=viz.SURFACE, lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels(T_.index, fontsize=8)
    ax.set_xlabel(TR("F1 (teste)"))
    ax.set_title(TR("F1 por estacao de vigilancia"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3)
    ax.grid(axis="y", visible=False)
    viz.despine(ax)
    viz.save(fig, os.path.join(FIG, "etapa7_fig3_por_estacao.png"))
