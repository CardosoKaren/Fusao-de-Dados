"""
gerar_figuras_etapas.py -- Figuras das Etapas 1 a 6.

Uso:  python3 Codigo/gerar_figuras_etapas.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as Fe
import figuras_resultados as FR
import haxr_io as H
import i18n
import pipeline as PL
from i18n import T as TR
import tratado_io as T
import viz

viz.apply_style()
import matplotlib.pyplot as plt

FIG = H.FIG_DIR
CENA = ("koehlbrandhoeft", "09")


def _cena_idx(d, n=1):
    return int(np.argsort([-len(d.targets(i)) for i in range(d.n_scan)])[n])


# ------------------------------------------------------------------ etapa 1
def etapa1_figs():
    import etapa1_filtro_cluswisard as E1
    FR.fig_ajuste("etapa1_ajuste_limiar.csv", "limiar", TR("limiar de decisao"),
                  "etapa1_fig1_ajuste.png", group="modo", glabel="modo",
                  titulo=TR("Ajuste do limiar de decisao da ClusWiSARD"))
    met = pd.read_csv(os.path.join(PL.RES_DIR, "etapa1_cluswisard_metricas.csv")).iloc[0]
    with open(E1.MODEL_PATH, "rb") as f:
        M = pickle.load(f)
    modo, thr = str(met["modo"]), float(met["limiar"])

    def mask(d, k):
        B, info = E1.retina(d, k)
        s = M.scores(B)
        r1, r0 = s.get(1, 0.0), s.get(0, 0.0)
        m = (r1 - r0) if modo == "dif" else (r1 - r0) / np.maximum(r1 + r0, 1e-9)
        return m > thr

    FR.cena(*CENA, mask, TR("Filtro ClusWiSARD -- celulas de video aprovadas"),
            "etapa1_fig2_cena.png", k=_cena_idx(T.Tratado(*CENA)))


# ------------------------------------------------------------------ etapa 2
def etapa2_figs():
    import etapa2_filtro_oscfar as E2
    tab = pd.read_csv(os.path.join(H.TAB_DIR, "etapa2_ajuste_cfar.csv"))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    g0 = tab[tab["guarda"] == tab.loc[tab["F1"].idxmax(), "guarda"]]
    for i, (q, t) in enumerate(g0.groupby("ordem")):
        t = t.sort_values("alpha")
        ax.plot(t["alpha"], t["F1"], marker="o", ms=4,
                color=viz.SERIES[i % len(viz.SERIES)], label=TR("ordem k = {q:.2f}").format(q=q))
    b = tab.loc[tab["F1"].idxmax()]
    ax.scatter([b["alpha"]], [b["F1"]], s=90, facecolor="none", edgecolor=viz.INK,
               lw=1.4, zorder=6)
    ax.annotate(TR("maximo F1={v:.3f}").format(v=b["F1"]), (b["alpha"], b["F1"]), xytext=(8, 8),
                textcoords="offset points", fontsize=8.5, color=viz.INK2)
    ax.set_xlabel(TR(r"fator de escala $\alpha$"))
    ax.set_ylabel(TR("F1 (particao de treino, 08 UTC)"))
    ax.set_title(TR("Ajuste do OS-CFAR (raio de guarda {g:.0f} m)").format(g=b["guarda"]))
    ax.legend(loc="best", ncol=2)
    viz.despine(ax)
    viz.save(fig, os.path.join(FIG, "etapa2_fig1_ajuste.png"))

    met = pd.read_csv(os.path.join(PL.RES_DIR, "etapa2_oscfar_metricas.csv")).iloc[0]
    g, q, a = float(met["guarda"]), float(met["ordem"]), float(met["alpha"])

    def mask(d, k):
        lev, info = E2.cfar_levels(d, k, (g,), (q,))
        return info["amp"].astype(float) > a * lev[(g, q)]

    FR.cena(*CENA, mask, TR("Filtro OS-CFAR -- celulas de video aprovadas"),
            "etapa2_fig2_cena.png", k=_cena_idx(T.Tratado(*CENA)))


# ------------------------------------------------- efeito da clusterizacao
def cluster_fig(src, dst, titulo, out, ajuste_csv=None, xcol=None, xlabel=None):
    a = PL.load_detections(src)
    b = PL.load_detections(dst)
    d = T.Tratado(*CENA)
    k = _cena_idx(d)
    key = (CENA[0], CENA[1], k)
    A = a[(a.station == key[0]) & (a.hour == key[1]) & (a.scan == key[2])]
    B = b[(b.station == key[0]) & (b.hour == key[1]) & (b.scan == key[2])]
    G = Fe.gt_xy(d, k) + np.array([d.x0, d.y0])
    az, ri, amp = d.cells(k)
    x, y = d.cell_xy(az, ri)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4))
    for ax, S, t in [(axes[0], A, TR("(a) antes -- {n} plots primitivos").format(n=len(A))),
                     (axes[1], B, TR("(b) depois -- {n} plots agrupados").format(n=len(B)))]:
        ax.scatter(x + d.x0, y + d.y0, s=0.7, color=viz.INK3, alpha=0.22, lw=0)
        ax.scatter(S["x"], S["y"], s=22, color=viz.SERIES[0], lw=0, label=TR("plots"))
        ax.scatter(G[:, 0], G[:, 1], s=80, facecolor="none", edgecolor=viz.SERIES[1],
                   lw=1.2, label=TR("ground truth AIS"))
        ax.set_aspect("equal")
        ax.set_title(t)
        ax.set_xlabel("X [m]")
        viz.despine(ax)
    axes[0].set_ylabel("Y [m]")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.suptitle(titulo, fontsize=11, color=viz.INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    viz.save(fig, os.path.join(FIG, out))
    d.close()
    if ajuste_csv:
        FR.fig_ajuste(ajuste_csv, xcol, xlabel, out.replace("cena", "ajuste"))


def etapa3_figs():
    cluster_fig("etapa1_cluswisard", "etapa3_cluswisard_cluster",
                TR("Clusterizador ClusWiSARD sobre os plots do filtro ClusWiSARD"),
                "etapa3_fig1_cena.png")
    FR.fig_ajuste("etapa3_ajuste_clusterizador.csv", "min_score",
                  TR("limiar de resposta do clusterizador"), "etapa3_fig2_ajuste.png",
                  titulo=TR("Ajuste do clusterizador ClusWiSARD"))


def etapa5_figs():
    cluster_fig("etapa2_oscfar", "etapa5_oscfar_dbscan",
                TR("DBSCAN sobre os plots do filtro OS-CFAR"), "etapa5_fig1_cena.png")
    FR.fig_ajuste("etapa5_ajuste_dbscan.csv", "eps", TR(r"raio $\varepsilon$ do DBSCAN [m]"),
                  "etapa5_fig2_ajuste.png", titulo=TR("Ajuste do DBSCAN"))


# ------------------------------------------------------------ cinematica
def cinematica_fig():
    import cluster_common as CC
    pl = PL.load_detections("etapa1_cluswisard")
    pl = CC.add_kinematics(pl, PL.scan_meta())
    gt = T.load_gt()
    gt = gt[gt["hour"].isin(T.TEST_HOURS)]
    v = np.hypot(pl["vx"], pl["vy"])
    vg = np.hypot(gt["vx"], gt["vy"])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    bins = np.arange(0, 10.2, 0.4)
    axes[0].hist(vg, bins=bins, color=viz.SERIES[1], edgecolor=viz.SURFACE, lw=1.2,
                 density=True, label=TR("AIS (referencia)"))
    axes[0].hist(v[v < 10], bins=bins, color=viz.SERIES[0], alpha=0.75, lw=1.2,
                 edgecolor=viz.SURFACE, density=True, label=TR("estimada dos plots"))
    axes[0].set_xlabel(TR("modulo da velocidade [m/s]"))
    axes[0].set_ylabel(TR("densidade"))
    axes[0].set_title(TR("(a) velocidade dos plots x velocidade AIS"))
    axes[0].legend(loc="upper right")
    viz.despine(axes[0])

    tab = pd.read_csv(os.path.join(H.TAB_DIR, "etapa4_ajuste_clusterizador.csv"))
    for i, (vb, t) in enumerate(tab.groupby("bits_velocidade")):
        t = t.sort_values("min_score")
        axes[1].plot(t["min_score"], t["F1"], marker="o", ms=4,
                     color=viz.SERIES[i % 3], label=TR("{vb} bits de velocidade").format(vb=vb))
    t3 = pd.read_csv(os.path.join(H.TAB_DIR, "etapa3_ajuste_clusterizador.csv")).sort_values("min_score")
    axes[1].plot(t3["min_score"], t3["F1"], color=viz.INK3, ls="--", marker="o", ms=4,
                 label=TR("sem cinematica (Etapa 3)"))
    axes[1].set_xlabel(TR("limiar de resposta do clusterizador"))
    axes[1].set_ylabel(TR("F1 (janela de validacao, 08 UTC)"))
    axes[1].set_title(TR("(b) efeito da cinematica no agrupamento"))
    axes[1].legend(loc="best", fontsize=7.5)
    viz.despine(axes[1])
    viz.save(fig, os.path.join(FIG, "etapa4_fig1_cinematica.png"))


if __name__ == "__main__":
    for fn in (etapa1_figs, etapa2_figs, etapa3_figs, etapa5_figs, cinematica_fig):
        try:
            fn()
        except Exception as e:
            print(f"[pulado] {fn.__name__}: {type(e).__name__}: {e}")
