"""
etapa0_figs.py -- Figuras e tabelas da Etapa 0.

Gera:
  fig1  cobertura das 13 estacoes no referencial cartesiano comum
  fig2  efeito do recorte: video bruto completo x DataSet_tratado
  fig3  atribuicao hungara possiveis-plots <-> AIS em uma cena
  fig4  deslocamento AIS<->eco e concordancia entre estacoes
  fig5  status de confirmacao do ground truth por estacao
  fig6  fusao multiestacao de um mesmo alvo
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import haxr_io as H
import i18n
import plots_eval as P
from i18n import T as TR
import tratado_io as T
import viz

viz.apply_style()
import matplotlib.pyplot as plt

FIG = H.FIG_DIR
os.makedirs(FIG, exist_ok=True)
GT = pd.read_csv(os.path.join(H.OUT_DIR, "ground_truth.csv"))
STN = H.load_stations()


# ---------------------------------------------------------------- figura 1
def fig1():
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for st in STN.index:
        d = T.Tratado(st, "08")
        prof = d.coverage
        az = np.arange(len(prof)) * (360.0 / len(prof))
        r = prof.copy()
        r[r == 0] = np.nan
        x = d.x0 + r * np.sin(np.deg2rad(az))
        y = d.y0 + r * np.cos(np.deg2rad(az))
        ax.fill(np.r_[x, x[:1]], np.r_[y, y[:1]], color=viz.SERIES[0], alpha=0.11, lw=0)
        ax.plot(np.r_[x, x[:1]], np.r_[y, y[:1]], color=viz.SERIES[0], lw=0.8, alpha=0.55)
        d.close()
    ax.scatter(STN["x"], STN["y"], s=34, color=viz.SERIES[1], zorder=5,
               edgecolor=viz.SURFACE, linewidth=1.6)
    # deslocamentos manuais: as estacoes se aglomeram no centro do porto
    POS = {"altona": (0, 10, "center"), "koehlbrandhoeft": (-11, -4, "right"),
           "landungsbruecken": (0, 10, "center"), "ellerholzhoeft": (0, -16, "center"),
           "amerikahoeft": (0, -16, "center"), "seemannshoeft": (0, 10, "center"),
           "parkhafen": (-11, -12, "right"), "sandauhafen": (-9, 6, "right"),
           "krusenbusch": (9, -16, "left"), "kattwyk": (-9, -16, "right"),
           "reiherstieg": (11, -4, "left"), "hohe_schaar": (0, -18, "center"),
           "nesssand": (0, 10, "center")}
    for st, row in STN.iterrows():
        dx, dy, ha = POS.get(st, (0, 10, "center"))
        ax.annotate(st, (row["x"], row["y"]), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=7.0, color=viz.INK2)
    ax.set_aspect("equal")
    ax.set_xlabel(TR("X [m]  (Leste)"))
    ax.set_ylabel(TR("Y [m]  (Norte)"))
    ax.set_title(TR("Envelopes de cobertura das 13 estacoes de vigilancia"))
    ax.plot([], [], color=viz.SERIES[0], lw=2,
               label=TR(r"envelope de cobertura $r_{max}(\theta)$"))
    ax.scatter([], [], s=34, color=viz.SERIES[1], label=TR("estacao"))
    ax.legend(loc="upper left")
    viz.despine(ax)
    viz.save(fig, os.path.join(FIG, "etapa0_fig1_cobertura.png"))


# ---------------------------------------------------------------- figura 2
def fig2(station="amerikahoeft", hour="08", k=40):
    d = T.Tratado(station, hour)
    cyc = int(d.scan["cycle"][k])
    rf = H.RadarFile(station, hour)
    s = rf.load_cycle(cyc)
    rf.close()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4))
    X = s.r * np.sin(np.deg2rad(s.az))
    Y = s.r * np.cos(np.deg2rad(s.az))
    axes[0].scatter(X, Y, c=s.amp, s=0.35, cmap=viz.CMAP_SEQ, vmin=0, vmax=150, lw=0)
    axes[0].set_title(TR("(a) video bruto -- varredura completa"))
    az, ri, amp = d.cells(k)
    x, y = d.cell_xy(az, ri)
    sc = axes[1].scatter(x, y, c=amp, s=0.9, cmap=viz.CMAP_SEQ, vmin=0, vmax=150, lw=0)
    tg = d.targets(k)
    tx = tg["r"] * np.sin(np.deg2rad(tg["az"]))
    ty = tg["r"] * np.cos(np.deg2rad(tg["az"]))
    th = np.linspace(0, 2 * np.pi, 90)
    for a, b in zip(tx, ty):
        axes[1].plot(a + d.roi * np.cos(th), b + d.roi * np.sin(th),
                     color=viz.SERIES[1], lw=0.7, alpha=0.8)
    axes[1].scatter(tx, ty, s=16, color=viz.SERIES[1], marker="x", lw=1.1)
    axes[1].set_title(TR("(b) DataSet_tratado -- ROI de 200 m em torno de alvos AIS"))
    axes[1].plot([], [], color=viz.SERIES[1], lw=1.4, label=TR("ROI / posicao AIS"))
    axes[1].legend(loc="upper right")
    lim = max(np.abs(X).max(), np.abs(Y).max()) * 1.02
    for a in axes:
        a.set_aspect("equal")
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_xlabel("x [m]")
        viz.despine(a)
    axes[0].set_ylabel("y [m]")
    cb = fig.colorbar(sc, ax=axes, fraction=0.026, pad=0.03)
    cb.set_label(TR("amplitude do video"), color=viz.INK2, fontsize=8.5)
    cb.outline.set_visible(False)
    fig.suptitle(TR("Recorte do dataset -- {station} {hour} UTC, varredura {k}")
                 .format(station=station, hour=hour, k=k),
                 fontsize=11, color=viz.INK, y=1.02)
    viz.save(fig, os.path.join(FIG, "etapa0_fig2_recorte.png"))
    d.close()


# ---------------------------------------------------------------- figura 3
def fig3(station="koehlbrandhoeft", hour="08"):
    d = T.Tratado(station, hour)
    # cena com muitos alvos e muitos possiveis plots
    best, bk = -1, 0
    for k in range(d.n_scan):
        n = len(d.targets(k))
        if n > best:
            best, bk = n, k
    az, ri, amp = d.cells(bk)
    x, y = d.cell_xy(az, ri)
    ps = P.extract_plots(x, y, amp)
    tg = d.targets(bk)
    gx = tg["r"].to_numpy() * np.sin(np.deg2rad(tg["az"].to_numpy()))
    gy = tg["r"].to_numpy() * np.cos(np.deg2rad(tg["az"].to_numpy()))
    pi, gi = P.match(ps.xy, np.c_[gx, gy], 100.0)

    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    ax.scatter(x, y, c=amp, s=0.8, cmap=viz.CMAP_SEQ, vmin=0, vmax=150, lw=0, alpha=0.85)
    ax.scatter(ps.x, ps.y, s=26, facecolor="none", edgecolor=viz.SERIES[1],
               lw=1.0, label=TR("possiveis plots (n={n})").format(n=len(ps)))
    ax.scatter(gx, gy, s=44, color=viz.SERIES[2], marker="+", lw=1.4,
               label=TR("AIS visivel (n={n})").format(n=len(gx)))
    for p, g in zip(pi, gi):
        ax.plot([ps.x[p], gx[g]], [ps.y[p], gy[g]], color=viz.INK2, lw=1.0, alpha=0.9)
    ax.plot([], [], color=viz.INK2, lw=1.4, label=TR("atribuicao hungara (n={n})").format(n=len(pi)))
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(TR(r"Atribuicao possivel-plot $\leftrightarrow$ AIS -- {station} {hour} UTC")
                 .format(station=station, hour=hour))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    viz.despine(ax)
    viz.save(fig, os.path.join(FIG, "etapa0_fig3_hungaro.png"))
    d.close()


# ---------------------------------------------------------------- figura 4
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    a = axes[0]
    v = GT["d_eco"].dropna().to_numpy()
    a.hist(v, bins=np.arange(0, 102, 4), color=viz.SERIES[0], edgecolor=viz.SURFACE, lw=1.2)
    a.axvline(np.median(v), color=viz.INK2, lw=1.2, ls="--")
    a.annotate(TR("mediana {v:.0f} m").format(v=np.median(v)), (np.median(v), a.get_ylim()[1] * 0.92),
               xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=viz.INK2)
    a.annotate(TR("p90 {v:.0f} m").format(v=np.percentile(v, 90)), (np.percentile(v, 90), a.get_ylim()[1] * 0.72),
               xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=viz.INK2)
    a.set_title(TR(r"(a) deslocamento AIS $\rightarrow$ eco associado"))
    a.set_xlabel(TR("distancia [m]"))
    a.set_ylabel(TR("alvos-varredura"))
    b = axes[1]
    w = GT["d_fusao"].dropna().to_numpy()
    b.hist(w, bins=np.arange(0, 102, 4), color=viz.SERIES[1], edgecolor=viz.SURFACE, lw=1.2)
    b.axvline(np.median(w), color=viz.INK2, lw=1.2, ls="--")
    b.annotate(TR("mediana {v:.0f} m").format(v=np.median(w)), (np.median(w), b.get_ylim()[1] * 0.92),
               xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=viz.INK2)
    b.set_title(TR("(b) discordancia entre estacoes sobrepostas"))
    b.set_xlabel(TR("distancia entre ecos atribuidos ao mesmo AIS [m]"))
    b.set_ylabel(TR("alvos-varredura"))
    for x in axes:
        viz.despine(x)
    viz.save(fig, os.path.join(FIG, "etapa0_fig4_offsets.png"))


# ---------------------------------------------------------------- figura 5
def fig5():
    piv = (GT.groupby(["station", "status"]).size().unstack(fill_value=0))
    order = ["confirmado_fusao", "confirmado_local", "nao_confirmado"]
    piv = piv.reindex(columns=order, fill_value=0)
    frac = piv.div(piv.sum(axis=1), axis=0) * 100
    frac = frac.sort_values("confirmado_fusao", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    left = np.zeros(len(frac))
    labels = {"confirmado_fusao": TR("confirmado por fusao multiestacao"),
              "confirmado_local": TR("confirmado apenas localmente"),
              "nao_confirmado": TR("sem eco associado")}
    for c in order:
        ax.barh(frac.index, frac[c], left=left, height=0.66,
                color=viz.STATUS[c], label=labels[c], edgecolor=viz.SURFACE, lw=1.4)
        left += frac[c].to_numpy()
    for i, st in enumerate(frac.index):
        ax.annotate(f"{100 - frac.loc[st, 'nao_confirmado']:.0f}%", (100.8, i),
                    va="center", fontsize=8, color=viz.INK2)
    ax.set_xlim(0, 108)
    ax.set_xlabel(TR("percentual dos alvos-varredura [%]"))
    ax.set_title(TR("Confirmacao do ground truth por estacao"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    ax.grid(axis="y", visible=False)
    viz.despine(ax)
    viz.save(fig, os.path.join(FIG, "etapa0_fig5_confirmacao.png"))


# ---------------------------------------------------------------- figura 6
def fig6():
    """Um mesmo alvo visto simultaneamente por varias estacoes sobrepostas."""
    sub = GT[np.isfinite(GT["X_eco"])].copy()
    cnt = sub.groupby(["hour", "uid", "tbin"])["station"].nunique().sort_values(ascending=False)
    key = cnt.index[0]
    g = sub[(sub.hour == key[0]) & (sub.uid == key[1]) & (sub.tbin == key[2])]
    g = g.groupby("station").first().reset_index()
    cx, cy = g["X_ais"].mean(), g["Y_ais"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    a = axes[0]
    for _, row in g.iterrows():
        sx, sy = STN.loc[row["station"], "x"], STN.loc[row["station"], "y"]
        a.plot([sx, row["X_eco"]], [sy, row["Y_eco"]], color=viz.SERIES[0],
               lw=0.7, alpha=0.45)
        a.scatter(sx, sy, s=30, color=viz.SERIES[0], zorder=4)
        a.annotate(row["station"], (sx, sy), xytext=(0, 8), textcoords="offset points",
                   ha="center", fontsize=7, color=viz.INK2)
    a.scatter(cx, cy, s=70, color=viz.SERIES[1], marker="+", lw=1.8, zorder=6)
    a.set_aspect("equal")
    a.set_xlabel("X [m]")
    a.set_ylabel("Y [m]")
    a.set_title(TR("(a) {n} estacoes observam o alvo {uid}").format(n=len(g), uid=key[1]))
    a.scatter([], [], s=30, color=viz.SERIES[0], label=TR("estacao / linha de visada"))
    a.scatter([], [], s=70, color=viz.SERIES[1], marker="+", lw=1.8, label=TR("posicao AIS"))
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    viz.despine(a)

    b = axes[1]
    b.scatter(g["X_eco"], g["Y_eco"], s=64, facecolor="none",
              edgecolor=viz.SERIES[0], lw=1.5, zorder=4)
    for _, row in g.iterrows():
        b.annotate(row["station"], (row["X_eco"], row["Y_eco"]), xytext=(8, 5),
                   textcoords="offset points", fontsize=7, color=viz.INK2)
    b.scatter(g["X_ais"], g["Y_ais"], s=80, color=viz.SERIES[1], marker="+", lw=1.8, zorder=5)
    W = 90
    b.set_xlim(cx - W, cx + W)
    b.set_ylim(cy - W, cy + W)
    b.set_aspect("equal")
    b.set_xlabel("X [m]")
    b.set_ylabel("Y [m]")
    d = np.hypot(g["X_eco"] - cx, g["Y_eco"] - cy)
    b.set_title(TR("(b) ecos atribuidos -- dispersao mediana {d:.0f} m").format(d=np.median(d)))
    b.scatter([], [], s=64, facecolor="none", edgecolor=viz.SERIES[0], lw=1.5,
              label=TR("eco atribuido por cada estacao"))
    b.scatter([], [], s=80, color=viz.SERIES[1], marker="+", lw=1.8, label=TR("posicao AIS"))
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    viz.despine(b)
    viz.save(fig, os.path.join(FIG, "etapa0_fig6_fusao.png"))


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
