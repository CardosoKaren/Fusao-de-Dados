"""
viz.py -- Parametros graficos comuns a todas as figuras do trabalho.

Paleta categorica validada (ordem fixa, nunca ciclada). Formas do tipo
"todos os pares" (dispersao, mapas) usam no maximo os tres primeiros
slots, que sao os validados para esse caso; formas com pares adjacentes
(barras, linhas) podem usar a ordem completa.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- paleta categorica (light) ---------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES3 = SERIES[:3]           # unico subconjunto validado para "todos os pares"

# --- tinta e superficie -----------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8880"
GRID = "#e3e2de"

# --- rampa sequencial de um unico matiz (azul 100 -> 700) -------------------
SEQ_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fe"] + SEQ_STEPS)
CMAP_SEQ2 = LinearSegmentedColormap.from_list(
    "seq_orange", ["#fdf3ee", "#f7c9b1", "#f0a37f", "#eb6834", "#c14a1d", "#8a3413"])

STATUS = {"confirmado_fusao": "#1baf7a", "confirmado_local": "#2a78d6",
          "nao_confirmado": "#e34948"}


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK3,
        "ytick.color": INK3,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": INK2,
        "lines.linewidth": 2.0,
        "font.size": 9,
        "font.family": "DejaVu Sans",
        "figure.dpi": 140,
    })


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=170)
    plt.close(fig)
    print("figura:", path)
