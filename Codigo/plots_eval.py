"""
plots_eval.py -- Extracao de plots e protocolo de avaliacao.

Duas funcoes canonicas, compartilhadas por todas as seis propostas, de modo
que a comparacao entre elas isole exatamente o que se deseja comparar:

  * `extract_plots`  : celulas de video -> plots primitivos (componentes
                       conexas em grade metrica). E o extrator classico de
                       plots de um radar de vigilancia.
  * `evaluate`       : plots -> (VP, FP, FN, precisao, recall, F1) contra o
                       ground truth, com associacao otima pelo algoritmo
                       hungaro sob porta de associacao.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

PIXEL = 5.0        # lado do pixel da grade metrica de aglutinacao [m]
MIN_CELLS = 3      # minimo de celulas para um plot primitivo
GATE = 100.0       # porta de associacao plot <-> ground truth [m]


@dataclass
class PlotSet:
    """Conjunto de plots de uma varredura."""

    x: np.ndarray          # centroide (m, relativo a estacao)
    y: np.ndarray
    amp: np.ndarray        # amplitude total
    n: np.ndarray          # numero de celulas
    ext: np.ndarray        # extensao (raio equivalente, m)
    vx: np.ndarray | None = None
    vy: np.ndarray | None = None

    def __len__(self):
        return len(self.x)

    @property
    def xy(self):
        return np.c_[self.x, self.y]


def extract_plots(x, y, amp, pixel=PIXEL, min_cells=MIN_CELLS, connectivity=2) -> PlotSet:
    """Aglutina celulas de video em plots por componentes conexas.

    As celulas sao rasterizadas em uma grade cartesiana de `pixel` metros e
    rotuladas com conectividade-8; cada componente com ao menos `min_cells`
    celulas produz um plot, cujo centroide e ponderado pela amplitude.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    amp = np.asarray(amp, float)
    if len(x) == 0:
        z = np.zeros(0)
        return PlotSet(z, z, z, z.astype(int), z)
    ix = np.floor((x - x.min()) / pixel).astype(np.int64)
    iy = np.floor((y - y.min()) / pixel).astype(np.int64)
    nx, ny = ix.max() + 1, iy.max() + 1
    if nx * ny > 60_000_000:      # protecao contra cenas muito extensas
        pixel *= 2
        ix = np.floor((x - x.min()) / pixel).astype(np.int64)
        iy = np.floor((y - y.min()) / pixel).astype(np.int64)
        nx, ny = ix.max() + 1, iy.max() + 1
    grid = np.zeros((nx, ny), bool)
    grid[ix, iy] = True
    struct = ndimage.generate_binary_structure(2, connectivity)
    lab, nlab = ndimage.label(grid, structure=struct)
    if nlab == 0:
        z = np.zeros(0)
        return PlotSet(z, z, z, z.astype(int), z)
    cl = lab[ix, iy]                       # rotulo de cada celula
    order = np.argsort(cl, kind="stable")
    cls = cl[order]
    bnd = np.r_[0, np.flatnonzero(np.diff(cls)) + 1, len(cls)]
    xs, ys, amps, ns, exts = [], [], [], [], []
    for k in range(len(bnd) - 1):
        sel = order[bnd[k]:bnd[k + 1]]
        if len(sel) < min_cells:
            continue
        w = amp[sel]
        sw = w.sum()
        if sw <= 0:
            continue
        cx = float((x[sel] * w).sum() / sw)
        cy = float((y[sel] * w).sum() / sw)
        xs.append(cx)
        ys.append(cy)
        amps.append(float(sw))
        ns.append(len(sel))
        exts.append(float(np.sqrt(((x[sel] - cx) ** 2 + (y[sel] - cy) ** 2).mean())))
    return PlotSet(np.array(xs), np.array(ys), np.array(amps),
                   np.array(ns, int), np.array(exts))


def match(plots_xy: np.ndarray, gt_xy: np.ndarray, gate: float = GATE):
    """Associacao otima (algoritmo hungaro) sob porta `gate`.

    Retorna (idx_plot, idx_gt) dos pares associados.
    """
    if len(plots_xy) == 0 or len(gt_xy) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    d = np.hypot(plots_xy[:, None, 0] - gt_xy[None, :, 0],
                 plots_xy[:, None, 1] - gt_xy[None, :, 1])
    big = gate * 1e3
    c = np.where(d <= gate, d, big)
    ri, ci = linear_sum_assignment(c)
    ok = c[ri, ci] <= gate
    return ri[ok], ci[ok]


def confusion(plots_xy: np.ndarray, gt_xy: np.ndarray, gate: float = GATE):
    """(VP, FP, FN) de uma varredura."""
    ri, ci = match(plots_xy, gt_xy, gate)
    vp = len(ri)
    return vp, len(plots_xy) - vp, len(gt_xy) - vp


def metrics(vp: int, fp: int, fn: int) -> dict:
    prec = vp / (vp + fp) if vp + fp else 0.0
    rec = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(VP=vp, FP=fp, FN=fn, precisao=prec, recall=rec, F1=f1)
