"""
pipeline.py -- Infraestrutura comum as Etapas 1 a 7.

* execucao paralela de um detector sobre os 39 recortes do DataSet_tratado;
* persistencia dos plots detectados (usados como entrada das etapas de
  clusterizacao);
* avaliacao contra o ground truth da Etapa 0, com associacao pelo algoritmo
  hungaro e agregacao de VP, FP, FN, precisao, recall e F1.

Todos os plots sao gravados no **referencial cartesiano comum** (absoluto),
o mesmo do ground truth.
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import haxr_io as H
import plots_eval as P
import tratado_io as T

RES_DIR = os.path.join(H.ROOT, "Resultados")
os.makedirs(RES_DIR, exist_ok=True)

PLOT_COLS = ["station", "hour", "scan", "x", "y", "amp", "n", "ext", "vx", "vy"]


# ---------------------------------------------------------------------------
# persistencia
# ---------------------------------------------------------------------------


def save_detections(df: pd.DataFrame, name: str) -> str:
    p = os.path.join(RES_DIR, f"{name}_plots.csv.gz")
    df.to_csv(p, index=False)
    return p


def load_detections(name: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(RES_DIR, f"{name}_plots.csv.gz"))
    d["hour"] = d["hour"].astype(str).str.zfill(2)
    return d


def plots_to_frame(station, hour, scan, ps: P.PlotSet, x0, y0) -> pd.DataFrame:
    n = len(ps)
    return pd.DataFrame({
        "station": np.repeat(station, n), "hour": np.repeat(hour, n),
        "scan": np.repeat(scan, n),
        "x": ps.x + x0, "y": ps.y + y0, "amp": ps.amp, "n": ps.n, "ext": ps.ext,
        "vx": ps.vx if ps.vx is not None else np.full(n, np.nan),
        "vy": ps.vy if ps.vy is not None else np.full(n, np.nan)})


# ---------------------------------------------------------------------------
# execucao
# ---------------------------------------------------------------------------


def run_chunks(worker, jobs=None, n_jobs=6, hours=None):
    """Executa `worker((station, hour))` em paralelo e concatena os DataFrames."""
    jobs = jobs or T.list_chunks(hours)
    with Pool(n_jobs) as p:
        parts = [x for x in p.map(worker, jobs) if x is not None and len(x)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PLOT_COLS)


# ---------------------------------------------------------------------------
# avaliacao
# ---------------------------------------------------------------------------


def evaluate(plots: pd.DataFrame, gt: pd.DataFrame, gate: float = P.GATE,
             hours=None, only_confirmed=False, window=None) -> tuple[dict, pd.DataFrame]:
    """Avalia plots contra o ground truth.

    `window` restringe a avaliacao a uma das janelas de 120 s de cada recorte
    -- usado para separar ajuste (janela 1) de treino da rede (janela 0).

    Retorna (metricas globais, tabela por varredura).
    """
    if hours is not None:
        gt = gt[gt["hour"].isin(hours)]
        plots = plots[plots["hour"].isin(hours)]
    if window is not None:
        w = scan_meta()
        keep = set(map(tuple, w.loc[w["window"] == window,
                                    ["station", "hour", "scan"]].to_numpy()))
        gt = gt[[tuple(r) in keep for r in gt[["station", "hour", "scan"]].to_numpy()]]
        plots = plots[[tuple(r) in keep
                       for r in plots[["station", "hour", "scan"]].to_numpy()]]
    if only_confirmed:
        gt = gt[gt["status"] != "nao_confirmado"]
    gp = {k: v for k, v in plots.groupby(["station", "hour", "scan"])}
    rows = []
    for key, g in gt.groupby(["station", "hour", "scan"]):
        pxy = gp[key][["x", "y"]].to_numpy() if key in gp else np.zeros((0, 2))
        gxy = g[["X_ais", "Y_ais"]].to_numpy()
        vp, fp, fn = P.confusion(pxy, gxy, gate)
        rows.append(dict(station=key[0], hour=key[1], scan=key[2],
                         VP=vp, FP=fp, FN=fn, n_plots=len(pxy), n_gt=len(gxy)))
    per = pd.DataFrame(rows)
    m = P.metrics(int(per.VP.sum()), int(per.FP.sum()), int(per.FN.sum()))
    m["varreduras"] = len(per)
    m["plots"] = int(per.n_plots.sum())
    m["alvos"] = int(per.n_gt.sum())
    m["plots_por_alvo"] = round(per.n_plots.sum() / max(per.n_gt.sum(), 1), 2)
    return m, per


def report(name: str, m: dict, per: pd.DataFrame, extra: dict | None = None):
    """Grava e imprime o resultado de uma proposta."""
    per.to_csv(os.path.join(RES_DIR, f"{name}_por_varredura.csv"), index=False)
    row = dict(proposta=name, **m, **(extra or {}))
    pd.DataFrame([row]).to_csv(os.path.join(RES_DIR, f"{name}_metricas.csv"), index=False)
    print(f"\n=== {name} ===")
    for k, v in row.items():
        if k == "proposta":
            continue
        print(f"  {k:18s} {v:.4f}" if isinstance(v, float) else f"  {k:18s} {v}")
    return row


def by_station(plots, gt, gate=P.GATE, hours=None):
    """Metricas desagregadas por estacao."""
    _, per = evaluate(plots, gt, gate, hours)
    g = per.groupby("station")[["VP", "FP", "FN"]].sum()
    out = g.apply(lambda r: pd.Series(P.metrics(int(r.VP), int(r.FP), int(r.FN))), axis=1)
    return out.round(4)


_META = None


def scan_meta() -> pd.DataFrame:
    """Tabela (station, hour, scan) -> window, t, periodo de rotacao."""
    global _META
    if _META is None:
        rows = []
        for s, h in T.list_chunks():
            d = T.Tratado(s, h)
            rows.append(pd.DataFrame(dict(
                station=s, hour=h, scan=np.arange(d.n_scan),
                window=d.scan["window"].to_numpy(), t=d.scan["t"].to_numpy(),
                periodo=d.period)))
            d.close()
        _META = pd.concat(rows, ignore_index=True)
    return _META


def add_scan_meta(plots: pd.DataFrame) -> pd.DataFrame:
    return plots.merge(scan_meta(), on=["station", "hour", "scan"], how="left")
