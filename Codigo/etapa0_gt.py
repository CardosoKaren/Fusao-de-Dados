"""
etapa0_gt.py -- Construcao e validacao do ground truth do DataSet_tratado.

Procedimento:
  1. De cada varredura extraem-se os *possiveis plots* diretamente do video
     bruto recortado (componentes conexas em grade metrica de 5 m), sem
     qualquer filtragem -- e a hipotese mais permissiva possivel.
  2. Em cada varredura pode haver varios possiveis plots e varios alvos AIS
     visiveis. A atribuicao "qual possivel plot pertence a qual AIS" e
     resolvida globalmente pelo *algoritmo hungaro* (custo = distancia
     euclidiana, porta de 100 m), e nao por vizinho mais proximo -- o que
     evita que dois AIS disputem o mesmo eco.
  3. As 13 estacoes possuem campos de visao sobrepostos. As atribuicoes de
     um mesmo `uid` obtidas por estacoes distintas em instantes proximos sao
     levadas ao referencial cartesiano comum e comparadas: a concordancia
     entre estacoes confirma, de forma independente, a qual eco o AIS
     corresponde.

Saidas: Tabelas/etapa0_gt*.csv, DataSet_tratado/ground_truth.csv, Figuras/etapa0_*.png
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

GATE_GT = 100.0        # porta AIS <-> possivel plot [m]
FUSE_DT = 1.6          # tolerancia temporal entre estacoes [s]
FUSE_TOL = 60.0        # concordancia espacial entre estacoes [m]


def process_chunk(args):
    station, hour = args
    d = T.Tratado(station, hour)
    rows = []
    for i in range(d.n_scan):
        az, ri, amp = d.cells(i)
        tg = d.targets(i)
        if len(tg) == 0:
            continue
        x, y = d.cell_xy(az, ri)
        ps = P.extract_plots(x, y, amp)
        gx = tg["r"].to_numpy() * np.sin(np.deg2rad(tg["az"].to_numpy()))
        gy = tg["r"].to_numpy() * np.cos(np.deg2rad(tg["az"].to_numpy()))
        gt_xy = np.c_[gx, gy]
        pi, gi = P.match(ps.xy, gt_xy, GATE_GT)
        assign = {int(g): int(p) for p, g in zip(pi, gi)}
        t = float(d.scan["t"][i])
        for k, (_, row) in enumerate(tg.iterrows()):
            j = assign.get(k, -1)
            rows.append(dict(
                station=station, hour=hour, scan=i, t=t, uid=row["uid"],
                X_ais=row["X"], Y_ais=row["Y"], r=row["r"], az=row["az"],
                vx=row["vx"], vy=row["vy"],
                n_plots=len(ps), n_tgt=len(tg),
                plot=j,
                X_eco=(ps.x[j] + d.x0) if j >= 0 else np.nan,
                Y_eco=(ps.y[j] + d.y0) if j >= 0 else np.nan,
                d_eco=float(np.hypot(ps.x[j] - gx[k], ps.y[j] - gy[k])) if j >= 0 else np.nan,
                n_cells_eco=int(ps.n[j]) if j >= 0 else 0,
                amp_eco=float(ps.amp[j]) if j >= 0 else 0.0,
                ext_eco=float(ps.ext[j]) if j >= 0 else 0.0,
            ))
    d.close()
    return pd.DataFrame(rows)


def fuse(gt: pd.DataFrame) -> pd.DataFrame:
    """Confirmacao cruzada entre estacoes com campo de visao sobreposto."""
    gt = gt.copy()
    gt["tbin"] = np.floor(gt["t"] / (2 * FUSE_DT)).astype(int)
    gt["n_est"] = 0
    gt["n_est_conf"] = 0
    gt["d_fusao"] = np.nan
    key = ["hour", "uid", "tbin"]
    idx = {k: v.index.to_numpy() for k, v in gt.groupby(key)}
    for k, ii in idx.items():
        sub = gt.loc[ii]
        gt.loc[ii, "n_est"] = len(sub)
        ok = sub[np.isfinite(sub["X_eco"])]
        if len(ok) < 2:
            continue
        xy = ok[["X_eco", "Y_eco"]].to_numpy()
        dm = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        np.fill_diagonal(dm, np.inf)
        near = dm.min(axis=1)
        gt.loc[ok.index, "n_est_conf"] = int((near <= FUSE_TOL).sum())
        gt.loc[ok.index, "d_fusao"] = near
    st = np.where(np.isfinite(gt["X_eco"]),
                  np.where(gt["n_est_conf"] >= 2, "confirmado_fusao", "confirmado_local"),
                  "nao_confirmado")
    gt["status"] = st
    return gt


def main():
    jobs = T.list_chunks()
    with Pool(6) as p:
        parts = p.map(process_chunk, jobs)
    gt = pd.concat(parts, ignore_index=True)
    gt = fuse(gt)
    os.makedirs(H.TAB_DIR, exist_ok=True)
    gt.to_csv(os.path.join(H.OUT_DIR, "ground_truth.csv"), index=False)

    # -- resumo por estacao ------------------------------------------------
    g = gt.groupby("station")
    res = pd.DataFrame({
        "alvos_varredura": g.size(),
        "uids": g["uid"].nunique(),
        "conf_local_%": g["status"].apply(lambda s: 100 * (s != "nao_confirmado").mean()),
        "conf_fusao_%": g["status"].apply(lambda s: 100 * (s == "confirmado_fusao").mean()),
        "d_eco_mediano_m": g["d_eco"].median(),
        "d_eco_p90_m": g["d_eco"].quantile(0.9),
        "plots_por_varredura": g["n_plots"].mean(),
        "alvos_por_varredura": g["n_tgt"].mean(),
    }).round(2)
    res.to_csv(os.path.join(H.TAB_DIR, "etapa0_gt_por_estacao.csv"))
    print(res.to_string())

    tot = pd.Series({
        "alvos_varredura": len(gt),
        "uids_distintos": gt["uid"].nunique(),
        "varreduras": gt.groupby(["station", "hour", "scan"]).ngroups,
        "conf_local_%": round(100 * (gt["status"] != "nao_confirmado").mean(), 2),
        "conf_fusao_%": round(100 * (gt["status"] == "confirmado_fusao").mean(), 2),
        "d_eco_mediano_m": round(gt["d_eco"].median(), 2),
        "d_eco_p90_m": round(gt["d_eco"].quantile(0.9), 2),
        "d_fusao_mediano_m": round(gt["d_fusao"].median(), 2),
        "obs_multiestacao_%": round(100 * (gt["n_est"] >= 2).mean(), 2),
    })
    tot.to_csv(os.path.join(H.TAB_DIR, "etapa0_gt_global.csv"))
    print("\n", tot.to_string())


if __name__ == "__main__":
    main()
