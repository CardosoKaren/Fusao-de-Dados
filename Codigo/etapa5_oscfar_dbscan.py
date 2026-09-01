"""
etapa5_oscfar_dbscan.py -- Proposta 5: filtro OS-CFAR + clusterizador DBSCAN.

Contraparte classica da Etapa 3. Os plots primitivos aprovados pelo OS-CFAR
(Etapa 2) sao reagrupados por DBSCAN (Ester et al., 1996) sobre as coordenadas
(x, y) de cada varredura, com `min_samples = 1` -- todo plot pertence a algum
agrupamento, pois a rejeicao de ruido ja foi feita pelo filtro. O raio `eps` e
ajustado na particao de treino, com o mesmo criterio (F1 de plots) e o mesmo
orcamento de busca usados no clusterizador ClusWiSARD.

Uso:  python3 Codigo/etapa5_oscfar_dbscan.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cluster_common as CC
import haxr_io as H
import pipeline as PL
import tratado_io as T

NAME = "etapa5_oscfar_dbscan"
SRC = "etapa2_oscfar"

EPS = (15.0, 25.0, 35.0, 45.0, 60.0, 80.0, 100.0, 130.0)


def dbscan_scan(g: pd.DataFrame, eps: float, k_vel: float = 0.0) -> pd.DataFrame:
    cols = ["station", "hour", "scan", "x", "y", "amp", "n", "ext", "vx", "vy"]
    if len(g) <= 1:
        return g[cols].copy()
    Z = [g["x"].to_numpy(float), g["y"].to_numpy(float)]
    if k_vel:
        Z += [np.nan_to_num(g["vx"].to_numpy(float)) * k_vel,
              np.nan_to_num(g["vy"].to_numpy(float)) * k_vel]
    lab = DBSCAN(eps=eps, min_samples=1).fit_predict(np.c_[tuple(Z)])
    return CC.merge_by_label(g, lab)


def cluster_all(plots, eps, k_vel=0.0):
    return CC.cluster_all(plots, dbscan_scan, eps=eps, k_vel=k_vel)


def tune(plots, gt, hours):
    rows, best = [], None
    sub = plots[plots["hour"].isin(hours)]
    for e in EPS:
        cl = cluster_all(sub, e)
        mm, _ = PL.evaluate(cl, gt, hours=hours, window=T.VAL_WINDOW)
        rows.append(dict(eps=e, **mm))
        print(f"  eps={e:5.0f} m  F1={mm['F1']:.4f} recall={mm['recall']:.4f} "
              f"precisao={mm['precisao']:.4f} plots/alvo={mm['plots_por_alvo']}")
        if best is None or mm["F1"] > best[1]:
            best = (e, mm["F1"])
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(H.TAB_DIR, "etapa5_ajuste_dbscan.csv"), index=False)
    return best[0], tab


def main():
    gt = T.load_gt()
    pl = PL.load_detections(SRC)
    print(f"entrada: {len(pl)} plots da Etapa 2")
    print("ajuste do DBSCAN (particao de treino, 08 UTC):")
    eps, tab = tune(pl, gt, T.TRAIN_HOURS)
    print(f"\nescolhido: eps={eps:.0f} m")

    t0 = time.time()
    cl = cluster_all(pl, eps)
    print(f"clusterizacao completa: {len(pl)} -> {len(cl)} plots ({time.time() - t0:.0f}s)")
    PL.save_detections(cl, NAME)

    mte, per = PL.evaluate(cl, gt, hours=T.TEST_HOURS)
    PL.report(NAME, mte, per, extra=dict(eps=eps))
    mval, _ = PL.evaluate(cl, gt, hours=T.TRAIN_HOURS, window=T.VAL_WINDOW)
    print("  (08 UTC janela de validacao: F1=%.4f recall=%.4f prec=%.4f)"
          % (mval["F1"], mval["recall"], mval["precisao"]))
    mc, _ = PL.evaluate(cl, gt, hours=T.TEST_HOURS, only_confirmed=True)
    print("  (teste, apenas alvos confirmados: F1=%.4f recall=%.4f)" % (mc["F1"], mc["recall"]))
    PL.by_station(cl, gt, hours=T.TEST_HOURS).to_csv(
        os.path.join(H.TAB_DIR, "etapa5_por_estacao.csv"))


if __name__ == "__main__":
    main()
