"""
etapa6_oscfar_dbscan_cinematica.py -- Proposta 6:
filtro OS-CFAR + DBSCAN + dados cinematicos.

Contraparte classica da Etapa 4. O DBSCAN passa a operar em quatro dimensoes,
(x, y, k*vx, k*vy), em que a velocidade e estimada pelo acompanhamento dos
plots ao longo das rotacoes de antena de uma janela de 2 minutos. O fator de
escala k [m / (m/s)] converte a diferenca de velocidade em uma distancia
equivalente, de modo que `eps` continue tendo unidade de metro; k e ajustado
na particao de treino junto com `eps`.

Uso:  python3 Codigo/etapa6_oscfar_dbscan_cinematica.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cluster_common as CC
import etapa5_oscfar_dbscan as E5
import haxr_io as H
import pipeline as PL
import tratado_io as T

NAME = "etapa6_oscfar_dbscan_cinematica"
SRC = "etapa2_oscfar"

EPS = (15.0, 25.0, 35.0, 45.0, 60.0, 80.0, 100.0, 130.0)
K_VEL = (5.0, 15.0, 40.0)


def cluster_all(plots, eps, k_vel):
    return CC.cluster_all(plots, E5.dbscan_scan, eps=eps, k_vel=k_vel)


def tune(plots, gt, hours):
    rows, best = [], None
    sub = plots[plots["hour"].isin(hours)]
    for k in K_VEL:
        for e in EPS:
            cl = cluster_all(sub, e, k)
            mm, _ = PL.evaluate(cl, gt, hours=hours, window=T.VAL_WINDOW)
            rows.append(dict(k_vel=k, eps=e, **mm))
            print(f"  k={k:5.0f} eps={e:5.0f} m  F1={mm['F1']:.4f} "
                  f"recall={mm['recall']:.4f} precisao={mm['precisao']:.4f} "
                  f"plots/alvo={mm['plots_por_alvo']}")
            if best is None or mm["F1"] > best[2]:
                best = (e, k, mm["F1"])
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(H.TAB_DIR, "etapa6_ajuste_dbscan.csv"), index=False)
    return best[0], best[1], tab


def main():
    gt = T.load_gt()
    pl = PL.load_detections(SRC)
    t0 = time.time()
    pl = CC.add_kinematics(pl, PL.scan_meta())
    v = np.hypot(pl["vx"], pl["vy"])
    print(f"cinematica estimada para {len(pl)} plots ({time.time() - t0:.0f}s); "
          f"|v| mediana {np.median(v):.2f} m/s, p90 {np.percentile(v, 90):.2f} m/s")
    print("ajuste do DBSCAN cinematico (particao de treino, 08 UTC):")
    eps, k, tab = tune(pl, gt, T.TRAIN_HOURS)
    print(f"\nescolhidos: eps={eps:.0f} m, k={k:.0f}")

    cl = cluster_all(pl, eps, k)
    print(f"clusterizacao completa: {len(pl)} -> {len(cl)} plots")
    PL.save_detections(cl, NAME)

    mte, per = PL.evaluate(cl, gt, hours=T.TEST_HOURS)
    PL.report(NAME, mte, per, extra=dict(eps=eps, k_vel=k))
    mval, _ = PL.evaluate(cl, gt, hours=T.TRAIN_HOURS, window=T.VAL_WINDOW)
    print("  (08 UTC janela de validacao: F1=%.4f recall=%.4f prec=%.4f)"
          % (mval["F1"], mval["recall"], mval["precisao"]))
    mc, _ = PL.evaluate(cl, gt, hours=T.TEST_HOURS, only_confirmed=True)
    print("  (teste, apenas alvos confirmados: F1=%.4f recall=%.4f)" % (mc["F1"], mc["recall"]))
    PL.by_station(cl, gt, hours=T.TEST_HOURS).to_csv(
        os.path.join(H.TAB_DIR, "etapa6_por_estacao.csv"))


if __name__ == "__main__":
    main()
