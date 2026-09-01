"""
etapa4_cluswisard_cluster_cinematica.py -- Proposta 4:
filtro ClusWiSARD + clusterizador ClusWiSARD + dados cinematicos.

Identica a Etapa 3, exceto pela adicao de dois eixos de atributo ao
clusterizador: a velocidade (vx, vy) de cada plot primitivo, estimada pelo
acompanhamento do plot ao longo das rotacoes de antena de uma janela de
2 minutos (`cluster_common.estimate_kinematics`).

Justificativa: duas manchas de eco vizinhas que se deslocam com rumos ou
velocidades distintas nao podem pertencer ao mesmo casco. A cinematica separa
esse caso -- que a posicao, sozinha, funde indevidamente -- e evita que
embarcacoes que se cruzam sejam colapsadas em um unico plot.

O numero de bits dedicado a cada eixo de velocidade controla o peso relativo
da cinematica na resposta da rede, e e ajustado na particao de treino junto
com o limiar de agrupamento.

Uso:  python3 Codigo/etapa4_cluswisard_cluster_cinematica.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cluster_common as CC
import haxr_io as H
import pipeline as PL
import tratado_io as T

NAME = "etapa4_cluswisard_cluster_cinematica"
SRC = "etapa1_cluswisard"

SCORES = (0.80, 0.85, 0.88, 0.91, 0.94, 0.96, 0.98)
VEL_BITS = (72, 144, 288)


def cluster_all(plots, min_score, vel_bits):
    return CC.cluster_all(plots, CC.cluswisard_cluster_scan,
                          min_score=min_score, vel_bits=vel_bits)


def tune(plots, gt, hours):
    rows, best = [], None
    sub = plots[plots["hour"].isin(hours)]
    for vb in VEL_BITS:
        for ms in SCORES:
            cl = cluster_all(sub, ms, vb)
            mm, _ = PL.evaluate(cl, gt, hours=hours, window=T.VAL_WINDOW)
            rows.append(dict(bits_velocidade=vb, min_score=ms, **mm))
            print(f"  bits_v={vb:3d} min_score={ms:.2f}  F1={mm['F1']:.4f} "
                  f"recall={mm['recall']:.4f} precisao={mm['precisao']:.4f} "
                  f"plots/alvo={mm['plots_por_alvo']}")
            if best is None or mm["F1"] > best[2]:
                best = (ms, vb, mm["F1"])
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(H.TAB_DIR, "etapa4_ajuste_clusterizador.csv"), index=False)
    return best[0], best[1], tab


def main():
    gt = T.load_gt()
    pl = PL.load_detections(SRC)
    t0 = time.time()
    pl = CC.add_kinematics(pl, PL.scan_meta())
    v = np.hypot(pl["vx"], pl["vy"])
    print(f"cinematica estimada para {len(pl)} plots ({time.time() - t0:.0f}s); "
          f"|v| mediana {np.median(v):.2f} m/s, p90 {np.percentile(v, 90):.2f} m/s")
    pd.DataFrame(dict(velocidade_mediana=[float(np.median(v))],
                      velocidade_p90=[float(np.percentile(v, 90))],
                      velocidade_p99=[float(np.percentile(v, 99))])
                 ).to_csv(os.path.join(H.TAB_DIR, "etapa4_cinematica.csv"), index=False)

    print("ajuste do clusterizador cinematico (particao de treino, 08 UTC):")
    ms, vb, tab = tune(pl, gt, T.TRAIN_HOURS)
    print(f"\nescolhidos: min_score={ms:.2f}, bits de velocidade={vb}")

    cl = cluster_all(pl, ms, vb)
    print(f"clusterizacao completa: {len(pl)} -> {len(cl)} plots")
    PL.save_detections(cl, NAME)

    mte, per = PL.evaluate(cl, gt, hours=T.TEST_HOURS)
    PL.report(NAME, mte, per, extra=dict(min_score=ms, bits_velocidade=vb))
    mval, _ = PL.evaluate(cl, gt, hours=T.TRAIN_HOURS, window=T.VAL_WINDOW)
    print("  (08 UTC janela de validacao: F1=%.4f recall=%.4f prec=%.4f)"
          % (mval["F1"], mval["recall"], mval["precisao"]))
    mc, _ = PL.evaluate(cl, gt, hours=T.TEST_HOURS, only_confirmed=True)
    print("  (teste, apenas alvos confirmados: F1=%.4f recall=%.4f)" % (mc["F1"], mc["recall"]))
    PL.by_station(cl, gt, hours=T.TEST_HOURS).to_csv(
        os.path.join(H.TAB_DIR, "etapa4_por_estacao.csv"))


if __name__ == "__main__":
    main()
