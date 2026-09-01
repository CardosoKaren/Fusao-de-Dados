"""
etapa3_cluswisard_cluster.py -- Proposta 3: filtro ClusWiSARD + clusterizador
ClusWiSARD.

Os plots aprovados pelo filtro da Etapa 1 sao reagrupados por uma **segunda
rede sem peso**, agora empregada de forma nao supervisionada: cada
discriminador criado dinamicamente representa uma embarcacao da cena.

Codificacao dos plots: cada coordenada e escrita por *campo receptivo* --
uma barra de largura 2*w centrada no valor (`cluswisard.interval_code`).
Dois plots compartilham bits enquanto |dx| < 2w e |dy| < 2w, e nenhum bit
alem disso; como as tuplas misturam bits de x e de y, a resposta da rede so e
alta quando **ambas** as coordenadas coincidem. A largura w e o limiar
`min_score` definem, juntos, a escala metrica do agrupamento -- ajustada na
particao de treino.

Uso:  python3 Codigo/etapa3_cluswisard_cluster.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cluster_common as CC
import cluswisard as C
import haxr_io as H
import pipeline as PL
import tratado_io as T

NAME = "etapa3_cluswisard_cluster"
SRC = "etapa1_cluswisard"

SCORES = (0.80, 0.85, 0.88, 0.91, 0.94, 0.96, 0.98)   # limiar de resposta do clusterizador


def cluster_all(plots, min_score):
    return CC.cluster_all(plots, CC.cluswisard_cluster_scan, min_score=min_score)


def tune(plots, gt, hours):
    rows, best = [], None
    sub = plots[plots["hour"].isin(hours)]
    for ms in SCORES:
        cl = cluster_all(sub, ms)
        mm, _ = PL.evaluate(cl, gt, hours=hours, window=T.VAL_WINDOW)
        raio = CC.SPAN_XY * (1 - ms ** (1 / CC.TUPLE))
        rows.append(dict(min_score=ms, raio_equiv_m=round(raio, 1), **mm))
        print(f"  min_score={ms:.2f} (raio ~{raio:5.0f} m)  F1={mm['F1']:.4f} "
              f"recall={mm['recall']:.4f} precisao={mm['precisao']:.4f} "
              f"plots/alvo={mm['plots_por_alvo']}")
        if best is None or mm["F1"] > best[1]:
            best = (ms, mm["F1"])
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(H.TAB_DIR, "etapa3_ajuste_clusterizador.csv"), index=False)
    return best[0], tab


def main():
    gt = T.load_gt()
    pl = PL.load_detections(SRC)
    print(f"entrada: {len(pl)} plots da Etapa 1")
    print("ajuste do clusterizador (particao de treino, 08 UTC):")
    ms, tab = tune(pl, gt, T.TRAIN_HOURS)
    print(f"\nescolhido: min_score={ms:.2f}")

    t0 = time.time()
    cl = cluster_all(pl, ms)
    print(f"clusterizacao completa: {len(pl)} -> {len(cl)} plots ({time.time() - t0:.0f}s)")
    PL.save_detections(cl, NAME)

    mte, per = PL.evaluate(cl, gt, hours=T.TEST_HOURS)
    PL.report(NAME, mte, per, extra=dict(min_score=ms))
    mval, _ = PL.evaluate(cl, gt, hours=T.TRAIN_HOURS, window=T.VAL_WINDOW)
    print("  (08 UTC janela de validacao: F1=%.4f recall=%.4f prec=%.4f)"
          % (mval["F1"], mval["recall"], mval["precisao"]))
    mc, _ = PL.evaluate(cl, gt, hours=T.TEST_HOURS, only_confirmed=True)
    print("  (teste, apenas alvos confirmados: F1=%.4f recall=%.4f)" % (mc["F1"], mc["recall"]))
    PL.by_station(cl, gt, hours=T.TEST_HOURS).to_csv(
        os.path.join(H.TAB_DIR, "etapa3_por_estacao.csv"))


if __name__ == "__main__":
    main()
