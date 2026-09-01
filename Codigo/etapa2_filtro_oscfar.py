"""
etapa2_filtro_oscfar.py -- Proposta 2: filtro OS-CFAR.

Detector classico de taxa constante de alarmes falsos com estatistica de
ordem (Rohling, 1983). Para cada celula sob teste (CUT), o nivel de fundo e
estimado pelo k-esimo menor valor entre as celulas de treinamento vizinhas, e
a celula e declarada deteccao quando

    amp(CUT) > alpha * Z_(k)

A estatistica de ordem -- em vez da media (CA-CFAR) -- e a escolha adequada
em ambiente portuario: ela e robusta a alvos interferentes dentro da janela de
treinamento, situacao permanente aqui, onde varias embarcacoes e estruturas de
cais coexistem a poucas dezenas de metros.

A janela de treinamento e definida em **metros** (coroa entre o raio de guarda
e o raio externo), amostrada no referencial distancia/travessia do radar. Isso
mantem a extensao fisica da janela constante em toda a cobertura -- a celula de
resolucao em travessia cresce com a distancia -- e iguala a extensao do contexto
usado pela retina da ClusWiSARD (Etapa 1), de modo que a comparacao entre os
dois filtros seja justa.

alpha e a ordem k sao ajustados na particao de treino (08 UTC) maximizando o
F1 de plots, exatamente o mesmo criterio e orcamento de ajuste da Etapa 1.

Uso:  python3 Codigo/etapa2_filtro_oscfar.py
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
import haxr_io as H
import pipeline as PL
import plots_eval as P
import tratado_io as T

NAME = "etapa2_oscfar"

WIN_K, WIN_STEP = 23, 8.0     # janela 23x23 amostras de 8 m  (+-88 m)
GUARDS = (24.0, 40.0, 56.0)   # raios de guarda testados [m]
ORDERS = (0.60, 0.75, 0.90, 0.98)
ALPHAS = (1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 6.0)


def cfar_levels(d, i, guards=GUARDS, orders=ORDERS):
    """Nivel de fundo Z_(k) por celula, para cada raio de guarda e ordem.

    O raio de guarda precisa exceder a extensao do proprio alvo: em ambiente
    portuario um navio de 200-300 m ocupa dezenas de celulas de treinamento e,
    com guarda insuficiente, mascara a si mesmo (self-masking). Por isso o raio
    de guarda tambem e ajustado na particao de treino.
    """
    S, dist, info = F.metric_window(d, i, WIN_K, WIN_STEP)
    if info is None:
        return None, None
    rmax = WIN_K // 2 * WIN_STEP + 1e-6
    lev = {}
    for g in guards:
        train = (dist > g) & (dist <= rmax)
        Tr = np.sort(S[:, train], axis=1)
        n = Tr.shape[1]
        for q in orders:
            lev[(g, q)] = Tr[:, min(int(np.ceil(q * n)) - 1, n - 1)].astype(np.float64)
    return lev, info


def detect_chunk(args):
    station, hour, guards, orders, alphas = args
    d = T.Tratado(station, hour)
    out = []
    for i in range(d.n_scan):
        lev, info = cfar_levels(d, i, guards, orders)
        if info is None:
            continue
        amp = info["amp"].astype(np.float64)
        for g in guards:
            for q in orders:
                for a in alphas:
                    k = amp > a * lev[(g, q)]
                    ps = P.extract_plots(info["x"][k], info["y"][k], info["amp"][k])
                    fr = PL.plots_to_frame(station, hour, i, ps, d.x0, d.y0)
                    fr["guarda"] = g
                    fr["ordem"] = q
                    fr["alpha"] = a
                    out.append(fr)
    d.close()
    return pd.concat(out, ignore_index=True) if out else None


def run(jobs, guards, orders, alphas, n_jobs=6):
    with Pool(n_jobs) as p:
        parts = [x for x in p.map(detect_chunk,
                                  [(s, h, guards, orders, alphas) for s, h in jobs])
                 if x is not None]
    return pd.concat(parts, ignore_index=True)


def tune(gt):
    t0 = time.time()
    pl = run(T.list_chunks(T.TRAIN_HOURS), GUARDS, ORDERS, ALPHAS)
    rows = []
    for g in GUARDS:
        for q in ORDERS:
            for a in ALPHAS:
                sub = pl[(pl["guarda"] == g) & (pl["ordem"] == q) & (pl["alpha"] == a)]
                mm, _ = PL.evaluate(sub, gt, hours=T.TRAIN_HOURS, window=T.VAL_WINDOW)
                rows.append(dict(guarda=g, ordem=q, alpha=a, **mm))
                print(f"  guarda={g:4.0f} ordem={q:.2f} alpha={a:.1f}  F1={mm['F1']:.4f} "
                      f"recall={mm['recall']:.4f} precisao={mm['precisao']:.4f}")
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(H.TAB_DIR, "etapa2_ajuste_cfar.csv"), index=False)
    b = tab.loc[tab["F1"].idxmax()]
    print(f"  ({time.time() - t0:.0f}s)")
    return float(b["guarda"]), float(b["ordem"]), float(b["alpha"]), tab


def main():
    gt = T.load_gt()
    print("ajuste de ordem e alpha (particao de treino, 08 UTC):")
    g, q, a, tab = tune(gt)
    print(f"\nescolhidos: guarda={g:.0f} m, ordem k={q:.2f}, alpha={a:.1f}")

    t0 = time.time()
    pl = run(T.list_chunks(), (g,), (q,), (a,)).drop(columns=["guarda", "ordem", "alpha"])
    print(f"deteccao em todas as varreduras: {len(pl)} plots ({time.time() - t0:.0f}s)")
    PL.save_detections(pl, NAME)

    mte, per = PL.evaluate(pl, gt, hours=T.TEST_HOURS)
    PL.report(NAME, mte, per, extra=dict(guarda=g, ordem=q, alpha=a))
    mtr, _ = PL.evaluate(pl, gt, hours=T.TRAIN_HOURS)
    print("  (treino 08 UTC: F1=%.4f recall=%.4f prec=%.4f)"
          % (mtr["F1"], mtr["recall"], mtr["precisao"]))
    mc, _ = PL.evaluate(pl, gt, hours=T.TEST_HOURS, only_confirmed=True)
    print("  (teste, apenas alvos confirmados: F1=%.4f recall=%.4f)" % (mc["F1"], mc["recall"]))
    PL.by_station(pl, gt, hours=T.TEST_HOURS).to_csv(
        os.path.join(H.TAB_DIR, "etapa2_por_estacao.csv"))


if __name__ == "__main__":
    main()
