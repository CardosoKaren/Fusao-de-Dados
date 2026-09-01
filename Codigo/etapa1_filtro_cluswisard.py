"""
etapa1_filtro_cluswisard.py -- Proposta 1: filtro ClusWiSARD.

A rede neural sem peso ClusWiSARD e treinada para decidir, celula a celula do
video radar recortado, se o retorno pertence a um alvo ou a ruido/clutter.
As celulas aprovadas sao aglutinadas em plots pelo extrator canonico
(componentes conexas em grade metrica de 5 m), e os plots resultantes sao
confrontados com o ground truth da Etapa 0.

Protocolo:
  treino  -> hora 08 UTC (13 estacoes)
  teste   -> horas 09 e 11 UTC (13 estacoes)   [metricas reportadas]

Uso:  python3 Codigo/etapa1_filtro_cluswisard.py
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cluswisard as C
import features as F
import haxr_io as H
import pipeline as PL
import plots_eval as P
import tratado_io as T

NAME = "etapa1_cluswisard"
MODEL_PATH = os.path.join(PL.RES_DIR, f"{NAME}_modelo.pkl")

# --- rotulagem das celulas de treino ---------------------------------------
D_POS = 50.0        # celula a menos de 50 m de um alvo AIS -> classe "alvo"
D_NEG = 90.0        # celula a mais de 90 m de qualquer alvo -> classe "ruido"
MAX_POS, MAX_NEG = 500, 900      # amostras por varredura
SCAN_STEP = 3                    # subamostragem de varreduras no treino
FIT_WINDOW = T.FIT_WINDOW        # janela de 120 s usada para ajustar a rede
VAL_WINDOW = T.VAL_WINDOW        # janela de 120 s usada para escolher o limiar

# --- hiperparametros da rede ------------------------------------------------
HP = dict(tuple_size=48, n_ram=128, table_bits=18, limit=4,
          min_score=0.5, threshold=32, balance=True, seed=1, dtype=np.uint8)


def retina(d, i):
    """Retina binaria com codificacao complementar (bit e seu complemento).

    Sem o complemento, a esparsidade do video (≈3 % de celulas com retorno)
    faz quase toda tupla enderecar a posicao zero e as respostas saturam.
    """
    B, info = F.scan_retina(d, i)
    if info is None:
        return None, None
    return np.concatenate([B, 1 - B], axis=1).astype(np.uint8), info


# ---------------------------------------------------------------------------
# treino
# ---------------------------------------------------------------------------


def collect_training(seed=0):
    rng = np.random.default_rng(seed)
    Bs, ys = [], []
    for st, hh in T.list_chunks(T.TRAIN_HOURS):
        d = T.Tratado(st, hh)
        sel = np.flatnonzero(d.scan["window"].to_numpy() == FIT_WINDOW)[::SCAN_STEP]
        for i in sel:
            B, info = retina(d, int(i))
            if info is None:
                continue
            dd = F.dist_to_gt(info, F.gt_xy(d, int(i)))
            pos = np.flatnonzero(dd < D_POS)
            neg = np.flatnonzero(dd > D_NEG)
            if len(pos) < 5 or len(neg) < 5:
                continue
            pi = rng.choice(pos, min(len(pos), MAX_POS), replace=False)
            ni = rng.choice(neg, min(len(neg), MAX_NEG), replace=False)
            Bs.append(B[pi]); ys.append(np.ones(len(pi)))
            Bs.append(B[ni]); ys.append(np.zeros(len(ni)))
        d.close()
    B = np.vstack(Bs); y = np.concatenate(ys)
    p = rng.permutation(len(B))
    return B[p], y[p]


def train():
    """Treina a rede e escolhe o bleaching pela AUC em particao retida."""
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    B, y = collect_training()
    n = int(0.8 * len(B))
    print(f"treino: {len(B)} celulas, {B.shape[1]} bits, "
          f"{100 * y.mean():.1f}% classe alvo  (coleta {time.time() - t0:.0f}s)")
    m = C.ClusWiSARD(B.shape[1], **HP).fit(B[:n], y[:n])
    print(f"rede treinada em {time.time() - t0:.0f}s -- discriminadores {m.n_disc}, "
          f"{m.mp.n_ram} RAMs de 2^{m.mp.table_bits} posicoes")
    rows = []
    for bl in (1, 2, 4):
        m.bleach = bl
        auc = roc_auc_score(y[n:], m.margin(B[n:]))
        rows.append(dict(bleach=bl, AUC=round(float(auc), 4)))
        print(f"  bleach={bl}: AUC (celula) = {auc:.4f}")
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(H.TAB_DIR, "etapa1_ajuste_bleaching.csv"), index=False)
    m.bleach = int(tab.loc[tab["AUC"].idxmax(), "bleach"])
    print(f"  bleaching escolhido: {m.bleach}")

    # ablacao: WiSARD classica (um unico discriminador por classe)
    hp1 = dict(HP); hp1.update(limit=1, balance=False)
    m1 = C.ClusWiSARD(B.shape[1], **hp1).fit(B[:n], y[:n])
    m1.bleach = m.bleach
    auc1 = roc_auc_score(y[n:], m1.margin(B[n:]))
    pd.DataFrame([dict(modelo="ClusWiSARD (6 discriminadores/classe)",
                       AUC=round(float(roc_auc_score(y[n:], m.margin(B[n:]))), 4)),
                  dict(modelo="WiSARD (1 discriminador/classe)", AUC=round(float(auc1), 4))]
                 ).to_csv(os.path.join(H.TAB_DIR, "etapa1_ablacao_discriminadores.csv"), index=False)
    print(f"  ablacao WiSARD (1 discriminador/classe): AUC = {auc1:.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(m, f, protocol=4)
    return m


# ---------------------------------------------------------------------------
# aplicacao
# ---------------------------------------------------------------------------

_M = None


def _init():
    global _M
    with open(MODEL_PATH, "rb") as f:
        _M = pickle.load(f)


def detect_chunk(args):
    """Aplica o filtro a um recorte e extrai plots para cada limiar pedido."""
    station, hour, thrs = args
    d = T.Tratado(station, hour)
    out = []
    for i in range(d.n_scan):
        B, info = retina(d, i)
        if info is None:
            continue
        s1 = _M.scores(B)
        r1, r0 = s1.get(1, 0.0), s1.get(0, 0.0)
        marg = {"dif": r1 - r0, "norm": (r1 - r0) / np.maximum(r1 + r0, 1e-9)}
        for modo, thr in thrs:
            k = marg[modo] > thr
            ps = P.extract_plots(info["x"][k], info["y"][k], info["amp"][k])
            fr = PL.plots_to_frame(station, hour, i, ps, d.x0, d.y0)
            fr["modo"] = modo
            fr["limiar"] = thr
            out.append(fr)
    d.close()
    return pd.concat(out, ignore_index=True) if out else None


def run(jobs, thrs, n_jobs=4):
    with Pool(n_jobs, initializer=_init) as p:
        parts = [x for x in p.map(detect_chunk, [(s, h, thrs) for s, h in jobs])
                 if x is not None]
    return pd.concat(parts, ignore_index=True)


def tune_threshold(gt):
    """Escolhe modo e limiar de decisao na **janela de validacao** (08 UTC).

    A rede e ajustada na janela 0 de cada recorte; o limiar e escolhido na
    janela 1, 30 min depois, que a rede nunca viu. Sem essa separacao o limiar
    e escolhido sobre respostas memorizadas e nao se transfere para as horas de
    teste -- a WiSARD e um classificador por memorizacao, e sua resposta em
    dados ja apresentados e sistematicamente mais alta.
    """
    t0 = time.time()
    cands = [("dif", float(v)) for v in np.round(np.arange(-0.12, 0.29, 0.02), 3)]
    cands += [("norm", float(v)) for v in np.round(np.arange(-0.10, 0.31, 0.02), 3)]
    pl = run(T.list_chunks(T.TRAIN_HOURS), cands)
    rows = []
    for modo, thr in cands:
        sub = pl[(pl["modo"] == modo) & (pl["limiar"] == thr)]
        mm, _ = PL.evaluate(sub, gt, hours=T.TRAIN_HOURS, window=VAL_WINDOW)
        rows.append(dict(modo=modo, limiar=thr, **mm))
        print(f"  modo={modo:4s} limiar={thr:+.2f}  F1={mm['F1']:.4f} "
              f"recall={mm['recall']:.4f} precisao={mm['precisao']:.4f}  "
              f"plots/alvo={mm['plots_por_alvo']}")
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(H.TAB_DIR, "etapa1_ajuste_limiar.csv"), index=False)
    b = tab.loc[tab["F1"].idxmax()]
    print(f"  ({time.time() - t0:.0f}s)")
    return str(b["modo"]), float(b["limiar"]), tab


def main():
    gt = T.load_gt()
    train()
    print("\najuste do limiar de decisao (particao de treino, 08 UTC):")
    modo, thr, tab = tune_threshold(gt)
    print(f"\nescolhidos: modo={modo}, limiar={thr:+.2f}")

    t0 = time.time()
    pl = run(T.list_chunks(), [(modo, thr)], n_jobs=4).drop(columns=["modo", "limiar"])
    print(f"deteccao em todas as varreduras: {len(pl)} plots ({time.time() - t0:.0f}s)")
    PL.save_detections(pl, NAME)

    with open(MODEL_PATH, "rb") as f:
        bl = pickle.load(f).bleach
    mte, per = PL.evaluate(pl, gt, hours=T.TEST_HOURS)
    PL.report(NAME, mte, per, extra=dict(bleach=bl, modo=modo, limiar=thr))
    mfit, _ = PL.evaluate(pl, gt, hours=T.TRAIN_HOURS, window=FIT_WINDOW)
    mval, _ = PL.evaluate(pl, gt, hours=T.TRAIN_HOURS, window=VAL_WINDOW)
    print("  (08 UTC janela de ajuste  : F1=%.4f recall=%.4f prec=%.4f)"
          % (mfit["F1"], mfit["recall"], mfit["precisao"]))
    print("  (08 UTC janela de validacao: F1=%.4f recall=%.4f prec=%.4f)"
          % (mval["F1"], mval["recall"], mval["precisao"]))
    mc, _ = PL.evaluate(pl, gt, hours=T.TEST_HOURS, only_confirmed=True)
    print("  (teste, apenas alvos confirmados: F1=%.4f recall=%.4f)" % (mc["F1"], mc["recall"]))
    PL.by_station(pl, gt, hours=T.TEST_HOURS).to_csv(
        os.path.join(H.TAB_DIR, "etapa1_por_estacao.csv"))


if __name__ == "__main__":
    main()
