"""
cluster_common.py -- Estagio de clusterizacao comum as Etapas 3 a 6.

Motivacao fisica: um navio longo nao produz um unico eco compacto. O feixe
varre o casco, a superestrutura e os guindastes de bordo em azimutes
diferentes, e o filtro (ClusWiSARD ou OS-CFAR) aprova celulas em manchas
separadas. As componentes conexas do extrator de plots resultam entao em
**varios plots para uma unica embarcacao** -- falsos positivos que nao
correspondem a nenhum alvo novo.

O estagio de clusterizacao agrupa os plots primitivos de uma varredura que
pertencem a mesma embarcacao e substitui cada grupo por um unico plot, no
centroide ponderado pela amplitude.

Duas variantes de atributos:
  * **posicional**  (Etapas 3 e 5): apenas (x, y);
  * **cinematico**  (Etapas 4 e 6): (x, y, vx, vy), com a velocidade estimada
    pelo acompanhamento do plot ao longo das rotacoes de antena de uma janela
    de 2 minutos. Duas manchas proximas mas com rumo/velocidade distintos nao
    podem pertencer a mesma embarcacao, e a cinematica separa esse caso.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# cinematica: velocidade de cada plot pelas rotacoes de antena
# ---------------------------------------------------------------------------

def estimate_kinematics(plots: pd.DataFrame, period: float, gate: float = 45.0,
                        n_back: int = 12, max_miss: int = 2) -> pd.DataFrame:
    """Estima (vx, vy) de cada plot pela observacao das rotacoes de antena.

    Implementa um rastreador de vizinho mais proximo com predicao e tolerancia
    a falhas de deteccao -- necessario porque os plots de um mesmo alvo piscam
    entre varreduras, e um encadeamento ingenuo produz trilhas curtas demais
    para uma estimativa de velocidade estavel.

    A cada rotacao:
      1. cada trilha viva e propagada para o instante da varredura pela sua
         velocidade corrente (predicao);
      2. resolve-se a associacao otima (algoritmo hungaro) entre as predicoes e
         os plots da varredura, sob porta `gate` -- compativel com o
         deslocamento maximo de uma embarcacao em um periodo de antena
         (2,9 s a 15 m/s = 44 m);
      3. plots nao associados iniciam novas trilhas; trilhas nao associadas
         seguem em voo cego (`coasting`) por ate `max_miss` rotacoes.

    A velocidade atribuida a cada plot e o coeficiente angular da regressao
    linear de x e y contra o tempo sobre as ultimas `n_back` posicoes da sua
    trilha (12 rotacoes ~ 35 s), dentro da janela de 2 min do DataSet_tratado.
    """
    plots = plots.sort_values(["window", "scan"]).reset_index(drop=True)
    X = plots["x"].to_numpy(float)
    Y = plots["y"].to_numpy(float)
    vx = np.zeros(len(plots))
    vy = np.zeros(len(plots))
    n_pontos = np.ones(len(plots), int)

    for _, wg in plots.groupby("window"):
        scans = sorted(wg["scan"].unique())
        idx = {sc: wg.index[wg["scan"] == sc].to_numpy() for sc in scans}
        tracks = []          # cada trilha: dict(t=[], x=[], y=[], miss=int, vx, vy)
        for sc in scans:
            ib = idx[sc]
            t = sc * period
            if len(tracks) and len(ib):
                px = np.array([tr["x"][-1] + tr["vx"] * (t - tr["t"][-1]) for tr in tracks])
                py = np.array([tr["y"][-1] + tr["vy"] * (t - tr["t"][-1]) for tr in tracks])
                d = np.hypot(px[:, None] - X[ib][None, :], py[:, None] - Y[ib][None, :])
                c = np.where(d <= gate, d, gate * 1e3)
                ri, ci = linear_sum_assignment(c)
                ok = c[ri, ci] <= gate
                ri, ci = ri[ok], ci[ok]
            else:
                ri = ci = np.zeros(0, int)
            usados = set(ci.tolist())
            vistos = set(ri.tolist())
            for a, b in zip(ri, ci):
                tr = tracks[a]
                j = ib[b]
                tr["t"].append(t); tr["x"].append(X[j]); tr["y"].append(Y[j])
                tr["miss"] = 0
                _fit_track(tr, n_back)
                vx[j], vy[j] = tr["vx"], tr["vy"]
                n_pontos[j] = len(tr["t"])
            for b in range(len(ib)):
                if b in usados:
                    continue
                j = ib[b]
                tracks.append(dict(t=[t], x=[X[j]], y=[Y[j]], miss=0, vx=0.0, vy=0.0))
            for a, tr in enumerate(tracks[:len(tracks)]):
                if a not in vistos and tr["t"][-1] < t:
                    tr["miss"] += 1
            tracks = [tr for tr in tracks if tr["miss"] <= max_miss]

    plots["vx"] = vx
    plots["vy"] = vy
    plots["n_pontos"] = n_pontos
    return plots


def _fit_track(tr, n_back):
    t = np.asarray(tr["t"][-n_back:])
    if len(t) < 3:
        tr["vx"] = tr["vy"] = 0.0
        return
    tc = t - t.mean()
    den = (tc * tc).sum()
    if den <= 0:
        tr["vx"] = tr["vy"] = 0.0
        return
    x = np.asarray(tr["x"][-n_back:])
    y = np.asarray(tr["y"][-n_back:])
    tr["vx"] = float((tc * (x - x.mean())).sum() / den)
    tr["vy"] = float((tc * (y - y.mean())).sum() / den)


def add_kinematics(plots: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Aplica `estimate_kinematics` por (estacao, hora)."""
    pl = plots.merge(meta, on=["station", "hour", "scan"], how="left")
    out = []
    for (st, hh), g in pl.groupby(["station", "hour"], sort=False):
        out.append(estimate_kinematics(g, float(g["periodo"].iloc[0])))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# fusao dos plots de um agrupamento
# ---------------------------------------------------------------------------

def merge_by_label(g: pd.DataFrame, lab: np.ndarray) -> pd.DataFrame:
    """Substitui cada agrupamento por um unico plot (centroide ponderado)."""
    g = g.copy()
    g["_lab"] = lab
    w = g["amp"].to_numpy(float)
    w = np.where(w > 0, w, 1.0)
    g["_wx"] = g["x"] * w
    g["_wy"] = g["y"] * w
    g["_w"] = w
    a = g.groupby("_lab").agg(
        x=("_wx", "sum"), y=("_wy", "sum"), _w=("_w", "sum"),
        amp=("amp", "sum"), n=("n", "sum"), ext=("ext", "max"),
        vx=("vx", "mean"), vy=("vy", "mean"),
        station=("station", "first"), hour=("hour", "first"), scan=("scan", "first"))
    a["x"] /= a["_w"]
    a["y"] /= a["_w"]
    return a.drop(columns=["_w"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# clusterizador ClusWiSARD (Etapas 3 e 4)
# ---------------------------------------------------------------------------

import cluswisard as C   # noqa: E402

SPAN_XY = 12000.0    # extensao do referencial comum coberta pela codificacao [m]
L_XY = 2048          # bits de termometro por eixo de posicao  (~5,9 m/bit)
V_MAX = 15.0         # extensao da codificacao de velocidade [m/s]
TUPLE, N_RAM = 24, 128


def encode_plots(g: pd.DataFrame, vel_bits: int = 0) -> np.ndarray:
    """Retina binaria dos plots de uma varredura (termometro).

    A codificacao termometro e densa (~50 % de bits ativos), o que evita a
    degeneracao das tuplas para o endereco nulo, e faz a resposta da rede
    decair de forma controlada com a distancia: duas posicoes separadas de d
    diferem em d/SPAN dos bits, e a resposta de uma tupla de n bits vale
    aproximadamente (1 - d/SPAN)^n. O limiar `min_score` do clusterizador
    corresponde, portanto, a um **raio metrico de agrupamento**
    d ~ SPAN * (1 - min_score^{1/n}) -- 40 m para min_score = 0,95 com n = 24.

    Com `vel_bits > 0` acrescentam-se dois eixos de velocidade; o numero de
    bits controla o peso relativo da cinematica na resposta.
    """
    B = [C.thermometer(g["x"].to_numpy(float), -SPAN_XY / 2, SPAN_XY / 2, L_XY),
         C.thermometer(g["y"].to_numpy(float), -SPAN_XY / 2, SPAN_XY / 2, L_XY)]
    if vel_bits:
        vx = np.nan_to_num(g["vx"].to_numpy(float))
        vy = np.nan_to_num(g["vy"].to_numpy(float))
        B += [C.thermometer(vx, -V_MAX, V_MAX, vel_bits),
              C.thermometer(vy, -V_MAX, V_MAX, vel_bits)]
    return np.concatenate(B, axis=1).astype(np.uint8)


def cluswisard_cluster_scan(g: pd.DataFrame, min_score: float, vel_bits: int = 0,
                            seed: int = 0) -> pd.DataFrame:
    if len(g) <= 1:
        return g[["station", "hour", "scan", "x", "y", "amp", "n", "ext", "vx", "vy"]].copy()
    B = encode_plots(g, vel_bits)
    cl = C.ClusWiSARDClusterer(B.shape[1], tuple_size=TUPLE, n_ram=N_RAM,
                               min_score=min_score, limit=max(len(g), 4),
                               n_passes=2, seed=seed, table_bits=16)
    return merge_by_label(g, cl.fit_predict(B))


def cluster_all(plots: pd.DataFrame, fn, **kw) -> pd.DataFrame:
    """Aplica um clusterizador varredura a varredura."""
    out = [fn(g, **kw) for _, g in plots.groupby(["station", "hour", "scan"], sort=False)]
    return pd.concat(out, ignore_index=True) if out else plots.copy()
