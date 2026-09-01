"""
haxr_io.py -- Camada de acesso ao dataset HAXR (Hamburg X-band Radar).

Fornece:
  * leitura dos arquivos .hdf5 (video radar bruto, esparso) e .csv (AIS);
  * conversao polar (estacao) <-> cartesiano local comum (frame do dataset);
  * selecao de varreduras (rotacoes de antena) nao sobrepostas;
  * mascara de visibilidade r_max(az) por estacao.

Convencoes geometricas (validadas empiricamente, ver etapa0.md):
    X = x_estacao + r * sin(az)      [metros, eixo Leste]
    Y = y_estacao + r * cos(az)      [metros, eixo Norte]
com az em graus, medido de Norte no sentido horario.

Autor: pipeline VF_FusaoDeDados
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Caminhos
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "DataSet_HAXR")
OUT_DIR = os.path.join(ROOT, "DataSet_tratado")
FIG_DIR = os.path.join(ROOT, "Figuras")
TAB_DIR = os.path.join(ROOT, "Tabelas")

HOURS = ["08", "09", "11"]

# ----------------------------------------------------------------------------
# Estacoes
# ----------------------------------------------------------------------------


def load_stations() -> pd.DataFrame:
    """Posicoes das 13 estacoes no frame cartesiano local comum."""
    df = pd.read_csv(os.path.join(RAW_DIR, "stations.csv"))
    df = df.rename(columns={"x (meters)": "x", "y (meters)": "y"})
    return df.set_index("station")


def station_list() -> list[str]:
    return list(load_stations().index)


# ----------------------------------------------------------------------------
# Geometria
# ----------------------------------------------------------------------------


def polar_to_local(r, az_deg, x0=0.0, y0=0.0):
    """(r, az) relativo a estacao -> (X, Y) no frame local comum."""
    a = np.deg2rad(az_deg)
    return x0 + r * np.sin(a), y0 + r * np.cos(a)


def local_to_polar(X, Y, x0=0.0, y0=0.0):
    """(X, Y) no frame local comum -> (r, az) relativo a estacao."""
    dx, dy = X - x0, Y - y0
    r = np.hypot(dx, dy)
    az = np.rad2deg(np.arctan2(dx, dy)) % 360.0
    return r, az


# ----------------------------------------------------------------------------
# AIS
# ----------------------------------------------------------------------------


def load_ais(station: str, hour: str, stations: pd.DataFrame | None = None) -> pd.DataFrame:
    """AIS de uma estacao/hora, com colunas r, az, X, Y, tod, uid.

    O CSV traz a posicao AIS ja reprojetada no referencial polar da estacao,
    reamostrada a 1 Hz.
    """
    if stations is None:
        stations = load_stations()
    p = os.path.join(RAW_DIR, f"{station}_{hour}-UTC.csv")
    d = pd.read_csv(p)
    d = d.rename(columns={"range (meters)": "r", "azimuth (degrees)": "az"})
    x0, y0 = stations.loc[station, "x"], stations.loc[station, "y"]
    d["X"], d["Y"] = polar_to_local(d["r"].to_numpy(float), d["az"].to_numpy(float), x0, y0)
    d["station"] = station
    return d


def ais_at_time(ais: pd.DataFrame, t: float, max_dt: float = 2.0) -> pd.DataFrame:
    """Interpola linearmente cada trilha AIS (uid) para o instante t.

    Retorna um DataFrame indexado por uid com r, az, X, Y, alem de vx, vy
    (velocidade em m/s estimada por diferencas centrais da trilha).
    """
    out = []
    for uid, g in ais.groupby("uid"):
        t_ = g["tod"].to_numpy(float)
        if t < t_.min() - max_dt or t > t_.max() + max_dt:
            continue
        j = np.searchsorted(t_, t)
        j = int(np.clip(j, 1, len(t_) - 1))
        if abs(t_[j] - t) > max_dt and abs(t_[j - 1] - t) > max_dt:
            continue
        X = np.interp(t, t_, g["X"].to_numpy(float))
        Y = np.interp(t, t_, g["Y"].to_numpy(float))
        # velocidade por regressao local (janela +-15 s)
        m = np.abs(t_ - t) <= 15.0
        if m.sum() >= 3:
            vx = np.polyfit(t_[m], g["X"].to_numpy(float)[m], 1)[0]
            vy = np.polyfit(t_[m], g["Y"].to_numpy(float)[m], 1)[0]
        else:
            vx = vy = 0.0
        out.append(dict(uid=uid, X=X, Y=Y, vx=vx, vy=vy))
    return pd.DataFrame(out).set_index("uid") if out else pd.DataFrame(
        columns=["X", "Y", "vx", "vy"], index=pd.Index([], name="uid")
    )


# ----------------------------------------------------------------------------
# Radar
# ----------------------------------------------------------------------------


@dataclass
class Scan:
    """Uma rotacao completa de antena."""

    station: str
    hour: str
    cycle: int
    t0: float
    t1: float
    az: np.ndarray  # graus, centro da celula
    r: np.ndarray  # metros, centro da celula
    amp: np.ndarray  # uint8
    tod: np.ndarray  # segundos UTC do dia

    @property
    def t(self) -> float:
        return 0.5 * (self.t0 + self.t1)


class RadarFile:
    """Acesso conveniente a um arquivo .hdf5 do HAXR."""

    def __init__(self, station: str, hour: str):
        self.station, self.hour = station, hour
        self.path = os.path.join(RAW_DIR, f"{station}_{hour}-UTC.hdf5")
        self.f = h5py.File(self.path, "r")
        g = self.f["cycle"]
        self.c_first = g["first"][:]
        self.c_last = g["last"][:]
        self.c_tod = g["tod"][:]

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # -- rotacao ---------------------------------------------------------
    def rotation_period(self, n=8) -> float:
        """Periodo de rotacao (s), estimado pelos saltos de azimute."""
        a, b = int(self.c_first[len(self.c_first) // 2]), None
        blk = self.f["az1"][a : a + 400000]
        tt = self.f["tod"][a : a + 400000]
        w = np.where(np.diff(blk) < -100)[0]
        if len(w) < 2:
            return 2.95
        return float(np.median(np.diff(tt[w])))

    def independent_cycles(self, t_start=None, t_end=None, period=None) -> np.ndarray:
        """Indices de ciclos nao sobrepostos (uma varredura por rotacao).

        A lista `cycle` do HAXR e uma janela deslizante (uma entrada a cada
        ~0.15 s); aqui selecionamos gulosamente ciclos separados por, ao menos,
        um periodo de rotacao.
        """
        if period is None:
            period = self.rotation_period()
        t = self.c_tod
        lo = t.min() if t_start is None else t_start
        hi = t.max() if t_end is None else t_end
        idx, last = [], -np.inf
        for i in np.where((t >= lo) & (t <= hi))[0]:
            if t[i] - last >= period * 0.995:
                idx.append(i)
                last = t[i]
        return np.asarray(idx, dtype=int)

    def load_cycle(self, i: int) -> Scan:
        a, b = int(self.c_first[i]), int(self.c_last[i])
        f = self.f
        az = 0.5 * (f["az1"][a:b] + f["az2"][a:b])
        r = 0.5 * (f["r1"][a:b] + f["r2"][a:b])
        amp = f["amp"][a:b]
        tod = f["tod"][a:b]
        return Scan(self.station, self.hour, i, float(tod[0]), float(tod[-1]),
                    az.astype(np.float32), r.astype(np.float32), amp, tod.astype(np.float32))

    # -- mascara de visibilidade -------------------------------------
    def coverage_profile(self, n_cycles: int = 120, n_az: int = 720) -> np.ndarray:
        """r_max(az): maior alcance com retorno registrado, por setor de azimute.

        Define a regiao fisicamente visivel pela estacao (o dataset e
        esparso: celulas sem retorno nao sao armazenadas, de modo que o
        envelope de alcance revela o horizonte de sombreamento imposto pela
        costa, cais e edificacoes).
        """
        prof = np.zeros(n_az)
        n = len(self.c_first)
        for i in np.linspace(5, n - 5, n_cycles).astype(int):
            a, b = int(self.c_first[i]), int(self.c_last[i])
            az = self.f["az1"][a:b]
            r2 = self.f["r2"][a:b]
            k = (az / (360.0 / n_az)).astype(int) % n_az
            np.maximum.at(prof, k, r2)
        # fechamento morfologico em azimute (preenche setores nao amostrados)
        k = 3
        pad = np.r_[prof[-k:], prof, prof[:k]]
        prof = np.array([pad[i : i + 2 * k + 1].max() for i in range(n_az)])
        return prof


def visible_mask(prof: np.ndarray, r, az) -> np.ndarray:
    """True quando (r, az) esta dentro do envelope de cobertura da estacao."""
    n_az = len(prof)
    k = (np.asarray(az) / (360.0 / n_az)).astype(int) % n_az
    return np.asarray(r) <= prof[k]
