"""
tratado_io.py -- Leitura do DataSet_tratado e rasterizacao polar.

O DataSet_tratado guarda, por (estacao, hora), apenas as celulas de video
radar contidas nas regioes de interesse (ROI) de raio R_ROI centradas em
alvos AIS visiveis. As celulas sao esparsas: uma celula ausente tem
amplitude 0 (o video do sensor ja e limiarizado em 1).
"""
from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import haxr_io as H

TRAIN_HOURS = ["08"]          # particao de treino
TEST_HOURS = ["09", "11"]     # particao de teste (avaliacao reportada)
FIT_WINDOW = 0                # janela de 120 s usada para ajustar modelos
VAL_WINDOW = 1                # janela de 120 s usada para escolher hiperparametros


def load_gt() -> pd.DataFrame:
    """Ground truth consolidado (Etapa 0)."""
    g = pd.read_csv(os.path.join(H.OUT_DIR, "ground_truth.csv"))
    g["hour"] = g["hour"].astype(str).str.zfill(2)
    return g


def chunk_path(station: str, hour: str) -> str:
    return os.path.join(H.OUT_DIR, f"{station}_{hour}-UTC_tratado.hdf5")


def list_chunks(hours=None):
    hours = hours or H.HOURS
    out = []
    for s in H.station_list():
        for h in hours:
            if os.path.exists(chunk_path(s, h)):
                out.append((s, h))
    return out


class Tratado:
    """Acesso a um recorte (estacao, hora) do DataSet_tratado."""

    def __init__(self, station: str, hour: str):
        self.station, self.hour = station, hour
        self.f = h5py.File(chunk_path(station, hour), "r")
        a = self.f.attrs
        self.x0, self.y0 = float(a["x0"]), float(a["y0"])
        self.r0, self.dr = float(a["r0"]), float(a["dr"])
        self.n_r, self.n_az = int(a["n_r"]), int(a["n_az"])
        self.az_step = float(a["az_step"])
        self.roi = float(a["roi_radius_m"])
        self.period = float(a["rot_period_s"])
        s = self.f["scan"]
        self.scan = pd.DataFrame({k: s[k][:] for k in ["cycle", "t0", "t1", "t", "window", "first", "last"]})
        t = self.f["tgt"]
        self.tgt = pd.DataFrame({
            "scan": t["scan"][:], "uid": [u.decode() for u in t["uid"][:]],
            "X": t["X"][:], "Y": t["Y"][:], "r": t["r"][:], "az": t["az"][:],
            "vx": t["vx"][:], "vy": t["vy"][:]})
        self._az = self.f["cell/az_idx"]
        self._ri = self.f["cell/r_idx"]
        self._amp = self.f["cell/amp"]
        self.coverage = self.f["coverage_prof"][:]
        self._tgt_by_scan = {k: v for k, v in self.tgt.groupby("scan")}

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    @property
    def n_scan(self) -> int:
        return len(self.scan)

    # -- acesso as celulas -------------------------------------------------
    def cells(self, i: int):
        """(az_idx, r_idx, amp) da varredura i."""
        a, b = int(self.scan["first"][i]), int(self.scan["last"][i])
        return self._az[a:b], self._ri[a:b], self._amp[a:b]

    def targets(self, i: int) -> pd.DataFrame:
        return self._tgt_by_scan.get(i, self.tgt.iloc[:0])

    # -- geometria ---------------------------------------------------------
    def cell_polar(self, az_idx, r_idx):
        return (self.r0 + np.asarray(r_idx, float) * self.dr,
                np.asarray(az_idx, float) * self.az_step)

    def cell_xy(self, az_idx, r_idx, absolute=False):
        r, az = self.cell_polar(az_idx, r_idx)
        a = np.deg2rad(az)
        x, y = r * np.sin(a), r * np.cos(a)
        if absolute:
            x, y = x + self.x0, y + self.y0
        return x, y

    def xy_to_cell(self, x, y):
        """(x, y) relativo a estacao -> (az_idx, r_idx) continuos."""
        r = np.hypot(x, y)
        az = np.rad2deg(np.arctan2(x, y)) % 360.0
        return az / self.az_step, (r - self.r0) / self.dr

    # -- rasterizacao ------------------------------------------------------
    def raster(self, i: int):
        """Imagem polar densa (n_az x n_r, uint8) da varredura i.

        Celulas fora das ROI valem 0 (nao foram preservadas); celulas dentro
        da ROI sem retorno tambem valem 0 (video do sensor limiarizado).
        """
        az, ri, amp = self.cells(i)
        img = np.zeros((self.n_az, self.n_r), np.uint8)
        img[az, ri] = amp
        return img

    def roi_mask(self, i: int, img_shape=None):
        """Mascara booleana (n_az x n_r) das celulas dentro de alguma ROI."""
        tg = self.targets(i)
        m = np.zeros((self.n_az, self.n_r), bool)
        if len(tg) == 0:
            return m
        rr = self.r0 + np.arange(self.n_r) * self.dr
        aa = np.arange(self.n_az) * self.az_step
        for _, t in tg.iterrows():
            # janela polar que contem o disco de raio ROI
            dr_bins = int(np.ceil(self.roi / self.dr)) + 1
            j0 = max(0, int((t.r - self.r0) / self.dr) - dr_bins)
            j1 = min(self.n_r, int((t.r - self.r0) / self.dr) + dr_bins + 1)
            if j1 <= j0:
                continue
            half = np.rad2deg(np.arcsin(min(1.0, self.roi / max(t.r, self.roi)))) if t.r > self.roi else 180.0
            k = np.arange(int(np.floor((t.az - half) / self.az_step)),
                          int(np.ceil((t.az + half) / self.az_step)) + 1) % self.n_az
            R, A = np.meshgrid(rr[j0:j1], np.deg2rad(aa[k]), indexing="ij")
            X, Y = R * np.sin(A), R * np.cos(A)
            tx, ty = t.r * np.sin(np.deg2rad(t.az)), t.r * np.cos(np.deg2rad(t.az))
            d = (X - tx) ** 2 + (Y - ty) ** 2 < self.roi ** 2
            sub = m[k, j0:j1]
            m[np.ix_(k, np.arange(j0, j1))] = sub | d.T
        return m
