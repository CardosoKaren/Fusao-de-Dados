"""
etapa0_build.py -- Construcao do DataSet_tratado.

Para cada par (estacao, hora) do HAXR:
  1. estima a mascara de visibilidade r_max(az) da estacao;
  2. seleciona janelas de 120 s de varreduras independentes (uma por rotacao);
  3. interpola o AIS para o instante em que o feixe ilumina cada alvo;
  4. mantem apenas as celulas de video radar contidas em um raio R_ROI em
     torno de alvos AIS visiveis -- preservando o clutter local -- e
     descarta o restante;
  5. grava o recorte em DataSet_tratado/<estacao>_<hora>-UTC_tratado.hdf5.

Uso:  python3 Codigo/etapa0_build.py [--jobs 12]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import haxr_io as H

# ----------------------------------------------------------------------------
# Parametros do recorte
# ----------------------------------------------------------------------------
R_ROI = 200.0        # raio da regiao de interesse em torno de cada alvo AIS [m]
WIN_SEC = 120.0      # duracao de cada janela (2 min -- exigida pela cinematica)
N_WIN = 2            # janelas por (estacao, hora)
N_AZ = 4096          # celulas de azimute por rotacao (0.0879 deg)
AZ_STEP = 360.0 / N_AZ


def _range_grid(rf: H.RadarFile, n_probe: int = 30):
    """Descobre a grade de alcance nativa (r0, dr, n_r) do sensor."""
    vals = []
    n = len(rf.c_first)
    for i in np.linspace(5, n - 5, n_probe).astype(int):
        a, b = int(rf.c_first[i]), int(rf.c_last[i])
        vals.append(rf.f["r1"][a:b])
    v = np.unique(np.concatenate(vals))
    d = np.diff(v)
    dr = float(np.median(d[d > 0.1]))
    r0 = float(v.min())
    n_r = int(np.ceil((v.max() - r0) / dr)) + 2
    return r0, dr, n_r


def build_chunk(args):
    station, hour = args
    t_ini = time.time()
    stns = H.load_stations()
    x0, y0 = float(stns.loc[station, "x"]), float(stns.loc[station, "y"])
    ais = H.load_ais(station, hour, stns)

    rf = H.RadarFile(station, hour)
    period = rf.rotation_period()
    prof = rf.coverage_profile(n_cycles=150, n_az=720)
    r0, dr, n_r = _range_grid(rf)

    # --- janelas de 120 s uniformemente distribuidas na hora ----------------
    t_lo, t_hi = float(rf.c_tod.min()), float(rf.c_tod.max())
    span = t_hi - t_lo
    starts = [t_lo + (k + 0.5) * span / N_WIN - WIN_SEC / 2 for k in range(N_WIN)]

    scan_rows, cell_az, cell_r, cell_amp, cell_tod, cell_first, cell_last = [], [], [], [], [], [], []
    tgt_rows = []
    ncell = 0

    for wid, ts in enumerate(starts):
        idx = rf.independent_cycles(ts, ts + WIN_SEC, period)
        for ci in idx:
            s = rf.load_cycle(int(ci))
            if len(s.amp) < 100:
                continue
            az_start = float(s.az[0])

            # --- posicao AIS no instante de iluminacao do feixe -------------
            tg = H.ais_at_time(ais, s.t, max_dt=3.0)
            if len(tg) == 0:
                continue
            X, Y = tg["X"].to_numpy(float), tg["Y"].to_numpy(float)
            vx, vy = tg["vx"].to_numpy(float), tg["vy"].to_numpy(float)
            for _ in range(2):  # iteracao ponto-fixo azimute <-> tempo
                r_, az_ = H.local_to_polar(X, Y, x0, y0)
                dt = ((az_ - az_start) % 360.0) / 360.0 * period
                t_beam = s.t0 + dt
                X = tg["X"].to_numpy(float) + vx * (t_beam - s.t)
                Y = tg["Y"].to_numpy(float) + vy * (t_beam - s.t)
            r_, az_ = H.local_to_polar(X, Y, x0, y0)

            vis = H.visible_mask(prof, r_, az_) & (r_ > 30.0)
            if not vis.any():
                continue
            uids = tg.index.to_numpy()[vis]
            X, Y, r_, az_ = X[vis], Y[vis], r_[vis], az_[vis]
            vx, vy = vx[vis], vy[vis]

            # --- recorte: celulas a menos de R_ROI de algum alvo visivel -
            cx = s.r * np.sin(np.deg2rad(s.az))
            cy = s.r * np.cos(np.deg2rad(s.az))
            tx = r_ * np.sin(np.deg2rad(az_))
            ty = r_ * np.cos(np.deg2rad(az_))
            keep = np.zeros(len(cx), dtype=bool)
            for j in range(len(tx)):
                box = (np.abs(cx - tx[j]) < R_ROI) & (np.abs(cy - ty[j]) < R_ROI)
                if not box.any():
                    continue
                k = np.where(box)[0]
                keep[k[(cx[k] - tx[j]) ** 2 + (cy[k] - ty[j]) ** 2 < R_ROI ** 2]] = True
            if keep.sum() == 0:
                continue

            ai = np.round(s.az[keep] / AZ_STEP).astype(np.uint16) % N_AZ
            ri = np.clip(np.round((s.r[keep] - r0) / dr), 0, n_r - 1).astype(np.uint16)
            sid = len(scan_rows)
            cell_first.append(ncell)
            ncell += int(keep.sum())
            cell_last.append(ncell)
            cell_az.append(ai)
            cell_r.append(ri)
            cell_amp.append(s.amp[keep])
            cell_tod.append(s.tod[keep])
            scan_rows.append((int(ci), s.t0, s.t1, s.t, wid))
            for j in range(len(uids)):
                tgt_rows.append((sid, uids[j], X[j], Y[j], r_[j], az_[j], vx[j], vy[j]))

    rf.close()
    if not scan_rows:
        return station, hour, 0, 0, 0, time.time() - t_ini

    os.makedirs(H.OUT_DIR, exist_ok=True)
    out = os.path.join(H.OUT_DIR, f"{station}_{hour}-UTC_tratado.hdf5")
    sc = np.array(scan_rows, dtype=object)
    tg = pd.DataFrame(tgt_rows, columns=["scan", "uid", "X", "Y", "r", "az", "vx", "vy"])
    with h5py.File(out, "w") as g:
        g.attrs.update(dict(station=station, hour=hour, x0=x0, y0=y0,
                            roi_radius_m=R_ROI, rot_period_s=period,
                            r0=r0, dr=dr, n_r=n_r, n_az=N_AZ, az_step=AZ_STEP,
                            win_sec=WIN_SEC, n_win=N_WIN))
        gs = g.create_group("scan")
        gs.create_dataset("cycle", data=np.array([r[0] for r in scan_rows], np.uint32))
        gs.create_dataset("t0", data=np.array([r[1] for r in scan_rows], np.float64))
        gs.create_dataset("t1", data=np.array([r[2] for r in scan_rows], np.float64))
        gs.create_dataset("t", data=np.array([r[3] for r in scan_rows], np.float64))
        gs.create_dataset("window", data=np.array([r[4] for r in scan_rows], np.uint8))
        gs.create_dataset("first", data=np.array(cell_first, np.uint64))
        gs.create_dataset("last", data=np.array(cell_last, np.uint64))
        gc = g.create_group("cell")
        gc.create_dataset("az_idx", data=np.concatenate(cell_az), compression="gzip", compression_opts=1)
        gc.create_dataset("r_idx", data=np.concatenate(cell_r), compression="gzip", compression_opts=1)
        gc.create_dataset("amp", data=np.concatenate(cell_amp), compression="gzip", compression_opts=1)
        gc.create_dataset("tod", data=np.concatenate(cell_tod).astype(np.float32), compression="gzip", compression_opts=1)
        gt = g.create_group("tgt")
        gt.create_dataset("scan", data=tg["scan"].to_numpy(np.uint32))
        gt.create_dataset("uid", data=np.array(tg["uid"].tolist(), dtype="S8"))
        for c in ["X", "Y", "r", "az", "vx", "vy"]:
            gt.create_dataset(c, data=tg[c].to_numpy(np.float64))
        g.create_dataset("coverage_prof", data=prof)
    return station, hour, len(scan_rows), ncell, len(tg), time.time() - t_ini


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(H.OUT_DIR, exist_ok=True)
    jobs = [(s, h) for s in H.station_list() for h in H.HOURS]
    rows = []
    with Pool(a.jobs) as p:
        for res in p.imap_unordered(build_chunk, jobs):
            st, hh, ns, nc, nt, el = res
            print(f"{st:18s} {hh}  varreduras={ns:5d}  celulas={nc:9d}  alvos={nt:6d}  {el:6.1f}s", flush=True)
            rows.append(dict(station=st, hour=hh, n_scan=ns, n_cell=nc, n_tgt=nt, sec=round(el, 1)))
    df = pd.DataFrame(rows).sort_values(["station", "hour"])
    os.makedirs(H.TAB_DIR, exist_ok=True)
    df.to_csv(os.path.join(H.TAB_DIR, "etapa0_dataset_tratado.csv"), index=False)
    print("\nTOTAL varreduras:", df.n_scan.sum(), " celulas:", df.n_cell.sum(), " alvos-varredura:", df.n_tgt.sum())


if __name__ == "__main__":
    main()
