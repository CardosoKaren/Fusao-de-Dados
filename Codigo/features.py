"""
features.py -- Retina binaria por celula de video radar.

A unidade de decisao dos filtros e a *celula com retorno* (o video do HAXR e
esparso; celulas sem retorno tem amplitude 0 e nao precisam ser classificadas).

Para cada celula constroi-se uma retina amostrada em um referencial local
alinhado aos eixos do radar -- distancia (radial) e travessia (tangencial) --
em duas escalas. Nesse referencial a conversao para indices da grade polar e
exata e barata:

    delta_indice_distancia = k_r * passo / dr                 (constante)
    delta_indice_azimute   = k_t * passo / (r * passo_azimute) (depende de r)

o que mantem a *extensao metrica* da retina constante em toda a cobertura,
apesar de a celula de resolucao em travessia crescer com a distancia.

A retina e complementada por atributos escalares (amplitude propria, distancia,
densidade local e amplitude media em cada escala), codificados em termometro.
"""
from __future__ import annotations

import numpy as np

import cluswisard as C

# escalas: (numero de amostras por eixo, passo em metros)
SCALES = ((9, 7.0), (7, 24.0))
LEVELS = (0, 64)          # limiares de binarizacao das amostras da retina
N_SCALAR = 12             # bits por atributo escalar
R_GUARD_FEAT = 24.0       # raio de guarda das estatisticas tipo CFAR [m]


def _sample(img, az_idx, r_idx, r_m, n_az, n_r, dr, az_step_rad, k, step):
    """Amostra a retina de uma escala: (M, k, k) uint8."""
    h = k // 2
    kk = np.arange(-h, h + 1)
    d_r = np.round(kk * step / dr).astype(np.int64)                    # (k,)
    d_t = np.round(kk[None, :] * step / (r_m[:, None] * az_step_rad)).astype(np.int64)  # (M,k)
    A = (az_idx[:, None].astype(np.int64) + d_t)[:, :, None] % n_az    # (M,k,1)
    R = np.clip(r_idx[:, None, None].astype(np.int64) + d_r[None, None, :], 0, n_r - 1)
    return img[A, R]


def scan_retina(d, i, scales=SCALES, levels=LEVELS):
    """Retina binaria e metadados das celulas da varredura i.

    Retorna (B, info) com B (M, n_bits) uint8 e info um dicionario com
    az_idx, r_idx, amp, x, y, r.
    """
    az_idx, r_idx, amp = d.cells(i)
    M = len(amp)
    if M == 0:
        return np.zeros((0, 1), np.uint8), None
    img = d.raster(i)
    az_idx = az_idx.astype(np.int64)
    r_idx = r_idx.astype(np.int64)
    r_m = d.r0 + r_idx * d.dr
    az_step_rad = np.deg2rad(d.az_step)

    bits, dens, meanamp = [], [], []
    coroa = None
    for k, step in scales:
        S = _sample(img, az_idx, r_idx, r_m, d.n_az, d.n_r, d.dr, az_step_rad, k, step)
        Fl = S.reshape(M, -1)
        for lv in levels:
            bits.append((Fl > lv).astype(np.uint8))
        dens.append((Fl > 0).mean(axis=1))
        meanamp.append(Fl.mean(axis=1))
        h = k // 2
        kk = np.arange(-h, h + 1) * step
        dist = np.hypot(*np.meshgrid(kk, kk, indexing="ij")).ravel()
        if coroa is None or step > scales[0][1]:
            sel = dist > R_GUARD_FEAT
            if sel.sum() >= 8:
                coroa = np.sort(Fl[:, sel].astype(np.float64), axis=1)

    # estatisticas tipo CFAR: razao entre a amplitude propria e o nivel de fundo
    # local, estimado por estatistica de ordem na coroa alem do raio de guarda.
    a = amp.astype(np.float64)
    if coroa is None:
        z98 = z75 = mz = np.zeros(M)
        rank = np.zeros(M)
    else:
        n = coroa.shape[1]
        z98 = coroa[:, min(int(np.ceil(0.98 * n)) - 1, n - 1)]
        z75 = coroa[:, min(int(np.ceil(0.75 * n)) - 1, n - 1)]
        mz = coroa.mean(axis=1)
        rank = (coroa < a[:, None]).mean(axis=1)

    sc = [C.thermometer(a, 0, 200, N_SCALAR),
          C.thermometer(r_m, 0, 6000, N_SCALAR),
          C.thermometer(dens[0], 0, 0.6, N_SCALAR),
          C.thermometer(dens[1], 0, 0.6, N_SCALAR),
          C.thermometer(meanamp[0], 0, 90, N_SCALAR),
          C.thermometer(meanamp[1], 0, 90, N_SCALAR),
          C.thermometer(a / (z98 + 1.0), 0, 6, N_SCALAR),
          C.thermometer(a / (z75 + 1.0), 0, 12, N_SCALAR),
          C.thermometer(a / (mz + 1.0), 0, 12, N_SCALAR),
          C.thermometer(a - z98, -120, 200, N_SCALAR),
          C.thermometer(rank, 0, 1, N_SCALAR)]
    B = np.concatenate(bits + sc, axis=1).astype(np.uint8)
    x, y = d.cell_xy(az_idx, r_idx)
    info = dict(az_idx=az_idx, r_idx=r_idx, amp=amp, x=x, y=y, r=r_m,
                dens_fina=dens[0], dens_grossa=dens[1], razao_cfar=a / (z98 + 1.0))
    return B, info


def n_bits(scales=SCALES, levels=LEVELS):
    n = sum(k * k * len(levels) for k, _ in scales)
    return n + 11 * N_SCALAR


def gt_xy(d, i):
    """Posicoes de ground truth (AIS visivel) da varredura i, rel. a estacao."""
    tg = d.targets(i)
    if len(tg) == 0:
        return np.zeros((0, 2))
    a = np.deg2rad(tg["az"].to_numpy())
    r = tg["r"].to_numpy()
    return np.c_[r * np.sin(a), r * np.cos(a)]


def dist_to_gt(info, G):
    """Distancia de cada celula ao alvo de ground truth mais proximo."""
    if len(G) == 0:
        return np.full(len(info["x"]), np.inf)
    return np.min(np.hypot(info["x"][:, None] - G[None, :, 0],
                           info["y"][:, None] - G[None, :, 1]), axis=1)


def metric_window(d, i, k=15, step=8.0):
    """Janela metrica (k x k, passo `step`) em torno de cada celula com retorno.

    Retorna (S, dist, info): S (M, k*k) amplitudes amostradas, dist (k*k)
    distancia de cada amostra ao centro em metros, e os metadados das celulas.
    Amostragem no referencial local distancia/travessia, de modo que a extensao
    metrica da janela e a mesma em toda a cobertura.
    """
    az_idx, r_idx, amp = d.cells(i)
    M = len(amp)
    if M == 0:
        return None, None, None
    img = d.raster(i)
    az_idx = az_idx.astype(np.int64)
    r_idx = r_idx.astype(np.int64)
    r_m = d.r0 + r_idx * d.dr
    S = _sample(img, az_idx, r_idx, r_m, d.n_az, d.n_r, d.dr,
                np.deg2rad(d.az_step), k, step)
    h = k // 2
    kk = np.arange(-h, h + 1) * step
    dist = np.hypot(*np.meshgrid(kk, kk, indexing="ij")).ravel()
    x, y = d.cell_xy(az_idx, r_idx)
    info = dict(az_idx=az_idx, r_idx=r_idx, amp=amp, x=x, y=y, r=r_m)
    return S.reshape(M, -1), dist, info
