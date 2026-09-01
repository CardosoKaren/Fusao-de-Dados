"""
etapa7_comparacao.py -- Comparacao das seis propostas.

Consolida as metricas das Etapas 1 a 6 em tabelas (CSV e LaTeX) e figuras:

  P1  Filtro ClusWiSARD
  P2  Filtro OS-CFAR
  P3  Filtro ClusWiSARD + clusterizador ClusWiSARD
  P4  Filtro ClusWiSARD + clusterizador ClusWiSARD + cinematica
  P5  Filtro OS-CFAR + DBSCAN
  P6  Filtro OS-CFAR + DBSCAN + cinematica

Alem da tabela principal (particao de teste, ground truth completo), sao
produzidas: a analise de sensibilidade restrita aos alvos com eco confirmado,
o desempenho por estacao, e a linha de base sem filtro algum (video bruto).

Uso:  python3 Codigo/etapa7_comparacao.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figuras_resultados as FR
import haxr_io as H
import pipeline as PL
import plots_eval as P
import tratado_io as T

ORDEM = FR.PROPOSTAS
ROTULO = {
    "etapa1_cluswisard": "P1  Filtro ClusWiSARD",
    "etapa2_oscfar": "P2  Filtro OS-CFAR",
    "etapa3_cluswisard_cluster": "P3  ClusWiSARD + clust. ClusWiSARD",
    "etapa4_cluswisard_cluster_cinematica": "P4  ClusWiSARD + clust. ClusWiSARD + cinematica",
    "etapa5_oscfar_dbscan": "P5  OS-CFAR + DBSCAN",
    "etapa6_oscfar_dbscan_cinematica": "P6  OS-CFAR + DBSCAN + cinematica",
}


def baseline(gt):
    """Linha de base: plots do video bruto recortado, sem filtro."""
    rows = []
    for st, hh in T.list_chunks():
        d = T.Tratado(st, hh)
        for i in range(d.n_scan):
            az, ri, amp = d.cells(i)
            x, y = d.cell_xy(az, ri)
            ps = P.extract_plots(x, y, amp)
            rows.append(PL.plots_to_frame(st, hh, i, ps, d.x0, d.y0))
        d.close()
    pl = pd.concat(rows, ignore_index=True)
    PL.save_detections(pl, "etapa7_baseline")
    m, per = PL.evaluate(pl, gt, hours=T.TEST_HOURS)
    PL.by_station(pl, gt, hours=T.TEST_HOURS).to_csv(
        os.path.join(H.TAB_DIR, "etapa7_baseline_por_estacao.csv"))
    return m, pl


def main():
    gt = T.load_gt()

    print("linha de base (video bruto recortado, sem filtro)...")
    mb, plb = baseline(gt)
    mbc, _ = PL.evaluate(plb, gt, hours=T.TEST_HOURS, only_confirmed=True)

    rows = [dict(proposta="B0  Video bruto (sem filtro)", **mb)]
    conf = [dict(proposta="B0  Video bruto (sem filtro)", **mbc)]
    for n, _ in ORDEM:
        f = os.path.join(PL.RES_DIR, f"{n}_metricas.csv")
        if not os.path.exists(f):
            print(f"  (ausente: {n})")
            continue
        m = pd.read_csv(f).iloc[0].to_dict()
        m["proposta"] = ROTULO[n]
        rows.append(m)
        pl = PL.load_detections(n)
        mc, _ = PL.evaluate(pl, gt, hours=T.TEST_HOURS, only_confirmed=True)
        conf.append(dict(proposta=ROTULO[n], **mc))

    cols = ["proposta", "VP", "FP", "FN", "precisao", "recall", "F1",
            "plots", "alvos", "plots_por_alvo"]
    tab = pd.DataFrame(rows)[cols].round(4)
    tab.to_csv(os.path.join(H.TAB_DIR, "etapa7_comparacao.csv"), index=False)
    tabc = pd.DataFrame(conf)[cols].round(4)
    tabc.to_csv(os.path.join(H.TAB_DIR, "etapa7_comparacao_confirmados.csv"), index=False)

    print("\n=== Etapa 7 -- comparacao das propostas (teste: 09 e 11 UTC) ===")
    print(tab.to_string(index=False))
    print("\n=== sensibilidade: apenas alvos com eco confirmado ===")
    print(tabc.to_string(index=False))

    # tabelas LaTeX
    with open(os.path.join(H.TAB_DIR, "etapa7_comparacao.tex"), "w") as f:
        f.write(to_latex(tab, "Desempenho das seis propostas na particao de teste "
                              "(09 e 11 UTC, 13 estacoes).", "tab:comparacao"))
    with open(os.path.join(H.TAB_DIR, "etapa7_comparacao_confirmados.tex"), "w") as f:
        f.write(to_latex(tabc, "Analise de sensibilidade: metricas restritas aos alvos "
                               "AIS com eco radar confirmado.", "tab:comparacao_conf"))

    # por estacao
    d = {}
    for n, _ in ORDEM:
        p = os.path.join(H.TAB_DIR, f"{n.split('_')[0]}_por_estacao.csv")
        if os.path.exists(p):
            d[ROTULO[n].split()[0]] = pd.read_csv(p, index_col=0)["F1"]
    if d:
        pe = pd.DataFrame(d).round(4)
        pe.to_csv(os.path.join(H.TAB_DIR, "etapa7_f1_por_estacao.csv"))
        print("\n=== F1 por estacao ===")
        print(pe.to_string())

    FR.fig_comparacao()
    FR.fig_pr()
    FR.fig_por_estacao()


def to_latex(t: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(t.columns)
    head = " & ".join(c.replace("_", r"\_") for c in cols) + r" \\"
    body = []
    for _, r in t.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v).replace("_", r"\_"))
        body.append(" & ".join(cells) + r" \\")
    return ("\\begin{table}[!ht]\n\\centering\n\\small\n\\caption{" + caption + "}\n"
            "\\label{" + label + "}\n\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}\n"
            "\\hline\n" + head + "\n\\hline\n" + "\n".join(body) +
            "\n\\hline\n\\end{tabular}\n\\end{table}\n")


if __name__ == "__main__":
    main()
