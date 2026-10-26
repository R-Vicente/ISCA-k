"""Reproduce one run of the distribution study (MCAR 20% and 40%) and compute its measures.

Usage:
    python reproduzir_distribuicoes.py --dataset Wine --condicao mcar_20 --metodo todos --repeticao 0

Measures (codigo/benchmark/medidas.py), over the hidden cells:
    missing_wasserstein_sd   Wasserstein distance in units of the SD of the complete column (W)
    missing_total_variation  total variation for categorical columns (TV)
    sd_ratio                 SD of the imputed values over SD of the true values (R)
    pearson_mae              mean absolute error of the Pearson correlations between numeric pairs (E_r)

Compare with resultados/distribuicoes/distribuicoes_repeticoes.csv and
resultados/distribuicoes/distribuicoes_por_variavel_celulas_ocultas.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
from threadpoolctl import threadpool_limits

import reproduzir as rp  # noqa: E402  (adds codigo/ to the import path)
from medidas import distributions, summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--condicao", required=True, choices=("mcar_20", "mcar_40"))
    p.add_argument("--metodo", required=True, choices=list(rp.METODOS) + ["todos"])
    p.add_argument("--repeticao", default="0")
    a = p.parse_args()

    ds = rp.carregar_dataset(a.dataset)
    mascaras, registos = rp.carregar_mascaras(a.dataset, a.condicao, ds)
    metodos = list(rp.METODOS.values()) if a.metodo == "todos" else [rp.METODOS[a.metodo]]
    X, tipos = ds["X"], ds["types"]
    for r in rp.intervalo(a.repeticao):
        mascara = mascaras[r]
        if rp.sha256_array(mascara) != registos[r]["mask_sha256"]:
            raise RuntimeError("Máscara individual alterada.")
        X_inc = X.copy()
        X_inc[mascara] = np.nan
        for metodo in metodos:
            with threadpool_limits(limits=1):
                rp.e.isk._MI_CACHE.clear()
                Y = rp.imputar(metodo, X_inc, ds)
            med = distributions(X, Y, mascara, tipos)
            s = summary(med)
            razoes = [c["missing"]["sd_ratio"] for c in med["columns"]
                      if c["type"] == "cont" and c["missing"] and c["missing"]["sd_ratio"] is not None]
            fmt = lambda v: "-" if v is None else f"{v:.4f}"  # noqa: E731
            print(f"{a.dataset:<18} {a.condicao:<8} {metodo:<12} rep {r:>2}  "
                  f"W {fmt(s['missing_wasserstein_sd'])}  TV {fmt(s['missing_total_variation'])}  "
                  f"R {fmt(float(np.mean(razoes)) if razoes else None)}  E_r {fmt(s['pearson_mae'])}", flush=True)


if __name__ == "__main__":
    main()
