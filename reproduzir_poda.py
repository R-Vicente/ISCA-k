"""Reproduce the ISCA-k candidate pruning trials (sections 5.7.2 and 5.7.3 of the dissertation).

Usage:
    python reproduzir_poda.py --dataset "California Housing" --variante todas --repeticao 0-4

Variants (MCAR 20%, replicates 0 to 4, the same masks as the main study):
    sem_poda                  ISCA-k with pruning switched off (full search)
    poda_original             ISCA-k with the pruning used in the main study
    poda_revista              revised pruning (codigo/src/isca_k_revised_pruning.py), automatic activation
    controlo_mediana_completa revised pruning with the low-overlap penalty median computed over all rows
                              (diagnostic control; its time does not measure a speed-up)

Times are comparable only when the variants run in sequence, in the same session, with a single
thread, as in the dissertation. Compare with resultados/poda_original/ and resultados/poda_revista/.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import reproduzir as rp  # noqa: F401  (adds codigo/ to the import path)
from reproduzir import e, h, intervalo
import isca_k_revised_pruning as revista  # noqa: E402

VARIANTES = ("sem_poda", "poda_original", "poda_revista", "controlo_mediana_completa")


def configuracao(v, tipos, politica):
    _, forcadas = h._iscak_type_contract(tipos, politica)
    cols = [f"v{j}" for j in range(len(tipos))]
    params = dict(adaptive_k=True, k_min=3, k_max=25, use_mi_weights=True, mi_n_neighbors=5,
                  mi_random_state=42, use_propagation=True, use_distance_shrinkage=True,
                  shrink_strength=.5, standardize_numeric=True, homogeneity_threshold=2.5,
                  stability_threshold=.01, use_candidate_pruning=v != "sem_poda", candidate_k=25,
                  force_categorical=[cols[j] for j in sorted(forcadas)] or None,
                  interactive_type_detection=False, verbose=False, record_diagnostics=False)
    modulo = e.isk if v in ("sem_poda", "poda_original") else revista
    if modulo is revista:
        params.update(pruning_reference="full" if v == "controlo_mediana_completa" else "sample",
                      pruning_diagnostics=False)
    return modulo, params, cols


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--variante", required=True, choices=list(VARIANTES) + ["todas"])
    p.add_argument("--repeticao", default="0-4")
    a = p.parse_args()

    ds = rp.carregar_dataset(a.dataset)
    mascaras, registos = rp.carregar_mascaras(a.dataset, "mcar_20", ds)
    variantes = VARIANTES if a.variante == "todas" else (a.variante,)
    for r in intervalo(a.repeticao):
        X = ds["X"]
        X_inc = X.copy()
        X_inc[mascaras[r]] = np.nan
        for v in variantes:
            modulo, params, cols = configuracao(v, ds["types"], ds["manifest"].get("iscak_type_policy"))
            with threadpool_limits(limits=1):
                modulo._MI_CACHE.clear()
                t0 = time.perf_counter()
                out = modulo.ISCAkImputer(modulo.ISCAkConfig(**params)).fit_transform(pd.DataFrame(X_inc, columns=cols))
                segundos = time.perf_counter() - t0
            X_imp = e.validate_output(X_inc, out[cols].to_numpy(dtype=float), ds["types"])
            m = h._metrics_for_imputation(X, X_imp, mascaras[r], ds["types"])
            print(f"{a.dataset:<18} rep {r}  {v:<26} NRMSE {m['nrmse_sd_mean']:.6f}  tempo {segundos:.3f} s", flush=True)


if __name__ == "__main__":
    main()
