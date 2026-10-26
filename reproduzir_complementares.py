"""Reproduce one run of the ISCA-k complementary studies (ablation, k sensitivity, recalibration).

Usage:
    python reproduzir_complementares.py --estudo ablacao --variante sem_MI --dataset Iris --repeticao 0
    python reproduzir_complementares.py --estudo sensibilidade_k --variante k_10 --dataset Wine --condicao mcar_40 --repeticao 0-4
    python reproduzir_complementares.py --estudo recalibracao --variante k_min_8 --dataset Titanic --condicao mar_20

Studies and variants (as defined in codigo/benchmark/experiments.py):
    ablacao          sem_MI, sem_k_adaptativo, sem_propagacao, sem_shrinkage, sem_poda       (mcar_20 only)
    sensibilidade_k  k_03, k_05, k_07, k_10, k_15, k_20, k_25, sqrtN                         (mcar_20, mcar_40)
    recalibracao     salvaguarda_sd, estabilidade_005, estabilidade_020, estabilidade_absoluta,
                     estabilidade_mista, k_min_8                          (mcar_20, mcar_40, mar_20, blockwise_20)

Compare with resultados/estudos_complementares/complementares_repeticoes.csv
(columns stage = ablation, ksens or calibration; variant; replicate).
"""
from __future__ import annotations

import argparse

import numpy as np
from threadpoolctl import threadpool_limits

import reproduzir as rp
from reproduzir import e, h, intervalo

ESTUDOS = {"ablacao": ("ablation", e.ABLATION, ("mcar_20",)),
           "sensibilidade_k": ("ksens", None, ("mcar_20", "mcar_40")),
           "recalibracao": ("calibration", e.CALIBRATION, ("mcar_20", "mcar_40", "mar_20", "blockwise_20"))}


def variantes(estudo, ds):
    if estudo == "sensibilidade_k":
        v = {f"k_{k:02}": {"adaptive_k": False, "k": k} for k in e.KGRID}
        v["sqrtN"] = {"adaptive_k": False, "k": int(h.k_bar(ds["X"].shape[0]))}
        return v
    return ESTUDOS[estudo][1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--estudo", required=True, choices=list(ESTUDOS))
    p.add_argument("--variante", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--condicao", default="mcar_20")
    p.add_argument("--repeticao", default="0")
    a = p.parse_args()

    etapa, _, condicoes = ESTUDOS[a.estudo]
    if a.condicao not in condicoes:
        raise SystemExit(f"O estudo {a.estudo} usa apenas: {', '.join(condicoes)}")
    ds = rp.carregar_dataset(a.dataset)
    vs = variantes(a.estudo, ds)
    if a.variante not in vs:
        raise SystemExit(f"Variante desconhecida. Opções: {', '.join(vs)}")
    mascaras, registos = rp.carregar_mascaras(a.dataset, a.condicao, ds)
    politica = ds["manifest"].get("iscak_type_policy")
    for r in intervalo(a.repeticao):
        if rp.sha256_array(mascaras[r]) != registos[r]["mask_sha256"]:
            raise RuntimeError("Máscara individual alterada.")
        X = ds["X"]
        X_inc = X.copy()
        X_inc[mascaras[r]] = np.nan
        with threadpool_limits(limits=1):
            e.isk._MI_CACHE.clear()
            X_imp, _, _ = e.fit_iscak(X_inc, ds["types"], politica, vs[a.variante])
        m = h._metrics_for_imputation(X, X_imp, mascaras[r], ds["types"])
        pfc = m.get("pfc_mean")
        pfc = f"{pfc:.6f}" if pfc is not None and np.isfinite(pfc) else "-"
        print(f"{etapa:<12} {a.dataset:<18} {a.condicao:<13} {a.variante:<22} rep {r:>2}  "
              f"NRMSE {m['nrmse_sd_mean']:.6f}  PFC {pfc}", flush=True)


if __name__ == "__main__":
    main()
