"""Reproduce one imputation of the main study and compute its metrics.

Usage:
    python reproduzir.py --dataset Iris --condicao mcar_20 --metodo ISCA-k --repeticao 0
    python reproduzir.py --dataset Iris --condicao mcar_20 --metodo todos --repeticao 0
    python reproduzir.py --dataset Iris --condicao mcar_20 --metodo SoftImpute --repeticao 0-74 --csv saida.csv

Conditions: mcar_05, mcar_10, mcar_20, mcar_30, mcar_40, mcar_50, blockwise_20, mar_20.
Methods: Media/Moda, k-NN, MICE, missForest, SoftImpute, ISCA-k (or "todos" for all).

Each output row can be compared with resultados/estudo_principal/benchmark_repeticoes.csv
(same dataset, condition, method and replicate).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

RAIZ = Path(__file__).resolve().parent
sys.path[:0] = [str(RAIZ / "codigo" / "src"), str(RAIZ / "codigo" / "benchmark")]

import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import bench_harness as h  # noqa: E402
import experiments as e  # noqa: E402
import softimpute as si  # noqa: E402
from harness_audit import load_dataset_snapshot, sha256_array, sha256_file  # noqa: E402

CONDICOES = ("mcar_05", "mcar_10", "mcar_20", "mcar_30", "mcar_40", "mcar_50", "blockwise_20", "mar_20")
METODOS = {"Media/Moda": "Mean/Mode", "k-NN": "k-NN", "MICE": "MICE",
           "missForest": "missForest", "SoftImpute": "SoftImpute", "ISCA-k": "ISCA-k"}


def carregar_dataset(nome):
    """Load the dataset snapshot and set the variable types used in the dissertation."""
    manifesto = json.loads((RAIZ / "dados" / "manifest.json").read_text(encoding="utf-8"))
    if nome not in manifesto["datasets"]:
        raise SystemExit(f"Dataset desconhecido: {nome}. Opções: {', '.join(manifesto['dataset_order'])}")
    entrada = dict(manifesto["datasets"][nome])
    entrada["snapshot_file"] = str(RAIZ / "dados" / entrada["snapshot_file"])
    ds = load_dataset_snapshot(entrada)          # checks the SHA-256 of the file and of the matrix
    ds["types"] = list(entrada["var_types"])     # protocol types (origin is categorical in Auto MPG)
    return ds


def carregar_mascaras(nome, condicao, ds):
    base = RAIZ / "dados" / "mascaras" / f"{nome.replace(' ', '_')}__{condicao}"
    info = json.loads(Path(f"{base}__manifest.json").read_text(encoding="utf-8"))
    ficheiro = Path(f"{base}__masks.npz")
    if sha256_file(ficheiro) != info["mask_bundle_sha256"]:
        raise RuntimeError(f"Ficheiro de máscaras alterado: {ficheiro.name}")
    if info["dataset_sha256"] != ds["manifest"]["snapshot_sha256"]:
        raise RuntimeError("Máscaras associadas a outro dataset.")
    with np.load(ficheiro, allow_pickle=False) as z:
        mascaras = np.asarray(z["masks"], dtype=bool)
    return mascaras, info["records"]


def imputar(metodo, X_inc, ds):
    """Run the method with the configuration used in the main study."""
    politica = ds["manifest"].get("iscak_type_policy")
    if metodo == "ISCA-k":
        X_imp, _, _ = e.fit_iscak(X_inc, ds["types"], politica)
    elif metodo == "SoftImpute":
        X_imp, _ = si.run_softimpute_tuned(X_inc, ds["types"])
    else:
        X_imp, _ = h.run_method_audited(metodo, X_inc, ds["types"], type_policy=politica)
    return e.validate_output(X_inc, X_imp, ds["types"])


def uma_repeticao(nome, condicao, metodo, r, ds, mascaras, registos):
    mascara = mascaras[r]
    if sha256_array(mascara) != registos[r]["mask_sha256"]:
        raise RuntimeError("Máscara individual alterada.")
    X = ds["X"]
    X_inc = X.copy()
    X_inc[mascara] = np.nan
    with threadpool_limits(limits=1):
        e.isk._MI_CACHE.clear()
        t0 = time.perf_counter()
        X_imp = imputar(metodo, X_inc, ds)
        segundos = time.perf_counter() - t0
    m = h._metrics_for_imputation(X, X_imp, mascara, ds["types"])
    pfc = m.get("pfc_mean")
    return {"dataset": nome, "condition": condicao, "method": metodo, "replicate": r,
            "nrmse": m["nrmse_sd_mean"], "pfc": "" if pfc is None or not np.isfinite(pfc) else pfc,
            "segundos": round(segundos, 4)}


def intervalo(texto):
    if "-" in texto:
        a, b = texto.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in texto.split(",")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--condicao", required=True, choices=CONDICOES)
    p.add_argument("--metodo", required=True, choices=list(METODOS) + ["todos"])
    p.add_argument("--repeticao", default="0", help="índice (0-74), lista '0,1,2' ou intervalo '0-74'")
    p.add_argument("--csv", help="grava os resultados neste ficheiro CSV")
    a = p.parse_args()

    ds = carregar_dataset(a.dataset)
    mascaras, registos = carregar_mascaras(a.dataset, a.condicao, ds)
    metodos = list(METODOS.values()) if a.metodo == "todos" else [METODOS[a.metodo]]
    linhas = []
    for r in intervalo(a.repeticao):
        for metodo in metodos:
            linha = uma_repeticao(a.dataset, a.condicao, metodo, r, ds, mascaras, registos)
            linhas.append(linha)
            pfc = f"{linha['pfc']:.6f}" if linha["pfc"] != "" else "-"
            print(f"{a.dataset:<18} {a.condicao:<13} {linha['method']:<12} rep {r:>2}  "
                  f"NRMSE {linha['nrmse']:.6f}  PFC {pfc}  ({linha['segundos']} s)", flush=True)
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0]))
            w.writeheader()
            w.writerows(linhas)


if __name__ == "__main__":
    main()
