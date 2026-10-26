"""Benchmark harness: runs the six methods of the main study and computes the metrics.

Mean/mode, k-NN, MICE and the metrics come from bench_utils, SoftImpute from
softimpute, and ISCA-k from codigo/src/isca_k.py. Each run returns the imputed
matrix together with an audit record of the effective configuration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Put codigo/src and codigo/benchmark first on the import path.
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
_SRC = _ROOT / "src"
_BENCH_DIR = _THIS.parent
for _p in (str(_SRC), str(_BENCH_DIR)):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path[:0] = [str(_SRC), str(_BENCH_DIR)]

import bench_utils as bu  # noqa: E402
from softimpute import run_softimpute_tuned  # noqa: E402


# Metrics and baseline imputers from bench_utils.
var_stats = bu.var_stats
per_var_metrics = bu.per_var_metrics
aggregate_by_type = bu.aggregate_by_type
full_matrix_rmse = bu.full_matrix_rmse
imputed_only_rmse = bu.imputed_only_rmse
get_type_indices = bu.get_type_indices
impute_mean_mode = bu.impute_mean_mode
impute_knn = bu.impute_knn
impute_mice = bu.impute_mice
k_bar = bu.k_bar
KNN_K_GRID = bu.KNN_K_GRID
MICE_IT_GRID = bu.MICE_IT_GRID
N_CV_MAX = bu.N_CV_MAX
_CV_SEED = bu._CV_SEED
_cv_rehide_mask = bu._cv_rehide_mask
_cv_score = bu._cv_score


def _cv_tune_audited(impute_fn, X_inc, var_types, grid, param_name, cv_rate=0.1):
    rng = np.random.default_rng(_CV_SEED)
    cv_mask = _cv_rehide_mask(X_inc, cv_rate, rng)
    audit = {"grid": [int(x) for x in grid], "cv_rate": float(cv_rate), "candidate_failures": []}
    if not cv_mask.any():
        chosen = grid[len(grid) // 2]
        audit.update({"chosen": int(chosen), "fallback": "empty_cv_mask"})
        return int(chosen), audit
    X_cv = X_inc.copy()
    X_cv[cv_mask] = np.nan
    best, best_err = None, np.inf
    for val in grid:
        try:
            X_imp = impute_fn(X_cv, var_types, **{param_name: val})
        except Exception as exc:
            audit["candidate_failures"].append({
                "value": int(val), "exception_type": type(exc).__name__, "message": str(exc)
            })
            continue
        err = _cv_score(X_inc, X_imp, cv_mask, var_types)
        if np.isfinite(err) and err < best_err:
            best_err, best = float(err), val
    if best is None:
        chosen = grid[len(grid) // 2]
        audit.update({"chosen": int(chosen), "fallback": "all_cv_candidates_failed"})
    else:
        chosen = best
        audit.update({"chosen": int(chosen), "best_cv_score": float(best_err), "fallback": None})
    return int(chosen), audit


def _impute_missforest_audited(X_inc, var_types, max_iter=10, n_trees=100, random_state=0):
    """missForest (Stekhoven and Bühlmann, 2012) with random forests from scikit-learn.

    Initial fill by mean or mode, columns in increasing order of missingness,
    at most max_iter iterations; stops when the relative change of the numeric
    imputations falls below 1e-4. A model-fit exception is recorded and raised.
    """
    X = X_inc.copy()
    n, p = X.shape
    cont_idx, cat_idx = get_type_indices(var_types)
    fallbacks = []
    for j in cont_idx:
        m = np.isnan(X[:, j])
        if m.any():
            X[m, j] = np.nanmean(X_inc[:, j])
    for j in cat_idx:
        m = np.isnan(X[:, j])
        if m.any():
            obs = X_inc[~m, j]
            if len(obs) > 0:
                uv, uc = np.unique(obs, return_counts=True)
                X[m, j] = uv[np.argmax(uc)]
    miss_count = np.isnan(X_inc).sum(axis=0)
    var_order = np.argsort(miss_count)
    X_prev = X.copy()
    iterations_run = 0
    for iteration in range(max_iter):
        iterations_run = iteration + 1
        for j in var_order:
            m = np.isnan(X_inc[:, j])
            if not m.any():
                continue
            obs_idx = ~m
            miss_idx = m
            oc = [c for c in range(p) if c != j]
            if obs_idx.sum() < 5:
                fallbacks.append({"column": int(j), "iteration": iteration, "reason": "observed_lt_5", "fallback": "initial_mean_or_mode"})
                continue
            try:
                if var_types[j] == "cont":
                    rf = bu.RandomForestRegressor(n_estimators=n_trees, random_state=random_state, n_jobs=1)
                    rf.fit(X[obs_idx][:, oc], X_inc[obs_idx, j])
                    X[miss_idx, j] = rf.predict(X[miss_idx][:, oc])
                else:
                    y_obs = X_inc[obs_idx, j].astype(int)
                    if len(np.unique(y_obs)) < 2:
                        fallbacks.append({"column": int(j), "iteration": iteration, "reason": "single_observed_class", "fallback": "initial_mode"})
                        continue
                    rf = bu.RandomForestClassifier(n_estimators=n_trees, random_state=random_state, n_jobs=1)
                    rf.fit(X[obs_idx][:, oc], y_obs)
                    X[miss_idx, j] = rf.predict(X[miss_idx][:, oc]).astype(float)
            except Exception as exc:
                err = RuntimeError(f"missForest fit failed at column={j}, iteration={iteration}: {type(exc).__name__}: {exc}")
                setattr(err, "audit_context", {
                    "column": int(j), "iteration": int(iteration), "exception_type": type(exc).__name__, "message": str(exc)
                })
                raise err from exc
        cont_mask = np.isnan(X_inc)
        if cont_idx:
            cont_miss = cont_mask[:, cont_idx]
            if cont_miss.any():
                rows_any = cont_miss.any(axis=1)
                diff_cont = np.sum((X[rows_any][:, cont_idx] - X_prev[rows_any][:, cont_idx]) ** 2)
                norm_cont = np.sum(X[:, cont_idx] ** 2) + 1e-10
                delta_cont = diff_cont / norm_cont
            else:
                delta_cont = 0
        else:
            delta_cont = 0
        if iteration > 0 and delta_cont < 1e-4:
            break
        X_prev = X.copy()
    return X, {
        "effective_config": {"max_iter": int(max_iter), "n_trees": int(n_trees), "random_state": int(random_state)},
        "fallbacks": fallbacks,
        "iterations_run": int(iterations_run),
    }


def _iscak_type_contract(var_types, type_policy=None):
    p = len(var_types)
    policy = dict(type_policy or {})
    auto_idx = {int(j) for j in policy.get("auto_detected_indices", [])}
    force_idx = {int(j) for j in policy.get("force_categorical_indices", [])}
    expected_cat = {j for j, t in enumerate(var_types) if t == "cat"}
    if not force_idx:
        force_idx = expected_cat - auto_idx
    if auto_idx & force_idx:
        raise ValueError("ISCA-k type policy cannot both force and auto-detect the same column")
    if (auto_idx | force_idx) != expected_cat:
        raise ValueError("ISCA-k categorical type policy must cover exactly the protocol categorical columns")
    if any(j < 0 or j >= p for j in auto_idx | force_idx):
        raise ValueError("ISCA-k type policy index out of bounds")
    return auto_idx, force_idx


def _validate_fitted_iscak_partition(imp, var_types, auto_idx, force_idx):
    cols = [f"v{j}" for j in range(len(var_types))]
    numeric = set(getattr(imp, "_numeric_set", set()))
    binary = set(getattr(imp, "_binary_set", set()))
    nominal = set(getattr(imp, "_nominal_set", set()))
    ordinal = set(getattr(imp, "_ordinal_set", set()))
    categorical = binary | nominal | ordinal
    rows = []
    mismatches = []
    for j, (c, expected) in enumerate(zip(cols, var_types)):
        if c in numeric:
            detected = "numeric"
        elif c in binary:
            detected = "binary"
        elif c in nominal:
            detected = "nominal"
        elif c in ordinal:
            detected = "ordinal"
        else:
            detected = "unknown"
        ok = (expected == "cont" and c in numeric) or (expected == "cat" and c in categorical)
        source = "auto_detected" if j in auto_idx else ("protocol" if j in force_idx else "auto_detected_numeric")
        row = {"index": j, "expected": expected, "detected": detected, "type_source": source, "match": bool(ok)}
        rows.append(row)
        if not ok:
            mismatches.append(row)
    if mismatches:
        raise RuntimeError(f"ISCA-k type partition does not match the dataset manifest: {mismatches}")
    return {"columns": rows, "exact_match": True}


def _impute_iscak(X_inc, var_types, *, type_policy=None):
    from isca_k import ISCAkConfig, ISCAkImputer
    p = X_inc.shape[1]
    cols = [f"v{j}" for j in range(p)]
    df = pd.DataFrame(np.asarray(X_inc, dtype=float), columns=cols)
    auto_idx, force_idx = _iscak_type_contract(var_types, type_policy)
    force_cat = [cols[j] for j in sorted(force_idx)] or None
    params = dict(
        adaptive_k=True, k_min=3, k_max=25,
        use_mi_weights=True, mi_n_neighbors=5, mi_random_state=42, use_propagation=True,
        use_distance_shrinkage=True, shrink_strength=0.5,
        standardize_numeric=True, homogeneity_threshold=2.5,
        stability_threshold=0.01, use_candidate_pruning=True,
        force_categorical=force_cat,
        interactive_type_detection=False, verbose=False,
    )
    imp = ISCAkImputer(ISCAkConfig(**params))
    out = imp.fit_transform(df)
    type_audit = _validate_fitted_iscak_partition(imp, var_types, auto_idx, force_idx)
    return out[cols].to_numpy(dtype=float), {
        "effective_config": params, "fallbacks": [], "type_validation": type_audit,
    }


def run_method_audited(method_name: str, X_inc, var_types, *, type_policy=None):
    if method_name == "Mean/Mode":
        return impute_mean_mode(X_inc, var_types), {"effective_config": {}, "fallbacks": [], "cv": None}
    if method_name == "k-NN":
        n = X_inc.shape[0]
        n_obs_min = int(np.min((~np.isnan(X_inc)).sum(axis=0)))
        if n <= N_CV_MAX:
            grid = tuple(k for k in KNN_K_GRID if k < max(2, n_obs_min)) or (min(5, n_obs_min - 1),)
            k, cv = _cv_tune_audited(impute_knn, X_inc, var_types, grid, "k")
        else:
            k, cv = k_bar(n), {"chosen": k_bar(n), "fallback": "large_n_k_bar", "candidate_failures": []}
        X_imp = impute_knn(X_inc, var_types, k=int(k))
        return X_imp, {"effective_config": {"k": int(k)}, "fallbacks": [], "cv": cv}
    if method_name == "MICE":
        it, cv = _cv_tune_audited(impute_mice, X_inc, var_types, MICE_IT_GRID, "max_iter")
        X_imp = impute_mice(X_inc, var_types, max_iter=int(it))
        return X_imp, {"effective_config": {"max_iter": int(it), "random_state": 0}, "fallbacks": [], "cv": cv}
    if method_name == "missForest":
        X_imp, audit = _impute_missforest_audited(X_inc, var_types, max_iter=10, n_trees=100, random_state=0)
        audit["cv"] = None
        return X_imp, audit
    if method_name == "SoftImpute":
        return run_softimpute_tuned(X_inc, var_types)
    if method_name == "ISCA-k":
        X_imp, audit = _impute_iscak(X_inc, var_types, type_policy=type_policy)
        audit["cv"] = None
        return X_imp, audit
    raise KeyError(f"unknown method: {method_name}")


def _nrmse_missforest_reference(X_true, X_imp, mask, var_types):
    """Reference pooled NRMSE matching missForest::nrmse semantics.

    Continuous entries that were artificially missing are pooled before the
    calculation. The numerator is the mean squared reconstruction error and the
    denominator is the sample variance of the corresponding true values
    (R ``var`` semantics, ddof=1). Reported for reference only; the study uses
    the per-variable metrics.
    """
    X_true = np.asarray(X_true, dtype=float)
    X_imp = np.asarray(X_imp, dtype=float)
    m = np.asarray(mask, dtype=bool).copy()
    cat_cols = [j for j, t in enumerate(var_types) if t == "cat"]
    if cat_cols:
        m[:, cat_cols] = False
    if int(m.sum()) < 2:
        return np.nan
    truth = X_true[m]
    imp = X_imp[m]
    denom = float(np.var(truth, ddof=1))
    if not np.isfinite(denom) or denom <= 0.0:
        return np.nan
    return float(np.sqrt(np.mean((truth - imp) ** 2) / denom))


def _metrics_for_imputation(X, X_imp, mask, var_types):
    st = var_stats(X)
    pv = per_var_metrics(X, X_imp, mask, st, var_types)
    result = {}
    for mk in ["nrmse_sd", "nrmse_range", "nrmse_iqr", "nrmse_mean", "rmse", "mae"]:
        result[f"{mk}_mean"] = aggregate_by_type(pv, mk, "mean")
        result[f"{mk}_wmean"] = aggregate_by_type(pv, mk, "weighted")
    result["pfc_mean"] = aggregate_by_type(pv, "pfc", "mean")
    result["pfc_wmean"] = aggregate_by_type(pv, "pfc", "weighted")
    result["rmse_full"] = full_matrix_rmse(X, X_imp, var_types)
    result["rmse_imponly"] = imputed_only_rmse(X, X_imp, mask, var_types)
    result["nrmse_missforest_reference"] = _nrmse_missforest_reference(X, X_imp, mask, var_types)
    result["miss_rate_actual"] = float(np.asarray(mask, dtype=bool).mean())
    # Per-variable components are kept so that other aggregations can be
    # computed later without repeating the imputation.
    result["per_var"] = pv
    return result


