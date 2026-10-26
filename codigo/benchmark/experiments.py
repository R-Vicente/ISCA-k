"""ISCA-k reference configuration and the variants of the complementary studies.

fit_iscak runs ISCA-k with the configuration of the main study, optionally
changed by one of the variants in ABLATION, CALIBRATION or the k grid (KGRID).
diagnostics computes the cell-level diagnostics of section 4.5 (k chosen,
stopping reasons, retrospective oracle); the true values are used only there,
after the imputation.
"""
from __future__ import annotations
from collections import Counter
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'benchmark')]
# One thread in the numerical libraries, for every method and every run.
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('PYTHONHASHSEED', '0')
import numpy as np
import pandas as pd
import bench_harness as h
import isca_k as isk

KGRID = (3, 5, 7, 10, 15, 20, 25)
ABLATION = {
    'sem_MI': {'use_mi_weights': False},
    'sem_k_adaptativo': {'adaptive_k': False, 'k': 10},
    'sem_propagacao': {'use_propagation': False},
    'sem_shrinkage': {'use_distance_shrinkage': False},
    'sem_poda': {'use_candidate_pruning': False},
}
CALIBRATION = {
    'salvaguarda_sd': {'safeguard_sd': True},
    'estabilidade_005': {'stability_threshold': .005},
    'estabilidade_020': {'stability_threshold': .02},
    'estabilidade_absoluta': {'stability_threshold': 0., 'stability_absolute': .01},
    'estabilidade_mista': {'stability_threshold': .005, 'stability_absolute': .005},
    'k_min_8': {'k_min': 8},
}


def validate_output(X_inc, X_imp, types):
    X_imp = np.asarray(X_imp, dtype=float).copy()
    if X_imp.shape != X_inc.shape or not np.isfinite(X_imp).all():
        raise ValueError('Imputacao incompleta ou nao finita.')
    observed = np.isfinite(X_inc)
    # Standardising and de-standardising in ISCA-k can change observed values by rounding error.
    if not np.allclose(X_imp[observed], X_inc[observed], rtol=1e-12, atol=1e-12):
        raise ValueError('O metodo alterou valores observados.')
    X_imp[observed] = X_inc[observed]
    for j, t in enumerate(types):
        if t != 'cont' and not np.isin(X_imp[:, j], np.unique(X_inc[observed[:, j], j])).all():
            raise ValueError(f'Categoria invalida na coluna {j}.')
    return X_imp

def fit_iscak(X_inc, types, type_policy=None, overrides=None, diagnostic=False):
    auto, forced = h._iscak_type_contract(types, type_policy)
    cols = [f'v{j}' for j in range(len(types))]
    params = dict(adaptive_k=True, k_min=3, k_max=25,
        use_mi_weights=True, mi_n_neighbors=5, mi_random_state=42,
        use_propagation=True, use_distance_shrinkage=True, shrink_strength=.5,
        standardize_numeric=True, homogeneity_threshold=2.5, stability_threshold=.01,
        use_candidate_pruning=True, candidate_k=25,
        force_categorical=[cols[j] for j in sorted(forced)] or None,
        interactive_type_detection=False, verbose=False, record_diagnostics=diagnostic)
    allowed = set(ABLATION['sem_MI']) | {'adaptive_k','k','use_propagation','use_distance_shrinkage',
        'use_candidate_pruning','safeguard_sd','stability_threshold','stability_absolute','k_min'}
    if set(overrides or {}) - allowed:
        raise ValueError('Override experimental nao autorizado.')
    params.update(overrides or {})
    imp = isk.ISCAkImputer(isk.ISCAkConfig(**params))
    out = imp.fit_transform(pd.DataFrame(X_inc, columns=cols))
    type_audit = h._validate_fitted_iscak_partition(imp, types, auto, forced)
    X_imp = validate_output(X_inc, out[cols].to_numpy(dtype=float), types)
    stats = imp.last_stats
    fallbacks = [{'reason': key, 'count': int(stats.get(key, 0))}
        for key in ('fallback_rows_all_missing','fallback_cells_no_neighbors','dropped_all_missing_cols')
        if stats.get(key, 0)]
    return X_imp, {'effective_config': params, 'type_validation': type_audit,
        'stats': stats, 'fallbacks': fallbacks}, imp

def entropy_weights(matrix):
    """Normalised entropy of the MI weights of each target; zero-weight predictors count, the target does not."""
    entropies = []
    zero_rows = 0
    if matrix is None:
        return {'mean_normalized_entropy': None, 'zero_weight_targets': 0}
    for j, row in enumerate(matrix):
        w = np.delete(np.asarray(row, float), j)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.)
        if len(w) <= 1:
            continue
        if w.sum() == 0:
            zero_rows += 1
            entropies.append(1.)  # ISCA-k then uses uniform weights
        else:
            p = w / w.sum()
            entropies.append(float(-np.sum(p[p > 0] * np.log(p[p > 0])) / np.log(len(w))))
    return {'mean_normalized_entropy': float(np.mean(entropies)) if entropies else None,
        'zero_weight_targets': zero_rows, 'target_entropies': entropies}

def diagnostics(imp, X, X_imp, mask, types):
    # The true values are used only here, after the imputation has finished.
    oracle = X_imp.copy()
    reasons = Counter(); direction = Counter(); seen = set(); numeric_cells = 0
    for trace in imp.cell_diagnostics:
        i, j = trace['row'], int(trace['column'][1:])
        if not mask[i, j] or (i, j) in seen:
            raise ValueError('Invalid or duplicate diagnostic cell.')
        seen.add((i, j))
        reasons[('cat_' if trace['categorical'] else 'cont_') + trace['reason']] += 1
        if 'predictions' not in trace:
            continue
        preds = np.asarray(trace['predictions'])
        ks = np.asarray(trace['ks'])
        selected = int(np.flatnonzero(ks == trace['k'])[0])
        if not np.isclose(preds[selected], X_imp[i, j], rtol=1e-10, atol=1e-10):
            raise ValueError('Diagnostic prediction differs from production output.')
        best = int(np.argmin((preds - X[i, j]) ** 2))
        oracle[i, j] = preds[best]
        direction['smaller_k' if ks[best] < trace['k'] else 'larger_k' if ks[best] > trace['k'] else 'same_k'] += 1
        numeric_cells += 1
    hist = Counter(imp._k_selections)
    return {'mi': entropy_weights(imp.mi_matrix), 'k_histogram': dict(hist),
        'k_mean': float(np.mean(imp._k_selections)) if imp._k_selections else None,
        'k_median': float(np.median(imp._k_selections)) if imp._k_selections else None,
        'stop_reasons': dict(reasons), 'traced_cells': len(seen),
        'missing_cells': int(mask.sum()), 'numeric_oracle_cells': numeric_cells,
        'oracle_direction': dict(direction),
        'cell_profile_sample': [imp.cell_diagnostics[i] for i in np.linspace(0, len(imp.cell_diagnostics)-1, min(32,len(imp.cell_diagnostics)), dtype=int)] if imp.cell_diagnostics else [],
        'oracle_kind': 'numeric only; fixed production donor sequence and propagation history; retrospective',
        'oracle_metrics': h._metrics_for_imputation(X, oracle, mask, types)}


