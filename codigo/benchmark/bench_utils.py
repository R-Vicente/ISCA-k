"""Shared benchmark utilities.

Dataset loaders used to build the snapshots in dados/snapshots, the type-aware
mean/mode, k-NN and MICE imputers, the internal validation helpers used to tune
them, and the per-variable error metrics (NRMSE and PFC).

Variable types are coded as 'cont' (numeric) and 'cat' (categorical); missing
cells are NaN.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_wine, load_iris, load_breast_cancer, load_diabetes,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: F401  (used by bench_harness)

warnings.filterwarnings('ignore')
# Worker processes start a fresh interpreter, which reads PYTHONWARNINGS at startup.
os.environ.setdefault("PYTHONWARNINGS", "ignore")


# ============================================================
# DATASET LOADERS
# ============================================================
def load_iris_ds():
    d = load_iris()
    types = ['cont'] * 4
    print(f"Iris: {d.data.shape[0]}x{d.data.shape[1]}, all continuous")
    return d.data, list(d.feature_names), types

def load_wine_ds():
    d = load_wine()
    types = ['cont'] * 13
    print(f"Wine: {d.data.shape[0]}x{d.data.shape[1]}, all continuous")
    return d.data, list(d.feature_names), types

def load_diabetes_ds():
    d = load_diabetes(scaled=False)
    names = list(d.feature_names)
    types = ['cat' if n == 'sex' else 'cont' for n in names]
    print(f"Diabetes: {d.data.shape[0]}x{d.data.shape[1]}, 9 cont + 1 cat (raw scale)")
    return d.data, names, types

def load_breastcancer_ds():
    d = load_breast_cancer()
    types = ['cont'] * 30
    print(f"Breast Cancer: {d.data.shape[0]}x{d.data.shape[1]}, all continuous")
    return d.data, list(d.feature_names), types

def load_heart_ds():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=45)
    df = ds.data.features.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    X = df.values.astype(float)
    names = list(df.columns)
    cat_vars = {'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'}
    types = ['cat' if n in cat_vars else 'cont' for n in names]
    print(f"Heart Disease: {X.shape[0]}x{X.shape[1]}, 5 cont + 8 cat")
    return X, names, types

def load_abalone_ds():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=1)
    df = ds.data.features.copy()
    df['Sex'] = df['Sex'].map({'M': 0, 'F': 1, 'I': 2})
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    X = df.values.astype(float)
    names = list(df.columns)
    types = ['cat' if n == 'Sex' else 'cont' for n in names]
    print(f"Abalone: {X.shape[0]}x{X.shape[1]}, 7 cont + 1 cat")
    return X, names, types

def load_california_housing_ds():
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing()
    X = d.data.astype(float)
    names = list(d.feature_names)
    types = ['cont'] * X.shape[1]
    print(f"California Housing: {X.shape[0]}x{X.shape[1]}, all continuous (large-n real-world)")
    return X, names, types

def load_penguins_ds():
    import seaborn as sns
    df = sns.load_dataset("penguins").dropna()
    X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].values.astype(float)
    names = ['bill_length', 'bill_depth', 'flipper_length', 'body_mass']
    types = ['cont'] * 4
    print(f"Penguins: {X.shape[0]}x{X.shape[1]}, all continuous (small, clean)")
    return X, names, types

def load_titanic_ds():
    import seaborn as sns
    df = sns.load_dataset("titanic")
    df = df[['age', 'fare', 'pclass', 'sibsp', 'parch', 'sex', 'embarked']].dropna()
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    X = df.values.astype(float)
    names = list(df.columns)
    types = ['cont', 'cont', 'cat', 'cont', 'cont', 'cat', 'cat']
    print(f"Titanic: {X.shape[0]}x{X.shape[1]}, mixed-type (classic)")
    return X, names, types

def load_auto_mpg_ds():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=9)
    df = ds.data.features.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    X = df.values.astype(float)
    names = list(df.columns)
    types = ['cont'] * X.shape[1]
    print(f"Auto MPG: {X.shape[0]}x{X.shape[1]}, continuous (small)")
    return X, names, types

def load_ionosphere_ds():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=52)
    X = ds.data.features.values.astype(float)
    names = list(ds.data.features.columns)
    types = ['cont'] * X.shape[1]
    print(f"Ionosphere: {X.shape[0]}x{X.shape[1]}, high-dim small-n continuous")
    return X, names, types

def load_heart_failure_ds():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=519)
    df = ds.data.features.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    X = df.values.astype(float)
    names = list(df.columns)
    cat_vars = {'anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking'}
    types = ['cat' if n in cat_vars else 'cont' for n in names]
    print(f"Heart Failure: {X.shape[0]}x{X.shape[1]}, mixed-type (clinical)")
    return X, names, types


DATASET_LOADERS = {
    'Iris':                load_iris_ds,
    'Wine':                load_wine_ds,
    'Diabetes':            load_diabetes_ds,
    'Breast Cancer':       load_breastcancer_ds,
    'Heart Disease':       load_heart_ds,
    'Abalone':             load_abalone_ds,
    'California Housing':  load_california_housing_ds,
    'Penguins':            load_penguins_ds,
    'Titanic':             load_titanic_ds,
    'Auto MPG':            load_auto_mpg_ds,
    'Ionosphere':          load_ionosphere_ds,
    'Heart Failure':       load_heart_failure_ds,
}

def load_available_datasets():
    loaded = {}
    for name, fn in DATASET_LOADERS.items():
        try:
            X, names, types = fn()
            loaded[name] = {'X': X, 'names': names, 'types': types}
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")
    return loaded

def get_type_indices(var_types):
    cont_idx = [j for j, t in enumerate(var_types) if t == 'cont']
    cat_idx = [j for j, t in enumerate(var_types) if t == 'cat']
    return cont_idx, cat_idx


# ============================================================
# TYPE-AWARE IMPUTATION METHODS
# ============================================================
def _get_unique_cats(X, cat_idx):
    cats = {}
    for j in cat_idx:
        vals = X[:, j]
        cats[j] = np.unique(vals[~np.isnan(vals)])
    return cats

def _round_to_categories(X_imp, cat_idx, cat_vals):
    X = X_imp.copy()
    for j in cat_idx:
        if j in cat_vals and len(cat_vals[j]) > 0:
            cats = cat_vals[j]
            col = X[:, j]
            diffs = np.abs(col[:, None] - cats[None, :])
            X[:, j] = cats[np.argmin(diffs, axis=1)]
    return X


def impute_mean_mode(X_inc, var_types):
    X = X_inc.copy()
    cont_idx, cat_idx = get_type_indices(var_types)
    for j in cont_idx:
        m = np.isnan(X[:, j])
        if m.any():
            X[m, j] = np.nanmean(X[:, j])
    for j in cat_idx:
        m = np.isnan(X[:, j])
        if m.any():
            obs = X[~m, j]
            if len(obs) > 0:
                vals, counts = np.unique(obs, return_counts=True)
                X[m, j] = vals[np.argmax(counts)]
    return X


def impute_knn(X_inc, var_types, k=5):
    """k-NN with mixed distance: normalised Euclidean (cont) + overlap (cat)."""
    X = X_inc.copy()
    n, p = X.shape
    cont_idx, cat_idx = get_type_indices(var_types)
    cont_idx = np.asarray(cont_idx)
    cat_idx = np.asarray(cat_idx)

    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0

    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    if np.isinf(X).any() or np.isnan(X).all(axis=0).any():
        raise ValueError("k-NN requires finite observed values in every column")
    fallback = mu.copy()
    for j in range(p):
        if var_types[j] != 'cont':
            observed = X_inc[np.isfinite(X_inc[:, j]), j]
            values, counts = np.unique(observed, return_counts=True)
            fallback[j] = values[np.argmax(counts)]
    is_obs = ~np.isnan(X)

    for j in range(p):
        miss = np.flatnonzero(~is_obs[:, j])
        if len(miss) == 0:
            continue
        obs = np.flatnonzero(is_obs[:, j])
        if len(obs) < k:
            if var_types[j] == 'cont':
                X[miss, j] = mu[j]
            else:
                vals_obs = X[obs, j]
                if len(vals_obs) > 0:
                    uv, uc = np.unique(vals_obs, return_counts=True)
                    X[miss, j] = uv[np.argmax(uc)]
                else:
                    X[miss, j] = mu[j]
            is_obs[miss, j] = True
            continue

        other_idx = np.delete(np.arange(p), j)
        for i in miss:
            usable = other_idx[is_obs[i, other_idx]]
            if len(usable) == 0:
                X[i, j] = fallback[j]
                continue
            donor_mask = np.all(is_obs[np.ix_(obs, usable)], axis=1)
            donors = obs[donor_mask]
            if len(donors) < k:
                X[i, j] = fallback[j]
                continue
            d = np.zeros(len(donors), dtype=float)
            usable_is_cont = np.isin(usable, cont_idx)
            usable_cont = usable[usable_is_cont]
            if len(usable_cont) > 0:
                vi = (X[i, usable_cont] - mu[usable_cont]) / sd[usable_cont]
                vd = (X[np.ix_(donors, usable_cont)] - mu[usable_cont]) / sd[usable_cont]
                d += np.sum((vd - vi)**2, axis=1)
            usable_cat = usable[~usable_is_cont]
            if len(usable_cat) > 0:
                ci = X[i, usable_cat]
                cd = X[np.ix_(donors, usable_cat)]
                d += np.sum(cd != ci, axis=1)
            nn = donors[np.argsort(d)[:k]]
            if var_types[j] == 'cont':
                X[i, j] = np.mean(X[nn, j])
            else:
                vals_nn = X[nn, j]
                uv, uc = np.unique(vals_nn, return_counts=True)
                X[i, j] = uv[np.argmax(uc)]
        is_obs[miss, j] = True
    if not np.isfinite(X).all():
        raise ValueError("k-NN produced non-finite values")
    observed = ~np.isnan(X_inc)
    if not np.array_equal(X[observed], X_inc[observed]):
        raise ValueError("k-NN changed observed cells")
    for j in range(p):
        if var_types[j] != 'cont' and not np.isin(X[:, j], np.unique(X_inc[observed[:, j], j])).all():
            raise ValueError(f"k-NN invalid category in column {j}")
    return X


def impute_mice(X_inc, var_types, max_iter=10, random_state=0):
    cat_idx = get_type_indices(var_types)[1]
    cat_vals = _get_unique_cats(X_inc, cat_idx) if cat_idx else {}
    imp = IterativeImputer(max_iter=max_iter, random_state=random_state,
                           sample_posterior=False)
    X_imp = imp.fit_transform(X_inc)
    if cat_idx:
        X_imp = _round_to_categories(X_imp, cat_idx, cat_vals)
    return X_imp


# ── Parameter selection on held-out observed cells ──
# About 10% of the observed cells of each column are hidden, imputed with each
# candidate value and scored against their known values; the best candidate is
# kept. The cells hidden by the experimental mask are never used.
N_CV_MAX = 5000          # k-NN is tuned only up to this many rows (the search is O(n^2 p));
                         # larger tables use k = k_bar(n).
_CV_SEED = 20240601      # fixed seed: the held-out cells depend only on X_inc
KNN_K_GRID = (1, 3, 5, 9, 15, 20)
MICE_IT_GRID = (5, 10, 20)


def _cv_rehide_mask(X_inc, cv_rate, rng):
    is_obs = ~np.isnan(X_inc)
    cv_mask = np.zeros(X_inc.shape, dtype=bool)
    for j in range(X_inc.shape[1]):
        obs = np.flatnonzero(is_obs[:, j])
        if obs.size < 2:
            continue
        n_hide = min(max(1, int(cv_rate * obs.size)), obs.size - 1)
        cv_mask[rng.choice(obs, size=n_hide, replace=False), j] = True
    return cv_mask


def _cv_score(X_obs_true, X_imp, cv_mask, var_types):
    """Mean over columns of: standardized RMSE (cont) / PFC (cat) on re-hidden cells."""
    errs = []
    for j in range(X_imp.shape[1]):
        mj = cv_mask[:, j]
        if not mj.any():
            continue
        t, pr = X_obs_true[mj, j], X_imp[mj, j]
        if var_types[j] == 'cont':
            col = X_obs_true[~np.isnan(X_obs_true[:, j]), j]
            sd = np.std(col, ddof=0)
            rmse = np.sqrt(np.mean((t - pr) ** 2))
            errs.append(rmse / sd if sd > 0 else rmse)
        else:
            errs.append(np.mean(t != pr))
    return float(np.mean(errs)) if errs else np.inf


# Fixed k for k-NN when the table is too large to tune: sqrt(n), bounded to [5, 20].
def k_bar(n):
    return int(np.clip(round(np.sqrt(n)), 5, 20))


# ============================================================
# TYPE-AWARE METRICS
# ============================================================
def var_stats(X):
    X = np.asarray(X, dtype=float)
    if X.size == 0 or X.shape[1] == 0:
        p = X.shape[1] if X.ndim > 1 else 0
        return {'sd': np.zeros(p), 'range': np.ones(p), 'iqr': np.ones(p), 'mean': np.ones(p)}
    sd = np.nanstd(X, axis=0, ddof=1)
    sd = np.where((sd == 0) | np.isnan(sd), 1.0, sd)
    col_max = np.nanmax(X, axis=0)
    col_min = np.nanmin(X, axis=0)
    range_vals = np.where(np.isnan(col_max - col_min), 1.0, col_max - col_min)
    q75 = np.nanpercentile(X, 75, axis=0)
    q25 = np.nanpercentile(X, 25, axis=0)
    iqr_vals = np.where(np.isnan(q75 - q25), 1.0, q75 - q25)
    return {'sd': sd, 'range': range_vals, 'iqr': iqr_vals,
            'mean': np.abs(np.nanmean(X, axis=0))}

def per_var_metrics(X_true, X_imp, mask, st, var_types):
    """Per-variable reconstruction error.

    Each NRMSE denominator is a spread statistic of the true values at the
    hidden cells of that column, as in Stekhoven and Bühlmann (2012), not of the
    whole column. The SD uses ddof=0, so NRMSE^sd = sqrt(mean(err^2) / var(true_hidden)).
    The `st` argument is not used."""
    p = X_true.shape[1]
    res = {}
    for j in range(p):
        mj = mask[:, j]
        if not mj.any():
            continue
        n_miss = int(mj.sum())
        if var_types[j] == 'cont':
            true_miss = X_true[mj, j]
            err = true_miss - X_imp[mj, j]
            rmse = np.sqrt(np.mean(err**2))
            mae = np.mean(np.abs(err))
            sd_d = np.std(true_miss, ddof=0)
            range_d = np.ptp(true_miss)
            iqr_d = np.percentile(true_miss, 75) - np.percentile(true_miss, 25)
            mean_d = abs(np.mean(true_miss))
            res[j] = {
                'type': 'cont', 'rmse': rmse, 'mae': mae,
                'nrmse_sd':    rmse / sd_d    if sd_d > 0    else np.nan,
                'nrmse_range': rmse / range_d if range_d > 0 else np.nan,
                'nrmse_iqr':   rmse / iqr_d   if iqr_d > 0   else np.nan,
                'nrmse_mean':  rmse / mean_d  if mean_d > 0  else np.nan,
                'pfc': np.nan, 'n_miss': n_miss,
            }
        else:
            pfc = np.mean(X_true[mj, j] != X_imp[mj, j])
            res[j] = {
                'type': 'cat', 'rmse': np.nan, 'mae': np.nan,
                'nrmse_sd': np.nan, 'nrmse_range': np.nan,
                'nrmse_iqr': np.nan, 'nrmse_mean': np.nan,
                'pfc': pfc, 'n_miss': n_miss,
            }
    return res

def aggregate_by_type(per_var, metric_key, strategy='mean'):
    vals = [per_var[j][metric_key] for j in per_var
            if not np.isnan(per_var[j][metric_key])]
    weights = [per_var[j]['n_miss'] for j in per_var
               if not np.isnan(per_var[j][metric_key])]
    if not vals:
        return np.nan
    if strategy == 'mean':
        return np.mean(vals)
    elif strategy == 'weighted':
        return np.average(vals, weights=weights)
    raise ValueError(strategy)

def full_matrix_rmse(X_true, X_imp, var_types):
    cont = [j for j, t in enumerate(var_types) if t == 'cont']
    if not cont:
        return np.nan
    return np.sqrt(np.mean((X_true[:, cont] - X_imp[:, cont])**2))

def imputed_only_rmse(X_true, X_imp, mask, var_types):
    cont = [j for j, t in enumerate(var_types) if t == 'cont']
    if not cont:
        return np.nan
    cont_mask = mask.copy()
    cat_cols = [j for j, t in enumerate(var_types) if t == 'cat']
    cont_mask[:, cat_cols] = False
    if not cont_mask.any():
        return np.nan
    return np.sqrt(np.mean((X_true[cont_mask] - X_imp[cont_mask])**2))


