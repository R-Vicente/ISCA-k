"""ISCA-k: Instance-based Similarity and Contextual Adaptation.

Nearest-neighbour imputation for mixed data:
- distances per variable type (generalised Gower with robust scales), weighted
  by the mutual information between each predictor and the target column, with
  a penalty for low overlap of observed variables;
- columns imputed in increasing order of missingness (ties broken by mean MI),
  rows within a column in increasing order of missingness, and imputed values
  propagated to the following columns;
- adaptive number of neighbours per cell (_select_k_nw_homogeneity);
- optional two-stage candidate pruning for large tables.

Variable types are detected and encoded by MixedDataHandler (type_detection.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from type_detection import MixedDataHandler

# Cache of MI matrices, keyed by the content and configuration of the encoded table,
# so the matrix is not recomputed when the same table is fitted more than once.
import hashlib as _hashlib

_MI_CACHE: dict = {}
_MI_CACHE_MAX_SIZE: int = 8   # maximum number of cached matrices

def _df_cache_key(
    df: pd.DataFrame,
    *,
    numeric_cols: List[str] | None = None,
    ordinal_cols: List[str] | None = None,
    nominal_cols: List[str] | None = None,
    binary_cols: List[str] | None = None,
    mi_n_neighbors: int | None = None,
    mi_random_state: int | None = None,
) -> str:
    """Key for the MI cache.

    SHA-256 of the complete encoded table (including the index), the column
    names and dtypes, the type partition and the MI parameters, so a cached
    matrix is reused only for the same data and configuration.
    """
    h = _hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    h.update(repr(tuple(df.columns)).encode("utf-8"))
    h.update(repr(tuple(str(dt) for dt in df.dtypes)).encode("utf-8"))
    h.update(repr(tuple(numeric_cols or ())).encode("utf-8"))
    h.update(repr(tuple(ordinal_cols or ())).encode("utf-8"))
    h.update(repr(tuple(nominal_cols or ())).encode("utf-8"))
    h.update(repr(tuple(binary_cols or ())).encode("utf-8"))
    h.update(repr((mi_n_neighbors, mi_random_state)).encode("utf-8"))
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _safe_mode(s: pd.Series):
    s2 = s.dropna()
    if s2.empty:
        return np.nan
    m = s2.mode(dropna=True)
    return m.iloc[0] if len(m) else s2.iloc[0]


def _safe_mode_array(arr: np.ndarray) -> Any:
    if arr.dtype == object:
        valid = arr[pd.notna(arr)]
    else:
        valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return np.nan
    unique, counts = np.unique(valid, return_counts=True)
    return unique[np.argmax(counts)]

def _select_k_nw_homogeneity(
    ordered_values: np.ndarray,
    ordered_distances: np.ndarray,
    k_min: int,
    k_max: int,
    homogeneity_threshold: float,
    stability_threshold: float,
    is_categorical: bool = False,
    col_scale: float = None,
    eps: float = 1e-12,
    stability_absolute: float = 0.0,
    safeguard_sd: bool = False,
    diagnostic: Optional[dict] = None,
) -> int:
    """Adaptive choice of k: stabilisation of the Nadaraya-Watson estimate, with a safeguard.

    Numeric and ordinal targets:
      - The window grows from k_min to k_max one neighbour at a time, with the
        current NW estimate (weights 1/d; exact matches, d = 0, take all the weight).
      - Main rule (stabilisation): stop when admitting the next neighbour changes
        the estimate by less than stability_threshold in relative terms,
        |y_new - y_prev| / (|y_prev| + eps) < stability_threshold.
        Because the imputer works on standardised values, |y_prev| is the
        distance of the estimate to the column mean, so the tolerance is
        proportional to that distance: the window grows longer near the mean
        and closes earlier in the tails (assumes standardize_numeric=True).
        With stability_absolute > 0 the rule becomes
        |y_new - y_prev| < stability_absolute + stability_threshold * (|y_prev| + eps).
      - Safeguard (outliers): the next neighbour is admitted only if
        |v - y_hat| / (col_scale + eps) <= homogeneity_threshold, where col_scale
        is the robust scale (p95 - p05)/sigma of the target column (or 1, one SD,
        with safeguard_sd=True). With the default value (2.5) the effective limit
        is about 6 to 9 standard deviations: it stops gross contamination and
        rarely acts on regular data.

    Categorical and binary targets:
      - Tracks the purity of the window (share of the modal value).
      - Stops when adding the next neighbour lowers purity by more than
        homogeneity_threshold * (1 - current purity), checked only when the
        current purity exceeds 0.5.

    Rationale: a growing window stopped by stabilisation, in the spirit of
    bandwidth selection in local regression (Fan & Gijbels, 1996), with the
    Nadaraya-Watson estimator at its core.

    Parameters
    ----------
    ordered_values    : neighbour values sorted by increasing distance (already filtered)
    ordered_distances : corresponding distances (finite, >= 0)
    k_min             : minimum number of neighbours
    k_max             : maximum number of neighbours
    homogeneity_threshold : safeguard limit in units of col_scale (numeric)
                            or allowed purity drop (categorical) (default 2.5)
    stability_threshold   : relative tolerance of the stabilisation rule (default 0.01)
    is_categorical    : True for nominal or binary targets
    eps               : small constant to avoid division by zero
    diagnostic        : optional dict that receives the chosen k and the stopping reason

    Returns
    -------
    effective k (int, in [k_min, k_max])
    """
    def finish(k, reason):
        if diagnostic is not None:
            diagnostic.update(k=int(k), reason=reason, categorical=bool(is_categorical))
        return int(k)

    n_available = len(ordered_values)
    k_max_eff = min(k_max, n_available)

    if k_max_eff <= k_min:
        return finish(np.clip(k_max_eff, 1, k_max), "few_donors")

    # ── Categorical target ──
    # Category counts are updated incrementally as the window grows. The initial
    # purity is divided by k_min, including missing donor values.
    if is_categorical:
        notna_mask = np.array([pd.notna(v) for v in ordered_values[:k_max_eff]], dtype=bool)
        init_valid = [ordered_values[i] for i in range(k_min) if notna_mask[i]]
        if len(init_valid) == 0:
            return finish(k_min, "no_valid_prefix")
        unique, counts = np.unique(init_valid, return_counts=True)
        best_k = k_min
        reason = "upper_limit"
        current_purity = float(counts.max()) / k_min   # denominator k_min, missing values included

        for k in range(k_min + 1, k_max_eff + 1):
            if not notna_mask[k - 1]:
                continue
            new_val = ordered_values[k - 1]
            # Update the counts with the new neighbour
            idx = np.searchsorted(unique, new_val)
            if idx < len(unique) and unique[idx] == new_val:
                counts = counts.copy()
                counts[idx] += 1
            else:
                unique = np.insert(unique, idx, new_val)
                counts = np.insert(counts, idx, 1)
            n_valid_k = int(notna_mask[:k].sum())
            new_purity = float(counts.max()) / n_valid_k

            purity_drop = current_purity - new_purity
            allowed_drop = homogeneity_threshold * (1.0 - current_purity) + eps
            if purity_drop > allowed_drop and current_purity > 0.5:
                reason = "purity"
                break

            best_k = k
            current_purity = new_purity

        return finish(np.clip(best_k, k_min, k_max), reason)

    # ── Numeric or ordinal target ──
    vals = np.asarray(ordered_values, dtype=float)
    dists = np.asarray(ordered_distances, dtype=float)

    # Donors with a finite value and a finite distance.
    valid = np.isfinite(vals) & np.isfinite(dists)

    def _nw_estimate(k: int) -> float:
        vk = vals[:k][valid[:k]]
        dk = dists[:k][valid[:k]]
        if len(vk) == 0:
            return float(np.nanmean(vals[:k]))
        exact = dk == 0.0
        if exact.any():
            return float(np.mean(vk[exact]))
        w = 1.0 / (dk + eps)
        return float(np.dot(vk, w) / w.sum())   # same as np.average(vk, weights=w)

    y_hat = _nw_estimate(k_min)

    if col_scale is not None and np.isfinite(col_scale) and col_scale > eps:
        scale = col_scale
    else:
        all_vals = vals[np.isfinite(vals)]
        if len(all_vals) > 3:
            q75, q25 = np.percentile(all_vals, [75, 25])
            scale = (q75 - q25) / 1.349
        else:
            scale = float(np.std(all_vals)) if len(all_vals) > 1 else 1.0
        if not np.isfinite(scale) or scale < eps:
            scale = float(np.abs(y_hat)) if abs(y_hat) > eps else 1.0

    if safeguard_sd:
        scale = 1.0
    reason = "upper_limit"
    best_k = k_min
    prev_estimate = y_hat

    for k in range(k_min + 1, k_max_eff + 1):
        j = k - 1
        if not valid[j]:           # skip donors without a finite value or distance
            continue

        deviation = abs(vals[j] - y_hat) / (scale + eps)
        if deviation > homogeneity_threshold:
            reason = "safeguard"
            break

        new_estimate = _nw_estimate(k)

        rel_change = abs(new_estimate - prev_estimate) / (abs(prev_estimate) + eps)
        stable = (rel_change < stability_threshold) if stability_absolute == 0.0 else (abs(new_estimate - prev_estimate) < stability_absolute + stability_threshold * (abs(prev_estimate) + eps))
        if stable and k > k_min:
            best_k = k
            reason = "stability"
            break

        best_k = k
        prev_estimate = new_estimate
        y_hat = new_estimate

    return finish(np.clip(best_k, k_min, k_max), reason)

def _fit_numeric_stats(df, cols):
    means, stds, ranges = {}, {}, {}
    for c in cols:
        x = df[c].astype(float)
        m = float(x.mean(skipna=True))
        s = float(x.std(skipna=True, ddof=0))
        if not np.isfinite(s) or s == 0.0:
            s = 1.0
        vmin = float(x.min(skipna=True))
        vmax = float(x.max(skipna=True))
        r = vmax - vmin
        if not np.isfinite(r) or r == 0.0:
            r = 1.0
        means[c] = m; stds[c] = s; ranges[c] = r
    return means, stds, ranges


def _fit_robust_scales(
    df_enc: pd.DataFrame,
    cols: List[str],
    stds: Optional[Dict[str, float]] = None,
    eps: float = 1e-12,
) -> Tuple[Dict[str, float], set]:
    """Compute robust per-variable scales using (p95 - p05) on observed values.

    If `stds` is provided, scales are converted to the standardized space by dividing by std.
    Returns (scales_dict, constant_cols_set) where constant_cols have ~zero scale.
    """
    scales: Dict[str, float] = {}
    constant_cols: set = set()
    for c in cols:
        x = pd.to_numeric(df_enc[c], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            scales[c] = 0.0
            constant_cols.add(c)
            continue
        p05 = float(np.nanpercentile(x, 5))
        p95 = float(np.nanpercentile(x, 95))
        sc = p95 - p05
        if stds is not None:
            sd = float(stds.get(c, 1.0) or 1.0)
            sc = sc / sd
        if not np.isfinite(sc) or sc <= eps:
            scales[c] = 0.0
            constant_cols.add(c)
        else:
            scales[c] = float(sc)
    return scales, constant_cols


def _weighted_mean(values: np.ndarray, distances: np.ndarray) -> float:
    """Weighted mean using inverse-distance weights. d=0 → full weight on that neighbor."""
    values = np.asarray(values, dtype=float)
    distances = np.asarray(distances, dtype=float)
    valid = np.isfinite(values) & np.isfinite(distances)
    if not valid.any():
        return float(np.nanmean(values))
    v = values[valid]
    d = distances[valid]
    # If any distance is exactly 0, return mean of those exact matches
    exact = d == 0.0
    if exact.any():
        return float(np.mean(v[exact]))
    w = 1.0 / d
    return float(np.average(v, weights=w))


def _weighted_mode(values: np.ndarray, distances: np.ndarray):
    """Weighted vote using inverse-distance weights."""
    acc: Dict[Any, float] = {}
    for v, d in zip(values, distances):
        if pd.isna(v):
            continue
        if not np.isfinite(d):
            continue
        w = 1.0 / d if d > 0 else 1e12  # near-infinite weight for exact match
        acc[v] = acc.get(v, 0.0) + w
    if not acc:
        return _safe_mode_array(np.asarray(values, dtype=object))
    return max(acc.items(), key=lambda kv: kv[1])[0]


def _standardize_numeric(df, cols, means, stds):
    out = df.copy()
    for c in cols:
        out[c] = (out[c].astype(float) - means[c]) / stds[c]
    return out


def _destandardize_numeric(df, cols, means, stds):
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype(float) * stds[c] + means[c]
    return out


# ─────────────────────────────────────────────────────────────
# Mutual information
# ─────────────────────────────────────────────────────────────
def _compute_mi_matrix(
    df: pd.DataFrame,
    numeric_cols: list, ordinal_cols: list,
    nominal_cols: list, binary_cols: list,
    n_neighbors: int = 5, random_state: int = 42,
) -> Tuple[np.ndarray, list, int, int]:
    cols = list(df.columns)
    p = len(cols)
    mi_matrix = np.zeros((p, p), dtype=np.float64)
    continuous_set = set(numeric_cols) | set(ordinal_cols)

    pairs_total = p * (p - 1) // 2
    pairs_valid = 0

    for i in range(p):
        for j in range(i + 1, p):
            col_i, col_j = cols[i], cols[j]
            mask = df[[col_i, col_j]].notna().all(axis=1)
            n_valid = int(mask.sum())
            if n_valid < 10:
                continue

            pairs_valid += 1

            x = df.loc[mask, col_i].to_numpy()
            y = df.loc[mask, col_j].to_numpy()
            i_cont = col_i in continuous_set
            j_cont = col_j in continuous_set
            k_nn = min(n_neighbors, n_valid - 1)

            if i_cont and j_cont:
                mi_val = mutual_info_regression(
                    x.reshape(-1, 1).astype(np.float64),
                    y.astype(np.float64),
                    n_neighbors=k_nn, random_state=random_state
                )[0]
            elif i_cont and not j_cont:
                mi_val = mutual_info_classif(
                    x.reshape(-1, 1).astype(np.float64),
                    y.astype(str),
                    discrete_features=False, n_neighbors=k_nn, random_state=random_state
                )[0]
            elif (not i_cont) and j_cont:
                mi_val = mutual_info_regression(
                    x.reshape(-1, 1),
                    y.astype(np.float64),
                    discrete_features=[True], n_neighbors=k_nn, random_state=random_state
                )[0]
            else:
                mi_val = mutual_info_classif(
                    x.reshape(-1, 1),
                    y.astype(str),
                    discrete_features=[True], n_neighbors=k_nn, random_state=random_state
                )[0]

            mi_matrix[i, j] = max(float(mi_val), 0.0)
            mi_matrix[j, i] = mi_matrix[i, j]

    return mi_matrix, cols, pairs_valid, pairs_total


# ─────────────────────────────────────────────────────────────
# Distance computation (MI-weighted, single target column)
# ─────────────────────────────────────────────────────────────
def _compute_distances_for_rows(
    data_numeric: np.ndarray,
    data_ordinal: np.ndarray,
    data_categorical: np.ndarray,
    mask_numeric: np.ndarray,
    mask_ordinal: np.ndarray,
    mask_categorical: np.ndarray,
    num_scales: np.ndarray,
    ord_scales: np.ndarray,
    target_rows: np.ndarray,
    w_numeric: np.ndarray,
    w_ordinal: np.ndarray,
    w_categorical: np.ndarray,
    use_distance_shrinkage: bool,
    shrink_strength: float,
) -> np.ndarray:
    """Compute distances from each target row to all rows.

    Robust generalized-Gower per-variable:
      - num/ord: diff = min(|a-b|/scale, 1)
      - cat/bin: diff = 0/1
    Then aggregate **per type**:
      d_type = sum(diff * w) / sum(w)  (only over common observed)
      D = mean(d_type over types present)

    This prevents a single type (e.g. many categoricals) from dominating the total distance.
    """
    target_rows = np.asarray(target_rows, dtype=int)
    n_rows = data_numeric.shape[0] if data_numeric.size else (
        data_ordinal.shape[0] if data_ordinal.size else data_categorical.shape[0]
    )
    dist_matrix = np.full((len(target_rows), n_rows), np.inf, dtype=float)

    # Effective weights: constant features (scale=0) contribute nothing
    if w_numeric.size and num_scales.size:
        w_num_eff = w_numeric * (num_scales > 0).astype(float)
        total_w_num = float(np.sum(w_num_eff))
    else:
        w_num_eff = w_numeric
        total_w_num = float(np.sum(w_numeric)) if w_numeric.size else 0.0

    if w_ordinal.size and ord_scales.size:
        w_ord_eff = w_ordinal * (ord_scales > 0).astype(float)
        total_w_ord = float(np.sum(w_ord_eff))
    else:
        w_ord_eff = w_ordinal
        total_w_ord = float(np.sum(w_ordinal)) if w_ordinal.size else 0.0

    total_w_cat = float(np.sum(w_categorical)) if w_categorical.size else 0.0

    total_w_all = total_w_num + total_w_ord + total_w_cat
    if total_w_all <= 0:
        return dist_matrix

    # Scale the data once per call; the loop below then handles one target row at a time.
    if data_numeric.size:
        sc_num = np.where(num_scales > 0, num_scales, 1.0)
        data_num_sc = np.ascontiguousarray(data_numeric / sc_num)
        # Missing entries become 0 so that NaN*0 does not reach the sums; they are
        # excluded anyway through the common-observed mask.
        np.nan_to_num(data_num_sc, nan=0.0, copy=False)
        # Work buffers reused for every target row.
        _diff_buf_n = np.empty_like(data_num_sc)
        _cw_buf_n = np.empty_like(data_num_sc)
        _common_buf_n = np.empty(data_num_sc.shape, dtype=bool)
    if data_ordinal.size:
        sc_ord = np.where(ord_scales > 0, ord_scales, 1.0)
        data_ord_sc = np.ascontiguousarray(data_ordinal / sc_ord)
        np.nan_to_num(data_ord_sc, nan=0.0, copy=False)

    for ti, r in enumerate(target_rows):
        # --- numeric ---
        if data_numeric.size:
            np.subtract(data_num_sc, data_num_sc[r], out=_diff_buf_n)
            np.abs(_diff_buf_n, out=_diff_buf_n)
            np.clip(_diff_buf_n, 0.0, 1.0, out=_diff_buf_n)
            np.logical_and(mask_numeric[r], mask_numeric, out=_common_buf_n)
            np.multiply(_common_buf_n, w_num_eff, out=_cw_buf_n)  # reused for the differences and the weight sums
            np.multiply(_diff_buf_n, _cw_buf_n, out=_diff_buf_n)  # zero where the pair is not observed in both rows
            num_sum  = _diff_buf_n.sum(axis=1)
            num_wsum = _cw_buf_n.sum(axis=1)
            d_num = np.full(n_rows, np.nan, dtype=float)
            np.divide(num_sum, num_wsum, out=d_num, where=num_wsum > 0)
        else:
            num_wsum = np.zeros(n_rows, dtype=float)
            d_num = np.full(n_rows, np.nan, dtype=float)

        # --- ordinal ---
        if data_ordinal.size:
            diffs = np.abs(data_ord_sc - data_ord_sc[r])
            np.clip(diffs, 0.0, 1.0, out=diffs)
            common_o = mask_ordinal[r] & mask_ordinal
            cw_o = common_o * w_ord_eff
            diffs *= cw_o
            ord_sum  = diffs.sum(axis=1)
            ord_wsum = cw_o.sum(axis=1)
            d_ord = np.full(n_rows, np.nan, dtype=float)
            np.divide(ord_sum, ord_wsum, out=d_ord, where=ord_wsum > 0)
        else:
            ord_wsum = np.zeros(n_rows, dtype=float)
            d_ord = np.full(n_rows, np.nan, dtype=float)

        # --- categorical/binary ---
        if data_categorical.size:
            common_c = mask_categorical[r] & mask_categorical   # (n_rows, n_cat)
            diffs_c  = (data_categorical != data_categorical[r]).astype(float)
            cw_c = common_c * w_categorical
            diffs_c *= cw_c
            cat_sum  = diffs_c.sum(axis=1)
            cat_wsum = cw_c.sum(axis=1)
            d_cat = np.full(n_rows, np.nan, dtype=float)
            np.divide(cat_sum, cat_wsum, out=d_cat, where=cat_wsum > 0)
        else:
            cat_wsum = np.zeros(n_rows, dtype=float)
            d_cat = np.full(n_rows, np.nan, dtype=float)

        # Combine types: each present type has equal vote
        types_present = (num_wsum > 0).astype(float) + (ord_wsum > 0).astype(float) + (cat_wsum > 0).astype(float)
        base = np.full(n_rows, np.inf, dtype=float)
        num_part = np.where(num_wsum > 0, d_num, 0.0)
        ord_part = np.where(ord_wsum > 0, d_ord, 0.0)
        cat_part = np.where(cat_wsum > 0, d_cat, 0.0)
        np.divide(num_part + ord_part + cat_part, types_present, out=base, where=types_present > 0)

        # Optional shrinkage: penalize low overlap (little shared evidence)
        if use_distance_shrinkage:
            overlap_w = num_wsum + ord_wsum + cat_wsum
            overlap = np.clip(overlap_w / total_w_all, 0.0, 1.0)
            nonself_finite = np.isfinite(base)
            nonself_finite[r] = False
            finite_base = base[nonself_finite]
            if finite_base.size > 0:
                typical = float(np.median(finite_base))
            else:
                typical = 1.0
            base = base + float(shrink_strength) * (1.0 - overlap) * typical

        base[r] = np.inf  # self-distance excluded
        dist_matrix[ti] = base

    return dist_matrix

def _extract_typed_arrays(
    df_work: pd.DataFrame,
    num_cols: List[str],
    ord_cols: List[str],
    cat_cols: List[str],
    num_scales_dict: Dict[str, float],
    ord_scales_dict: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract numpy arrays and masks by type from the working DataFrame."""
    n_rows = len(df_work)

    if num_cols:
        data_num = df_work[num_cols].to_numpy(dtype=float)
        mask_num = np.isfinite(data_num)
        num_scales_arr = np.array([float(num_scales_dict.get(c, 0.0)) for c in num_cols], dtype=float)
    else:
        data_num = np.empty((n_rows, 0), dtype=float)
        mask_num = np.empty((n_rows, 0), dtype=bool)
        num_scales_arr = np.empty((0,), dtype=float)

    if ord_cols:
        data_ord = df_work[ord_cols].to_numpy(dtype=float)
        mask_ord = np.isfinite(data_ord)
        ord_scales_arr = np.array([float(ord_scales_dict.get(c, 0.0)) for c in ord_cols], dtype=float)
    else:
        data_ord = np.empty((n_rows, 0), dtype=float)
        mask_ord = np.empty((n_rows, 0), dtype=bool)
        ord_scales_arr = np.empty((0,), dtype=float)

    if cat_cols:
        data_cat = df_work[cat_cols].to_numpy(dtype=object)
        mask_cat = pd.notna(df_work[cat_cols]).to_numpy(dtype=bool)
    else:
        data_cat = np.empty((n_rows, 0), dtype=object)
        mask_cat = np.empty((n_rows, 0), dtype=bool)

    return data_num, mask_num, num_scales_arr, data_ord, mask_ord, ord_scales_arr, data_cat, mask_cat



# ─────────────────────────────────────────────────────────────
# Config & Imputer
# ─────────────────────────────────────────────────────────────
@dataclass
class ISCAkConfig:
    k: int = 5
    standardize_numeric: bool = True
    interactive_type_detection: bool = False
    categorical_threshold: int = 10
    force_categorical: Optional[list] = None
    force_ordinal: Optional[dict] = None
    verbose: bool = False
    type_verbose: bool = False
    # Mutual information weights
    use_mi_weights: bool = True
    mi_n_neighbors: int = 5
    mi_random_state: int = 42
    # Propagation of imputed values to later columns
    use_propagation: bool = True
    # Adaptive k (see _select_k_nw_homogeneity)
    adaptive_k: bool = True
    k_min: int = 3
    k_max: int = 25
    # homogeneity_threshold: outlier safeguard, in units of the robust scale of the
    #   target column (numeric), or allowed purity drop (categorical). Deliberately
    #   conservative for numeric targets (2.5 is about 6 to 9 SD).
    homogeneity_threshold: float = 2.5
    # stability_threshold: relative tolerance of the main stopping rule (stabilisation
    #   of the NW estimate); proportional to the distance of the estimate to the
    #   column mean in standardised space, so it assumes standardize_numeric=True.
    stability_threshold: float = 0.01
    # Low-overlap penalty on the distances
    use_distance_shrinkage: bool = True
    shrink_strength: float = 0.5  # >=0, higher = stronger penalty for low overlap
    # Two-stage candidate pruning
    use_candidate_pruning: bool = True
    # Settings used by the complementary studies; the defaults give the standard method.
    candidate_k: Optional[int] = None
    stability_absolute: float = 0.0
    safeguard_sd: bool = False
    record_diagnostics: bool = False



# ─────────────────────────────────────────────────────────────
# Two-stage candidate pruning helpers
# ─────────────────────────────────────────────────────────────

def _compute_global_mi_weights(
    mi_matrix: np.ndarray,
    mi_col_order: list,
    num_cols: list,
    ord_cols: list,
    cat_cols: list,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute global MI centrality weights: w_global[f] = mean_t MI(f,t).

    Returns weight arrays (w_num, w_ord, w_cat) aligned with the column lists,
    normalised so they sum to 1.  Falls back to uniform if MI is unavailable.
    """
    col_to_mi_idx = {c: i for i, c in enumerate(mi_col_order)}

    def _centrality(col_name: str) -> float:
        idx = col_to_mi_idx.get(col_name)
        if idx is None:
            return 0.0
        row = mi_matrix[idx]
        # mean across all *other* columns (diagonal = 0 anyway)
        return float(np.mean(row))

    w_num = np.array([_centrality(c) for c in num_cols], dtype=float) if num_cols else np.zeros(0, dtype=float)
    w_ord = np.array([_centrality(c) for c in ord_cols], dtype=float) if ord_cols else np.zeros(0, dtype=float)
    w_cat = np.array([_centrality(c) for c in cat_cols], dtype=float) if cat_cols else np.zeros(0, dtype=float)

    total = w_num.sum() + w_ord.sum() + w_cat.sum()
    if total <= 0:
        # uniform fallback
        if num_cols:
            w_num[:] = 1.0
        if ord_cols:
            w_ord[:] = 1.0
        if cat_cols:
            w_cat[:] = 1.0
        total = w_num.sum() + w_ord.sum() + w_cat.sum()

    if total > 0:
        w_num /= total
        w_ord /= total
        w_cat /= total

    return w_num, w_ord, w_cat


# Stage A processes target rows in batches of this many, so peak memory is about
# B * n_rows * 8 bytes instead of n_missing * n_rows * 8 (for n = 20640 the full
# matrix would take several GiB). Batching does not change any distance.
_CANDIDATE_BATCH_ROWS = 1024


def _compute_candidate_sets(
    rows_with_any_missing: np.ndarray,
    k_max: int,
    n_rows: int,
    data_numeric: np.ndarray,
    data_ordinal: np.ndarray,
    data_categorical: np.ndarray,
    mask_numeric: np.ndarray,
    mask_ordinal: np.ndarray,
    mask_categorical: np.ndarray,
    num_scales: np.ndarray,
    ord_scales: np.ndarray,
    w_num_global: np.ndarray,
    w_ord_global: np.ndarray,
    w_cat_global: np.ndarray,
    use_distance_shrinkage: bool,
    shrink_strength: float,
    verbose: bool = False,
) -> Dict[int, Optional[np.ndarray]]:
    """Stage A: build candidate set C(i) for each row with any missing value.

    Uses global MI-centrality weights to compute Gower distances from each
    missing row to all other rows, then keeps top-Kcand nearest.

    Kcand = min(n_rows-1, max(10*k_max, int(10*sqrt(n_rows))))
    If Kcand >= 0.9*(n_rows-1), returns None (meaning: use all rows).

    Returns dict  {row_index: ndarray of candidate row indices | None}.
    None means the full dataset should be used (no pruning).
    """
    Kcand = min(n_rows - 1, max(10 * k_max, int(10 * np.sqrt(n_rows))))

    if Kcand >= 0.9 * (n_rows - 1):
        # Pruning would keep at least 90% of the rows: use the full search
        if verbose:
            print(f"[ISCA-k pruning] Kcand={Kcand} >= 0.9*(n-1)={0.9*(n_rows-1):.0f} → no pruning")
        return {int(ri): None for ri in rows_with_any_missing}, Kcand

    if verbose:
        print(f"[ISCA-k pruning] Kcand={Kcand}, rows_to_prune={len(rows_with_any_missing)}")

    # Distances from each row with missing values to all rows, with the global
    # weights, in batches of _CANDIDATE_BATCH_ROWS target rows.
    rows_with_any_missing = np.asarray(rows_with_any_missing, dtype=int)
    candidates: Dict[int, Optional[np.ndarray]] = {}

    for b0 in range(0, len(rows_with_any_missing), _CANDIDATE_BATCH_ROWS):
        chunk = rows_with_any_missing[b0:b0 + _CANDIDATE_BATCH_ROWS]
        dist_chunk = _compute_distances_for_rows(
            data_numeric, data_ordinal, data_categorical,
            mask_numeric, mask_ordinal, mask_categorical,
            num_scales, ord_scales,
            chunk,
            w_num_global, w_ord_global, w_cat_global,
            use_distance_shrinkage, shrink_strength,
        )
        for ti, ri in enumerate(chunk):
            dists = dist_chunk[ti]
            finite_mask = np.isfinite(dists)
            n_finite = int(finite_mask.sum())
            if n_finite <= Kcand:
                # Not enough finite distances to prune → use all
                candidates[int(ri)] = None
            else:
                # Partial sort: get indices of Kcand smallest distances
                # argpartition is O(n) vs O(n log n) for full sort
                idx_all = np.where(finite_mask)[0]
                dists_finite = dists[idx_all]
                part_idx = np.argpartition(dists_finite, Kcand)[:Kcand]
                candidates[int(ri)] = idx_all[part_idx]
        del dist_chunk

    return candidates, Kcand


def _dist_row_to_subset(
    row_idx: int,
    subset_idx: np.ndarray,
    data_numeric: np.ndarray,
    data_ordinal: np.ndarray,
    data_categorical: np.ndarray,
    mask_numeric: np.ndarray,
    mask_ordinal: np.ndarray,
    mask_categorical: np.ndarray,
    num_scales: np.ndarray,
    ord_scales: np.ndarray,
    w_numeric: np.ndarray,
    w_ordinal: np.ndarray,
    w_categorical: np.ndarray,
    use_distance_shrinkage: bool,
    shrink_strength: float,
) -> np.ndarray:
    """Compute Gower distances from row_idx to a subset of rows.

    Same computation as _compute_distances_for_rows, restricted to the rows in
    subset_idx. Returns a 1D array of length len(subset_idx).
    """
    subset_idx = np.asarray(subset_idx, dtype=int)
    n_sub = len(subset_idx)
    eps = 1e-12

    # Effective weights (same logic as _compute_distances_for_rows)
    if w_numeric.size and num_scales.size:
        w_num_eff = w_numeric * (num_scales > 0).astype(float)
        total_w_num = float(np.sum(w_num_eff))
    else:
        w_num_eff = w_numeric
        total_w_num = float(np.sum(w_numeric)) if w_numeric.size else 0.0

    if w_ordinal.size and ord_scales.size:
        w_ord_eff = w_ordinal * (ord_scales > 0).astype(float)
        total_w_ord = float(np.sum(w_ord_eff))
    else:
        w_ord_eff = w_ordinal
        total_w_ord = float(np.sum(w_ordinal)) if w_ordinal.size else 0.0

    total_w_cat = float(np.sum(w_categorical)) if w_categorical.size else 0.0
    total_w_all = total_w_num + total_w_ord + total_w_cat

    if total_w_all <= 0:
        return np.full(n_sub, np.inf, dtype=float)

    r = row_idx

    # --- numeric ---
    if data_numeric.size:
        sc_num = np.where(num_scales > 0, num_scales, 1.0)
        row_num = data_numeric[r] / sc_num
        np.nan_to_num(row_num, nan=0.0, copy=False)
        sub_num = data_numeric[subset_idx] / sc_num
        np.nan_to_num(sub_num, nan=0.0, copy=False)

        diffs = np.abs(sub_num - row_num)
        np.clip(diffs, 0.0, 1.0, out=diffs)
        common_n = mask_numeric[r] & mask_numeric[subset_idx]  # (n_sub, n_num)
        cw_n = common_n * w_num_eff
        diffs *= cw_n
        num_sum = diffs.sum(axis=1)
        num_wsum = cw_n.sum(axis=1)
        d_num = np.full(n_sub, np.nan, dtype=float)
        np.divide(num_sum, num_wsum, out=d_num, where=num_wsum > 0)
    else:
        num_wsum = np.zeros(n_sub, dtype=float)
        d_num = np.full(n_sub, np.nan, dtype=float)

    # --- ordinal ---
    if data_ordinal.size:
        sc_ord = np.where(ord_scales > 0, ord_scales, 1.0)
        row_ord = data_ordinal[r] / sc_ord
        np.nan_to_num(row_ord, nan=0.0, copy=False)
        sub_ord = data_ordinal[subset_idx] / sc_ord
        np.nan_to_num(sub_ord, nan=0.0, copy=False)

        diffs = np.abs(sub_ord - row_ord)
        np.clip(diffs, 0.0, 1.0, out=diffs)
        common_o = mask_ordinal[r] & mask_ordinal[subset_idx]
        cw_o = common_o * w_ord_eff
        diffs *= cw_o
        ord_sum = diffs.sum(axis=1)
        ord_wsum = cw_o.sum(axis=1)
        d_ord = np.full(n_sub, np.nan, dtype=float)
        np.divide(ord_sum, ord_wsum, out=d_ord, where=ord_wsum > 0)
    else:
        ord_wsum = np.zeros(n_sub, dtype=float)
        d_ord = np.full(n_sub, np.nan, dtype=float)

    # --- categorical/binary ---
    if data_categorical.size:
        common_c = mask_categorical[r] & mask_categorical[subset_idx]
        diffs_c = (data_categorical[subset_idx] != data_categorical[r]).astype(float)
        cw_c = common_c * w_categorical
        diffs_c *= cw_c
        cat_sum = diffs_c.sum(axis=1)
        cat_wsum = cw_c.sum(axis=1)
        d_cat = np.full(n_sub, np.nan, dtype=float)
        np.divide(cat_sum, cat_wsum, out=d_cat, where=cat_wsum > 0)
    else:
        cat_wsum = np.zeros(n_sub, dtype=float)
        d_cat = np.full(n_sub, np.nan, dtype=float)

    # Combine types
    types_present = (num_wsum > 0).astype(float) + (ord_wsum > 0).astype(float) + (cat_wsum > 0).astype(float)
    base = np.full(n_sub, np.inf, dtype=float)
    num_part = np.where(num_wsum > 0, d_num, 0.0)
    ord_part = np.where(ord_wsum > 0, d_ord, 0.0)
    cat_part = np.where(cat_wsum > 0, d_cat, 0.0)
    np.divide(num_part + ord_part + cat_part, types_present, out=base, where=types_present > 0)

    # The row itself does not enter the typical-distance statistic.
    self_mask = (subset_idx == r)

    # Low-overlap penalty
    if use_distance_shrinkage:
        overlap_w = num_wsum + ord_wsum + cat_wsum
        overlap = np.clip(overlap_w / total_w_all, 0.0, 1.0)
        finite_base = base[np.isfinite(base) & ~self_mask]
        typical = float(np.median(finite_base)) if finite_base.size > 0 else 1.0
        base = base + float(shrink_strength) * (1.0 - overlap) * typical

    # Exclude self if present in subset
    base[self_mask] = np.inf

    return base


class ISCAkImputer:
    """ISCA-k imputer.

    Columns are imputed in increasing order of missingness (ties broken by
    decreasing mean MI), rows within a column in increasing order of total
    missingness, and each imputed column is used when imputing the next ones.
    """

    def __init__(self, config: ISCAkConfig = ISCAkConfig()):
        self.cfg = config
        self.handler = MixedDataHandler(categorical_threshold=config.categorical_threshold)

        self.type_info: Optional[Dict[str, Any]] = None
        self.numeric_cols: List[str] = []
        self.binary_cols: List[str] = []
        self.nominal_cols: List[str] = []
        self.ordinal_cols: List[str] = []

        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.num_ranges: Dict[str, float] = {}
        # Robust per-variable scales for distance (p95-p05)
        self.num_scales: Dict[str, float] = {}
        self.ord_scales: Dict[str, float] = {}
        self._constant_cols: set = set()
        # MI diagnostics
        self._mi_pairs_valid: int = 0
        self._mi_pairs_total: int = 0
        self.global_fill: Dict[str, Any] = {}
        self.last_stats: Dict[str, Any] = {}
        self._fitted: bool = False

        self._numeric_set: set = set()
        self._ordinal_set: set = set()
        self._nominal_set: set = set()
        self._binary_set: set = set()

        self.mi_matrix: Optional[np.ndarray] = None
        self.mi_col_order: Optional[List[str]] = None
        # Effective k used in each imputed cell (diagnostics)
        self._k_selections: List[int] = []
        self.cell_diagnostics: list = []
        self._last_impute_result: Tuple[str, Any, int] = ("", None, 0)


    def fit(self, df: pd.DataFrame):
        if self.cfg.adaptive_k and not self.cfg.standardize_numeric:
            import warnings
            warnings.warn(
                "adaptive_k=True com standardize_numeric=False: o critério de "
                "estabilização da vizinhança está definido sobre valores "
                "padronizados; sem padronização, a paragem passa a depender da "
                "posição absoluta dos valores e degenera (janelas mínimas em "
                "dados afastados de zero). Recomenda-se standardize_numeric=True.",
                UserWarning,
            )
        df_enc, type_info = self.handler.fit_transform(
            df,
            force_categorical=self.cfg.force_categorical,
            force_ordinal=self.cfg.force_ordinal,
            interactive=self.cfg.interactive_type_detection,
            verbose=self.cfg.type_verbose,
        )
        self.type_info = type_info
        self.numeric_cols = self.handler.numeric_cols
        self.binary_cols = self.handler.binary_cols
        self.nominal_cols = self.handler.nominal_cols
        self.ordinal_cols = self.handler.ordinal_cols

        self._numeric_set = set(self.numeric_cols)
        self._ordinal_set = set(self.ordinal_cols)
        self._nominal_set = set(self.nominal_cols)
        self._binary_set = set(self.binary_cols)

        self.means, self.stds, self.num_ranges = _fit_numeric_stats(df_enc, self.numeric_cols)

        # Robust scales for distances (computed on encoded data; converted to standardized space if needed)
        stds_for_scale = self.stds if self.cfg.standardize_numeric else None
        self.num_scales, const_num = _fit_robust_scales(df_enc, self.numeric_cols, stds=stds_for_scale)
        # Ordinals are encoded as scalars (typically in [0,1]); still compute robust scales to detect constants
        self.ord_scales, const_ord = _fit_robust_scales(df_enc, self.ordinal_cols, stds=None)
        self._constant_cols = set(const_num) | set(const_ord)

        for c in df.columns:
            if c in self._numeric_set:
                self.global_fill[c] = float(pd.to_numeric(df[c], errors="coerce").mean(skipna=True))
            else:
                self.global_fill[c] = _safe_mode(df[c])

        if self.cfg.use_mi_weights:
            if self.cfg.verbose:
                print("[ISCA-k] Computing MI pairwise matrix...")
            # Reuse the MI matrix if this table and configuration were already seen.
            _cache_key = _df_cache_key(
                df_enc,
                numeric_cols=self.numeric_cols,
                ordinal_cols=self.ordinal_cols,
                nominal_cols=self.nominal_cols,
                binary_cols=self.binary_cols,
                mi_n_neighbors=self.cfg.mi_n_neighbors,
                mi_random_state=self.cfg.mi_random_state,
            )
            if _cache_key in _MI_CACHE:
                (self.mi_matrix, self.mi_col_order,
                 self._mi_pairs_valid, self._mi_pairs_total) = _MI_CACHE[_cache_key]
                if self.cfg.verbose:
                    print("[ISCA-k] MI matrix: cache hit.")
            else:
                self.mi_matrix, self.mi_col_order, self._mi_pairs_valid, self._mi_pairs_total = _compute_mi_matrix(
                    df_enc,
                    numeric_cols=self.numeric_cols, ordinal_cols=self.ordinal_cols,
                    nominal_cols=self.nominal_cols, binary_cols=self.binary_cols,
                    n_neighbors=self.cfg.mi_n_neighbors, random_state=self.cfg.mi_random_state,
                )
                # Store in the cache, dropping the oldest entry when it is full
                if len(_MI_CACHE) >= _MI_CACHE_MAX_SIZE:
                    _MI_CACHE.pop(next(iter(_MI_CACHE)))
                _MI_CACHE[_cache_key] = (
                    self.mi_matrix, self.mi_col_order,
                    self._mi_pairs_valid, self._mi_pairs_total
                )
            if self.cfg.verbose:
                print(f"[ISCA-k] MI matrix shape: {self.mi_matrix.shape}")

        self._fitted = True
        return self

    def fit_transform(self, df_missing: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df_missing).transform(df_missing)

    def transform(self, df_missing: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("ISCAkImputer must be fitted first.")
        self.cell_diagnostics = []
        # Reset k diagnostics for this transform call
        self._k_selections = []
        mi_uniform_targets = 0
        mi_total_targets = 0


        # ═══════════════════════════════════════════════════════
        # PHASE 1: Encoding
        # ═══════════════════════════════════════════════════════
        df_enc = df_missing.copy()
        for c in self.nominal_cols:
            if c in df_enc.columns:
                mapping = self.handler.nominal_mappings.get(c, {}).get("forward", {})
                df_enc[c] = df_enc[c].map(mapping)
        for c in self.ordinal_cols:
            if c in df_enc.columns:
                mapping = self.handler.ordinal_mappings.get(c, {}).get("forward", {})
                df_enc[c] = df_enc[c].map(mapping)

        # ═══════════════════════════════════════════════════════
        # PHASE 2: Handle fully-missing columns
        # ═══════════════════════════════════════════════════════
        all_missing_cols = [c for c in df_enc.columns if df_enc[c].isna().all()]
        if all_missing_cols:
            df_work = df_enc.drop(columns=all_missing_cols)
        else:
            df_work = df_enc.copy()

        # ═══════════════════════════════════════════════════════
        # PHASE 3: Standardize numeric columns
        # ═══════════════════════════════════════════════════════
        if self.cfg.standardize_numeric and self.numeric_cols:
            cols_present = [c for c in self.numeric_cols if c in df_work.columns]
            df_work = _standardize_numeric(df_work, cols_present, self.means, self.stds)

        # ═══════════════════════════════════════════════════════
        # PHASE 4: Fallback values
        # ═══════════════════════════════════════════════════════
        fallback_values = {}
        for c in df_work.columns:
            if c in self._numeric_set:
                val = pd.to_numeric(df_work[c], errors="coerce").mean(skipna=True)
                fallback_values[c] = float(val) if np.isfinite(val) else 0.0
            else:
                fallback_values[c] = _safe_mode(df_work[c])

        # ═══════════════════════════════════════════════════════
        # PHASE 5: Column imputation order
        # ═══════════════════════════════════════════════════════
        col_list = list(df_work.columns)
        col_to_idx = {c: i for i, c in enumerate(col_list)}

        num_cols_present = [c for c in self.numeric_cols if c in col_to_idx]
        ord_cols_present = [c for c in self.ordinal_cols if c in col_to_idx]
        nom_cols_present = [c for c in self.nominal_cols if c in col_to_idx]
        bin_cols_present = [c for c in self.binary_cols if c in col_to_idx]
        cat_cols_present = nom_cols_present + bin_cols_present

        mi_col_to_idx = None
        if self.cfg.use_mi_weights and self.mi_matrix is not None:
            mi_col_to_idx = {c: i for i, c in enumerate(self.mi_col_order)}

        missing_per_col = df_work.isna().sum()
        cols_with_missing = [c for c in col_list if missing_per_col[c] > 0]

        n_rows = len(df_work)

        # Statistics
        fallback_rows_all_missing = 0
        fallback_cells_no_neighbors = 0
        neighbors_scanned_total = 0
        imputed_cells = 0

        # Handle fully-missing rows with fallback
        mask_init = pd.notna(df_work).to_numpy()
        for ri in range(n_rows):
            if not np.any(mask_init[ri]):
                fallback_rows_all_missing += 1
                for ci, c in enumerate(col_list):
                    if not mask_init[ri, ci]:
                        df_work.iat[ri, ci] = fallback_values[c]

        # ═══════════════════════════════════════════════════════
        # PHASE 6+7: Imputation (propagation or non-propagation)
        # ═══════════════════════════════════════════════════════
        if self.cfg.use_propagation and cols_with_missing:
            # Sort columns: missingness ascending, mean MI descending
            def _col_sort_key(c):
                n_miss = missing_per_col[c]
                if mi_col_to_idx is not None and c in mi_col_to_idx:
                    mi_mean = float(self.mi_matrix[mi_col_to_idx[c]].mean())
                else:
                    mi_mean = 0.0
                return (n_miss, -mi_mean)

            cols_with_missing.sort(key=_col_sort_key)

            if self.cfg.verbose:
                print(f"[ISCA-k] Imputation order ({len(cols_with_missing)} cols):")
                for c in cols_with_missing[:10]:
                    print(f"  {c}: {missing_per_col[c]} missings")
                if len(cols_with_missing) > 10:
                    print(f"  ... and {len(cols_with_missing) - 10} more")

            # Number of missing cells per row, decremented as cells are imputed.
            row_miss_count = df_work.isna().sum(axis=1).to_numpy(dtype=int).copy()

            # ── Stage A: candidate pruning (global, per-row) ──
            _kcand_val = None
            _use_pruning = self.cfg.use_candidate_pruning and len(cols_with_missing) > 0
            _candidate_sets: Dict[int, Optional[np.ndarray]] = {}
            

            if _use_pruning:
                # Identify all rows with any missing value
                _rows_any_missing = np.where(row_miss_count > 0)[0]
                if len(_rows_any_missing) > 0:
                    # Need initial arrays for global distance computation
                    _data_full_init = df_work.to_numpy(dtype=object)
                    _mask_full_init = pd.notna(df_work).to_numpy()
                    _n_r = len(df_work)

                    if num_cols_present:
                        _num_ci = [col_to_idx[c] for c in num_cols_present]
                        _dn = _data_full_init[:, _num_ci].astype(float)
                        _mn = _mask_full_init[:, _num_ci]
                        _ns = np.array([float(self.num_scales.get(c, 0.0)) for c in num_cols_present], dtype=float)
                    else:
                        _dn = np.empty((_n_r, 0), dtype=float)
                        _mn = np.empty((_n_r, 0), dtype=bool)
                        _ns = np.empty((0,), dtype=float)

                    if ord_cols_present:
                        _oci = [col_to_idx[c] for c in ord_cols_present]
                        _do = _data_full_init[:, _oci].astype(float)
                        _mo = _mask_full_init[:, _oci]
                        _os = np.array([float(self.ord_scales.get(c, 0.0)) for c in ord_cols_present], dtype=float)
                    else:
                        _do = np.empty((_n_r, 0), dtype=float)
                        _mo = np.empty((_n_r, 0), dtype=bool)
                        _os = np.empty((0,), dtype=float)

                    if cat_cols_present:
                        _cci = [col_to_idx[c] for c in cat_cols_present]
                        _dc = _data_full_init[:, _cci]
                        _mc = _mask_full_init[:, _cci]
                    else:
                        _dc = np.empty((_n_r, 0), dtype=object)
                        _mc = np.empty((_n_r, 0), dtype=bool)

                    # Global MI centrality weights
                    if (self.cfg.use_mi_weights and self.mi_matrix is not None
                            and self.mi_col_order is not None):
                        _wgn, _wgo, _wgc = _compute_global_mi_weights(
                            self.mi_matrix, self.mi_col_order,
                            num_cols_present, ord_cols_present, cat_cols_present)
                    else:
                        # uniform fallback
                        _tn = len(num_cols_present) + len(ord_cols_present) + len(cat_cols_present)
                        _tn = max(_tn, 1)
                        _wgn = np.full(len(num_cols_present), 1.0 / _tn, dtype=float)
                        _wgo = np.full(len(ord_cols_present), 1.0 / _tn, dtype=float)
                        _wgc = np.full(len(cat_cols_present), 1.0 / _tn, dtype=float)

                    _candidate_sets, _kcand_val = _compute_candidate_sets(
                        _rows_any_missing,
                        k_max=(self.cfg.candidate_k if self.cfg.candidate_k is not None else (self.cfg.k_max if self.cfg.adaptive_k else self.cfg.k)),
                        n_rows=n_rows,
                        data_numeric=_dn, data_ordinal=_do, data_categorical=_dc,
                        mask_numeric=_mn, mask_ordinal=_mo, mask_categorical=_mc,
                        num_scales=_ns, ord_scales=_os,
                        w_num_global=_wgn, w_ord_global=_wgo, w_cat_global=_wgc,
                        use_distance_shrinkage=self.cfg.use_distance_shrinkage,
                        shrink_strength=self.cfg.shrink_strength,
                        verbose=self.cfg.verbose,
                    )

            # Column-by-column with propagation
            for target_col in cols_with_missing:
                target_ci = col_to_idx[target_col]

                # Which rows still need imputation?
                missing_mask_col = df_work[target_col].isna()
                rows_needing = np.where(missing_mask_col.to_numpy())[0]
                if len(rows_needing) == 0:
                    continue

                # Sort rows by current total missingness, ascending
                rows_needing = rows_needing[np.argsort(row_miss_count[rows_needing])]

                # Extract fresh arrays (reflects propagation from prior columns)
                data_full = df_work.to_numpy(dtype=object)
                mask_full = pd.notna(df_work).to_numpy()

                n_r = len(df_work)
                if num_cols_present:
                    num_ci = [col_to_idx[c] for c in num_cols_present]
                    data_num = data_full[:, num_ci].astype(float)
                    mask_num = mask_full[:, num_ci]
                    num_scales_arr = np.array(
                        [float(self.num_scales.get(c, 0.0)) for c in num_cols_present], dtype=float)
                else:
                    data_num = np.empty((n_r, 0), dtype=float)
                    mask_num = np.empty((n_r, 0), dtype=bool)
                    num_scales_arr = np.empty((0,), dtype=float)

                if ord_cols_present:
                    ord_ci = [col_to_idx[c] for c in ord_cols_present]
                    data_ord = data_full[:, ord_ci].astype(float)
                    mask_ord = mask_full[:, ord_ci]
                    ord_scales_arr = np.array(
                        [float(self.ord_scales.get(c, 0.0)) for c in ord_cols_present], dtype=float)
                else:
                    data_ord = np.empty((n_r, 0), dtype=float)
                    mask_ord = np.empty((n_r, 0), dtype=bool)
                    ord_scales_arr = np.empty((0,), dtype=float)

                if cat_cols_present:
                    cat_ci = [col_to_idx[c] for c in cat_cols_present]
                    data_cat = data_full[:, cat_ci]
                    mask_cat = mask_full[:, cat_ci]
                else:
                    data_cat = np.empty((n_r, 0), dtype=object)
                    mask_cat = np.empty((n_r, 0), dtype=bool)

                # MI weights for this target
                w_num, w_ord, w_cat, mi_uniform = self._get_column_weights(
                    target_col, num_cols_present, ord_cols_present,
                    cat_cols_present, col_list, mi_col_to_idx)

                mi_total_targets += 1
                if mi_uniform:
                    mi_uniform_targets += 1

                # ── Stage B: per-row imputation with candidate pruning ──
                # Decide whether to use batch (no pruning) or per-row subset distances
                _any_pruned = _use_pruning and any(
                    _candidate_sets.get(int(ri)) is not None for ri in rows_needing
                )

                if not _any_pruned:
                    # No pruning: distances from all target rows to all rows at once
                    dist_matrix = _compute_distances_for_rows(
                        data_num, data_ord, data_cat,
                        mask_num, mask_ord, mask_cat,
                        num_scales_arr, ord_scales_arr,
                        rows_needing, w_num, w_ord, w_cat,
                        self.cfg.use_distance_shrinkage, self.cfg.shrink_strength)

                    for ti, ri in enumerate(rows_needing):
                        distances = dist_matrix[ti]
                        self._impute_single_cell_from_distances(
                            ri, target_ci, target_col, distances, mask_full, data_full,
                            fallback_values, num_cols_present, ord_cols_present,
                            num_scales_arr, ord_scales_arr, row_miss_count,
                        )
                        status, value, _nc = self._last_impute_result
                        if status == "fallback" and _nc == 0:
                            fallback_cells_no_neighbors += 1
                        df_work.iat[ri, target_ci] = value
                        row_miss_count[ri] -= 1
                        neighbors_scanned_total += n_rows
                        imputed_cells += 1
                else:
                    # Pruning: distances to each row's candidate set
                    for ri in rows_needing:
                        cand = _candidate_sets.get(int(ri))
                        if cand is None:
                            # No candidate set for this row: full search
                            dist_1 = _compute_distances_for_rows(
                                data_num, data_ord, data_cat,
                                mask_num, mask_ord, mask_cat,
                                num_scales_arr, ord_scales_arr,
                                np.array([ri]), w_num, w_ord, w_cat,
                                self.cfg.use_distance_shrinkage, self.cfg.shrink_strength)[0]
                            self._impute_single_cell_from_distances(
                                ri, target_ci, target_col, dist_1, mask_full, data_full,
                                fallback_values, num_cols_present, ord_cols_present,
                                num_scales_arr, ord_scales_arr, row_miss_count,
                            )
                            status, value, _nc = self._last_impute_result
                            if status == "fallback" and _nc == 0:
                                fallback_cells_no_neighbors += 1
                            df_work.iat[ri, target_ci] = value
                            row_miss_count[ri] -= 1
                            neighbors_scanned_total += n_rows
                        else:
                            # Filter C(i): keep only candidates with observed target
                            cand_obs_mask = mask_full[cand, target_ci]
                            cand_observed = cand[cand_obs_mask]

                            if len(cand_observed) == 0:
                                # No candidates with observed target → fallback
                                fallback_cells_no_neighbors += 1
                                df_work.iat[ri, target_ci] = fallback_values[target_col]
                                row_miss_count[ri] -= 1
                                self._k_selections.append(0)
                                imputed_cells += 1
                                neighbors_scanned_total += 0
                                continue

                            # Compute target-aware distances only to this subset
                            sub_dists = _dist_row_to_subset(
                                ri, cand_observed,
                                data_num, data_ord, data_cat,
                                mask_num, mask_ord, mask_cat,
                                num_scales_arr, ord_scales_arr,
                                w_num, w_ord, w_cat,
                                self.cfg.use_distance_shrinkage, self.cfg.shrink_strength,
                            )

                            # Build full-sized distance vector (inf everywhere except subset)
                            full_dists = np.full(n_rows, np.inf, dtype=float)
                            full_dists[cand_observed] = sub_dists

                            self._impute_single_cell_from_distances(
                                ri, target_ci, target_col, full_dists, mask_full, data_full,
                                fallback_values, num_cols_present, ord_cols_present,
                                num_scales_arr, ord_scales_arr, row_miss_count,
                            )
                            status, value, _nc = self._last_impute_result
                            if status == "fallback" and _nc == 0:
                                fallback_cells_no_neighbors += 1
                            df_work.iat[ri, target_ci] = value
                            row_miss_count[ri] -= 1
                            neighbors_scanned_total += len(cand_observed)

                        imputed_cells += 1

                # Column done: df_work now holds its imputed values for the next columns

        else:
            # ── Without propagation: every column is imputed from the original data ──
            mask_full = pd.notna(df_work).to_numpy()
            data_full = df_work.to_numpy(dtype=object)
            data_imputed = data_full.copy()

            (data_num, mask_num, num_scales_arr,
             data_ord, mask_ord, ord_scales_arr,
             data_cat, mask_cat) = _extract_typed_arrays(
                df_work, num_cols_present, ord_cols_present,
                cat_cols_present, self.num_scales, self.ord_scales)

            missing_cols_map = {}
            for ri in range(n_rows):
                if not np.any(mask_full[ri]):
                    continue
                for ci, c in enumerate(col_list):
                    if not mask_full[ri, ci]:
                        missing_cols_map.setdefault(c, []).append(ri)

            # ── Stage A: candidate pruning for non-propagation path ──
            _kcand_val = None
            _use_pruning_np = self.cfg.use_candidate_pruning and len(missing_cols_map) > 0
            _use_pruning = _use_pruning_np
            _candidate_sets_np: Dict[int, Optional[np.ndarray]] = {}
            

            if _use_pruning_np:
                _rows_any_missing_np = np.array(sorted(set(
                    ri for rows in missing_cols_map.values() for ri in rows
                )), dtype=int)
                if len(_rows_any_missing_np) > 0:
                    if (self.cfg.use_mi_weights and self.mi_matrix is not None
                            and self.mi_col_order is not None):
                        _wgn, _wgo, _wgc = _compute_global_mi_weights(
                            self.mi_matrix, self.mi_col_order,
                            num_cols_present, ord_cols_present, cat_cols_present)
                    else:
                        _tn = len(num_cols_present) + len(ord_cols_present) + len(cat_cols_present)
                        _tn = max(_tn, 1)
                        _wgn = np.full(len(num_cols_present), 1.0 / _tn, dtype=float)
                        _wgo = np.full(len(ord_cols_present), 1.0 / _tn, dtype=float)
                        _wgc = np.full(len(cat_cols_present), 1.0 / _tn, dtype=float)

                    _candidate_sets_np, _kcand_val = _compute_candidate_sets(
                        _rows_any_missing_np,
                        k_max=(self.cfg.candidate_k if self.cfg.candidate_k is not None else (self.cfg.k_max if self.cfg.adaptive_k else self.cfg.k)),
                        n_rows=n_rows,
                        data_numeric=data_num, data_ordinal=data_ord, data_categorical=data_cat,
                        mask_numeric=mask_num, mask_ordinal=mask_ord, mask_categorical=mask_cat,
                        num_scales=num_scales_arr, ord_scales=ord_scales_arr,
                        w_num_global=_wgn, w_ord_global=_wgo, w_cat_global=_wgc,
                        use_distance_shrinkage=self.cfg.use_distance_shrinkage,
                        shrink_strength=self.cfg.shrink_strength,
                        verbose=self.cfg.verbose,
                    )

            for target_col, target_rows_list in missing_cols_map.items():
                target_rows_arr = np.array(sorted(set(target_rows_list)))
                target_ci = col_to_idx[target_col]

                w_num, w_ord, w_cat, mi_uniform = self._get_column_weights(
                    target_col, num_cols_present, ord_cols_present,
                    cat_cols_present, col_list, mi_col_to_idx)

                mi_total_targets += 1
                if mi_uniform:
                    mi_uniform_targets += 1

                # Check if any row in this target has pruning active
                _any_pruned_np = _use_pruning_np and any(
                    _candidate_sets_np.get(int(ri)) is not None for ri in target_rows_arr
                )

                if not _any_pruned_np:
                    # No pruning: distances from all target rows to all rows at once
                    dist_matrix = _compute_distances_for_rows(
                        data_num, data_ord, data_cat,
                        mask_num, mask_ord, mask_cat,
                        num_scales_arr, ord_scales_arr,
                        target_rows_arr, w_num, w_ord, w_cat,
                        self.cfg.use_distance_shrinkage, self.cfg.shrink_strength)

                    for ti, ri in enumerate(target_rows_arr):
                        distances = dist_matrix[ti]
                        valid_mask = np.isfinite(distances)
                        if not np.any(valid_mask):
                            fallback_cells_no_neighbors += 1
                            data_imputed[ri, target_ci] = fallback_values[target_col]
                            imputed_cells += 1
                            continue
                        sorted_idx = np.argsort(distances)
                        ordered_neighbors = sorted_idx[np.isfinite(distances[sorted_idx])]
                        chosen = []
                        chosen_dists = []
                        scanned = 0

                        max_needed = self.cfg.k
                        if self.cfg.adaptive_k:
                            max_needed = max(self.cfg.k_max, self.cfg.k_min)

                        for rj in ordered_neighbors:
                            scanned += 1
                            if mask_full[rj, target_ci]:
                                chosen.append(rj)
                                chosen_dists.append(float(distances[rj]))
                                if len(chosen) >= max_needed:
                                    break

                        if self.cfg.adaptive_k and len(chosen) > 0:
                            is_cat = target_col not in self._numeric_set and target_col not in self._ordinal_set
                            vals_for_k = data_full[np.array(chosen), target_ci]
                            if target_col in num_cols_present:
                                col_scale_val = float(num_scales_arr[num_cols_present.index(target_col)])
                            elif target_col in ord_cols_present:
                                col_scale_val = float(ord_scales_arr[ord_cols_present.index(target_col)])
                            else:
                                col_scale_val = None
                            k_eff = _select_k_nw_homogeneity(
                                ordered_values=vals_for_k,
                                ordered_distances=np.asarray(chosen_dists, dtype=float),
                                k_min=self.cfg.k_min,
                                k_max=self.cfg.k_max,
                                homogeneity_threshold=self.cfg.homogeneity_threshold,
                                stability_threshold=self.cfg.stability_threshold,
                                is_categorical=is_cat,
                                col_scale=col_scale_val,
                stability_absolute=self.cfg.stability_absolute,
                safeguard_sd=self.cfg.safeguard_sd,
                            )
                            chosen = chosen[:k_eff]
                            chosen_dists = chosen_dists[:k_eff]
                        elif not self.cfg.adaptive_k:
                            k_eff = min(self.cfg.k, len(chosen))
                            chosen = chosen[:k_eff]
                            chosen_dists = chosen_dists[:k_eff]
                        else:
                            k_eff = 0

                        self._k_selections.append(int(k_eff))
                        neighbors_scanned_total += scanned
                        imputed_cells += 1
                        if not chosen:
                            fallback_cells_no_neighbors += 1
                            data_imputed[ri, target_ci] = fallback_values[target_col]
                            continue
                        vals = data_full[np.array(chosen), target_ci]
                        dists_chosen = np.array(chosen_dists[:len(chosen)], dtype=float)

                        if target_col in self._numeric_set or target_col in self._ordinal_set:
                            vals_float = pd.to_numeric(vals, errors='coerce')
                            valid_v = ~np.isnan(vals_float)
                            vals_clean = vals_float[valid_v]
                            dists_clean = dists_chosen[valid_v]
                            if len(vals_clean) == 0:
                                data_imputed[ri, target_ci] = fallback_values[target_col]
                            else:
                                data_imputed[ri, target_ci] = _weighted_mean(vals_clean, dists_clean)
                        else:
                            data_imputed[ri, target_ci] = _weighted_mode(vals, dists_chosen)
                else:
                    # Pruning active → per-row
                    for ri in target_rows_arr:
                        cand = _candidate_sets_np.get(int(ri))
                        if cand is None:
                            # No pruning → full distance
                            dist_1 = _compute_distances_for_rows(
                                data_num, data_ord, data_cat,
                                mask_num, mask_ord, mask_cat,
                                num_scales_arr, ord_scales_arr,
                                np.array([ri]), w_num, w_ord, w_cat,
                                self.cfg.use_distance_shrinkage, self.cfg.shrink_strength)[0]
                            full_dists = dist_1
                            scanned_count = n_rows
                        else:
                            cand_obs_mask = mask_full[cand, target_ci]
                            cand_observed = cand[cand_obs_mask]
                            if len(cand_observed) == 0:
                                fallback_cells_no_neighbors += 1
                                data_imputed[ri, target_ci] = fallback_values[target_col]
                                self._k_selections.append(0)
                                imputed_cells += 1
                                continue
                            sub_dists = _dist_row_to_subset(
                                ri, cand_observed,
                                data_num, data_ord, data_cat,
                                mask_num, mask_ord, mask_cat,
                                num_scales_arr, ord_scales_arr,
                                w_num, w_ord, w_cat,
                                self.cfg.use_distance_shrinkage, self.cfg.shrink_strength,
                            )
                            full_dists = np.full(n_rows, np.inf, dtype=float)
                            full_dists[cand_observed] = sub_dists
                            scanned_count = len(cand_observed)

                        # Same imputation logic as non-pruned path
                        valid_mask = np.isfinite(full_dists)
                        if not np.any(valid_mask):
                            fallback_cells_no_neighbors += 1
                            data_imputed[ri, target_ci] = fallback_values[target_col]
                            imputed_cells += 1
                            continue

                        sorted_idx = np.argsort(full_dists)
                        ordered_neighbors = sorted_idx[np.isfinite(full_dists[sorted_idx])]
                        chosen = []
                        chosen_dists = []
                        scanned = 0

                        max_needed = self.cfg.k
                        if self.cfg.adaptive_k:
                            max_needed = max(self.cfg.k_max, self.cfg.k_min)

                        for rj in ordered_neighbors:
                            scanned += 1
                            if mask_full[rj, target_ci]:
                                chosen.append(rj)
                                chosen_dists.append(float(full_dists[rj]))
                                if len(chosen) >= max_needed:
                                    break

                        if self.cfg.adaptive_k and len(chosen) > 0:
                            is_cat = target_col not in self._numeric_set and target_col not in self._ordinal_set
                            vals_for_k = data_full[np.array(chosen), target_ci]
                            if target_col in num_cols_present:
                                col_scale_val = float(num_scales_arr[num_cols_present.index(target_col)])
                            elif target_col in ord_cols_present:
                                col_scale_val = float(ord_scales_arr[ord_cols_present.index(target_col)])
                            else:
                                col_scale_val = None
                            k_eff = _select_k_nw_homogeneity(
                                ordered_values=vals_for_k,
                                ordered_distances=np.asarray(chosen_dists, dtype=float),
                                k_min=self.cfg.k_min,
                                k_max=self.cfg.k_max,
                                homogeneity_threshold=self.cfg.homogeneity_threshold,
                                stability_threshold=self.cfg.stability_threshold,
                                is_categorical=is_cat,
                                col_scale=col_scale_val,
                stability_absolute=self.cfg.stability_absolute,
                safeguard_sd=self.cfg.safeguard_sd,
                            )
                            chosen = chosen[:k_eff]
                            chosen_dists = chosen_dists[:k_eff]
                        elif not self.cfg.adaptive_k:
                            k_eff = min(self.cfg.k, len(chosen))
                            chosen = chosen[:k_eff]
                            chosen_dists = chosen_dists[:k_eff]
                        else:
                            k_eff = 0

                        self._k_selections.append(int(k_eff))
                        neighbors_scanned_total += scanned_count
                        imputed_cells += 1
                        if not chosen:
                            fallback_cells_no_neighbors += 1
                            data_imputed[ri, target_ci] = fallback_values[target_col]
                            continue
                        vals = data_full[np.array(chosen), target_ci]
                        dists_chosen = np.array(chosen_dists[:len(chosen)], dtype=float)

                        if target_col in self._numeric_set or target_col in self._ordinal_set:
                            vals_float = pd.to_numeric(vals, errors='coerce')
                            valid_v = ~np.isnan(vals_float)
                            vals_clean = vals_float[valid_v]
                            dists_clean = dists_chosen[valid_v]
                            if len(vals_clean) == 0:
                                data_imputed[ri, target_ci] = fallback_values[target_col]
                            else:
                                data_imputed[ri, target_ci] = _weighted_mean(vals_clean, dists_clean)
                        else:
                            data_imputed[ri, target_ci] = _weighted_mode(vals, dists_chosen)

            df_work = pd.DataFrame(data_imputed, columns=col_list, index=df_work.index)

        # ═══════════════════════════════════════════════════════
        # PHASE 9: Destandardize
        # ═══════════════════════════════════════════════════════
        if self.cfg.standardize_numeric and self.numeric_cols:
            cols_present = [c for c in self.numeric_cols if c in df_work.columns]
            df_work = _destandardize_numeric(df_work, cols_present, self.means, self.stds)

        # ═══════════════════════════════════════════════════════
        # PHASE 10: Reattach fully-missing columns
        # ═══════════════════════════════════════════════════════
        if all_missing_cols:
            for c in all_missing_cols:
                df_work[c] = self.global_fill.get(c, np.nan)
            df_work = df_work[df_enc.columns]

        # ═══════════════════════════════════════════════════════
        # PHASE 11: Decode
        # ═══════════════════════════════════════════════════════
        df_dec = self.handler.inverse_transform(df_work)

        # ═══════════════════════════════════════════════════════
        # PHASE 12: Fill residual NaNs
        # ═══════════════════════════════════════════════════════
        for c in df_dec.columns:
            if df_dec[c].isna().any():
                df_dec[c] = df_dec[c].fillna(self.global_fill.get(c, _safe_mode(df_dec[c])))

        # ═══════════════════════════════════════════════════════
        # PHASE 13: Stats
        # ═══════════════════════════════════════════════════════
        try:
            mask_final = pd.notna(df_missing).to_numpy()
            total_cells = int(df_missing.size)
            missing_cells_count = int((~mask_final).sum())
            rows_with_missing_count = int((~mask_final).any(axis=1).sum())
            avg_neighbors = float(neighbors_scanned_total / max(1, imputed_cells))

            self.last_stats = {
                "rows": int(df_missing.shape[0]),
                "cols": int(df_missing.shape[1]),
                "total_cells": total_cells,
                "missing_cells": missing_cells_count,
                "missing_pct": float(missing_cells_count / max(1, total_cells)),
                "rows_with_missing": rows_with_missing_count,
                # Report configured k and actually used k distribution (if adaptive)
                "k_config": int(self.cfg.k),
                "k_used_mean": float(np.mean(self._k_selections)) if self._k_selections else float(self.cfg.k),
                "k_used_median": float(np.median(self._k_selections)) if self._k_selections else float(self.cfg.k),
                "k_used_min": int(np.min(self._k_selections)) if self._k_selections else int(self.cfg.k),
                "k_used_max": int(np.max(self._k_selections)) if self._k_selections else int(self.cfg.k),

                "adaptive_k": bool(self.cfg.adaptive_k),
                "k_min": int(self.cfg.k_min),
                "k_max": int(self.cfg.k_max),
                "fallback_rows_all_missing": int(fallback_rows_all_missing),
                "imputed_cells": int(imputed_cells),
                "fallback_cells_no_neighbors": int(fallback_cells_no_neighbors),
                "avg_neighbors_scanned_per_cell": avg_neighbors,
                "candidate_pruning": bool(self.cfg.use_candidate_pruning),
                "dropped_all_missing_cols": len(all_missing_cols),
                "mi_weighted": self.cfg.use_mi_weights,
                "propagation": self.cfg.use_propagation,
                "scale_mode": bool(_use_pruning and _kcand_val is not None),
                "kcand": _kcand_val,
                "k_selections_len": len(self._k_selections),
                "k_selections_sample": self._k_selections[:5],
    
                "mi_uniform_used_count": mi_uniform_targets,
                "mi_weights_total_count": mi_total_targets,
            }

            if self.cfg.verbose:
                fb_pct = 100.0 * (fallback_cells_no_neighbors / max(1, missing_cells_count))
                mi_label = "MI-weighted" if self.cfg.use_mi_weights else "uniform"
                prop_label = "propagation" if self.cfg.use_propagation else "no-propagation"
                print(
                    f"[ISCA-k] rows_with_missing={rows_with_missing_count} | "
                    f"avg_neighbors_checked={avg_neighbors:.2f} | "
                    f"fallback_cells={fb_pct:.2f}% | {mi_label} | {prop_label}"
                )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.last_stats = {}

        return df_dec

    # ── Helper: impute a single cell given a full distance vector ──

    def _impute_single_cell_from_distances(
        self,
        ri: int,
        target_ci: int,
        target_col: str,
        distances: np.ndarray,
        mask_full: np.ndarray,
        data_full: np.ndarray,
        fallback_values: dict,
        num_cols_present: list,
        ord_cols_present: list,
        num_scales_arr: np.ndarray,
        ord_scales_arr: np.ndarray,
        row_miss_count: np.ndarray,
    ):
        """Impute cell (ri, target_ci) from the given distance vector.

        Sorts the neighbours, keeps those with an observed target, chooses k and
        computes the weighted estimate. The result is stored in
        self._last_impute_result as (status, value, n_neighbours); the caller
        writes the value into the working table.
        """
        valid_mask = np.isfinite(distances)

        if not np.any(valid_mask):
            self._last_impute_result = ("fallback", fallback_values[target_col], 0)
            return

        sorted_idx = np.argsort(distances)
        ordered_neighbors = sorted_idx[np.isfinite(distances[sorted_idx])]

        chosen = []
        chosen_dists = []

        max_needed = self.cfg.k
        if self.cfg.adaptive_k:
            max_needed = max(self.cfg.k_max, self.cfg.k_min)

        for rj in ordered_neighbors:
            if mask_full[rj, target_ci]:
                chosen.append(rj)
                chosen_dists.append(float(distances[rj]))
                if len(chosen) >= max_needed:
                    break

        if self.cfg.adaptive_k and len(chosen) > 0:
            is_cat = target_col not in self._numeric_set and target_col not in self._ordinal_set
            vals_for_k = data_full[np.array(chosen), target_ci]
            if target_col in num_cols_present:
                col_scale_val = float(num_scales_arr[num_cols_present.index(target_col)])
            elif target_col in ord_cols_present:
                col_scale_val = float(ord_scales_arr[ord_cols_present.index(target_col)])
            else:
                col_scale_val = None
            trace = {} if self.cfg.record_diagnostics else None
            k_eff = _select_k_nw_homogeneity(
                ordered_values=vals_for_k,
                ordered_distances=np.asarray(chosen_dists, dtype=float),
                k_min=self.cfg.k_min,
                k_max=self.cfg.k_max,
                homogeneity_threshold=self.cfg.homogeneity_threshold,
                stability_threshold=self.cfg.stability_threshold,
                is_categorical=is_cat,
                diagnostic=trace,
                col_scale=col_scale_val,
                stability_absolute=self.cfg.stability_absolute,
                safeguard_sd=self.cfg.safeguard_sd,
            )
            if trace is not None:
                trace.update(row=int(ri), column=target_col)
                trace["ks"] = list(range(min(self.cfg.k_min, len(chosen)), len(chosen) + 1))
                if not is_cat:
                    vals = np.asarray(vals_for_k, dtype=float)
                    ds = np.asarray(chosen_dists, dtype=float)
                    predictions = [_weighted_mean(vals[:kk], ds[:kk]) for kk in trace["ks"]]
                    if target_col in self._numeric_set and self.cfg.standardize_numeric:
                        predictions = [v * self.stds[target_col] + self.means[target_col] for v in predictions]
                    trace["predictions"] = [float(v) for v in predictions]
                self.cell_diagnostics.append(trace)
            chosen = chosen[:k_eff]
            chosen_dists = chosen_dists[:k_eff]
        elif not self.cfg.adaptive_k:
            k_eff = min(self.cfg.k, len(chosen))
            chosen = chosen[:k_eff]
            chosen_dists = chosen_dists[:k_eff]
        else:
            k_eff = 0

        self._k_selections.append(int(k_eff))

        if not chosen:
            self._last_impute_result = ("fallback", fallback_values[target_col], 0)
            return

        vals = data_full[np.array(chosen), target_ci]
        dists_chosen = np.array(chosen_dists[:len(chosen)], dtype=float)

        if target_col in self._numeric_set or target_col in self._ordinal_set:
            vals_float = pd.to_numeric(vals, errors='coerce')
            valid_v = ~np.isnan(vals_float)
            vals_clean = vals_float[valid_v]
            dists_clean = dists_chosen[valid_v]
            if len(vals_clean) == 0:
                self._last_impute_result = ("fallback", fallback_values[target_col], len(chosen))
            else:
                self._last_impute_result = ("imputed", _weighted_mean(vals_clean, dists_clean), len(chosen))
        else:
            self._last_impute_result = ("imputed", _weighted_mode(vals, dists_chosen), len(chosen))

    # ── Helper: get MI weight vectors for a target column ──

    def _get_column_weights(
        self,
        target_col: str,
        num_cols_present: list,
        ord_cols_present: list,
        cat_cols_present: list,
        col_list: list,
        mi_col_to_idx: dict | None,
    ):
        """Return the weight vectors (w_num, w_ord, w_cat) for the target column.

        With MI available, the weight of each predictor is MI(target, predictor);
        otherwise the weights are uniform. The target itself always gets weight 0.
        Also returns used_uniform (True when MI was not used).
        """
    
        def _uniform_non_target_weights():
            w_num = np.array([0.0 if c == target_col else 1.0 for c in num_cols_present], dtype=float)
            w_ord = np.array([0.0 if c == target_col else 1.0 for c in ord_cols_present], dtype=float)
            w_cat = np.array([0.0 if c == target_col else 1.0 for c in cat_cols_present], dtype=float)
            total = w_num.sum() + w_ord.sum() + w_cat.sum()
            if total > 0:
                w_num /= total
                w_ord /= total
                w_cat /= total
            return w_num, w_ord, w_cat, True

        # --- Uniform weights over the predictors (never the target) ---
        if (
            (not getattr(self.cfg, "use_mi_weights", False))
            or (self.mi_matrix is None)
            or (self.mi_col_order is None)
            or (mi_col_to_idx is None)
        ):
            return _uniform_non_target_weights()
    
        # --- Uniform weights also when the target is not in the MI matrix ---
        if target_col not in mi_col_to_idx:
            return _uniform_non_target_weights()
    
        used_uniform = False
        t_idx = mi_col_to_idx[target_col]
    
        # MI weight of one column (0 if not available)
        def _mi_weight(col_name: str) -> float:
            if col_name == target_col:
                return 0.0
            j = mi_col_to_idx.get(col_name, None)
            if j is None:
                return 0.0
            v = float(self.mi_matrix[t_idx, j])
            if not np.isfinite(v) or v < 0:
                return 0.0
            return v
    
        # --- Weights by type ---
        w_num = np.array([_mi_weight(c) for c in num_cols_present], dtype=float) if num_cols_present else np.zeros((0,), dtype=float)
        w_ord = np.array([_mi_weight(c) for c in ord_cols_present], dtype=float) if ord_cols_present else np.zeros((0,), dtype=float)
        w_cat = np.array([_mi_weight(c) for c in cat_cols_present], dtype=float) if cat_cols_present else np.zeros((0,), dtype=float)
    
        # If all weights are 0 (e.g. too few pairs for MI), use uniform weights
        if (w_num.sum() + w_ord.sum() + w_cat.sum()) <= 0:
            return _uniform_non_target_weights()

        # Normalise by the total, so that total_w_all in _compute_distances_for_rows
        # does not depend on the scale of the MI values and the overlap share used by
        # the low-overlap penalty is computed on a common scale.
        total_w = w_num.sum() + w_ord.sum() + w_cat.sum()
        if total_w > 0:
            w_num = w_num / total_w
            w_ord = w_ord / total_w
            w_cat = w_cat / total_w

        return w_num, w_ord, w_cat, used_uniform