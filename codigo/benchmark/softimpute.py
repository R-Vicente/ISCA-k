"""SoftImpute (Mazumder, Hastie and Tibshirani, 2010, Algorithm 1) with lambda chosen on held-out observed cells.

- lambda = lam_frac x lambda_0, where lambda_0 is the largest singular value of
  the standardised matrix with missing entries set to zero (Mazumder et al.,
  2010, sec. 9.4; lambda0() in the R package softImpute);
- lam_frac is chosen in {1/1.5, 1/2, 1/5, 1/10, 1/20, 1/50, 1/100} on about 10%
  of the observed cells, hidden and scored as for k-NN and MICE;
- stopping rule of Algorithm 1: ||Z_new - Z_old||_F^2 / ||Z_old||_F^2 < eps,
  with eps = 1e-5 (thresh in R) and at most 100 iterations (maxit in R);
- initial fill by mean or mode, standardisation (as biScale in R); categorical
  columns are treated as numeric and rounded to the nearest observed code.
"""
import numpy as np

LAM_FRAC_GRID = (1/1.5, 1/2, 1/5, 1/10, 1/20, 1/50, 1/100)
MAX_ITER = 100
EPS = 1e-5
CV_RATE = 0.1
CV_SEED = 20240601   # same as for k-NN and MICE


def _get_unique_cats(X_inc, cat_idx):
    return {j: np.unique(X_inc[~np.isnan(X_inc[:, j]), j]) for j in cat_idx}


def _round_to_categories(X_imp, cat_idx, cat_vals):
    X = X_imp.copy()
    for j in cat_idx:
        cats = cat_vals.get(j)
        if cats is not None and len(cats) > 0:
            X[:, j] = cats[np.argmin(np.abs(X[:, j][:, None] - cats[None, :]), axis=1)]
    return X


def impute_softimpute(X_inc, var_types, lam_frac, max_iter=MAX_ITER, eps=EPS, return_info=False):
    cat_idx = [j for j, t in enumerate(var_types) if t != 'cont']
    cat_vals = _get_unique_cats(X_inc, cat_idx)
    X = X_inc.copy()
    mask = np.isnan(X)
    for j in range(X.shape[1]):
        mj = mask[:, j]
        if not mj.any():
            continue
        if var_types[j] == 'cont':
            X[mj, j] = np.nanmean(X_inc[:, j])
        else:
            uv, uc = np.unique(X_inc[~mj, j], return_counts=True)
            X[mj, j] = uv[np.argmax(uc)]
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Zf = (X - mu) / sd                    # completed matrix (observed + current estimate)
    P = np.where(mask, 0.0, Zf)           # P_Omega(X) in the standardised space
    lam0 = np.linalg.svd(P, compute_uv=False)[0]
    lam = lam_frac * lam0
    Z_old = None
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        U, s, Vt = np.linalg.svd(Zf, full_matrices=False)
        Z_new = (U * np.maximum(s - lam, 0.0)) @ Vt
        if Z_old is not None:
            den = np.sum(Z_old ** 2)
            rel = np.sum((Z_new - Z_old) ** 2) / den if den > 0 else 0.0
            if rel < eps:
                converged = True
                Z_old = Z_new
                break
        Z_old = Z_new
        Zf = np.where(mask, Z_new, Zf)
    Zf = np.where(mask, Z_old, Zf)
    X_imp = Zf * sd + mu
    X_imp[~mask] = X_inc[~mask]
    if cat_idx:
        X_imp = _round_to_categories(X_imp, cat_idx, cat_vals)
        X_imp[~mask] = X_inc[~mask]
    info = {"lam_frac": float(lam_frac), "lam0": float(lam0), "lam": float(lam),
            "iterations": int(it), "converged": bool(converged),
            "rank": int(np.linalg.matrix_rank(Z_old, tol=1e-8)) if Z_old is not None else 0}
    return (X_imp, info) if return_info else X_imp


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
    errs = []
    for j in range(X_imp.shape[1]):
        mj = cv_mask[:, j]
        if not mj.any():
            continue
        t, pr = X_obs_true[mj, j], X_imp[mj, j]
        if var_types[j] == 'cont':
            col = X_obs_true[~np.isnan(X_obs_true[:, j]), j]
            s = np.std(col, ddof=0)
            r = np.sqrt(np.mean((t - pr) ** 2))
            errs.append(r / s if s > 0 else r)
        else:
            errs.append(np.mean(t != pr))
    return float(np.mean(errs)) if errs else np.inf


def run_softimpute_tuned(X_inc, var_types, grid=LAM_FRAC_GRID):
    """Choose lam_frac on held-out observed cells, then impute with the chosen value."""
    rng = np.random.default_rng(CV_SEED)
    cv_mask = _cv_rehide_mask(X_inc, CV_RATE, rng)
    audit = {"grid": [float(g) for g in grid], "cv_rate": CV_RATE, "scores": {}, "candidate_failures": []}
    best, best_err = None, np.inf
    if cv_mask.any():
        X_cv = X_inc.copy(); X_cv[cv_mask] = np.nan
        for g in grid:
            try:
                Xi = impute_softimpute(X_cv, var_types, g)
            except Exception as exc:
                audit["candidate_failures"].append({"value": float(g), "exception": repr(exc)}); continue
            err = _cv_score(X_inc, Xi, cv_mask, var_types)
            audit["scores"][f"{g:.6g}"] = err
            if np.isfinite(err) and err < best_err:
                best, best_err = g, err
    if best is None:
        best = grid[len(grid) // 2]; audit["fallback"] = "sem_cv"
    else:
        audit["fallback"] = None; audit["best_cv_score"] = best_err
    X_imp, info = impute_softimpute(X_inc, var_types, best, return_info=True)
    audit["chosen"] = float(best)
    return X_imp, {"effective_config": {"lam_frac": float(best), "max_iter": MAX_ITER, "eps": EPS, **info},
                   "fallbacks": [], "cv": audit}
