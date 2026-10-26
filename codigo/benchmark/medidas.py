"""Distribution diagnostics, evaluated only AFTER imputation.

The full original column supplies a common numerical scale (population SD).
No statistic in this module is supplied to an imputation algorithm.
"""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

QUANTILES = np.array([.05, .25, .5, .75, .95])


def numerical(reference, estimate, scale):
    a, b = np.asarray(reference, float), np.asarray(estimate, float)
    if not len(a) or not len(b):
        return None
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite distribution input')
    aq, bq = np.quantile(a, QUANTILES), np.quantile(b, QUANTILES)
    sd_a, sd_b = float(a.std(ddof=0)), float(b.std(ddof=0))
    w = float(wasserstein_distance(a, b))
    bias = float(b.mean() - a.mean())
    return dict(n_original=len(a), n_compared=len(b), wasserstein=w,
                wasserstein_sd=w / scale if scale > 0 else None,
                mean_bias=bias, mean_bias_sd=bias / scale if scale > 0 else None,
                sd_original=sd_a, sd_compared=sd_b,
                sd_ratio=sd_b / sd_a if sd_a > 0 else None,
                sd_absolute_error_scaled=abs(sd_b-sd_a)/scale if scale > 0 else None,
                quantiles_original=aq.tolist(), quantiles_compared=bq.tolist(),
                quantile_mae_sd=float(np.abs(aq-bq).mean()/scale) if scale > 0 else None,
                scale_full_original_sd=float(scale))


def categorical(reference, estimate, levels):
    a, b = np.asarray(reference), np.asarray(estimate)
    if not len(a) or not len(b):
        return None
    if not np.isin(a, levels).all() or not np.isin(b, levels).all():
        raise ValueError('Unknown category')
    pa = np.array([(a == v).mean() for v in levels])
    pb = np.array([(b == v).mean() for v in levels])
    return dict(n_original=len(a), n_compared=len(b),
                total_variation=float(.5*np.abs(pa-pb).sum()),
                levels=np.asarray(levels).tolist(),
                proportions_original=pa.tolist(), proportions_compared=pb.tolist())


def correlation_error(original, compared, types):
    """Pairwise complete observations for amputated data; full rows otherwise.

    Original constant columns are excluded from correlation targets. If a
    originally defined pair becomes undefined, report it and do not produce
    a deceptively favourable mean over just the remaining pairs.
    """
    cols = [j for j, t in enumerate(types) if t == 'cont' and np.std(original[:, j]) > 0]
    result = {}
    if len(cols) < 2:
        return {k: dict(mae=None, rmse=None, maximum=None, pairs_original=0,
                        pairs_valid=0, pairs_undefined=0) for k in ('pearson', 'spearman')}
    a, b = pd.DataFrame(original[:, cols]), pd.DataFrame(compared[:, cols])
    ix = np.triu_indices(len(cols), 1)
    for method in ('pearson', 'spearman'):
        ca = a.corr(method=method, min_periods=3).to_numpy()[ix]
        cb = b.corr(method=method, min_periods=3).to_numpy()[ix]
        target = np.isfinite(ca)
        valid = target & np.isfinite(cb)
        bad = int((target & ~valid).sum())
        delta = cb[valid]-ca[valid]
        defined = len(delta) > 0 and bad == 0
        result[method] = dict(mae=float(np.abs(delta).mean()) if defined else None,
            rmse=float(np.sqrt(np.mean(delta**2))) if defined else None,
            maximum=float(np.abs(delta).max()) if defined else None,
            pairs_original=int(target.sum()), pairs_valid=int(valid.sum()), pairs_undefined=bad)
    return result


def distributions(X, Y, mask, types):
    columns = []
    for j, t in enumerate(types):
        m = mask[:, j]
        if t == 'cont':
            scale = float(X[:, j].std(ddof=0))
            full = numerical(X[:, j], Y[:, j], scale)
            hidden = numerical(X[m, j], Y[m, j], scale)
        else:
            levels = np.unique(X[:, j])
            full = categorical(X[:, j], Y[:, j], levels)
            hidden = categorical(X[m, j], Y[m, j], levels)
        columns.append(dict(index=j, type=t, n_missing=int(m.sum()),
                            full=full, missing=hidden))
    return dict(columns=columns, correlations=correlation_error(X, Y, types))


def observed_distributions(X, mask, types):
    columns = []
    for j, t in enumerate(types):
        obs = X[~mask[:, j], j]
        val = (numerical(X[:, j], obs, float(X[:, j].std(ddof=0)))
               if t == 'cont' else categorical(X[:, j], obs, np.unique(X[:, j])))
        columns.append(dict(index=j, type=t, n_observed=len(obs), observed=val))
    incomplete = X.copy()
    incomplete[mask] = np.nan
    return dict(columns=columns, correlations=correlation_error(X, incomplete, types))


def average(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return (float(np.mean(vals)) if vals else None), len(vals)


def summary(measures, scopes=('full', 'missing')):
    out = {}
    for scope in scopes:
        for metric, kind in [('wasserstein_sd', 'cont'), ('total_variation', 'cat'),
                             ('sd_absolute_error_scaled', 'cont'), ('quantile_mae_sd', 'cont')]:
            vals = [(c.get(scope) or {}).get(metric) for c in measures['columns'] if c['type'] == kind]
            value, n = average(vals)
            out[f'{scope}_{metric}'] = value
            out[f'{scope}_{metric}_n_columns'] = n
    for kind, values in measures['correlations'].items():
        for key, value in values.items():
            out[f'{kind}_{key}'] = value
    return out
