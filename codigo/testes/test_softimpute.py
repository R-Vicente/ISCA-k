import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
import softimpute as si  # noqa: E402


def _dados(seed=0, n=120):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    X = np.column_stack([z[:, 0], z[:, 0] + 0.3 * rng.normal(size=n), z[:, 1] * 5 + 10,
                         (z[:, 1] > 0).astype(float), rng.integers(0, 3, n).astype(float)])
    tipos = ["cont", "cont", "cont", "cat", "cat"]
    M = rng.random(X.shape) < 0.2
    for i in np.flatnonzero(M.all(axis=1)):
        M[i, 0] = False
    Xm = X.copy()
    Xm[M] = np.nan
    return X, Xm, M, tipos


def test_observados_intactos_e_categorias_validas():
    X, Xm, M, tipos = _dados()
    Y, audit = si.run_softimpute_tuned(Xm, tipos)
    assert np.isfinite(Y).all()
    assert np.array_equal(Y[~M], X[~M])
    for j in (3, 4):
        assert np.isin(Y[:, j], np.unique(Xm[~M[:, j], j])).all()
    assert audit["cv"]["chosen"] in si.LAM_FRAC_GRID


def test_deterministico():
    _, Xm, _, tipos = _dados(1)
    a, _ = si.run_softimpute_tuned(Xm, tipos)
    b, _ = si.run_softimpute_tuned(Xm, tipos)
    assert np.array_equal(a, b)


def test_lambda0_e_solucao_nula():
    # With lam >= lambda_0 the low-rank solution is zero: numeric cells take the observed mean.
    _, Xm, M, tipos = _dados(2)
    Y, info = si.impute_softimpute(Xm, tipos, lam_frac=1.0, return_info=True)
    assert info["rank"] == 0
    for j in (0, 1, 2):
        assert np.allclose(Y[M[:, j], j], np.nanmean(Xm[:, j]))


def test_ponto_fixo_do_algoritmo_1():
    # At convergence, Z ~ S_lambda(P_Omega(X) + P_Omega_perp(Z)) (Mazumder et al., 2010).
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 3)) @ rng.normal(size=(3, 6)) + 0.1 * rng.normal(size=(80, 6))
    M = rng.random(X.shape) < 0.25
    Xm = X.copy(); Xm[M] = np.nan
    tipos = ["cont"] * 6
    Y, info = si.impute_softimpute(Xm, tipos, lam_frac=0.1, max_iter=100000, eps=1e-14, return_info=True)
    assert info["converged"]
    filled = np.where(M, np.nanmean(Xm, axis=0), Xm)
    m, s = filled.mean(axis=0), filled.std(axis=0)
    Z = (Y - m) / s
    U, d, Vt = np.linalg.svd(Z, full_matrices=False)
    Znew = (U * np.maximum(d - info["lam"], 0)) @ Vt
    assert np.allclose(Znew[M], Z[M], atol=1e-5)


def test_criterio_de_paragem_igual_ao_do_R():
    # ratio = ||Z_old - Z_new||_F^2 / ||Z_old||_F^2 (Frob in softImpute::simpute.svd); with eps = 0 it never stops before max_iter.
    _, Xm, _, tipos = _dados(4)
    _, info = si.impute_softimpute(Xm, tipos, lam_frac=0.1, max_iter=7, eps=0.0, return_info=True)
    assert info["iterations"] == 7 and not info["converged"]
