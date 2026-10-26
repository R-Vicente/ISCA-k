"""MAR missingness generator with a single, always observed driver column.

The driver is the first column marked continuous in the manifest that is
finite and non-constant; its index is recorded in the dataset manifest and
passed explicitly to ``formal_mar_mask``. Given the driver, the mask depends
only on the observed driver values and on independent logistic noise, so
P(R|X) = P(R|X_obs) for every generated pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class MARInfo:
    n: int
    p: int
    prop: float
    budget: int
    driver_col: int
    beta: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_mar_driver(X, var_types: Sequence[str]) -> int:
    """Resolve the first continuous, finite, non-constant manifest column."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError("X must be a 2D array")
    if len(var_types) != arr.shape[1]:
        raise ValueError("var_types length must equal number of columns")

    for j, t in enumerate(var_types):
        if t != "cont":
            continue
        col = np.asarray(arr[:, j], dtype=float)
        if not np.isfinite(col).all():
            continue
        sd = float(np.std(col, ddof=0))
        if np.isfinite(sd) and sd > 0.0:
            return int(j)
    raise ValueError("formal MAR requires at least one continuous, finite, non-constant driver")


def _validate_driver(X, var_types: Sequence[str], driver_col: int) -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError("X must be a 2D array")
    n, p = arr.shape
    if len(var_types) != p:
        raise ValueError("var_types length must equal number of columns")
    driver_col = int(driver_col)
    if driver_col < 0 or driver_col >= p:
        raise ValueError("driver_col out of bounds")
    if var_types[driver_col] != "cont":
        raise ValueError("MAR driver must be marked 'cont' in the manifest")
    driver = np.asarray(arr[:, driver_col], dtype=float)
    if not np.isfinite(driver).all():
        raise ValueError("MAR driver must contain only finite values")
    driver_sd = float(np.std(driver, ddof=0))
    if not np.isfinite(driver_sd) or driver_sd <= 0.0:
        raise ValueError("MAR driver must have strictly positive finite variance")
    return driver


def formal_mar_mask(
    X,
    var_types: Sequence[str],
    *,
    driver_col: int,
    prop: float = 0.20,
    seed: int = 42,
    beta: float = 1.0,
    return_info: bool = False,
):
    """Generate a MAR mask with a fixed budget of round(prop*n*p) cells.

    ``driver_col`` is the driver recorded in the dataset manifest and is never
    hidden. Each cell of the other columns gets the score
    ``beta * z(driver) + Logistic(0,1)``; every target column first loses its
    highest-scoring cell, and the remaining budget goes to the highest scores
    overall, keeping at least one observed cell per column.
    """
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError("X must be a 2D array")
    n, p = arr.shape
    if n <= 1 or p < 2:
        raise ValueError("formal MAR requires n >= 2 and p >= 2")
    if len(var_types) != p:
        raise ValueError("var_types length must equal number of columns")
    prop = float(prop)
    beta = float(beta)
    seed = int(seed)
    driver_col = int(driver_col)
    if not np.isfinite(prop) or not (0.0 < prop < 1.0):
        raise ValueError("prop must be finite and strictly between 0 and 1")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and strictly positive")

    driver = _validate_driver(arr, var_types, driver_col)
    driver_sd = float(np.std(driver, ddof=0))
    z = (driver - float(np.mean(driver))) / driver_sd

    budget = int(round(prop * n * p))
    target_cols = np.array([j for j in range(p) if j != driver_col], dtype=int)
    n_targets = len(target_cols)
    min_budget = n_targets
    max_budget = n_targets * (n - 1)
    if budget < min_budget:
        raise ValueError("missingness budget too small to mask every target column")
    if budget > max_budget:
        raise ValueError("missingness budget too large while preserving one observed cell per target")

    rng = np.random.default_rng(seed)
    noise = rng.logistic(loc=0.0, scale=1.0, size=(n, n_targets))
    scores = beta * z[:, None] + noise

    mask = np.zeros((n, p), dtype=bool)
    counts = np.zeros(n_targets, dtype=int)

    for k, col in enumerate(target_cols):
        row = int(np.argmax(scores[:, k]))
        mask[row, col] = True
        counts[k] = 1

    remaining = budget - n_targets
    if remaining:
        flat_scores = scores.ravel()
        flat_idx = np.arange(flat_scores.size, dtype=np.int64)
        order = np.lexsort((flat_idx, -flat_scores))
        added = 0
        for flat in order:
            row = int(flat // n_targets)
            k = int(flat % n_targets)
            col = int(target_cols[k])
            if mask[row, col]:
                continue
            if counts[k] >= n - 1:
                continue
            mask[row, col] = True
            counts[k] += 1
            added += 1
            if added == remaining:
                break
        if added != remaining:
            raise RuntimeError("internal invariant failed: unable to fill MAR missingness budget")

    if mask[:, driver_col].any():
        raise RuntimeError("internal invariant failed: MAR driver was masked")
    if int(mask.sum()) != budget:
        raise RuntimeError("internal invariant failed: missingness budget mismatch")
    target_counts = mask[:, target_cols].sum(axis=0)
    if np.any(target_counts <= 0) or np.any(target_counts >= n):
        raise RuntimeError("internal invariant failed: target column degeneracy")
    if np.any(mask.all(axis=1)):
        raise RuntimeError("internal invariant failed: fully missing row generated")

    info = MARInfo(
        n=int(n), p=int(p), prop=prop, budget=budget,
        driver_col=driver_col, beta=beta, seed=seed,
    )
    return (mask, info) if return_info else mask


