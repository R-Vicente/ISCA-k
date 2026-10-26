"""Blockwise missingness generator.

Two blocks of cells, an exact global budget of round(prop*n*p) cells, disjoint
row groups, and no fully missing rows or columns.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BlockwiseInfo:
    n: int
    p: int
    prop: float
    budget: int
    block_width: int
    block_budgets: tuple[int, int]
    block_columns: tuple[tuple[int, ...], tuple[int, ...]]
    block_rows: tuple[tuple[int, ...], tuple[int, ...]]
    partial_sizes: tuple[int, int]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_shape_and_prop(n: int, p: int, prop: float) -> None:
    if n <= 0:
        raise ValueError("n must be positive")
    if p < 3:
        raise ValueError("calibrated blockwise requires p >= 3")
    if not np.isfinite(prop) or not (0.0 < prop < 1.0):
        raise ValueError("prop must be finite and strictly between 0 and 1")


def calibrated_blockwise_mask(
    n: int,
    p: int,
    *,
    prop: float = 0.20,
    seed: int = 42,
    return_info: bool = False,
):
    """Generate a calibrated two-block missingness mask.

    The total missing-cell budget is exactly ``round(prop*n*p)``. Columns are
    permuted once; each block uses ``ceil(p/2)`` consecutive positions in that
    permutation, with wrap-around for block 2. Row groups are disjoint.
    """
    n = int(n)
    p = int(p)
    prop = float(prop)
    seed = int(seed)
    _validate_shape_and_prop(n, p, prop)

    budget = int(round(prop * n * p))
    if budget <= 0 or budget >= n * p:
        raise ValueError("rounded missingness budget must be between 1 and n*p-1")

    width = int(np.ceil(p / 2.0))
    # width < p for every p >= 3, so a full block can never erase a whole row.
    if width >= p:
        raise ValueError("block width must remain strictly smaller than p")

    b1 = budget // 2
    b2 = budget - b1
    block_budgets = (b1, b2)

    needed_rows = []
    qr = []
    for b in block_budgets:
        q, r = divmod(b, width)
        needed_rows.append(q + int(r > 0))
        qr.append((q, r))
    if sum(needed_rows) > n:
        raise ValueError("not enough rows to form two disjoint calibrated blocks")

    rng = np.random.default_rng(seed)
    col_perm = rng.permutation(p)
    c1 = col_perm[np.arange(width)]
    c2 = col_perm[(np.arange(width) + width) % p]

    # With width=ceil(p/2), both block column sets together must cover all p columns.
    if len(set(c1.tolist()) | set(c2.tolist())) != p:
        raise RuntimeError("internal invariant failed: block columns do not cover all variables")

    row_perm = rng.permutation(n)
    mask = np.zeros((n, p), dtype=bool)
    block_rows: list[tuple[int, ...]] = []
    partial_sizes: list[int] = []
    cursor = 0

    for cols, (q, r), rows_needed in zip((c1, c2), qr, needed_rows):
        rows = row_perm[cursor : cursor + rows_needed]
        cursor += rows_needed
        full_rows = rows[:q]
        if q:
            mask[np.ix_(full_rows, cols)] = True
        if r:
            partial_row = int(rows[q])
            mask[partial_row, cols[:r]] = True
        block_rows.append(tuple(int(x) for x in rows))
        partial_sizes.append(int(r))

    if int(mask.sum()) != budget:
        raise RuntimeError("internal invariant failed: missingness budget mismatch")
    if np.any(mask.all(axis=1)):
        raise RuntimeError("internal invariant failed: fully missing row generated")
    if np.any(mask.all(axis=0)):
        raise RuntimeError("internal invariant failed: fully missing column generated")

    info = BlockwiseInfo(
        n=n,
        p=p,
        prop=prop,
        budget=budget,
        block_width=width,
        block_budgets=block_budgets,
        block_columns=(tuple(int(x) for x in c1), tuple(int(x) for x in c2)),
        block_rows=(block_rows[0], block_rows[1]),
        partial_sizes=(partial_sizes[0], partial_sizes[1]),
        seed=seed,
    )
    return (mask, info) if return_info else mask


