"""Hashes, JSON writing and dataset snapshots.

SHA-256 of files and arrays (used to check snapshots and masks), atomic JSON
writes, and saving and loading of the dataset snapshots in dados/snapshots.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Mapping

import numpy as np


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr) -> str:
    a = np.asarray(arr)
    # Canonical benchmark matrices/hashes are float64 or bool in C order.
    if a.dtype == bool:
        a = np.ascontiguousarray(a, dtype=np.uint8)
        dtype_tag = b"bool"
    else:
        a = np.ascontiguousarray(a, dtype=np.float64)
        dtype_tag = b"float64"
    h = hashlib.sha256()
    h.update(dtype_tag)
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def mask_sha256(mask) -> str:
    m = np.asarray(mask, dtype=bool)
    return sha256_array(m)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def atomic_write_json(path: str | os.PathLike, obj) -> None:
    """Atomically write JSON, retrying transient Windows sharing/permission locks.

    The payload is fully written and fsynced to ``<path>.tmp`` before replacement.
    On Windows, a short-lived reader (for example antivirus or the search
    indexer) can briefly block ``os.replace``. These sharing violations are not
    data errors, so the write is retried for about 25 s before failing. The .tmp
    file is kept if the final attempt fails.
    """
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, indent=2, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())

    delay = 0.05
    attempts = 60
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except OSError as exc:
            # WinError 5=ACCESS_DENIED, 32=SHARING_VIOLATION, 33=LOCK_VIOLATION.
            # PermissionError on Windows is normally WinError 5.  Do not retry
            # unrelated I/O errors because those may indicate a real filesystem issue.
            winerror = getattr(exc, "winerror", None)
            transient = isinstance(exc, PermissionError) or winerror in {5, 32, 33}
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 1.5, 0.5)


def validate_snapshot_matrix(X, *, allow_nan: bool = False) -> dict:
    a = np.asarray(X, dtype=float)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError("snapshot matrix must be a non-empty 2D numeric array")
    posinf = int(np.isposinf(a).sum())
    neginf = int(np.isneginf(a).sum())
    nan = int(np.isnan(a).sum())
    if posinf or neginf:
        raise ValueError(f"snapshot contains infinity: +inf={posinf}, -inf={neginf}")
    if nan and not allow_nan:
        raise ValueError(f"snapshot contains {nan} NaN values but allow_nan=False")
    return {"nan_count": nan, "posinf_count": posinf, "neginf_count": neginf}


def save_dataset_snapshot(
    snapshot_dir: str | os.PathLike,
    dataset_name: str,
    X,
    names,
    var_types,
    *,
    source: str,
    preprocessing: str,
    mar_driver: int,
    allow_nan: bool = False,
    categorical_mappings: Mapping | None = None,
    loader_source_sha256: str | None = None,
) -> dict:
    X = np.asarray(X, dtype=float)
    if len(names) != X.shape[1] or len(var_types) != X.shape[1]:
        raise ValueError("column names/types length mismatch")
    counts = validate_snapshot_matrix(X, allow_nan=allow_nan)
    if not (0 <= int(mar_driver) < X.shape[1]):
        raise ValueError("invalid MAR driver index")
    if var_types[int(mar_driver)] != "cont":
        raise ValueError("MAR driver must be marked cont in the dataset manifest")
    driver = X[:, int(mar_driver)]
    if not np.isfinite(driver).all() or float(np.std(driver, ddof=0)) <= 0:
        raise ValueError("MAR driver must be finite and have positive variance")

    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    safe = dataset_name.replace(" ", "_").replace("/", "_")
    path = snapshot_dir / f"{safe}.npz"
    np.savez_compressed(
        path,
        X=X,
        names=np.asarray(list(names), dtype=str),
        var_types=np.asarray(list(var_types), dtype=str),
    )
    return {
        "dataset_id": dataset_name,
        "source": source,
        "preprocessing": preprocessing,
        "preprocessing_sha256": hashlib.sha256(preprocessing.encode("utf-8")).hexdigest(),
        "loader_source_sha256": loader_source_sha256,
        "snapshot_file": str(path),
        "snapshot_sha256": sha256_file(path),
        "matrix_sha256": sha256_array(X),
        "n": int(X.shape[0]),
        "p": int(X.shape[1]),
        "column_names": list(names),
        "dtypes": [str(X.dtype)] * X.shape[1],
        "var_types": list(var_types),
        "categorical_mappings": dict(categorical_mappings or {}),
        "allow_nan": bool(allow_nan),
        **counts,
        "mar_driver": int(mar_driver),
        "mar_driver_name": str(names[int(mar_driver)]),
    }


def load_dataset_snapshot(entry: Mapping) -> dict:
    path = entry["snapshot_file"]
    if sha256_file(path) != entry["snapshot_sha256"]:
        raise RuntimeError(f"snapshot file hash mismatch: {entry['dataset_id']}")
    with np.load(path, allow_pickle=False) as z:
        X = np.asarray(z["X"], dtype=float)
        names = [str(x) for x in z["names"].tolist()]
        types = [str(x) for x in z["var_types"].tolist()]
    if sha256_array(X) != entry["matrix_sha256"]:
        raise RuntimeError(f"snapshot matrix hash mismatch: {entry['dataset_id']}")
    validate_snapshot_matrix(X, allow_nan=bool(entry.get("allow_nan", False)))
    return {"X": X, "names": names, "types": types, "manifest": dict(entry)}
