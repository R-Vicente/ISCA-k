"""Generation and storage of the missingness masks (75 per dataset and condition, seeds 42 to 116)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from harness_audit import atomic_write_json, mask_sha256, sha256_file
from missingness_blockwise import calibrated_blockwise_mask
from missingness_mar import formal_mar_mask

BASE_SEED = 42
N_REPS = 75

CONDITIONS = (
    {"id": "mcar_05", "kind": "mcar", "prop": 0.05, "generator": "uniform_mcar"},
    {"id": "mcar_10", "kind": "mcar", "prop": 0.10, "generator": "uniform_mcar"},
    {"id": "mcar_20", "kind": "mcar", "prop": 0.20, "generator": "uniform_mcar"},
    {"id": "mcar_30", "kind": "mcar", "prop": 0.30, "generator": "uniform_mcar"},
    {"id": "mcar_40", "kind": "mcar", "prop": 0.40, "generator": "uniform_mcar"},
    {"id": "mcar_50", "kind": "mcar", "prop": 0.50, "generator": "uniform_mcar"},
    {"id": "blockwise_20", "kind": "blockwise", "prop": 0.20, "generator": "blockwise"},
    {"id": "mar_20", "kind": "mar", "prop": 0.20, "generator": "mar"},
)

CONDITION_BY_ID = {c["id"]: c for c in CONDITIONS}


def uniform_mcar_mask(X, prop: float, seed: int) -> np.ndarray:
    """MCAR mask: each cell is hidden with probability prop; a row left fully hidden gets one random cell back."""
    X = np.asarray(X)
    rng = np.random.default_rng(int(seed))
    mask = rng.random(X.shape) < float(prop)
    for i in range(X.shape[0]):
        if mask[i].all():
            mask[i, rng.integers(X.shape[1])] = False
    return mask


def generate_mask(X, var_types: Sequence[str], condition: Mapping, seed: int, *, mar_driver: int | None = None):
    cid = condition["id"]
    prop = float(condition["prop"])
    if condition["kind"] == "mcar":
        return uniform_mcar_mask(X, prop, seed), {"generator": "uniform_mcar"}
    if condition["kind"] == "blockwise":
        mask, info = calibrated_blockwise_mask(
            np.asarray(X).shape[0], np.asarray(X).shape[1], prop=prop, seed=int(seed), return_info=True
        )
        return mask, {"generator": "blockwise", "generator_info": info.to_dict()}
    if condition["kind"] == "mar":
        if mar_driver is None:
            raise ValueError("MAR masks require the mar_driver recorded in the dataset manifest")
        mask, info = formal_mar_mask(
            X, var_types, driver_col=int(mar_driver), prop=prop, seed=int(seed), beta=1.0, return_info=True
        )
        if int(info.driver_col) != int(mar_driver):
            raise RuntimeError(
                f"MAR driver mismatch: dataset manifest={mar_driver}, generator={info.driver_col}"
            )
        return mask, {"generator": "mar", "generator_info": info.to_dict()}
    raise KeyError(f"unknown condition: {cid}")


def _mask_record(dataset_id, dataset_sha256, condition, replicate, seed, mask, extra):
    mask = np.asarray(mask, dtype=bool)
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    rec = {
        "dataset_id": dataset_id,
        "dataset_sha256": dataset_sha256,
        "condition_id": condition["id"],
        "generator_name": condition["generator"],
        "generator_version": "1.0",
        "parameters": {"prop": float(condition["prop"]), "beta": 1.0 if condition["kind"] == "mar" else None},
        "replicate": int(replicate),
        "seed": int(seed),
        "shape": [int(mask.shape[0]), int(mask.shape[1])],
        "missing_count": int(mask.sum()),
        "effective_rate": float(mask.mean()),
        "mask_sha256": mask_sha256(mask),
        "min_missing_per_row": int(row_counts.min()),
        "max_missing_per_row": int(row_counts.max()),
        "min_missing_per_col": int(col_counts.min()),
        "max_missing_per_col": int(col_counts.max()),
        "row_all_missing": bool(mask.all(axis=1).any()),
        "col_all_missing": bool(mask.all(axis=0).any()),
    }
    if condition["kind"] == "mar":
        rec["mar_driver"] = int(extra["generator_info"]["driver_col"])
    return rec


def create_mask_bundle(
    X,
    var_types,
    *,
    dataset_id: str,
    dataset_sha256: str,
    mar_driver: int,
    condition_id: str,
    out_dir: str | os.PathLike,
    n_reps: int = N_REPS,
    base_seed: int = BASE_SEED,
):
    condition = CONDITION_BY_ID[condition_id]
    masks = []
    records = []
    for r in range(int(n_reps)):
        seed = int(base_seed) + r
        mask, extra = generate_mask(X, var_types, condition, seed, mar_driver=mar_driver)
        masks.append(np.asarray(mask, dtype=bool))
        records.append(_mask_record(dataset_id, dataset_sha256, condition, r, seed, mask, extra))

    masks_arr = np.stack(masks, axis=0)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = dataset_id.replace(" ", "_").replace("/", "_")
    npz_path = out_dir / f"{safe}__{condition_id}__masks.npz"
    np.savez_compressed(npz_path, masks=masks_arr)
    manifest = {
        "dataset_id": dataset_id,
        "dataset_sha256": dataset_sha256,
        "condition": dict(condition),
        "n_reps": int(n_reps),
        "base_seed": int(base_seed),
        "mask_bundle_file": str(npz_path),
        "mask_bundle_sha256": sha256_file(npz_path),
        "records": records,
    }
    manifest_path = out_dir / f"{safe}__{condition_id}__manifest.json"
    atomic_write_json(manifest_path, manifest)
    return masks_arr, manifest, str(manifest_path)


def load_mask_bundle(manifest: Mapping):
    path = manifest["mask_bundle_file"]
    if sha256_file(path) != manifest["mask_bundle_sha256"]:
        raise RuntimeError("mask bundle file hash mismatch")
    with np.load(path, allow_pickle=False) as z:
        masks = np.asarray(z["masks"], dtype=bool)
    if len(masks) != len(manifest["records"]):
        raise RuntimeError("mask bundle/manifest replicate count mismatch")
    for r, rec in enumerate(manifest["records"]):
        if mask_sha256(masks[r]) != rec["mask_sha256"]:
            raise RuntimeError(f"mask hash mismatch at replicate {r}")
    return masks


def ensure_mask_bundle(
    X, var_types, *, dataset_id: str, dataset_sha256: str, mar_driver: int,
    condition_id: str, out_dir: str | os.PathLike, n_reps: int = N_REPS, base_seed: int = BASE_SEED,
):
    out_dir = Path(out_dir)
    safe = dataset_id.replace(" ", "_").replace("/", "_")
    manifest_path = out_dir / f"{safe}__{condition_id}__manifest.json"
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = CONDITION_BY_ID[condition_id]
        if manifest.get("dataset_sha256") != dataset_sha256:
            raise RuntimeError("existing mask bundle dataset hash mismatch")
        if manifest.get("condition", {}).get("id") != condition_id:
            raise RuntimeError("existing mask bundle condition mismatch")
        if int(manifest.get("n_reps", -1)) != int(n_reps) or int(manifest.get("base_seed", -1)) != int(base_seed):
            raise RuntimeError("existing mask bundle replicate/seed contract mismatch")
        if manifest.get("condition", {}).get("generator") != expected["generator"]:
            raise RuntimeError("existing mask bundle generator mismatch")
        masks = load_mask_bundle(manifest)
        return masks, manifest, str(manifest_path)
    return create_mask_bundle(
        X, var_types, dataset_id=dataset_id, dataset_sha256=dataset_sha256,
        mar_driver=mar_driver, condition_id=condition_id, out_dir=out_dir,
        n_reps=n_reps, base_seed=base_seed,
    )
