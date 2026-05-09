#!/usr/bin/env python3
"""Bootstrap 95% CIs for ensemble AUC table.

Reproduces the z-normalized ensemble construction from
``compute_raw_ensemble_auc.py``, then on a stratified sub-sample of the
held-out parquet draws B bootstrap resamples and reports the
2.5/50/97.5 percentile AUC for each ensemble + each baseline.

Output: ``ensemble_auc_ci_table.csv`` with columns
``model, mean_auc, ci_lo, ci_hi, n_bootstrap, n_subsample, auc_source``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from compute_raw_ensemble_auc import (
    AUC_TABLE_PATH,
    LABEL_COL,
    PARQUET_PATH,
    RAW_SCORE_COLS,
    load_parquet_columns,
    z_normalize,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUTPUT = os.path.join(
    REPO_ROOT,
    "outputs/supervisor_recovery_20260329/ensemble",
    "ensemble_auc_ci_table.csv",
)

ENSEMBLE_SPECS: list[tuple[str, list[str]]] = [
    ("Ensemble_GV", ["GraphyloVar"]),
    ("Ensemble_GV_PhastCons", ["GraphyloVar", "PhastCons"]),
    ("Ensemble_GV_PhyloP", ["GraphyloVar", "PhyloP"]),
    ("Ensemble_GV_CADD_inverted", ["GraphyloVar", "CADD"]),
    ("Ensemble_GV_PhastCons_CADD_inverted", ["GraphyloVar", "PhastCons", "CADD"]),
    ("Ensemble_AllAvail", [name for col, name, _ in RAW_SCORE_COLS]),
]


def _bootstrap_auc(
    y: np.ndarray,
    score: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> Optional[tuple[float, float, float, float]]:
    from sklearn.metrics import roc_auc_score

    valid = ~np.isnan(score)
    yv = y[valid].astype(np.int8)
    sv = score[valid].astype(np.float32)
    if len(sv) < 2 or len(np.unique(yv)) < 2:
        return None

    n = len(sv)
    pos_idx = np.where(yv == 1)[0]
    neg_idx = np.where(yv == 0)[0]
    if len(pos_idx) < 1 or len(neg_idx) < 1:
        return None

    aucs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pi = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        ni = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pi, ni])
        aucs[b] = roc_auc_score(yv[idx], sv[idx])

    return (
        float(np.mean(aucs)),
        float(np.percentile(aucs, 2.5)),
        float(np.percentile(aucs, 97.5)),
        float(n),
    )


def _build_ensemble_score(per_model_z: dict[str, np.ndarray], members: list[str]) -> np.ndarray:
    arrays = [per_model_z[m] for m in members if m in per_model_z]
    stack = np.stack(arrays, axis=1)
    n_valid = np.sum(~np.isnan(stack), axis=1)
    return np.where(n_valid > 0, np.nanmean(stack, axis=1), np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap 95% CIs for ensemble AUC.")
    parser.add_argument("--parquet", default=PARQUET_PATH)
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT)
    parser.add_argument("--n_subsample", type=int, default=200_000,
                        help="Sub-sample size for bootstrap (default 200K : tighter CIs, fits in ~40 min).")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.parquet):
        print(f"ERROR: parquet not found at {args.parquet}", file=sys.stderr)
        sys.exit(1)

    raw_cols = [col for col, _, _ in RAW_SCORE_COLS]
    needed = [LABEL_COL] + raw_cols
    data = load_parquet_columns(args.parquet, needed)
    y_full = data[LABEL_COL].astype(np.float32)
    n_full = len(y_full)

    rng = np.random.default_rng(args.seed)
    if args.n_subsample > 0 and args.n_subsample < n_full:
        sub_idx = rng.choice(n_full, size=args.n_subsample, replace=False)
        sub_idx.sort()
    else:
        sub_idx = np.arange(n_full)
    y = y_full[sub_idx]
    print(f"Sub-sampled {len(sub_idx):,} of {n_full:,} rows.", flush=True)

    per_model_z: dict[str, np.ndarray] = {}
    rows: list[dict] = []

    print("\n--- Per-baseline bootstrap ---", flush=True)
    for col, name, flip in RAW_SCORE_COLS:
        arr = data.get(col, np.full_like(y_full, np.nan))[sub_idx]
        z = z_normalize(arr)
        if flip:
            z = np.where(~np.isnan(z), -z, np.nan)
        per_model_z[name] = z

        result = _bootstrap_auc(y, arr if not flip else -arr, rng, args.n_bootstrap)
        if result is None:
            continue
        mean, lo, hi, nv = result
        display = f"{name}_inverted" if flip else name
        print(f"  {display:<28s}  mean={mean:.4f}  CI=[{lo:.4f}, {hi:.4f}]  n_valid={int(nv):,}", flush=True)
        rows.append({
            "model": display,
            "mean_auc": f"{mean:.6f}",
            "ci_lo": f"{lo:.6f}",
            "ci_hi": f"{hi:.6f}",
            "n_bootstrap": str(args.n_bootstrap),
            "n_subsample": str(int(nv)),
            "auc_source": "raw_z_inverted" if flip else "raw_z",
        })

    print("\n--- Ensemble bootstrap ---", flush=True)
    for ens_name, members in ENSEMBLE_SPECS:
        if not all(m in per_model_z for m in members):
            continue
        score = _build_ensemble_score(per_model_z, members)
        result = _bootstrap_auc(y, score, rng, args.n_bootstrap)
        if result is None:
            continue
        mean, lo, hi, nv = result
        print(f"  {ens_name:<45s}  mean={mean:.4f}  CI=[{lo:.4f}, {hi:.4f}]  n_valid={int(nv):,}", flush=True)
        rows.append({
            "model": ens_name,
            "mean_auc": f"{mean:.6f}",
            "ci_lo": f"{lo:.6f}",
            "ci_hi": f"{hi:.6f}",
            "n_bootstrap": str(args.n_bootstrap),
            "n_subsample": str(int(nv)),
            "auc_source": "raw_z_ensemble",
        })

    fieldnames = ["model", "mean_auc", "ci_lo", "ci_hi", "n_bootstrap", "n_subsample", "auc_source"]
    with open(args.output_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote {len(rows)} rows to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
