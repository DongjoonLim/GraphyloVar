#!/usr/bin/env python3
"""Pairwise DeLong tests for held-out ensemble vs baseline AUC.

Implements the Sun & Xu (2014) fast DeLong algorithm in pure NumPy
(O(n log n) per pair via midrank). For each ensemble defined in
``compute_raw_ensemble_auc.py`` we compare against each individual
baseline (GraphyloVar, CADD-inverted, PhyloP-inverted, PhastCons-inverted)
and report the two-sided p-value plus the AUC difference and its
standard error.

Output: ``pairwise_delong_pvalues.csv`` with columns
``model_a, auc_a, model_b, auc_b, delta_auc, se, z, p_value, n_used``.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from compute_raw_ensemble_auc import (
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
    "pairwise_delong_pvalues.csv",
)

ENSEMBLE_SPECS: list[tuple[str, list[str]]] = [
    ("Ensemble_GV_CADD_inverted", ["GraphyloVar", "CADD"]),
    ("Ensemble_GV_PhastCons_CADD_inverted", ["GraphyloVar", "PhastCons", "CADD"]),
    ("Ensemble_AllAvail", [name for col, name, _ in RAW_SCORE_COLS]),
]


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midrank: tied ranks averaged. O(n log n)."""
    n = len(x)
    j = np.argsort(x, kind="mergesort")
    z = x[j]
    t = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1.0  # 1-indexed midrank
        i = k
    out = np.empty(n, dtype=np.float64)
    out[j] = t
    return out


def fast_delong(predictions_sorted: np.ndarray, label_1_count: int):
    """Sun & Xu (2014) fast DeLong.

    predictions_sorted shape: (k_models, m+n) with positives first.
    Returns (aucs, delongcov) : aucs shape (k,), delongcov shape (k, k).
    """
    m = label_1_count
    n = predictions_sorted.shape[1] - m
    positive_examples = predictions_sorted[:, :m]
    negative_examples = predictions_sorted[:, m:]
    k = predictions_sorted.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if sx.ndim == 0:
        sx = np.array([[float(sx)]])
        sy = np.array([[float(sy)]])
    delongcov = sx / m + sy / n
    return aucs, delongcov


def _build_score(per_model_z: dict[str, np.ndarray], members: list[str]) -> np.ndarray:
    arrays = [per_model_z[m] for m in members if m in per_model_z]
    stack = np.stack(arrays, axis=1)
    n_valid = np.sum(~np.isnan(stack), axis=1)
    return np.where(n_valid > 0, np.nanmean(stack, axis=1), np.nan)


def _delong_pvalue(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray):
    """Two-sided DeLong test for AUC_a - AUC_b on common-valid rows."""
    valid = (~np.isnan(score_a)) & (~np.isnan(score_b))
    yv = y[valid].astype(np.int8)
    a = score_a[valid].astype(np.float64)
    b = score_b[valid].astype(np.float64)
    if len(yv) < 4 or len(np.unique(yv)) < 2:
        return None

    # Reorder so positives come first (DeLong convention).
    order = np.argsort(-yv, kind="mergesort")
    yv = yv[order]
    a = a[order]
    b = b[order]
    m = int(yv.sum())
    if m == 0 or m == len(yv):
        return None

    preds = np.vstack([a, b])
    aucs, cov = fast_delong(preds, m)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
                "delta": diff, "se": 0.0, "z": float("nan"),
                "p": float("nan"), "n_used": int(len(yv))}
    se = math.sqrt(var)
    z = diff / se
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
            "delta": diff, "se": se, "z": z, "p": p, "n_used": int(len(yv))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise DeLong test for AUC.")
    parser.add_argument("--parquet", default=PARQUET_PATH)
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT)
    parser.add_argument("--n_subsample", type=int, default=500_000,
                        help="Sub-sample for tractable midrank cost (default 500K).")
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
    for col, name, flip in RAW_SCORE_COLS:
        arr = data.get(col, np.full_like(y_full, np.nan))[sub_idx]
        z = z_normalize(arr)
        if flip:
            z = np.where(~np.isnan(z), -z, np.nan)
        per_model_z[name] = z

    baseline_models = [
        ("GraphyloVar", per_model_z["GraphyloVar"]),
        ("CADD_inverted", per_model_z["CADD"]),
        ("PhyloP_inverted", per_model_z["PhyloP"]),
        ("PhastCons_inverted", per_model_z["PhastCons"]),
    ]

    rows: list[dict] = []
    print("\n--- Pairwise DeLong (ensemble vs baseline) ---", flush=True)
    for ens_name, members in ENSEMBLE_SPECS:
        if not all(m in per_model_z for m in members):
            continue
        ens_score = _build_score(per_model_z, members)
        for base_name, base_score in baseline_models:
            res = _delong_pvalue(y, ens_score, base_score)
            if res is None:
                continue
            print(
                f"  {ens_name:<45s} vs {base_name:<22s}  "
                f"AUC_a={res['auc_a']:.4f}  AUC_b={res['auc_b']:.4f}  "
                f"Δ={res['delta']:+.4f}  SE={res['se']:.4f}  "
                f"z={res['z']:+.2f}  p={res['p']:.3e}  n={res['n_used']:,}",
                flush=True,
            )
            rows.append({
                "model_a": ens_name,
                "auc_a": f"{res['auc_a']:.6f}",
                "model_b": base_name,
                "auc_b": f"{res['auc_b']:.6f}",
                "delta_auc": f"{res['delta']:+.6f}",
                "se": f"{res['se']:.6f}",
                "z": f"{res['z']:+.4f}",
                "p_value": f"{res['p']:.6e}",
                "n_used": str(res["n_used"]),
            })

    fieldnames = ["model_a", "auc_a", "model_b", "auc_b", "delta_auc", "se", "z", "p_value", "n_used"]
    with open(args.output_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote {len(rows)} rows to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
