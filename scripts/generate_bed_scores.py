#!/usr/bin/env python3
"""Generate a BED-like TSV with per-variant GraphyloVar, CADD, and GV+CADD z-score ensemble.

Output columns (tab-separated, 0-based BED coordinates):
  chrom  start  end  ref  alt  maf  region  gv_score  cadd_score  gv_cadd_ensemble

The GV+CADD ensemble is the parameter-free z-score combination used in Figure 5:
  1. z-normalize gv_score  over all ~149M variants
  2. z-normalize cadd_score over all rows where cadd is non-NaN
  3. negate the CADD z-score (CADD is oriented higher=more-deleterious; we flip so
     higher = more common, matching GV direction)
  4. ensemble = (z_gv + z_cadd_flipped) / 2  (NaN if either is missing)

Usage:
  python scripts/generate_bed_scores.py \\
    [--out /path/to/output.bed.gz]  \\
    [--csv /path/to/supervisor_heldout_variant_scores_chr13_22.csv.gz]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
from typing import Optional

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_CSV = os.path.join(
    REPO_ROOT,
    "outputs/supervisor_recovery_20260329/ensemble",
    "supervisor_heldout_variant_scores_chr13_22.csv.gz",
)
DEFAULT_OUT = os.path.join(
    REPO_ROOT,
    "outputs/supervisor_recovery_20260329/ensemble",
    "graphylovar_variant_scores.bed.gz",
)

HEADER = "#chrom\tstart\tend\tref\talt\tmaf\tregion\tgv_score\tcadd_score\tgv_cadd_ensemble"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate BED-like score file.")
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--chunk_size", type=int, default=5_000_000,
                   help="Rows to load per pass for z-normalization (default 5M)")
    return p.parse_args()


def _nan(v: str) -> Optional[float]:
    try:
        x = float(v)
        return None if (x != x) else x  # catches nan strings
    except (ValueError, TypeError):
        return None


def compute_znorm_params(csv_path: str) -> tuple[float, float, float, float]:
    """Two-pass: compute mean/std for gv and cadd over the full file."""
    print("Pass 1: computing z-normalization parameters...", flush=True)
    gv_sum = gv_sq = gv_n = 0.0
    cadd_sum = cadd_sq = cadd_n = 0.0

    opener = gzip.open if csv_path.endswith(".gz") else open
    with opener(csv_path, "rt", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            gv = _nan(row.get("graphylovar_combined_score_raw", ""))
            cadd = _nan(row.get("cadd_score_raw", ""))
            if gv is not None:
                gv_sum += gv
                gv_sq += gv * gv
                gv_n += 1
            if cadd is not None:
                cadd_sum += cadd
                cadd_sq += cadd * cadd
                cadd_n += 1
            if (i + 1) % 10_000_000 == 0:
                print(f"  {i+1:,} rows scanned...", flush=True)

    gv_mean = gv_sum / gv_n
    gv_std  = ((gv_sq / gv_n) - gv_mean ** 2) ** 0.5
    cadd_mean = cadd_sum / cadd_n
    cadd_std  = ((cadd_sq / cadd_n) - cadd_mean ** 2) ** 0.5
    print(f"  GV:   mean={gv_mean:.6f}  std={gv_std:.6f}  n={int(gv_n):,}", flush=True)
    print(f"  CADD: mean={cadd_mean:.6f}  std={cadd_std:.6f}  n={int(cadd_n):,}", flush=True)
    return gv_mean, gv_std, cadd_mean, cadd_std


def write_bed(csv_path: str, out_path: str,
              gv_mean: float, gv_std: float,
              cadd_mean: float, cadd_std: float) -> None:
    print("Pass 2: writing BED file...", flush=True)
    opener_in  = gzip.open if csv_path.endswith(".gz") else open
    opener_out = gzip.open if out_path.endswith(".gz") else open

    with opener_in(csv_path, "rt", encoding="utf-8") as fin, \
         opener_out(out_path, "wt", encoding="utf-8") as fout:
        fout.write(HEADER + "\n")
        reader = csv.DictReader(fin)
        written = 0
        for i, row in enumerate(reader):
            chrom = row.get("chrom", "")
            pos0  = row.get("pos0", "")
            ref   = row.get("ref", "")
            alt   = row.get("alt", "")
            maf   = row.get("maf", "")
            region = row.get("region", "")

            try:
                start = int(pos0)
                end   = start + 1
            except (ValueError, TypeError):
                continue

            gv   = _nan(row.get("graphylovar_combined_score_raw", ""))
            cadd = _nan(row.get("cadd_score_raw", ""))

            gv_z   = (gv   - gv_mean)   / gv_std   if gv   is not None else None
            # CADD negated so higher = more common (aligns with GV direction)
            cadd_z = -((cadd - cadd_mean) / cadd_std) if cadd is not None else None

            if gv_z is not None and cadd_z is not None:
                ens = (gv_z + cadd_z) / 2.0
                ens_str = f"{ens:.6f}"
            else:
                ens_str = "nan"

            gv_str   = f"{gv:.6f}"   if gv   is not None else "nan"
            cadd_str = f"{cadd:.6f}" if cadd is not None else "nan"

            fout.write(
                f"{chrom}\t{start}\t{end}\t{ref}\t{alt}\t{maf}\t{region}"
                f"\t{gv_str}\t{cadd_str}\t{ens_str}\n"
            )
            written += 1
            if (i + 1) % 10_000_000 == 0:
                print(f"  {i+1:,} rows processed, {written:,} written...", flush=True)

    print(f"Done. Wrote {written:,} variants to {out_path}", flush=True)


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.csv):
        print(f"ERROR: input CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    gv_mean, gv_std, cadd_mean, cadd_std = compute_znorm_params(args.csv)
    write_bed(args.csv, args.out, gv_mean, gv_std, cadd_mean, cadd_std)


if __name__ == "__main__":
    main()
