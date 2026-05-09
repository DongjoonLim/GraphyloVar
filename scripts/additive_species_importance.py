#!/usr/bin/env python3
"""Additive species importance: each species alone vs all-gap baseline.

For each variant in the sample:
  baseline  = loss when ALL 58 species are masked (all-gap input)
  additive  = loss when ONLY species i is unmasked (all others remain gap)
  delta_i   = baseline_loss - additive_loss   (always >= 0)

This answers "how much information does species i carry in isolation?", which
guarantees all-positive values for every species, including Chimpanzee. It is
complementary to the standard leave-one-out analysis: LOO shows *marginal*
contributions (what each species adds on top of all others), while additive
shows *isolated* contributions (the information each species carries alone).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
# Allow GPU execution for massive inference batches
# os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graphylovar.phylogeny import NAMES, SPECIES, branch_length_to_human
from graphylovar.topmed import (
    extract_batch_examples_from_encoded,
    load_alignment_encoded_cache,
)
from scripts.perturbation_species_importance import (
    GAP_TOKEN,
    COMMON_NAMES,
    CLADE_COLORS,
    load_model,
    load_run_spec,
)
from scripts.subtree_perturbation_importance import (
    ce_sum_and_count,
    infer_nucleotide,
    parse_chromosome_spec,
    resolve_existing_dir,
    species_plot_group,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT_DIR = os.path.join(
    REPO_ROOT,
    "outputs/supervisor_recovery_20260329/interpretability/additive_species",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_tag", default="v3flank16")
    p.add_argument("--samples_per_chrom", type=int, default=5_000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--chromosomes", default="13-22")
    p.add_argument("--compact_dir", default="")
    p.add_argument("--align_cache_dir", default="")
    p.add_argument("--output_dir", default=DEFAULT_OUT_DIR)
    return p.parse_args()


def build_species_specs():
    specs = []
    for ucsc in SPECIES:
        specs.append({
            "ucsc_name": ucsc,
            "common_name": COMMON_NAMES.get(ucsc, ucsc),
            "node_idx": NAMES.index(ucsc),
            "additive_loss_sum": 0.0,
        })
    return specs


def iter_batches(run_spec, chroms, samples_per_chrom, batch_size):
    rng = np.random.default_rng(42)
    compact_dir = run_spec["effective_compact_dir"]
    align_cache = run_spec["effective_align_cache"]
    context = int(run_spec["context"])
    context_flank = int(run_spec["context_flank"])

    for chrom in chroms:
        pos_path = os.path.join(compact_dir, f"positions_graphylovar_topmed_chr{chrom}.npy")
        tgt_path = os.path.join(compact_dir, f"y_graphylovar_topmed_chr{chrom}.npy")
        meta_path = os.path.join(compact_dir, f"metadata_graphylovar_topmed_chr{chrom}.json")
        if not all(os.path.exists(p) for p in [pos_path, tgt_path, meta_path]):
            print(f"  [additive] skipping chr{chrom} (missing files)", flush=True)
            continue

        with open(meta_path) as fh:
            meta = json.load(fh)
        encoded_alignment, _ = load_alignment_encoded_cache(
            alignment_path=meta["alignment_path"],
            cache_dir=align_cache,
            chromosome=chrom,
        )

        positions = np.load(pos_path, mmap_mode="r")
        targets = np.load(tgt_path, mmap_mode="r")
        if 0 < samples_per_chrom < len(positions):
            idx = rng.choice(len(positions), size=samples_per_chrom, replace=False)
            idx.sort()
        else:
            idx = np.arange(len(positions))

        for start in range(0, len(idx), batch_size):
            chosen = idx[start:start + batch_size]
            pos_batch = positions[chosen].astype(np.int64) - 1
            tgt_batch = targets[chosen].astype(np.float32)
            x_batch, valid = extract_batch_examples_from_encoded(
                encoded_alignment=encoded_alignment,
                positions_zero_based=pos_batch,
                context=context,
                context_flank=context_flank,
                mask_indices=None,
            )
            if x_batch.shape[0]:
                yield chrom, x_batch[valid], tgt_batch[valid, :5]


def score_additive(model_wrapper, run_spec, chroms, samples_per_chrom, batch_size):
    specs = build_species_specs()
    n_species = len(specs)

    allgap_sum = 0.0
    n_total = 0
    batch_counter = 0

    for chrom, x_batch, y_batch in iter_batches(run_spec, chroms, samples_per_chrom, batch_size):
        B, S, F = x_batch.shape

        # All-gap baseline: every species row is GAP
        x_allgap = np.full_like(x_batch, GAP_TOKEN)
        gap_pred = infer_nucleotide(model_wrapper, x_allgap)
        gap_sum, n_batch = ce_sum_and_count(gap_pred, y_batch)
        allgap_sum += gap_sum
        n_total += n_batch

        # Sequential per-species evaluation (memory-safe for MHA layers)
        for spec in specs:
            x_add = np.full_like(x_batch, GAP_TOKEN)
            x_add[:, spec["node_idx"], :] = x_batch[:, spec["node_idx"], :]
            pred = infer_nucleotide(model_wrapper, x_add)
            add_sum, _ = ce_sum_and_count(pred, y_batch)
            spec["additive_loss_sum"] += add_sum

        batch_counter += 1
        if batch_counter % 50 == 0:
            print(f"[additive] chr={chrom} examples={n_total} batches={batch_counter}", flush=True)

    return specs, allgap_sum, n_total


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    run_spec = load_run_spec(args.run_tag)
    model_wrapper = load_model(run_spec)
    chroms = parse_chromosome_spec(args.chromosomes)
    probe = chroms[0]
    run_spec["effective_compact_dir"] = resolve_existing_dir(
        args.compact_dir or run_spec["compact_dir"],
        os.path.join(REPO_ROOT, "topmed_compact_full"),
        f"positions_graphylovar_topmed_chr{probe}.npy",
    )
    run_spec["effective_align_cache"] = resolve_existing_dir(
        args.align_cache_dir or run_spec["align_cache"],
        os.path.join(REPO_ROOT, "topmed_alignment_cache"),
        f"alignment_encoded_chr{probe}.npy",
    )

    print(f"[additive] starting: {len(chroms)} chroms × {args.samples_per_chrom} samples/chrom", flush=True)
    specs, allgap_sum, n_total = score_additive(
        model_wrapper, run_spec, chroms, args.samples_per_chrom, args.batch_size
    )

    allgap_mean = allgap_sum / n_total
    print(f"[additive] done. n={n_total} allgap_loss={allgap_mean:.6f}", flush=True)

    # delta_i = allgap_loss - loss_with_species_i  (positive: species reduces loss)
    rows = []
    for spec in specs:
        add_mean = spec["additive_loss_sum"] / n_total
        delta = allgap_mean - add_mean
        rows.append({
            "ucsc_name": spec["ucsc_name"],
            "common_name": spec["common_name"],
            "clade": species_plot_group(spec["common_name"]),
            "branch_length_to_human": float(branch_length_to_human(spec["ucsc_name"])),
            "delta_additive": delta,
            "allgap_loss": allgap_mean,
            "additive_loss": add_mean,
        })

    rows.sort(key=lambda r: -r["delta_additive"])
    print("\nTop 20 species (additive importance):")
    for r in rows[:20]:
        print(f"  {r['common_name']:25s}  {r['delta_additive']:+.6e}  ({r['clade']})")
    print(f"\nMin delta: {min(r['delta_additive'] for r in rows):.6e}")
    print(f"Negative count: {sum(1 for r in rows if r['delta_additive'] < 0)}")

    # Write CSV
    csv_path = os.path.join(args.output_dir, "additive_species_importance_scores.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV -> {csv_path}")

    # Write summary JSON
    summary = {
        "status": "completed",
        "run_tag": run_spec["run_tag"],
        "n_examples": n_total,
        "n_species": len(rows),
        "allgap_baseline_loss": allgap_mean,
        "method": "additive_from_allgap_baseline",
        "interpretation": (
            "delta_additive = allgap_baseline_loss - loss_with_species_i. "
            "Positive values guaranteed: each species reduces loss vs all-gap. "
            "Shows isolated information content, complementary to LOO (marginal) analysis."
        ),
        "top10": rows[:10],
    }
    with open(os.path.join(args.output_dir, "additive_species_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
