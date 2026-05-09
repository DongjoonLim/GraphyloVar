#!/usr/bin/env python3
"""Position-occlusion saliency across the default v3flank16 context window.

For each column in the 33 bp window, mask that column with the GAP token
across all alignment rows and measure the change in held-out nucleotide
cross-entropy. Produces a single-panel bar chart indexed by offset from
the variant position.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "-1") or "-1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graphylovar.topmed import (
    extract_batch_examples_from_encoded,
    load_alignment_encoded_cache,
)
from scripts.perturbation_species_importance import (
    GAP_TOKEN,
    load_model,
    load_run_spec,
)
from scripts.subtree_perturbation_importance import (
    ce_sum_and_count,
    infer_nucleotide,
    parse_chromosome_spec,
    resolve_existing_dir,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT_DIR = os.path.join(
    REPO_ROOT,
    "outputs/supervisor_recovery_20260329/interpretability",
)
LATEX_FIG_DIR = os.path.join(
    REPO_ROOT, "latex/graphylovar_submission/figures"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Position-occlusion saliency sweep.")
    parser.add_argument("--run_tag", default="v3flank16")
    parser.add_argument("--samples_per_chrom", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--chromosomes", default="13-22")
    parser.add_argument("--compact_dir", default="")
    parser.add_argument("--align_cache_dir", default="")
    parser.add_argument("--output_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--status_json",
        default=os.path.join(DEFAULT_OUT_DIR, "position_occlusion_status.json"),
    )
    return parser.parse_args()


def write_status(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("last_heartbeat", datetime.now().astimezone().isoformat())
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


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
        if not (os.path.exists(pos_path) and os.path.exists(tgt_path) and os.path.exists(meta_path)):
            continue

        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
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


def score_positions(model_wrapper, run_spec, chroms, samples_per_chrom, batch_size, status_path):
    baseline_sum = 0.0
    loss_sums = None
    n_positions = None
    n_total = 0
    batch_counter = 0

    for chrom, x_batch, y_batch in iter_batches(run_spec, chroms, samples_per_chrom, batch_size):
        if loss_sums is None:
            n_positions = int(x_batch.shape[2])
            loss_sums = np.zeros(n_positions, dtype=np.float64)

        baseline_pred = infer_nucleotide(model_wrapper, x_batch)
        loss_sum, n_batch = ce_sum_and_count(baseline_pred, y_batch)
        baseline_sum += loss_sum
        n_total += n_batch

        for p in range(n_positions):
            x_pert = x_batch.copy()
            x_pert[:, :, p] = GAP_TOKEN
            pred = infer_nucleotide(model_wrapper, x_pert)
            pert_sum, _ = ce_sum_and_count(pred, y_batch)
            loss_sums[p] += pert_sum

        batch_counter += 1
        write_status(status_path, {
            "status": "in_progress",
            "run_tag": run_spec["run_tag"],
            "processed_examples": n_total,
            "last_chromosome": chrom,
            "batch_counter": batch_counter,
            "samples_per_chrom": samples_per_chrom,
            "heldout_chromosomes": list(chroms),
            "n_positions": n_positions,
            "masking_policy": "position_occlusion_all_rows",
        })
        if batch_counter % 50 == 0:
            print(
                f"[position-occlusion] processed_examples={n_total} last_chr={chrom} batches={batch_counter}",
                flush=True,
            )

    if n_total == 0:
        raise RuntimeError("No held-out examples were loaded.")

    baseline_mean = baseline_sum / n_total
    write_status(status_path, {
        "status": "completed",
        "run_tag": run_spec["run_tag"],
        "processed_examples": n_total,
        "batch_counter": batch_counter,
        "heldout_chromosomes": list(chroms),
        "samples_per_chrom": samples_per_chrom,
        "n_positions": n_positions,
        "baseline_loss": baseline_mean,
        "masking_policy": "position_occlusion_all_rows",
    })
    return loss_sums, baseline_mean, n_total, n_positions


def build_rows(loss_sums, baseline_mean, n_total, n_positions, context_flank):
    rows = []
    center = context_flank
    for p in range(n_positions):
        pert_mean = float(loss_sums[p]) / float(n_total)
        delta = pert_mean - baseline_mean
        rows.append({
            "position": p,
            "offset_from_variant": p - center,
            "delta_loss": delta,
            "delta_loss_clipped": max(delta, 0.0),
            "baseline_loss": baseline_mean,
            "perturbed_loss": pert_mean,
        })
    return rows


def write_csv(rows, output_dir):
    csv_path = os.path.join(output_dir, "position_occlusion_scores.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("position,offset_from_variant,delta_loss,delta_loss_clipped,baseline_loss,perturbed_loss\n")
        for row in rows:
            handle.write(
                f"{row['position']},{row['offset_from_variant']},"
                f"{row['delta_loss']:.9f},{row['delta_loss_clipped']:.9f},"
                f"{row['baseline_loss']:.9f},{row['perturbed_loss']:.9f}\n"
            )
    return csv_path


def plot_positions(rows, output_dir, latex_dir):
    fig, ax = plt.subplots(figsize=(12.0, 4.5))
    offsets = [r["offset_from_variant"] for r in rows]
    values = [r["delta_loss_clipped"] for r in rows]
    cmap = plt.cm.viridis
    max_abs = max(abs(o) for o in offsets) or 1
    colors = [cmap(1.0 - abs(o) / max_abs) for o in offsets]
    ax.bar(offsets, values, color=colors, edgecolor="#173040", alpha=0.95)
    ax.axvline(0.0, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("Offset from variant position (bp)")
    ax.set_ylabel(r"$\Delta$ nucleotide cross-entropy")
    ax.set_title("Position-occlusion saliency across the 33 bp default input window")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    primary_path = os.path.join(output_dir, "position_occlusion_plot.png")
    fig.savefig(primary_path, dpi=220, bbox_inches="tight")
    os.makedirs(latex_dir, exist_ok=True)
    latex_path = os.path.join(latex_dir, "figS2_position_occlusion.png")
    fig.savefig(latex_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return primary_path, latex_path


def write_summary(rows, run_spec, chroms, samples_per_chrom, n_examples, baseline_mean, n_positions, output_dir):
    summary = {
        "status": "current",
        "run_tag": run_spec["run_tag"],
        "checkpoint_path": run_spec["model_path"],
        "heldout_chromosomes": list(chroms),
        "samples_per_chrom": samples_per_chrom,
        "n_examples": n_examples,
        "n_positions": n_positions,
        "baseline_loss_mean": baseline_mean,
        "masking_policy": "position_occlusion_all_rows",
        "interpretation_scope": "causal_perturbation",
        "perturbation_method": (
            "Each of the context window columns masked individually to GAP "
            "across all alignment rows; delta nucleotide cross-entropy on "
            "held-out variants used as saliency score per position."
        ),
        "top5": sorted(rows, key=lambda r: r["delta_loss_clipped"], reverse=True)[:5],
        "rows": rows,
    }
    out_path = os.path.join(output_dir, "position_occlusion_summary.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return out_path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    run_spec = load_run_spec(args.run_tag)
    model_wrapper = load_model(run_spec)
    chroms = parse_chromosome_spec(args.chromosomes)
    probe_chrom = chroms[0]
    run_spec["effective_compact_dir"] = resolve_existing_dir(
        args.compact_dir or run_spec["compact_dir"],
        os.path.join(REPO_ROOT, "topmed_compact_full"),
        f"positions_graphylovar_topmed_chr{probe_chrom}.npy",
    )
    run_spec["effective_align_cache"] = resolve_existing_dir(
        args.align_cache_dir or run_spec["align_cache"],
        os.path.join(REPO_ROOT, "topmed_alignment_cache"),
        f"alignment_encoded_chr{probe_chrom}.npy",
    )
    context_flank = int(run_spec["context_flank"])

    loss_sums, baseline_mean, n_examples, n_positions = score_positions(
        model_wrapper, run_spec, chroms, args.samples_per_chrom, args.batch_size, args.status_json
    )
    rows = build_rows(loss_sums, baseline_mean, n_examples, n_positions, context_flank)

    csv_path = write_csv(rows, args.output_dir)
    primary_path, latex_path = plot_positions(rows, args.output_dir, LATEX_FIG_DIR)
    summary_path = write_summary(
        rows, run_spec, chroms, args.samples_per_chrom, n_examples, baseline_mean, n_positions, args.output_dir
    )

    print(json.dumps({
        "csv": csv_path,
        "summary": summary_path,
        "figure_primary": primary_path,
        "figure_latex": latex_path,
        "status_json": args.status_json,
        "n_examples": n_examples,
    }, indent=2))


if __name__ == "__main__":
    main()
