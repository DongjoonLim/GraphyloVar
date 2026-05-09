#!/usr/bin/env python3
"""Per-species perturbation across ALL 58 leaf placental mammals.

Complementary to subtree_perturbation_importance.py. The canonical script
ablates non-primate clades as subtrees and only primates+tree shrew as
single species. This script ablates EVERY leaf species individually so
non-primate species can be ranked at single-species resolution.

Operates on a sampled subset of held-out variants (default 200K per chromosome,
~2M total) to fit within reasonable GPU 4 wall-time.
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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graphylovar.phylogeny import NAMES, SPECIES, branch_length_to_human
from graphylovar.topmed import (
    extract_batch_examples_from_encoded,
    load_alignment_encoded_cache,
)
from scripts.perturbation_species_importance import (
    CLADE_COLORS,
    CLADES,
    COMMON_NAMES,
    GAP_TOKEN,
    extract_nucleotide_predictions,
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
    "outputs/supervisor_recovery_20260329/interpretability",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-species perturbation for all 58 leaves.")
    parser.add_argument("--run_tag", default="v3flank16")
    parser.add_argument("--samples_per_chrom", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--chromosomes", default="13-22")
    parser.add_argument("--compact_dir", default="")
    parser.add_argument("--align_cache_dir", default="")
    parser.add_argument("--output_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--status_json",
        default=os.path.join(DEFAULT_OUT_DIR, "all58_species_perturbation_status.json"),
    )
    return parser.parse_args()


def write_status(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("last_heartbeat", datetime.now().astimezone().isoformat())
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def iter_batches(run_spec: dict, chroms, samples_per_chrom: int, batch_size: int):
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


def score_all58(model_wrapper, run_spec, chroms, samples_per_chrom, batch_size, status_path):
    species_specs = []
    for ucsc_name in SPECIES:
        species_specs.append({
            "ucsc_name": ucsc_name,
            "common_name": COMMON_NAMES.get(ucsc_name, ucsc_name),
            "node_idx": NAMES.index(ucsc_name),
            "loss_sum": 0.0,
        })

    baseline_sum = 0.0
    n_total = 0
    batch_counter = 0
    last_chrom = None

    for chrom, x_batch, y_batch in iter_batches(run_spec, chroms, samples_per_chrom, batch_size):
        baseline_pred = infer_nucleotide(model_wrapper, x_batch)
        loss_sum, n_batch = ce_sum_and_count(baseline_pred, y_batch)
        baseline_sum += loss_sum
        n_total += n_batch

        for spec in species_specs:
            x_pert = x_batch.copy()
            x_pert[:, spec["node_idx"], :] = GAP_TOKEN
            pred = infer_nucleotide(model_wrapper, x_pert)
            pert_sum, _ = ce_sum_and_count(pred, y_batch)
            spec["loss_sum"] += pert_sum

        batch_counter += 1
        last_chrom = chrom
        write_status(status_path, {
            "status": "in_progress",
            "run_tag": run_spec["run_tag"],
            "processed_examples": n_total,
            "last_chromosome": chrom,
            "batch_counter": batch_counter,
            "samples_per_chrom": samples_per_chrom,
            "heldout_chromosomes": list(chroms),
            "n_species": len(species_specs),
            "masking_policy": "all58_leaf_species_singletons",
        })
        if batch_counter % 50 == 0:
            print(
                f"[all58-perturbation] processed_examples={n_total} last_chr={chrom} batches={batch_counter}",
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
        "n_species": len(species_specs),
        "baseline_loss": baseline_mean,
        "masking_policy": "all58_leaf_species_singletons",
    })
    return species_specs, baseline_mean, n_total


def build_rows(species_specs, baseline_mean, n_total):
    rows = []
    for spec in species_specs:
        pert_mean = float(spec["loss_sum"]) / float(n_total)
        delta = pert_mean - baseline_mean
        rows.append({
            "ucsc_name": spec["ucsc_name"],
            "common_name": spec["common_name"],
            "clade": species_plot_group(spec["common_name"]),
            "branch_length_to_human": float(branch_length_to_human(spec["ucsc_name"])),
            "delta_loss": delta,
            "delta_loss_clipped": max(delta, 0.0),
            "baseline_loss": baseline_mean,
            "perturbed_loss": pert_mean,
        })
    return rows


def write_csv(rows, output_dir):
    csv_path = os.path.join(output_dir, "all58_species_perturbation_scores.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write(
            "ucsc_name,common_name,clade,branch_length_to_human,"
            "delta_loss,delta_loss_clipped,baseline_loss,perturbed_loss\n"
        )
        for row in rows:
            handle.write(
                f"{row['ucsc_name']},{row['common_name']},{row['clade']},"
                f"{row['branch_length_to_human']:.6f},"
                f"{row['delta_loss']:.9f},{row['delta_loss_clipped']:.9f},"
                f"{row['baseline_loss']:.9f},{row['perturbed_loss']:.9f}\n"
            )
    return csv_path


def plot_ranking(rows, output_dir):
    ranking = sorted(rows, key=lambda r: float(r["delta_loss_clipped"]), reverse=True)
    fig, ax = plt.subplots(figsize=(15.0, 6.2))
    values = [float(r["delta_loss_clipped"]) for r in ranking]
    colors = [CLADE_COLORS.get(str(r["clade"]), "#888888") for r in ranking]
    ax.bar(range(len(ranking)), values, color=colors, edgecolor="#173040", alpha=0.95)
    ax.set_xticks(range(len(ranking)))
    ax.set_xticklabels([str(r["common_name"]) for r in ranking], rotation=70, ha="right", fontsize=7)
    ax.set_ylabel(r"Perturbation Importance ($\Delta$ nucleotide cross-entropy)")
    ax.set_title("Per-species perturbation importance : all 58 placental mammals (held-out, sampled)")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    legend_patches = [mpatches.Patch(facecolor=color, label=clade) for clade, color in CLADE_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "all58_species_perturbation_ranking.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_summary(rows, run_spec, chroms, samples_per_chrom, n_examples, baseline_mean, output_dir):
    summary = {
        "status": "current",
        "run_tag": run_spec["run_tag"],
        "checkpoint_path": run_spec["model_path"],
        "heldout_chromosomes": list(chroms),
        "samples_per_chrom": samples_per_chrom,
        "n_examples": n_examples,
        "n_species": len(rows),
        "baseline_loss_mean": baseline_mean,
        "masking_policy": "all58_leaf_species_singletons",
        "interpretation_scope": "causal_perturbation",
        "perturbation_method": (
            "Each of 58 extant placental mammal leaves masked individually "
            "to GAP token (input row replaced with all-GAP); delta nucleotide "
            "cross-entropy on held-out variants used as importance score."
        ),
        "top10": sorted(rows, key=lambda r: float(r["delta_loss_clipped"]), reverse=True)[:10],
    }
    out_path = os.path.join(output_dir, "all58_species_perturbation_summary.json")
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

    species_specs, baseline_mean, n_examples = score_all58(
        model_wrapper, run_spec, chroms, args.samples_per_chrom, args.batch_size, args.status_json
    )
    rows = build_rows(species_specs, baseline_mean, n_examples)

    csv_path = write_csv(rows, args.output_dir)
    fig_path = plot_ranking(rows, args.output_dir)
    summary_path = write_summary(
        rows, run_spec, chroms, args.samples_per_chrom, n_examples, baseline_mean, args.output_dir
    )

    print(json.dumps({
        "csv": csv_path,
        "summary": summary_path,
        "figure": fig_path,
        "status_json": args.status_json,
        "n_examples": n_examples,
    }, indent=2))


if __name__ == "__main__":
    main()
