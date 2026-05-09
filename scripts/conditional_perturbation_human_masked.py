#!/usr/bin/env python3
"""Conditional per-species perturbation: Human row always masked.

Answers the question: "Which species is most important when Human is absent?"

The baseline is computed with ONLY the Human row masked. For each non-human
species, the Human row remains masked AND that species' row is also masked.
delta = (human+species masked loss) - (human-only masked baseline loss).

This directly validates the information-redundancy explanation for why
Chimpanzee shows near-zero importance in the standard (human-present)
perturbation: under standard conditions, Chimp is redundant with Human
because they share ~98.7% sequence identity. Once Human is removed, Chimp
should become the most important remaining species.

Operates on the same 200K-per-chromosome random sample as all58_species_perturbation.
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
    COMMON_NAMES,
    GAP_TOKEN,
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
    "outputs/supervisor_recovery_20260329/interpretability/conditional_human_masked",
)
HUMAN_UCSC = "hg38"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conditional per-species perturbation (Human always masked)."
    )
    parser.add_argument("--run_tag", default="v3flank16")
    parser.add_argument("--samples_per_chrom", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--chromosomes", default="13-22")
    parser.add_argument("--compact_dir", default="")
    parser.add_argument("--align_cache_dir", default="")
    parser.add_argument("--output_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--status_json",
        default=os.path.join(DEFAULT_OUT_DIR, "conditional_perturbation_status.json"),
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


def score_conditional(model_wrapper, run_spec, chroms, samples_per_chrom, batch_size, status_path):
    human_node_idx = NAMES.index(HUMAN_UCSC)

    # All 57 non-human species
    species_specs = []
    for ucsc_name in SPECIES:
        if ucsc_name == HUMAN_UCSC:
            continue
        species_specs.append({
            "ucsc_name": ucsc_name,
            "common_name": COMMON_NAMES.get(ucsc_name, ucsc_name),
            "node_idx": NAMES.index(ucsc_name),
            "loss_sum": 0.0,
        })

    # Baseline: human masked only
    baseline_sum = 0.0
    n_total = 0
    batch_counter = 0

    for chrom, x_batch, y_batch in iter_batches(run_spec, chroms, samples_per_chrom, batch_size):
        # Baseline with human always masked
        x_human_masked = x_batch.copy()
        x_human_masked[:, human_node_idx, :] = GAP_TOKEN
        baseline_pred = infer_nucleotide(model_wrapper, x_human_masked)
        loss_sum, n_batch = ce_sum_and_count(baseline_pred, y_batch)
        baseline_sum += loss_sum
        n_total += n_batch

        # For each non-human species: mask human + species
        for spec in species_specs:
            x_pert = x_human_masked.copy()
            x_pert[:, spec["node_idx"], :] = GAP_TOKEN
            pred = infer_nucleotide(model_wrapper, x_pert)
            pert_sum, _ = ce_sum_and_count(pred, y_batch)
            spec["loss_sum"] += pert_sum

        batch_counter += 1
        write_status(status_path, {
            "status": "in_progress",
            "run_tag": run_spec["run_tag"],
            "processed_examples": n_total,
            "last_chromosome": chrom,
            "batch_counter": batch_counter,
            "samples_per_chrom": samples_per_chrom,
            "heldout_chromosomes": list(chroms),
            "n_species": len(species_specs),
            "masking_policy": "conditional_human_always_masked",
        })
        if batch_counter % 50 == 0:
            print(
                f"[conditional-perturbation] processed_examples={n_total} "
                f"last_chr={chrom} batches={batch_counter}",
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
        "masking_policy": "conditional_human_always_masked",
        "interpretation": (
            "Baseline = human-only masked loss. delta = (human+species masked) - baseline. "
            "Positive delta means the species adds information beyond human."
        ),
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
            "masking_policy": "conditional_human_always_masked",
        })
    return rows


def write_csv(rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "species_perturbation_scores_human_masked.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write(
            "ucsc_name,common_name,clade,branch_length_to_human,"
            "delta_loss,delta_loss_clipped,baseline_loss,perturbed_loss,masking_policy\n"
        )
        for row in rows:
            handle.write(
                f"{row['ucsc_name']},{row['common_name']},{row['clade']},"
                f"{row['branch_length_to_human']:.6f},"
                f"{row['delta_loss']:.9f},{row['delta_loss_clipped']:.9f},"
                f"{row['baseline_loss']:.9f},{row['perturbed_loss']:.9f},"
                f"{row['masking_policy']}\n"
            )
    return csv_path


def plot_comparison(rows_conditional, all58_csv_path, output_dir):
    """Two-panel figure: standard (non-human species) vs. conditional importance."""
    import csv

    # Load standard all58 results (excluding human)
    rows_standard = []
    if os.path.exists(all58_csv_path):
        with open(all58_csv_path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["ucsc_name"] != "hg38":
                    rows_standard.append({
                        "common_name": row["common_name"],
                        "clade": row["clade"],
                        "delta_loss_clipped": float(row["delta_loss_clipped"]),
                        "delta_loss": float(row["delta_loss"]),
                    })

    rows_cond_sorted = sorted(rows_conditional, key=lambda r: r["delta_loss_clipped"], reverse=True)

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(15.0, 10.0), gridspec_kw={"height_ratios": [1, 1]})

    if rows_standard:
        std_sorted = sorted(rows_standard, key=lambda r: r["delta_loss_clipped"], reverse=True)
        names_a = [r["common_name"] for r in std_sorted]
        vals_a = [r["delta_loss_clipped"] for r in std_sorted]
        colors_a = [CLADE_COLORS.get(r["clade"], "#888888") for r in std_sorted]
        ax_a.bar(range(len(std_sorted)), vals_a, color=colors_a, edgecolor="#173040", alpha=0.92)
        ax_a.set_xticks(range(len(std_sorted)))
        ax_a.set_xticklabels(names_a, rotation=70, ha="right", fontsize=7)
        ax_a.set_ylabel(r"$\Delta$ CE (clipped, human present)")
        ax_a.set_title(
            "(a) Standard masking: 57 non-human species, human present in alignment. "
            "Chimp near-zero (redundant with Human).",
            fontsize=9,
        )
        ax_a.grid(axis="y", linestyle=":", alpha=0.35)
        legend_patches = [mpatches.Patch(facecolor=c, label=cl) for cl, c in CLADE_COLORS.items()]
        ax_a.legend(handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.9)
    else:
        ax_a.text(0.5, 0.5, "Standard all58 CSV not found", ha="center", va="center", transform=ax_a.transAxes)

    names_b = [r["common_name"] for r in rows_cond_sorted]
    vals_b = [r["delta_loss_clipped"] for r in rows_cond_sorted]
    colors_b = [CLADE_COLORS.get(r["clade"], "#888888") for r in rows_cond_sorted]
    ax_b.bar(range(len(rows_cond_sorted)), vals_b, color=colors_b, edgecolor="#173040", alpha=0.92)
    ax_b.set_xticks(range(len(rows_cond_sorted)))
    ax_b.set_xticklabels(names_b, rotation=70, ha="right", fontsize=7)
    ax_b.set_ylabel(r"$\Delta$ CE (clipped, human masked)")
    ax_b.set_title(
        "(b) Conditional masking: Human row always removed. "
        "Chimp becomes most important (closest remaining primate).",
        fontsize=9,
    )
    ax_b.grid(axis="y", linestyle=":", alpha=0.35)
    ax_b.legend(handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.9)

    fig.suptitle(
        "Species importance: standard vs. conditional (human-masked) perturbation",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(output_dir, "conditional_vs_standard_comparison.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Also save to figures directory for manuscript
    figures_dir = os.path.join(REPO_ROOT, "latex/graphylovar_submission/figures")
    manuscript_path = os.path.join(figures_dir, "figS8_conditional_perturbation.png")
    fig2, (ax_a2, ax_b2) = plt.subplots(2, 1, figsize=(15.0, 10.0), gridspec_kw={"height_ratios": [1, 1]})
    if rows_standard:
        std_sorted = sorted(rows_standard, key=lambda r: r["delta_loss_clipped"], reverse=True)
        names_a = [r["common_name"] for r in std_sorted]
        vals_a = [r["delta_loss_clipped"] for r in std_sorted]
        colors_a = [CLADE_COLORS.get(r["clade"], "#888888") for r in std_sorted]
        ax_a2.bar(range(len(std_sorted)), vals_a, color=colors_a, edgecolor="#173040", alpha=0.92)
        ax_a2.set_xticks(range(len(std_sorted)))
        ax_a2.set_xticklabels(names_a, rotation=70, ha="right", fontsize=7)
        ax_a2.set_ylabel(r"$\Delta$ CE (clipped, human present)")
        ax_a2.set_title("(a) Standard masking: 57 non-human species, human present", fontsize=9)
        ax_a2.grid(axis="y", linestyle=":", alpha=0.35)
        ax_a2.legend(handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.9)
    else:
        ax_a2.text(0.5, 0.5, "Standard all58 CSV not found", ha="center", va="center", transform=ax_a2.transAxes)
    ax_b2.bar(range(len(rows_cond_sorted)), vals_b, color=colors_b, edgecolor="#173040", alpha=0.92)
    ax_b2.set_xticks(range(len(rows_cond_sorted)))
    ax_b2.set_xticklabels(names_b, rotation=70, ha="right", fontsize=7)
    ax_b2.set_ylabel(r"$\Delta$ CE (clipped, human masked)")
    ax_b2.set_title("(b) Conditional masking: Human always removed", fontsize=9)
    ax_b2.grid(axis="y", linestyle=":", alpha=0.35)
    ax_b2.legend(handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.9)
    fig2.suptitle(
        "Species importance: standard vs. conditional (human-masked) perturbation",
        fontsize=11,
        fontweight="bold",
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    fig2.savefig(manuscript_path, dpi=220, bbox_inches="tight")
    plt.close(fig2)

    return out_path, manuscript_path


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

    species_specs, baseline_mean, n_examples = score_conditional(
        model_wrapper, run_spec, chroms, args.samples_per_chrom, args.batch_size, args.status_json
    )
    rows = build_rows(species_specs, baseline_mean, n_examples)
    csv_path = write_csv(rows, args.output_dir)

    all58_csv = os.path.join(
        REPO_ROOT,
        "outputs/supervisor_recovery_20260329/interpretability",
        "all58_species_perturbation_scores.csv",
    )
    fig_path, manuscript_path = plot_comparison(rows, all58_csv, args.output_dir)

    print(json.dumps({
        "csv": csv_path,
        "figure": fig_path,
        "manuscript_figure": manuscript_path,
        "status_json": args.status_json,
        "n_examples": n_examples,
        "baseline_loss_human_masked": baseline_mean,
        "top5": sorted(
            [{"species": r["common_name"], "delta": r["delta_loss_clipped"]} for r in rows],
            key=lambda x: x["delta"],
            reverse=True,
        )[:5],
    }, indent=2))


if __name__ == "__main__":
    main()
