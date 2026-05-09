#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "-1") or "-1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.stats import linregress

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graphylovar.phylogeny import ANCESTORS, EDGES, NAMES, SPECIES, branch_length_to_human
from graphylovar.topmed import extract_batch_examples_from_encoded, load_alignment_encoded_cache
from scripts.perturbation_species_importance import (
    CLADE_COLORS,
    CLADES,
    COMMON_NAMES,
    GAP_TOKEN,
    extract_nucleotide_predictions,
    load_model,
    load_run_spec,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(REPO_ROOT, "figures")
DEFAULT_TEST_CHROMS = tuple(range(13, 23))

NON_PRIMATE_GROUPS = {
    "Glires": ["speTri2", "jacJac1", "micOch1", "criGri1", "mesAur1", "mm10", "rn6", "hetGla2", "cavPor3", "chiLan1", "octDeg1", "oryCun2", "ochPri3"],
    "Cetartiodactyla": ["susScr3", "vicPac2", "camFer1", "turTru2", "orcOrc1", "panHod1", "bosTau8", "oviAri3", "capHir1"],
    "Carnivora + Perissodactyla": ["equCab2", "cerSim1", "felCat8", "canFam3", "musFur1", "ailMel1", "odoRosDiv1", "lepWed1"],
    "Chiroptera": ["pteAle1", "pteVam1", "eptFus1", "myoDav1", "myoLuc2"],
    "Eulipotyphla": ["eriEur2", "sorAra2", "conCri1"],
    "Afrotheria + Xenarthra": ["loxAfr3", "eleEdw1", "triMan1", "chrAsi1", "echTel2", "oryAfe1", "dasNov3"],
}

PRIMATE_AND_TREE_SHREW_SPECIES = [
    "hg38",
    "panTro4",
    "gorGor3",
    "ponAbe2",
    "nomLeu3",
    "rheMac3",
    "macFas5",
    "papAnu2",
    "chlSab2",
    "calJac3",
    "saiBol1",
    "otoGar3",
    "tupChi1",
]

SPECIES_PLOT_COLORS = {
    **CLADE_COLORS,
    "Tree shrew": "#6b6b6b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid held-out perturbation for GraphyloVar.")
    parser.add_argument("--run_tag", default="v3flank16")
    parser.add_argument("--samples_per_chrom", type=int, default=0, help="0 means all held-out rows.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chromosomes", default="13-22")
    parser.add_argument("--compact_dir", default="", help="Optional override for the compact example directory.")
    parser.add_argument("--align_cache_dir", default="", help="Optional override for the alignment cache directory.")
    parser.add_argument("--output_dir", default=OUT_DIR)
    parser.add_argument("--status_json", default="")
    return parser.parse_args()


def parse_chromosome_spec(spec: str) -> tuple[int, ...]:
    chroms: list[int] = []
    for chunk in str(spec).split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            chroms.extend(range(int(start), int(end) + 1))
        else:
            chroms.append(int(token))
    return tuple(sorted(dict.fromkeys(chroms))) or DEFAULT_TEST_CHROMS


def resolve_existing_dir(preferred: str, fallback: str, probe_name: str) -> str:
    if preferred and os.path.exists(os.path.join(preferred, probe_name)):
        return preferred
    if os.path.exists(os.path.join(fallback, probe_name)):
        return fallback
    return preferred


def rooted_descendants(root: str) -> dict[str, set[str]]:
    adj: dict[str, list[str]] = defaultdict(list)
    for a, b in EDGES:
        adj[a].append(b)
        adj[b].append(a)

    parent = {root: None}
    order = [root]
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for nxt in adj[node]:
            if nxt in parent:
                continue
            parent[nxt] = node
            order.append(nxt)
            queue.append(nxt)

    descendants: dict[str, set[str]] = {}
    for node in reversed(order):
        if node in SPECIES:
            descendants[node] = {node}
        else:
            leaves = set()
            for nxt in adj[node]:
                if parent.get(nxt) == node:
                    leaves |= descendants[nxt]
            descendants[node] = leaves
    return descendants


def subtree_indices(group_species: list[str], descendants: dict[str, set[str]]) -> tuple[list[int], list[str], list[str]]:
    leaf_set = set(group_species)
    internal = [name for name in ANCESTORS if descendants.get(name) and descendants[name].issubset(leaf_set)]
    node_names = sorted(leaf_set) + sorted(internal)
    return [NAMES.index(name) for name in node_names], sorted(leaf_set), sorted(internal)


def species_plot_group(common_name: str) -> str:
    if common_name == "Tree shrew":
        return "Tree shrew"
    for clade, members in CLADES.items():
        if common_name in members:
            return clade
    return "Other"


def iter_batches(run_spec: dict, chroms: tuple[int, ...], samples_per_chrom: int, batch_size: int):
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
        if samples_per_chrom > 0 and samples_per_chrom < len(positions):
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


def infer_nucleotide(model_wrapper: dict, x: np.ndarray) -> np.ndarray:
    tensor = tf.convert_to_tensor(x, dtype=tf.uint8)
    if model_wrapper["mode"] == "savedmodel-signature":
        preds = model_wrapper["callable"](**{model_wrapper["input_key"]: tensor})
    else:
        preds = model_wrapper["callable"](tensor, training=False)
    return extract_nucleotide_predictions(preds)


def ce_sum_and_count(pred: np.ndarray, target: np.ndarray) -> tuple[float, int]:
    pred = np.clip(pred, 1e-7, 1.0)
    losses = -np.sum(target * np.log(pred), axis=-1)
    return float(np.sum(losses)), int(losses.shape[0])


def write_status(path: str, payload: dict) -> None:
    from datetime import datetime

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    if not payload.get("last_heartbeat"):
        payload["last_heartbeat"] = datetime.now().astimezone().isoformat()
    if payload.get("current_shard_path") is None:
        payload["current_shard_path"] = ""
    payload.setdefault("current_shard_path", "")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_mask_specs() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    descendants = rooted_descendants(ANCESTORS[-1])
    subtree_specs = []
    for group_name, species in NON_PRIMATE_GROUPS.items():
        idxs, leaf_species, internal_nodes = subtree_indices(species, descendants)
        subtree_specs.append(
            {
                "kind": "subtree",
                "group_name": group_name,
                "idxs": idxs,
                "leaf_species": leaf_species,
                "internal_nodes": internal_nodes,
                "loss_sum": 0.0,
            }
        )
    species_specs = []
    for ucsc_name in PRIMATE_AND_TREE_SHREW_SPECIES:
        species_specs.append(
            {
                "kind": "species",
                "ucsc_name": ucsc_name,
                "common_name": COMMON_NAMES.get(ucsc_name, ucsc_name),
                "idxs": [NAMES.index(ucsc_name)],
                "loss_sum": 0.0,
            }
        )
    return subtree_specs, species_specs


def score_masks_streaming(
    model_wrapper: dict,
    run_spec: dict,
    chroms: tuple[int, ...],
    samples_per_chrom: int,
    batch_size: int,
    status_path: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], float, int]:
    subtree_specs, species_specs = build_mask_specs()
    baseline_sum = 0.0
    n_total = 0
    batch_counter = 0
    current_mask_name = ""

    for chrom, x_batch, y_batch in iter_batches(run_spec, chroms, samples_per_chrom, batch_size):
        baseline_pred = infer_nucleotide(model_wrapper, x_batch)
        loss_sum, n_batch = ce_sum_and_count(baseline_pred, y_batch)
        baseline_sum += loss_sum
        n_total += n_batch

        for spec in subtree_specs + species_specs:
            x_perturbed = x_batch.copy()
            x_perturbed[:, spec["idxs"], :] = GAP_TOKEN
            pred = infer_nucleotide(model_wrapper, x_perturbed)
            pert_sum, _ = ce_sum_and_count(pred, y_batch)
            spec["loss_sum"] += pert_sum
            current_mask_name = str(spec.get("group_name", spec.get("common_name", "")))

        batch_counter += 1
        if status_path:
            write_status(
                status_path,
                {
                    "status": "in_progress",
                    "run_tag": run_spec["run_tag"],
                    "processed_examples": n_total,
                    "last_chromosome": chrom,
                    "batch_counter": batch_counter,
                    "samples_per_chrom": samples_per_chrom,
                    "heldout_chromosomes": list(chroms),
                    "current_mask_name": current_mask_name,
                    "masking_policy": "non_primate_subtrees_plus_primate_species",
                    "current_shard_path": f"streaming://chr{chrom}/batch{batch_counter}",
                },
            )
        print(f"[hybrid-perturbation] processed_examples={n_total} last_chr={chrom}", flush=True)

    if n_total == 0:
        raise RuntimeError("No held-out examples were loaded for hybrid perturbation.")

    baseline_mean = baseline_sum / n_total
    if status_path:
        write_status(
            status_path,
            {
                "status": "completed",
                "run_tag": run_spec["run_tag"],
                "processed_examples": n_total,
                "batch_counter": batch_counter,
                "heldout_chromosomes": list(chroms),
                "samples_per_chrom": samples_per_chrom,
                "baseline_loss": baseline_mean,
                "masking_policy": "non_primate_subtrees_plus_primate_species",
                "current_shard_path": "",
            },
        )
    return subtree_specs, species_specs, baseline_mean, n_total


def build_species_rows(species_specs: list[dict[str, object]], baseline_mean: float) -> list[dict[str, object]]:
    rows = []
    for spec in species_specs:
        pert_sum = float(spec["loss_sum"])
        rows.append(
            {
                "ucsc_name": str(spec["ucsc_name"]),
                "common_name": str(spec["common_name"]),
                "clade": species_plot_group(str(spec["common_name"])),
                "branch_length_to_human": float(branch_length_to_human(str(spec["ucsc_name"]))),
                "delta_loss": 0.0,
                "delta_loss_clipped": 0.0,
                "baseline_loss": baseline_mean,
                "perturbed_loss": pert_sum,
                "masking_policy": "species_only",
                "n_masked_nodes": 1,
                "includes_internal_ancestors": "no",
            }
        )
    return rows


def build_subtree_rows(subtree_specs: list[dict[str, object]], baseline_mean: float) -> list[dict[str, object]]:
    rows = []
    for spec in subtree_specs:
        pert_sum = float(spec["loss_sum"])
        rows.append(
            {
                "group_name": str(spec["group_name"]),
                "delta_loss": 0.0,
                "delta_loss_clipped": 0.0,
                "n_leaf_species": len(spec["leaf_species"]),
                "n_internal_nodes": len(spec["internal_nodes"]),
                "leaf_species": ";".join(COMMON_NAMES.get(name, name) for name in spec["leaf_species"]),
                "internal_nodes": ";".join(spec["internal_nodes"]),
                "includes_internal_ancestors": "yes" if spec["internal_nodes"] else "no",
                "baseline_loss": baseline_mean,
                "perturbed_loss": pert_sum,
                "masking_policy": "subtree_only",
            }
        )
    return rows


def finalize_loss_deltas(species_rows: list[dict[str, object]], subtree_rows: list[dict[str, object]], n_total: int) -> None:
    for row in species_rows + subtree_rows:
        pert_mean = float(row["perturbed_loss"]) / float(n_total)
        delta = float(pert_mean - float(row["baseline_loss"]))
        row["perturbed_loss"] = pert_mean
        row["delta_loss"] = delta
        row["delta_loss_clipped"] = max(delta, 0.0)


def write_species_csv(rows: list[dict[str, object]], output_dir: str) -> str:
    csv_path = os.path.join(output_dir, "species_perturbation_scores.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write(
            "ucsc_name,common_name,clade,branch_length_to_human,delta_loss,delta_loss_clipped,baseline_loss,perturbed_loss,masking_policy,n_masked_nodes,includes_internal_ancestors\n"
        )
        for row in rows:
            handle.write(
                f"{row['ucsc_name']},{row['common_name']},{row['clade']},{row['branch_length_to_human']:.6f},"
                f"{row['delta_loss']:.9f},{row['delta_loss_clipped']:.9f},{row['baseline_loss']:.9f},{row['perturbed_loss']:.9f},"
                f"{row['masking_policy']},{row['n_masked_nodes']},{row['includes_internal_ancestors']}\n"
            )
    return csv_path


def write_subtree_csv(rows: list[dict[str, object]], output_dir: str) -> str:
    csv_path = os.path.join(output_dir, "subtree_perturbation_scores.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write(
            "group_name,delta_loss,delta_loss_clipped,n_leaf_species,n_internal_nodes,includes_internal_ancestors,leaf_species,internal_nodes,baseline_loss,perturbed_loss,masking_policy\n"
        )
        for row in rows:
            handle.write(
                f"{row['group_name']},{row['delta_loss']:.9f},{row['delta_loss_clipped']:.9f},"
                f"{row['n_leaf_species']},{row['n_internal_nodes']},{row['includes_internal_ancestors']},"
                f"\"{row['leaf_species']}\",\"{row['internal_nodes']}\",{row['baseline_loss']:.9f},{row['perturbed_loss']:.9f},{row['masking_policy']}\n"
            )
    return csv_path


def plot_species_outputs(rows: list[dict[str, object]], output_dir: str) -> tuple[str, str, float]:
    plot_rows = sorted(rows, key=lambda row: float(row["branch_length_to_human"]))
    x = np.asarray([float(row["branch_length_to_human"]) for row in plot_rows], dtype=np.float64)
    y = np.asarray([float(row["delta_loss_clipped"]) for row in plot_rows], dtype=np.float64)
    labels = [str(row["common_name"]) for row in plot_rows]
    groups = [str(row["clade"]) for row in plot_rows]

    slope, intercept, r, *_ = linregress(x, y)
    r2 = float(r ** 2)

    scatter_path = os.path.join(output_dir, "gcn_phylo_variance_scatter.png")
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for group_name, color in SPECIES_PLOT_COLORS.items():
        idxs = [i for i, value in enumerate(groups) if value == group_name]
        if idxs:
            ax.scatter(x[idxs], y[idxs], color=color, edgecolors="k", linewidths=0.4, s=60, alpha=0.9, label=group_name, zorder=3)
    xs = np.linspace(float(x.min()), float(x.max()), 200)
    ax.plot(xs, slope * xs + intercept, color="#333333", linestyle="--", linewidth=1.8, label=f"$R^2 = {r2:.3f}$", zorder=4)
    for label, xv, yv in zip(labels, x, y):
        ax.annotate(label, (xv, yv), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Distance to Human (UCSC branch length)")
    ax.set_ylabel("Perturbation Importance (Δ nucleotide cross-entropy)")
    ax.set_title("Held-out Primate + Tree Shrew Perturbation vs Distance to Human")
    ax.grid(True, linestyle=":", alpha=0.35)
    legend_patches = [mpatches.Patch(facecolor=color, edgecolor="none", label=group) for group, color in SPECIES_PLOT_COLORS.items()]
    ax.legend(handles=legend_patches + [ax.lines[-1]], loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    bar_path = os.path.join(output_dir, "gcn_species_importance.png")
    ranking = sorted(rows, key=lambda row: float(row["delta_loss_clipped"]), reverse=True)
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    values = [float(row["delta_loss_clipped"]) for row in ranking]
    colors = [SPECIES_PLOT_COLORS.get(str(row["clade"]), "#999999") for row in ranking]
    ax.bar(range(len(ranking)), values, color=colors, edgecolor="#173040", alpha=0.95)
    ax.set_xticks(range(len(ranking)))
    ax.set_xticklabels([str(row["common_name"]) for row in ranking], rotation=35, ha="right")
    ax.set_ylabel("Perturbation Importance (Δ nucleotide cross-entropy)")
    ax.set_title("Held-out Species Perturbation Importance (Primates + Tree Shrew)")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(bar_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return scatter_path, bar_path, r2


def plot_subtree_outputs(rows: list[dict[str, object]], output_dir: str) -> str:
    ranking = sorted(rows, key=lambda row: float(row["delta_loss_clipped"]), reverse=True)
    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    labels = [str(row["group_name"]) for row in ranking]
    values = [float(row["delta_loss_clipped"]) for row in ranking]
    ax.bar(range(len(labels)), values, color="#2f6b8a", edgecolor="#173040", alpha=0.95)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Perturbation Importance (Δ nucleotide cross-entropy)")
    ax.set_title("Held-out Non-primate Subtree Perturbation Impact")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "subtree_perturbation_major_clades.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_summaries(
    *,
    species_rows: list[dict[str, object]],
    subtree_rows: list[dict[str, object]],
    run_spec: dict,
    chroms: tuple[int, ...],
    samples_per_chrom: int,
    n_examples: int,
    r2: float,
    output_dir: str,
) -> tuple[str, str]:
    species_summary = {
        "status": "current",
        "run_tag": run_spec["run_tag"],
        "config_path": run_spec["config_path"],
        "checkpoint_path": run_spec["model_path"],
        "species_masking": run_spec["species_masking"],
        "config_compact_dir": run_spec["compact_dir"],
        "config_alignment_cache_dir": run_spec["align_cache"],
        "effective_compact_dir": run_spec["effective_compact_dir"],
        "effective_alignment_cache_dir": run_spec["effective_align_cache"],
        "heldout_chromosomes": list(chroms),
        "samples_per_chrom": samples_per_chrom,
        "n_examples": n_examples,
        "masking_policy": "hybrid_non_primate_subtrees_plus_primate_species",
        "primate_mode": "species_only",
        "tree_shrew_mode": "single_species_leaf",
        "interpretation_scope": "causal_perturbation",
        "perturbation_method": "Species row ablation to GAP/N token for primates plus tree shrew; non-primate taxa are handled separately as subtree perturbations.",
        "phylogeny_distance_r2": r2,
        "explicit_species_scores": {str(row["ucsc_name"]): float(row["delta_loss_clipped"]) for row in species_rows},
        "top_species": sorted(species_rows, key=lambda row: float(row["delta_loss_clipped"]), reverse=True),
    }
    species_summary_path = os.path.join(output_dir, "perturbation_importance_summary.json")
    with open(species_summary_path, "w", encoding="utf-8") as handle:
        json.dump(species_summary, handle, indent=2)

    subtree_summary = {
        "status": "current",
        "run_tag": run_spec["run_tag"],
        "config_path": run_spec["config_path"],
        "checkpoint_path": run_spec["model_path"],
        "species_masking": run_spec["species_masking"],
        "config_compact_dir": run_spec["compact_dir"],
        "config_alignment_cache_dir": run_spec["align_cache"],
        "effective_compact_dir": run_spec["effective_compact_dir"],
        "effective_alignment_cache_dir": run_spec["effective_align_cache"],
        "heldout_chromosomes": list(chroms),
        "samples_per_chrom": samples_per_chrom,
        "n_examples": n_examples,
        "masking_policy": "hybrid_non_primate_subtrees_plus_primate_species",
        "primate_mode": "species_only",
        "tree_shrew_mode": "single_species_leaf",
        "interpretation_scope": "causal_perturbation",
        "perturbation_method": "Subtree ablation to GAP/N token for non-primate major clades only. Primates and tree shrew are excluded from subtree aggregation and scored separately in species_perturbation_scores.csv.",
        "views": {"major_clades": list(NON_PRIMATE_GROUPS)},
        "top_subtrees": sorted(subtree_rows, key=lambda row: float(row["delta_loss_clipped"]), reverse=True),
    }
    subtree_summary_path = os.path.join(output_dir, "subtree_perturbation_summary.json")
    with open(subtree_summary_path, "w", encoding="utf-8") as handle:
        json.dump(subtree_summary, handle, indent=2)
    return species_summary_path, subtree_summary_path


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    status_path = args.status_json or os.path.join(args.output_dir, "subtree_perturbation_status.json")
    run_spec = load_run_spec(args.run_tag)
    model_wrapper = load_model(run_spec)
    chroms = parse_chromosome_spec(args.chromosomes)
    probe_chrom = chroms[0]
    compact_probe = f"positions_graphylovar_topmed_chr{probe_chrom}.npy"
    align_probe = f"alignment_encoded_chr{probe_chrom}.npy"
    run_spec["effective_compact_dir"] = resolve_existing_dir(
        args.compact_dir or run_spec["compact_dir"],
        os.path.join(REPO_ROOT, "topmed_compact_full"),
        compact_probe,
    )
    run_spec["effective_align_cache"] = resolve_existing_dir(
        args.align_cache_dir or run_spec["align_cache"],
        os.path.join(REPO_ROOT, "topmed_alignment_cache"),
        align_probe,
    )

    subtree_specs, species_specs, baseline_mean, n_examples = score_masks_streaming(
        model_wrapper, run_spec, chroms, args.samples_per_chrom, args.batch_size, status_path
    )
    species_rows = build_species_rows(species_specs, baseline_mean)
    subtree_rows = build_subtree_rows(subtree_specs, baseline_mean)
    finalize_loss_deltas(species_rows, subtree_rows, n_examples)

    species_csv_path = write_species_csv(species_rows, args.output_dir)
    subtree_csv_path = write_subtree_csv(subtree_rows, args.output_dir)
    scatter_path, bar_path, r2 = plot_species_outputs(species_rows, args.output_dir)
    subtree_plot_path = plot_subtree_outputs(subtree_rows, args.output_dir)
    species_summary_path, subtree_summary_path = write_summaries(
        species_rows=species_rows,
        subtree_rows=subtree_rows,
        run_spec=run_spec,
        chroms=chroms,
        samples_per_chrom=args.samples_per_chrom,
        n_examples=n_examples,
        r2=r2,
        output_dir=args.output_dir,
    )

    print(
        json.dumps(
            {
                "species_csv": species_csv_path,
                "species_summary": species_summary_path,
                "species_scatter": scatter_path,
                "species_bar": bar_path,
                "subtree_csv": subtree_csv_path,
                "subtree_summary": subtree_summary_path,
                "subtree_plot": subtree_plot_path,
                "status_json": status_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
