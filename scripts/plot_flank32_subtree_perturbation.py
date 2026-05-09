#!/usr/bin/env python3
"""Generate Supplementary Figure S7: flank=32 subtree perturbation importance.

Three-panel layout:
  (a) All primate species + Tree shrew on full scale : Human dominates.
  (b) Non-human primate species + Tree shrew on signed micro-scale : shows
      the ordering and confirms Chimpanzee near-zero despite closest distance.
  (c) Non-primate clade subtrees : confirms same pattern for distant clades.

Reads species_perturbation_scores.csv and subtree_perturbation_scores.csv.
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUBTREE_DIR = os.path.join(
    REPO_ROOT,
    "outputs/supervisor_recovery_20260329/interpretability/flank32_subtree",
)
SPECIES_CSV = os.path.join(SUBTREE_DIR, "species_perturbation_scores.csv")
CLADE_CSV   = os.path.join(SUBTREE_DIR, "subtree_perturbation_scores.csv")
OUT_PATH    = os.path.join(
    REPO_ROOT,
    "latex/graphylovar_submission/figures",
    "figS7_flank32_subtree.png",
)

CLADE_COLORS = {
    "Primates": "#e6194b",
    "Tree shrew": "#6b6b6b",
    "Glires": "#f58231",
    "Cetartiodactyla": "#911eb4",
    "Carnivora + Perissodactyla": "#3cb44b",
    "Chiroptera": "#f032e6",
    "Eulipotyphla": "#4363d8",
    "Afrotheria + Xenarthra": "#42d4f4",
}

CLADE_ORDER = [
    "Glires",
    "Eulipotyphla",
    "Cetartiodactyla",
    "Carnivora + Perissodactyla",
    "Chiroptera",
    "Afrotheria + Xenarthra",
]


def main() -> None:
    for path in (SPECIES_CSV, CLADE_CSV):
        if not os.path.exists(path):
            print(f"ERROR: not found: {path}", file=sys.stderr)
            sys.exit(1)

    species_rows = []
    with open(SPECIES_CSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            species_rows.append({
                "name":          row["common_name"],
                "clade":         row["clade"],
                "delta_clipped": float(row["delta_loss_clipped"]),
                "delta_signed":  float(row["delta_loss"]),
            })
    species_rows.sort(key=lambda r: r["delta_clipped"], reverse=True)
    non_human = [r for r in species_rows if r["name"] != "Human"]
    non_human_signed = sorted(non_human, key=lambda r: r["delta_signed"], reverse=True)

    clade_map: dict[str, float] = {}
    with open(CLADE_CSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            clade_map[row["group_name"]] = float(row["delta_loss_clipped"])

    fig, axes = plt.subplots(
        3, 1, figsize=(10.0, 13.0),
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.9]},
    )
    ax_a, ax_b, ax_c = axes

    # ── Panel (a): all species, full scale ─────────────────────────────────
    names_a = [r["name"] for r in species_rows]
    vals_a  = [r["delta_clipped"] for r in species_rows]
    colors_a = [CLADE_COLORS.get(r["clade"], "#888888") for r in species_rows]
    ax_a.bar(range(len(species_rows)), vals_a, color=colors_a,
             edgecolor="#173040", alpha=0.92)
    ax_a.set_xticks(range(len(species_rows)))
    ax_a.set_xticklabels(names_a, rotation=50, ha="right", fontsize=8)
    ax_a.set_ylabel(r"$\Delta$CE (clipped)", fontsize=10)
    ax_a.set_title(
        "(a) All primates + Tree shrew (flank = 32): Human dominates;\n"
        "all non-human species clipped to zero at this scale",
        fontsize=10,
    )
    ax_a.grid(axis="y", linestyle=":", alpha=0.35)
    legend_patches = [
        mpatches.Patch(facecolor=CLADE_COLORS["Primates"], label="Primates"),
        mpatches.Patch(facecolor=CLADE_COLORS["Tree shrew"], label="Tree shrew"),
    ]
    ax_a.legend(handles=legend_patches, loc="upper right", fontsize=9, framealpha=0.9)

    # ── Panel (b): non-human species, signed micro-scale ───────────────────
    names_b = [r["name"] for r in non_human_signed]
    vals_b  = [r["delta_signed"] for r in non_human_signed]
    colors_b = [CLADE_COLORS.get(r["clade"], "#888888") for r in non_human_signed]
    ax_b.bar(range(len(non_human_signed)), vals_b, color=colors_b,
             edgecolor="#173040", alpha=0.92)
    ax_b.axhline(0.0, color="black", linewidth=0.7)
    ax_b.set_xticks(range(len(non_human_signed)))
    ax_b.set_xticklabels(names_b, rotation=50, ha="right", fontsize=8)
    ax_b.set_ylabel(r"$\Delta$CE (signed, auto-scale)", fontsize=10)
    max_b = max(abs(v) for v in vals_b) if vals_b else 1e-9
    ax_b.set_ylim(-max_b * 1.4, max_b * 1.5)
    ax_b.set_title(
        "(b) Non-human primates + Tree shrew (signed, zoomed) : "
        "Chimpanzee near-zero despite closest phylogenetic distance;\n"
        "information redundancy with Human (~98.7% identity) suppresses its contribution",
        fontsize=10,
    )
    ax_b.grid(axis="y", linestyle=":", alpha=0.35)
    # Annotate Chimp explicitly
    for i, r in enumerate(non_human_signed):
        if r["name"] == "Chimp":
            y = vals_b[i]
            ax_b.annotate(
                f"Chimp\n({y:.2e})",
                xy=(i, y),
                xytext=(0, 12 if y >= 0 else -18),
                textcoords="offset points",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=7.5,
                color="#b00000",
                fontweight="bold",
            )
        elif r["name"] in ("Gorilla", "Orangutan", "Green monkey") and abs(vals_b[i]) > max_b * 0.2:
            y = vals_b[i]
            ax_b.text(i, y + (max_b * 0.05 if y >= 0 else -max_b * 0.08),
                      r["name"], ha="center", va="bottom" if y >= 0 else "top",
                      fontsize=7, color="#1a1a1a")
    ax_b.legend(handles=legend_patches, loc="upper right", fontsize=9, framealpha=0.9)

    # ── Panel (c): non-primate clade subtrees ──────────────────────────────
    clade_vals   = [clade_map.get(c, 0.0) for c in CLADE_ORDER]
    clade_colors = [CLADE_COLORS.get(c, "#888888") for c in CLADE_ORDER]
    ax_c.bar(range(len(CLADE_ORDER)), clade_vals, color=clade_colors,
             edgecolor="#173040", alpha=0.92)
    ax_c.set_xticks(range(len(CLADE_ORDER)))
    ax_c.set_xticklabels(CLADE_ORDER, rotation=20, ha="right", fontsize=9)
    ax_c.set_ylabel(r"$\Delta$CE (clipped)", fontsize=10)
    ax_c.set_title(
        "(c) Non-primate mammalian clades: subtree perturbation (flank = 32)",
        fontsize=10,
    )
    ax_c.grid(axis="y", linestyle=":", alpha=0.35)
    clade_legend_patches = [
        mpatches.Patch(facecolor=CLADE_COLORS[c], label=c) for c in CLADE_ORDER
    ]
    ax_c.legend(handles=clade_legend_patches, loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout(pad=2.0)
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
