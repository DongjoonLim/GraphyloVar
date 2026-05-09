#!/usr/bin/env python3
"""Generate Supplementary Figure S8: standard vs conditional (human-masked) species importance.

Two-panel bar chart showing how Chimpanzee importance changes when Human is masked,
directly demonstrating information redundancy.

Panel (a): Standard masking : each non-human species masked individually (from all58 CSV).
Panel (b): Conditional masking : Human also always masked (from conditional_human_masked CSV).
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
INTERP_DIR = os.path.join(
    REPO_ROOT, "outputs/supervisor_recovery_20260329/interpretability"
)
ALL58_CSV = os.path.join(INTERP_DIR, "all58_species_perturbation_scores.csv")
COND_CSV = os.path.join(
    INTERP_DIR,
    "conditional_human_masked/species_perturbation_scores_human_masked.csv",
)
OUT_PATH = os.path.join(
    REPO_ROOT,
    "latex/graphylovar_submission/figures",
    "figS8_conditional_perturbation.png",
)

CLADE_COLORS = {
    "Primates":                  "#e6194b",
    "Tree shrew":                "#6b6b6b",
    "Glires":                    "#f58231",
    "Cetartiodactyla":           "#911eb4",
    "Carnivora + Perissodactyla": "#3cb44b",
    "Chiroptera":                "#f032e6",
    "Eulipotyphla":              "#4363d8",
    "Afrotheria + Xenarthra":    "#42d4f4",
}


def _load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def _parse_species(rows: list[dict], exclude_human: bool = True) -> list[dict]:
    out = []
    for r in rows:
        if exclude_human and r.get("ucsc_name", r.get("ucsc", "")) == "hg38":
            continue
        out.append({
            "name":   r.get("common_name", r.get("ucsc_name", "")),
            "clade":  r.get("clade", ""),
            "delta":  float(r.get("delta_loss", r.get("delta_loss_clipped", 0))),
        })
    return out


def _bar_panel(ax, species: list[dict], title: str, ylabel: str,
               annotate_chimp: bool = True, use_clipped: bool = False) -> None:
    species_sorted = sorted(species, key=lambda r: r["delta"], reverse=True)
    names  = [r["name"]  for r in species_sorted]
    vals   = [r["delta"] for r in species_sorted]
    colors = [CLADE_COLORS.get(r["clade"], "#888888") for r in species_sorted]

    ax.bar(range(len(species_sorted)), vals, color=colors, edgecolor="#173040", alpha=0.92)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xticks(range(len(species_sorted)))
    ax.set_xticklabels(names, rotation=50, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    if annotate_chimp:
        for i, r in enumerate(species_sorted):
            if r["name"] == "Chimp":
                y = vals[i]
                max_abs = max(abs(v) for v in vals) if vals else 1e-9
                ax.annotate(
                    f"Chimp\n({y:.2e})",
                    xy=(i, y),
                    xytext=(0, 14 if y >= 0 else -20),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if y >= 0 else "top",
                    fontsize=7.5,
                    color="#b00000",
                    fontweight="bold",
                )
                break


def main() -> None:
    for path in (ALL58_CSV, COND_CSV):
        if not os.path.exists(path):
            print(f"ERROR: not found: {path}", file=sys.stderr)
            sys.exit(1)

    all58_rows   = _load_csv(ALL58_CSV)
    cond_rows    = _load_csv(COND_CSV)

    panel_a = _parse_species(all58_rows, exclude_human=True)
    panel_b = _parse_species(cond_rows,  exclude_human=True)

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(14.0, 10.0))

    _bar_panel(
        ax_a, panel_a,
        title=(
            "(a) Standard masking: each species masked individually\n"
            "Human dominates; Chimpanzee near-zero despite closest phylogenetic distance"
        ),
        ylabel=r"$\Delta$CE (signed)",
        annotate_chimp=True,
    )

    _bar_panel(
        ax_b, panel_b,
        title=(
            "(b) Conditional masking: Human row also masked\n"
            "Chimpanzee becomes highest-importance species, confirming information redundancy"
        ),
        ylabel=r"$\Delta$CE (signed, Human absent)",
        annotate_chimp=True,
    )

    # Shared legend
    legend_patches = [
        mpatches.Patch(facecolor=CLADE_COLORS[c], label=c)
        for c in CLADE_COLORS
    ]
    ax_a.legend(handles=legend_patches, loc="upper right", fontsize=8,
                framealpha=0.9, ncol=2)

    fig.tight_layout(pad=2.5)
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")

    # Print top-5 for each panel
    a_sorted = sorted(panel_a, key=lambda r: r["delta"], reverse=True)
    b_sorted = sorted(panel_b, key=lambda r: r["delta"], reverse=True)
    print("\nPanel (a) top-5 (standard masking):")
    for r in a_sorted[:5]:
        print(f"  {r['name']:30s}  delta={r['delta']:+.6f}")
    print("\nPanel (b) top-5 (conditional masking, human absent):")
    for r in b_sorted[:5]:
        print(f"  {r['name']:30s}  delta={r['delta']:+.6f}")


if __name__ == "__main__":
    main()
