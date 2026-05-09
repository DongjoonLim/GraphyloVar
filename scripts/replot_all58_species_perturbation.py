#!/usr/bin/env python3
"""Regenerate the all-58 perturbation figure with a readable two-panel layout.

Panel (a): all 58 species, clipped Delta cross-entropy, linear scale. Shows the
Human self-row dominance story.
Panel (b): 57 non-Human species, signed Delta cross-entropy, auto-scaled.
Reveals the primate gradient and the near-zero signed noise floor for the
non-primate clades. Chimpanzee is explicitly labeled even though its value is
near-zero, demonstrating the information-redundancy finding.

Reads the canonical CSV produced by per_species_all58_perturbation.py.
Writes a new PNG alongside the CSV and also into the LaTeX figures directory.
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(
    REPO_ROOT,
    "outputs/supervisor_recovery_20260329/interpretability",
    "all58_species_perturbation_scores.csv",
)
OUT_PATH_PRIMARY = os.path.join(
    REPO_ROOT,
    "outputs/supervisor_recovery_20260329/interpretability",
    "all58_species_perturbation_ranking.png",
)
OUT_PATH_FIGURES = os.path.join(
    REPO_ROOT,
    "latex/graphylovar_submission/figures",
    "figS6_all58_species_perturbation.png",
)

CLADE_COLORS = {
    "Primates": "#e6194b",
    "Tree shrew": "#6b6b6b",
    "Glires": "#f58231",
    "Cetartiodactyla": "#911eb4",
    "Carnivora & Perissodactyla": "#3cb44b",
    "Chiroptera": "#f032e6",
    "Eulipotyphla": "#4363d8",
    "Afrotheria & Xenarthra": "#42d4f4",
}

# Key species to always label regardless of bar height
ALWAYS_LABEL = {
    "Human", "Chimp", "Gorilla", "Orangutan", "Green monkey",
    "Gibbon", "Rhesus macaque",
}


def main() -> None:
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(
                {
                    "name": r["common_name"],
                    "clade": r["clade"],
                    "delta_clipped": float(r["delta_loss_clipped"]),
                    "delta_signed": float(r["delta_loss"]),
                }
            )

    if len(rows) != 58:
        print(f"WARN: expected 58 rows, got {len(rows)}", file=sys.stderr)

    ranking_full = sorted(rows, key=lambda r: r["delta_clipped"], reverse=True)
    non_human = [r for r in ranking_full if r["name"] != "Human"]
    ranking_bottom = sorted(non_human, key=lambda r: r["delta_signed"], reverse=True)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(15.0, 10.8), gridspec_kw={"height_ratios": [1.0, 1.25]}
    )

    # ── Panel (a): all 58, clipped scale ───────────────────────────────────
    vals_top = [r["delta_clipped"] for r in ranking_full]
    names_top = [r["name"] for r in ranking_full]
    colors_top = [CLADE_COLORS.get(r["clade"], "#888888") for r in ranking_full]
    ax_top.bar(range(len(ranking_full)), vals_top, color=colors_top, edgecolor="#173040", alpha=0.95)
    ax_top.set_xticks(range(len(ranking_full)))
    ax_top.set_xticklabels(names_top, rotation=70, ha="right", fontsize=7)
    ax_top.set_ylabel(r"$\Delta$ nucleotide cross-entropy (clipped)")
    ax_top.set_title(
        "(a) All 58 placental mammals : Human dominates because the variant position lies in the Human row;\n"
        "Chimpanzee clipped to zero despite closest phylogenetic distance (information redundancy with Human)"
    )
    ax_top.grid(axis="y", linestyle=":", alpha=0.35)
    max_top = max(vals_top) if vals_top else 1.0
    for i, r in enumerate(ranking_full):
        if r["name"] in ALWAYS_LABEL and vals_top[i] < max_top * 0.02:
            ax_top.annotate(
                r["name"],
                xy=(i, vals_top[i] + max_top * 0.005),
                xytext=(i, max_top * 0.06),
                textcoords="data",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#b00000",
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#b00000", lw=0.8),
            )
    legend_patches = [mpatches.Patch(facecolor=color, label=clade) for clade, color in CLADE_COLORS.items()]
    ax_top.legend(handles=legend_patches, loc="upper right", fontsize=8, framealpha=0.9)

    # ── Panel (b): 57 non-Human, signed, auto-scaled ───────────────────────
    vals_bot = [r["delta_signed"] for r in ranking_bottom]
    names_bot = [r["name"] for r in ranking_bottom]
    colors_bot = [CLADE_COLORS.get(r["clade"], "#888888") for r in ranking_bottom]
    ax_bot.bar(range(len(ranking_bottom)), vals_bot, color=colors_bot, edgecolor="#173040", alpha=0.95)
    ax_bot.axhline(0.0, color="black", linewidth=0.6)
    ax_bot.set_xticks(range(len(ranking_bottom)))
    ax_bot.set_xticklabels(names_bot, rotation=70, ha="right", fontsize=7)
    ax_bot.set_ylabel(r"$\Delta$ nucleotide cross-entropy (signed)")
    max_bot = max(abs(v) for v in vals_bot) if vals_bot else 1e-6
    ax_bot.set_ylim(-max_bot * 1.3, max_bot * 1.4)
    ax_bot.set_title(
        "(b) 57 non-Human species (signed, auto-scaled) : primate gradient visible;\n"
        "Chimpanzee near-zero/negative: information redundancy with Human (~98.7% sequence identity)"
    )
    ax_bot.grid(axis="y", linestyle=":", alpha=0.35)
    # Label key primates explicitly
    for i, r in enumerate(ranking_bottom):
        if r["name"] in ALWAYS_LABEL:
            y = vals_bot[i]
            offset = max_bot * 0.12 if y >= 0 else -max_bot * 0.18
            ax_bot.annotate(
                r["name"],
                xy=(i, y),
                xytext=(0, 10 if y >= 0 else -16),
                textcoords="offset points",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=7.5,
                color="#b00000" if abs(y) < max_bot * 0.05 else "#1a1a1a",
                fontweight="bold",
            )
    ax_bot.legend(handles=legend_patches, loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_PATH_PRIMARY, dpi=220, bbox_inches="tight")
    fig.savefig(OUT_PATH_FIGURES, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH_PRIMARY}")
    print(f"Wrote {OUT_PATH_FIGURES}")


if __name__ == "__main__":
    main()
