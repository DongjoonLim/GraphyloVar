#!/usr/bin/env python3
"""Generate Supplementary Figure S6: two scatter panels.

  (a) SE-gate magnitude vs evolutionary distance from human : figS6a_se_attention_scatter.png
  (b) Causal perturbation importance vs evolutionary distance : figS6b_perturbation_branch_length.png

Key species (especially Chimpanzee) are explicitly labeled.
Chimpanzee appears near-zero in BOTH panels because it is nearly identical to
Human (~98.7% sequence identity) and thus provides no non-redundant signal.
"""
from __future__ import annotations

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import linregress

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SE_CSV   = os.path.join(REPO_ROOT, "outputs/supervisor_recovery_20260329/interpretability",
                        "actual_se_attention_scores.csv")
PERT_CSV = os.path.join(REPO_ROOT, "outputs/supervisor_recovery_20260329/interpretability",
                        "all58_species_perturbation_scores.csv")
FIG_DIR  = os.path.join(REPO_ROOT, "latex/graphylovar_submission/figures")

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

ALWAYS_LABEL = {
    "Human", "Chimp", "Gorilla", "Orangutan", "Green monkey",
    "Gibbon", "Rhesus macaque", "Tree shrew",
}

# Scatter-plot label offsets for selected species
LABEL_OFFSETS = {
    "Human":       (6,   8,  "left"),
    "Chimp":       (6,  -12, "left"),
    "Gorilla":     (6,   6,  "left"),
    "Orangutan":   (6,  -10, "left"),
    "Green monkey":(-6,  8,  "right"),
    "Gibbon":      (6,   4,  "left"),
    "Rhesus macaque": (-6, -12, "right"),
    "Tree shrew":  (6,   6,  "left"),
}


def load_se(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            # support both column name variants
            gate = r.get("mean_se_attention_gate") or r.get("mean_se_gated_magnitude")
            rows.append({
                "name":   r["common_name"],
                "clade":  r["clade"],
                "branch": float(r["branch_length_to_human"]),
                "gate":   float(gate),
            })
    return rows


def load_pert(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "name":   r["common_name"],
                "clade":  r["clade"],
                "branch": float(r["branch_length_to_human"]),
                "delta":  float(r["delta_loss"]),
            })
    return rows


def scatter_panel(ax, rows, y_key, ylabel, title, highlight_zero=False):
    x = np.array([r["branch"] for r in rows])
    y = np.array([r[y_key]   for r in rows])
    clades  = [r["clade"] for r in rows]
    names   = [r["name"]  for r in rows]

    # Scatter by clade
    for clade, color in CLADE_COLORS.items():
        idx = [i for i, c in enumerate(clades) if c == clade]
        if idx:
            ax.scatter(x[idx], y[idx], color=color, edgecolors="k",
                       linewidths=0.4, s=55, alpha=0.88, label=clade, zorder=3)

    # Linear fit
    slope, intercept, r, *_ = linregress(x, y)
    r2 = r ** 2
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, slope * xs + intercept, color="#333333", linestyle="--",
            linewidth=1.6, label=f"Linear fit ($R^2={r2:.3f}$)", zorder=4)

    if highlight_zero:
        ax.axhline(0, color="black", linewidth=0.7, linestyle=":")

    # Label key species
    yrange = y.max() - y.min() if y.max() != y.min() else 1.0
    for i, name in enumerate(names):
        if name in ALWAYS_LABEL:
            dx, dy, ha = LABEL_OFFSETS.get(name, (5, 5, "left"))
            near_zero = abs(y[i]) < yrange * 0.04
            color = "#b00000" if (near_zero and name != "Human") else "#1a1a1a"
            ax.annotate(
                name,
                xy=(x[i], y[i]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha, va="center",
                fontsize=8,
                color=color,
                fontweight="bold" if near_zero else "normal",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.7) if abs(dx) > 8 or abs(dy) > 8 else None,
            )

    ax.set_xlabel("Evolutionary Distance from Human\n(UCSC branch length, substitutions/site)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.35)

    legend_patches = [mpatches.Patch(facecolor=color, label=clade)
                      for clade, color in CLADE_COLORS.items()]
    # find the linear-fit Line2D (has a real label set above)
    fit_line = next(l for l in ax.lines if l.get_label().startswith("Linear"))
    ax.legend(handles=legend_patches + [fit_line], loc="upper right",
              fontsize=8, framealpha=0.9)


def main():
    se_rows   = load_se(SE_CSV)
    pert_rows = load_pert(PERT_CSV)

    # ── Figure S6a: SE gate vs branch length ─────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    scatter_panel(
        ax, se_rows, "gate",
        ylabel="Mean SE-Gate Magnitude\n(post-gate feature activation per species)",
        title=(
            "SE-Gate Magnitude vs Evolutionary Distance from Human\n"
            "Chimpanzee near-zero despite closest distance : information redundancy with Human"
        ),
    )
    fig.tight_layout()
    out_a = os.path.join(FIG_DIR, "figS6a_se_attention_scatter.png")
    fig.savefig(out_a, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_a}")

    # ── Figure S6b: perturbation vs branch length ─────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    scatter_panel(
        ax, pert_rows, "delta",
        ylabel=r"Causal Perturbation Importance ($\Delta$ CE when masked)",
        title=(
            "Causal Perturbation Importance vs Evolutionary Distance from Human\n"
            "Chimpanzee near-zero (slightly negative): information redundancy, not low importance"
        ),
        highlight_zero=True,
    )
    fig.tight_layout()
    out_b = os.path.join(FIG_DIR, "figS6b_perturbation_branch_length.png")
    fig.savefig(out_b, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_b}")


if __name__ == "__main__":
    main()
