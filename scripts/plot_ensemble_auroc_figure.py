#!/usr/bin/env python3
"""Generate ensemble AUROC comparison figure for §3.4.

Uses full-dataset AUROCs (149M variants) where available, bootstrap CIs
from ensemble_auc_ci_table.csv for error bars.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_PATH = os.path.join(REPO_ROOT, "latex/graphylovar_submission/figures", "fig_ensemble_auroc.png")

# Full-dataset AUROCs (from 149M held-out variants, as reported in manuscript)
# CIs from bootstrap (n~200K subsample) scaled appropriately
METHODS = [
    # (label, auroc, ci_lo, ci_hi, color, group)
    ("GraphyloVar",      0.6246, 0.6174, 0.6368, "#5b2c8d", "individual"),
    ("CADD",             0.5546, 0.546,  0.563,  "#c44e52", "individual"),
    ("GV + CADD",        0.6442, 0.6395, 0.6587, "#2d8b57", "ensemble"),
]

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor("#fafafa")
fig.patch.set_facecolor("#fafafa")

y_positions = np.arange(len(METHODS))
labels = [m[0] for m in METHODS]
aucs   = [m[1] for m in METHODS]
errs_lo = [m[1] - m[2] for m in METHODS]
errs_hi = [m[3] - m[1] for m in METHODS]
colors  = [m[4] for m in METHODS]

# Draw bars with premium styling
bars = ax.barh(
    y_positions, aucs,
    xerr=[errs_lo, errs_hi],
    color=colors, edgecolor="#222222", linewidth=1.2,
    height=0.6, alpha=0.95,
    error_kw=dict(elinewidth=1.8, capsize=6, ecolor="#444444"),
    zorder=3
)

# Annotate AUROC values cleanly
for i, (auc, ci_lo, ci_hi) in enumerate(zip(aucs, [m[2] for m in METHODS], [m[3] for m in METHODS])):
    ax.text(auc + 0.005, i, f"{auc:.4f}", va="center", ha="left", fontsize=11, fontweight="bold", color="#111111")

# Mark delta for GV+CADD vs GV
delta = 0.6442 - 0.6246
gv_idx = 0
ens_idx = 2
gv_x = aucs[gv_idx]
ens_x = aucs[ens_idx]
y_mid = (y_positions[gv_idx] + y_positions[ens_idx]) / 2
ax.annotate(
    f"$+{delta:.4f}$\n($+{delta:.1%}$)",
    xy=(ens_x, y_positions[ens_idx]),
    xytext=(gv_x + (ens_x - gv_x) * 0.5, y_mid - 0.55),
    ha="center", va="top", fontsize=10, color="#196338", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#196338", lw=1.5, connectionstyle="arc3,rad=-0.1"),
)

# Vertical chance line
ax.axvline(0.5, color="#888888", linestyle="--", linewidth=1.2, alpha=0.7, zorder=1)
ax.text(0.502, y_positions[0]-0.4, "Chance (0.5)", fontsize=9, color="#666666", va="center")

# Separator between individual and ensemble
ax.axhline(y_positions[ens_idx] - 0.5, color="#cccccc", linestyle="-", linewidth=1.5, zorder=2)
ax.text(0.422, y_positions[ens_idx] - 0.25, "Ensemble", fontsize=10, color="#666666", va="center", fontweight="bold")
ax.text(0.422, y_positions[gv_idx] + 0.25, "Individual", fontsize=10, color="#666666", va="center", fontweight="bold")

# Axis styling
ax.set_yticks(y_positions)
ax.set_yticklabels(labels, fontsize=12, fontweight="medium")
ax.set_xlabel("Held-out AUROC (TOPMed chromosomes 13-22, common vs. rare variant discrimination)", fontsize=11, fontweight="medium", labelpad=10)
ax.set_xlim(0.42, 0.72)
ax.set_title(
    "Ensemble Complementarity: GraphyloVar + CADD\n"
    r"(full $\sim$149M-variant test set; error bars = bootstrap 95% CI, $n \approx 200{,}000$)",
    fontsize=14, fontweight="bold", pad=15
)

# Clean grid and spines
ax.grid(axis="x", color="#e0e0e0", linestyle="--", linewidth=1, alpha=0.7, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("#cccccc")

# p-value annotation
ax.text(0.6442 + 0.005, y_positions[2] + 0.35,
        r"$p < 10^{-15}$ vs. GraphyloVar alone (DeLong)", fontsize=9, color="#196338", fontweight="bold")

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Wrote {OUT_PATH}")
