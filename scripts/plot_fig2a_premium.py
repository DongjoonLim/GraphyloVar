#!/usr/bin/env python3
"""Generate a premium grouped bar chart for Figure 2A."""

from __future__ import annotations

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def parse_auc_table(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip().replace("\\textbf{", "").replace("}", "")
            if not line.startswith(("All", "Coding", "3'UTR", "cCREs", "TE", "Others")):
                continue
            if "&" not in line:
                continue
            left, _ = line.split("\\\\", 1)
            parts = [p.strip() for p in left.split("&")]
            if len(parts) != 8:
                continue
            rows.append({
                "Region": parts[0],
                "GraphyloVar": float(parts[1]),
                "CADD": float(parts[2]),
                "Enformer": float(parts[3]),
                "PhastCons": float(parts[4]),
                "PhyloP": float(parts[5]),
                "GPN-MSA": float(parts[6]),
                "GPN-Star": float(parts[7]),
            })
    return pd.DataFrame(rows)

def plot_premium_auc_regions(auc_df: pd.DataFrame, out_png: str) -> None:
    # Modern, striking palette
    # Make GraphyloVar a vibrant electric purple
    colors = {
        "GraphyloVar": "#7b2cbf", # vibrant purple
        "GPN-MSA": "#457b9d",     # muted blue
        "GPN-Star": "#a8dadc",    # light muted cyan
        "CADD": "#e63946",        # muted red
        "Enformer": "#f4a261",    # muted orange
        "PhyloP": "#8d99ae",      # slate
        "PhastCons": "#b7b7a4",   # light slate
    }
    
    methods = ["GraphyloVar", "GPN-MSA", "GPN-Star", "CADD", "Enformer", "PhyloP", "PhastCons"]
    
    # Sort the dataframe to have a logical order of regions
    region_order = ["Coding", "3'UTR", "cCREs", "TE"]
    auc_df = auc_df[auc_df["Region"].isin(region_order)]
    auc_df["Region"] = pd.Categorical(auc_df["Region"], categories=region_order, ordered=True)
    auc_df = auc_df.sort_values("Region")

    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#fafafa")
    
    x = np.arange(len(auc_df))
    width = 0.11
    
    for i, m in enumerate(methods):
        offset = (i - 3) * width
        
        # Add edge color and drop shadow logic for GraphyloVar to make it stand out
        if m == "GraphyloVar":
            ax.bar(
                x + offset, auc_df[m], width=width, label=m,
                color=colors[m], edgecolor="#3c096c", linewidth=1.8, zorder=3
            )
            # Add text labels on top of GraphyloVar bars
            for j, val in enumerate(auc_df[m]):
                ax.text(x[j] + offset, val + 0.002, f"{val:.3f}", 
                        ha="center", va="bottom", fontsize=10, fontweight="bold", color="#3c096c")
        else:
            ax.bar(
                x + offset, auc_df[m], width=width, label=m,
                color=colors[m], edgecolor="#333333", linewidth=0.5, alpha=0.85, zorder=3
            )

    ax.set_xticks(x)
    ax.set_xticklabels(auc_df["Region"], fontsize=13, fontweight="medium")
    ax.set_ylim(0.5, 0.70)
    
    ax.set_ylabel("Held-out AUROC", fontsize=14, fontweight="medium", labelpad=12)
    ax.set_title("Common vs. Rare Variant Classification by Genomic Region", fontsize=18, fontweight="bold", pad=20)
    
    # Clean up grid and spines
    ax.grid(axis="y", color="#e0e0e0", linestyle="--", linewidth=1, alpha=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    # Legend formatting
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.1),
        ncol=7, frameon=False, fontsize=12, columnspacing=1.5
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote premium Figure 2A to {out_png}")

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    auc_path = os.path.join(repo_root, "latex", "auc_table.tex")
    
    out_dir = os.path.join(repo_root, "latex", "graphylovar_submission", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "combined_auc_bar_chart.png")
    
    auc_df = parse_auc_table(auc_path)
    if auc_df.empty:
        print("Error: Could not parse auc_table.tex")
        return
        
    plot_premium_auc_regions(auc_df, out_png)

if __name__ == "__main__":
    main()
