"""
plot_tree_importance.py : Plot perturbation importance on the phylogenetic tree.

Reads figures/perturbation_importance.npy (previously computed) and overlays
the species importance scores on the UCSC 100-vertebrate phylogenetic tree layout,
similar to gcn_phylo_tree_importance.png.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graphylovar.phylogeny import NAMES, build_graph

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Prefer the definitive all-58 CSV; fall back to the legacy npy
ALL58_CSV = os.path.join(
    REPO_ROOT, "outputs/supervisor_recovery_20260329/interpretability",
    "all58_species_perturbation_scores.csv"
)
IMPORTANCE_NPY = os.path.join(REPO_ROOT, "figures", "perturbation_importance.npy")
OUT_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

COMMON_NAMES = {
    "hg38": "Human", "panTro4": "Chimp", "gorGor3": "Gorilla",
    "ponAbe2": "Orangutan", "nomLeu3": "Gibbon",
    "rheMac3": "Rhesus macaque", "macFas5": "Crab-eating macaque",
    "papAnu2": "Baboon", "chlSab2": "Green monkey",
    "calJac3": "Marmoset", "saiBol1": "Squirrel monkey",
    "otoGar3": "Bushbaby", "tupChi1": "Tree shrew",
    "speTri2": "Squirrel", "jacJac1": "Jerboa", "micOch1": "Prairie vole",
    "criGri1": "Chinese hamster", "mesAur1": "Golden hamster",
    "mm10": "Mouse", "rn6": "Rat", "hetGla2": "Naked mole-rat",
    "cavPor3": "Guinea pig", "chiLan1": "Chinchilla", "octDeg1": "Degu",
    "oryCun2": "Rabbit", "ochPri3": "Pika",
    "susScr3": "Pig", "vicPac2": "Alpaca", "camFer1": "Bactrian camel",
    "turTru2": "Dolphin", "orcOrc1": "Killer whale",
    "panHod1": "Tibetan antelope", "bosTau8": "Cow",
    "oviAri3": "Sheep", "capHir1": "Goat", "equCab2": "Horse",
    "cerSim1": "White rhinoceros", "felCat8": "Cat",
    "canFam3": "Dog", "musFur1": "Ferret", "ailMel1": "Panda",
    "odoRosDiv1": "Walrus", "lepWed1": "Weddell seal",
    "pteAle1": "Black flying fox", "pteVam1": "Large flying fox",
    "eptFus1": "Big brown bat", "myoDav1": "David's myotis",
    "myoLuc2": "Little brown bat", "eriEur2": "Hedgehog",
    "sorAra2": "Shrew", "conCri1": "Star-nosed mole",
    "loxAfr3": "Elephant", "eleEdw1": "Cape elephant shrew",
    "triMan1": "Manatee", "chrAsi1": "Cape golden mole",
    "echTel2": "Tenrec", "oryAfe1": "Aardvark", "dasNov3": "Armadillo",
}


def main():
    # Load importance scores : prefer all58 CSV over legacy npy
    name_to_imp: dict[str, float] = {}
    if os.path.exists(ALL58_CSV):
        import csv as _csv
        with open(ALL58_CSV, encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                ucsc = row["ucsc_name"]
                name_to_imp[ucsc] = float(row["delta_loss"])
        print(f"Loaded importance from {ALL58_CSV} ({len(name_to_imp)} species)")
    elif os.path.exists(IMPORTANCE_NPY):
        importance = np.load(IMPORTANCE_NPY)
        for i, n in enumerate(NAMES):
            name_to_imp[n] = float(importance[i])
        print(f"Loaded importance from {IMPORTANCE_NPY} (legacy)")
    else:
        raise FileNotFoundError(
            f"Missing both {ALL58_CSV} and {IMPORTANCE_NPY}. "
            "Run per_species_all58_perturbation.py first."
        )

    # Load phylogenetic graph
    G, A = build_graph()

    # ── Compute spring layout ──────────────────────────────────────────────
    import networkx as nx
    np.random.seed(42)
    pos = nx.spring_layout(G, seed=42, k=2.5, iterations=200)

    # ── Separate leaf vs internal nodes ───────────────────────────────────
    leaf_nodes  = [n for n in G.nodes if not n.startswith("_")]
    inner_nodes = [n for n in G.nodes if n.startswith("_")]

    leaf_vals = np.array([name_to_imp.get(n, 0.0) for n in leaf_nodes])
    # Clip negatives to 0 for colour mapping (negative Δloss = not important)
    leaf_vals_clipped = np.clip(leaf_vals, 0, None)

    # ── Colour mapping ─────────────────────────────────────────────────────
    vmin = 0.0
    vmax = max(leaf_vals_clipped.max(), 1e-9)
    norm  = Normalize(vmin=vmin, vmax=vmax)
    cmap  = cm.get_cmap("YlOrRd")
    leaf_colors = [cmap(norm(v)) for v in leaf_vals_clipped]

    # ── Node sizes: proportional to importance, min size for zeros ─────────
    size_min, size_max = 100, 900
    sizes = size_min + (leaf_vals_clipped / (vmax + 1e-9)) * (size_max - size_min)

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_facecolor("white")
    ax.axis("off")

    # Edges (grey)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4,
                           edge_color="#aaaaaa", width=0.8, arrows=False)

    # Internal nodes (small grey squares)
    if inner_nodes:
        inner_pos = {n: pos[n] for n in inner_nodes if n in pos}
        nx.draw_networkx_nodes(G, inner_pos, nodelist=list(inner_pos.keys()),
                               ax=ax, node_shape="s",
                               node_size=30, node_color="#cccccc", alpha=0.6)

    # Leaf nodes (coloured circles)
    for node, color, size in zip(leaf_nodes, leaf_colors, sizes):
        if node not in pos:
            continue
        nx.draw_networkx_nodes(G, {node: pos[node]}, nodelist=[node],
                               ax=ax, node_shape="o",
                               node_size=size, node_color=[color],
                               edgecolors="k", linewidths=0.6, alpha=0.92)

    # Labels: use common names where available
    leaf_labels = {n: COMMON_NAMES.get(n, n) for n in leaf_nodes if n in pos}
    nx.draw_networkx_labels(G, pos, labels=leaf_labels, ax=ax,
                            font_size=6.5, font_color="#333333",
                            font_weight="normal")

    # Explicitly annotate Chimpanzee with a note about near-zero importance
    chimp_ucsc = "panTro4"
    if chimp_ucsc in pos:
        cx, cy = pos[chimp_ucsc]
        ax.annotate(
            "Chimp\n(near-zero:\ninformation redundancy\nwith Human)",
            xy=(cx, cy),
            xytext=(cx + 0.18, cy - 0.12),
            fontsize=7,
            color="#b00000",
            fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="#b00000", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#b00000", alpha=0.85),
        )

    # ── Colorbar ───────────────────────────────────────────────────────────
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, aspect=30)
    cbar.set_label("Occlusion Importance\n(Δ nucleotide cross-entropy)", fontsize=11)

    ax.set_title(
        "Phylogenetic Tree with Species Importance\n"
        "(Color = perturbation importance, Size = importance magnitude)",
        fontsize=14, fontweight="bold", pad=12,
    )

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "gcn_phylo_tree_importance.png")
    fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor="white")
    # Also save to LaTeX figures directory as figS5
    figs_dir = os.path.join(REPO_ROOT, "latex/graphylovar_submission/figures")
    figS5_path = os.path.join(figs_dir, "figS5_gcn_tree_importance.png")
    fig.savefig(figS5_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_path}")
    print(f"Saved → {figS5_path}")

    # ── Print top-10 by importance ─────────────────────────────────────────
    pairs = [(n, name_to_imp.get(n, 0.0)) for n in leaf_nodes]
    pairs.sort(key=lambda p: p[1], reverse=True)
    print("\nTop-10 species by perturbation importance:")
    for name, val in pairs[:10]:
        common = COMMON_NAMES.get(name, name)
        print(f"  {common:30s}  {name:15s}  Δloss = {val:+.6f}")

    print("\nBottom-10 (lowest / negative = masking doesn't hurt):")
    for name, val in pairs[-10:]:
        common = COMMON_NAMES.get(name, name)
        print(f"  {common:30s}  {name:15s}  Δloss = {val:+.6f}")


if __name__ == "__main__":
    main()
