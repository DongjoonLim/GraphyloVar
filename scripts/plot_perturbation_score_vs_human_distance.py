#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graphylovar.phylogeny import NAMES, branch_length_to_human


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMPORTANCE_PATH = os.path.join(REPO_ROOT, "figures", "perturbation_importance.npy")
OUT_PATH = os.path.join(REPO_ROOT, "figures", "perturbation_score_vs_human_distance.png")

COMMON_NAMES = {
    "hg38": "Human",
    "panTro4": "Chimp",
    "gorGor3": "Gorilla",
    "ponAbe2": "Orangutan",
    "nomLeu3": "Gibbon",
    "rheMac3": "Rhesus macaque",
    "macFas5": "Crab-eating macaque",
    "papAnu2": "Baboon",
    "chlSab2": "Green monkey",
    "calJac3": "Marmoset",
    "saiBol1": "Squirrel monkey",
    "otoGar3": "Bushbaby",
    "tupChi1": "Tree shrew",
    "speTri2": "Squirrel",
    "jacJac1": "Jerboa",
    "micOch1": "Prairie vole",
    "criGri1": "Chinese hamster",
    "mesAur1": "Golden hamster",
    "mm10": "Mouse",
    "rn6": "Rat",
    "hetGla2": "Naked mole-rat",
    "cavPor3": "Guinea pig",
    "chiLan1": "Chinchilla",
    "octDeg1": "Degu",
    "oryCun2": "Rabbit",
    "ochPri3": "Pika",
    "susScr3": "Pig",
    "vicPac2": "Alpaca",
    "camFer1": "Bactrian camel",
    "turTru2": "Dolphin",
    "orcOrc1": "Killer whale",
    "panHod1": "Tibetan antelope",
    "bosTau8": "Cow",
    "oviAri3": "Sheep",
    "capHir1": "Goat",
    "equCab2": "Horse",
    "cerSim1": "White rhinoceros",
    "felCat8": "Cat",
    "canFam3": "Dog",
    "musFur1": "Ferret",
    "ailMel1": "Panda",
    "odoRosDiv1": "Walrus",
    "lepWed1": "Weddell seal",
    "pteAle1": "Black flying fox",
    "pteVam1": "Large flying fox",
    "eptFus1": "Big brown bat",
    "myoDav1": "David's myotis",
    "myoLuc2": "Little brown bat",
    "eriEur2": "Hedgehog",
    "sorAra2": "Shrew",
    "conCri1": "Star-nosed mole",
    "loxAfr3": "Elephant",
    "eleEdw1": "Cape elephant shrew",
    "triMan1": "Manatee",
    "chrAsi1": "Cape golden mole",
    "echTel2": "Tenrec",
    "oryAfe1": "Aardvark",
    "dasNov3": "Armadillo",
}

CLADES = {
    "Primates": [
        "Human",
        "Chimp",
        "Gorilla",
        "Orangutan",
        "Gibbon",
        "Rhesus macaque",
        "Crab-eating macaque",
        "Baboon",
        "Green monkey",
        "Marmoset",
        "Squirrel monkey",
        "Bushbaby",
        "Tree shrew",
    ],
    "Glires": [
        "Squirrel",
        "Jerboa",
        "Prairie vole",
        "Chinese hamster",
        "Golden hamster",
        "Mouse",
        "Rat",
        "Naked mole-rat",
        "Guinea pig",
        "Chinchilla",
        "Degu",
        "Rabbit",
        "Pika",
    ],
    "Cetartiodactyla": [
        "Alpaca",
        "Bactrian camel",
        "Pig",
        "Dolphin",
        "Killer whale",
        "Sheep",
        "Goat",
        "Cow",
        "Tibetan antelope",
    ],
    "Carnivora & Perissodactyla": [
        "Horse",
        "White rhinoceros",
        "Walrus",
        "Weddell seal",
        "Panda",
        "Ferret",
        "Dog",
        "Cat",
    ],
    "Chiroptera": [
        "Black flying fox",
        "Large flying fox",
        "David's myotis",
        "Little brown bat",
        "Big brown bat",
    ],
    "Eulipotyphla": ["Shrew", "Star-nosed mole", "Hedgehog"],
    "Afrotheria & Xenarthra": [
        "Elephant",
        "Cape elephant shrew",
        "Manatee",
        "Cape golden mole",
        "Tenrec",
        "Aardvark",
        "Armadillo",
    ],
}

CLADE_COLORS = {
    "Primates": "#e6194b",
    "Glires": "#f58231",
    "Cetartiodactyla": "#911eb4",
    "Carnivora & Perissodactyla": "#3cb44b",
    "Chiroptera": "#f032e6",
    "Eulipotyphla": "#4363d8",
    "Afrotheria & Xenarthra": "#42d4f4",
}

KEY_PRIMATES = {"hg38", "panTro4", "gorGor3", "ponAbe2", "nomLeu3", "rheMac3"}
CUSTOM_LABEL_OFFSETS = {
    "sorAra2": ((6, -2), "left"),
    "micOch1": ((6, 10), "left"),
    "criGri1": ((6, 2), "left"),
    "mesAur1": ((6, -8), "left"),
    "rn6": ((6, -14), "left"),
    "mm10": ((6, 8), "left"),
    "echTel2": ((6, 12), "left"),
    "eriEur2": ((6, -2), "left"),
    "ochPri3": ((6, 6), "left"),
    "octDeg1": ((6, -6), "left"),
}
LABEL_OFFSETS = [(4, 5), (4, -8), (-4, 5), (-4, -8), (6, 10), (-6, 10)]


def get_clade(common_name: str) -> str:
    for clade, members in CLADES.items():
        if common_name in members:
            return clade
    return "Other"


def label_style(name: str, idx: int) -> tuple[tuple[int, int], str]:
    if name in CUSTOM_LABEL_OFFSETS:
        return CUSTOM_LABEL_OFFSETS[name]
    base_dx, base_dy = LABEL_OFFSETS[idx % len(LABEL_OFFSETS)]
    if name in KEY_PRIMATES:
        base_dx = 6 if base_dx >= 0 else -6
        base_dy = 8 if base_dy >= 0 else -10
    ha = "left" if base_dx >= 0 else "right"
    return (base_dx, base_dy), ha


def main() -> None:
    if not os.path.exists(IMPORTANCE_PATH):
        raise FileNotFoundError(f"Missing importance array: {IMPORTANCE_PATH}")

    importance = np.load(IMPORTANCE_PATH)
    xs, ys, names, common_names = [], [], [], []
    for i, name in enumerate(NAMES):
        if name.startswith("_"):
            continue
        try:
            distance = branch_length_to_human(name)
        except Exception:
            continue
        xs.append(float(distance))
        ys.append(float(importance[i]))
        names.append(name)
        common_names.append(COMMON_NAMES.get(name, name))

    x = np.asarray(xs)
    y = np.asarray(ys)
    slope, intercept, r, *_ = linregress(x, y)
    r2 = float(r**2)
    clades = [get_clade(label) for label in common_names]

    fig, ax = plt.subplots(figsize=(15.5, 9.5))

    for clade, color in CLADE_COLORS.items():
        idx = [i for i, value in enumerate(clades) if value == clade]
        if idx:
            sizes = [92 if names[i] in KEY_PRIMATES else 46 for i in idx]
            ax.scatter(
                x[idx],
                y[idx],
                s=sizes,
                color=color,
                alpha=0.84,
                edgecolors="white",
                linewidths=0.5,
                label=clade,
                zorder=2,
            )

    order = np.argsort(y)
    for rank, idx in enumerate(order):
        name = names[idx]
        label = common_names[idx]
        (dx, dy), ha = label_style(name, rank)
        ax.annotate(
            label,
            (x[idx], y[idx]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.2 if name in KEY_PRIMATES else 5.6,
            fontweight="bold" if name in KEY_PRIMATES else "normal",
            color="#1f1f1f",
            alpha=0.95 if name in KEY_PRIMATES else 0.78,
            ha=ha,
            va="center",
            zorder=4,
        )

    xline = np.linspace(x.min(), x.max(), 200)
    ax.plot(
        xline,
        slope * xline + intercept,
        linestyle="--",
        linewidth=1.8,
        color="#333333",
        label=f"Linear fit ($R^2={r2:.3f}$)",
        zorder=3,
    )
    ax.set_xlabel("Distance to Human (UCSC branch length)", fontsize=12)
    ax.set_ylabel("Delta Prediction Score", fontsize=12)
    ax.set_title(
        "Species Perturbation Score vs Distance to Human (All 58 Species)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_xlim(float(x.min()) - 0.02, float(x.max()) + 0.045)
    ax.set_ylim(float(y.min()) - 0.001, float(y.max()) * 1.06)

    legend_patches = [
        mpatches.Patch(facecolor=color, edgecolor="none", label=clade)
        for clade, color in CLADE_COLORS.items()
    ]
    line_handle = ax.lines[-1]
    ax.legend(
        handles=legend_patches + [line_handle],
        loc="upper right",
        fontsize=8.8,
        framealpha=0.92,
    )

    fig.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.1)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
