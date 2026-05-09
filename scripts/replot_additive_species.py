"""Regenerate Figure 6: Species Importance (additive perturbation, 500k variants).
Shows ALL values including negatives using a symlog y-axis in a single plot.
"""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CSV_PATH = os.path.join(
    REPO_ROOT,
    "outputs/supervisor_recovery_20260329/interpretability/additive_species",
    "additive_species_importance_scores.csv",
)
OUT_PATH = os.path.join(
    REPO_ROOT,
    "latex/graphylovar_submission/figures/gcn_species_importance.png",
)

CLADE_COLORS = {
    "Primate":                    "#e63946",
    "Euarchontoglires":           "#2a9d8f",
    "Laurasiatheria":             "#f4a261",
    "Afrotheria":                 "#457b9d",
    "Mammal":                     "#adb5bd",
    "Other":                      "#adb5bd",
}

if not os.path.exists(CSV_PATH):
    print(f"ERROR: CSV not found: {CSV_PATH}", file=sys.stderr)
    sys.exit(1)

NAME_FIXES = {
    "Rhesus macaque": "Rhesus",
    "Tree shrew": "Chinese tree shrew",
    "Jerboa": "Lesser Egyptian jerboa",
    "Goat": "Domestic goat",
    "Walrus": "Pacific walrus",
    "Black flying fox": "Black flying-fox",
    "Large flying fox": "Megabat",
    "David's myotis": "David's myotis (bat)",
    "Bottlenose dolphin": "Dolphin",
}

rows = []
with open(CSV_PATH) as fh:
    reader = csv.DictReader(fh)
    for r in reader:
        val = float(r["delta_additive"])
        cname = r["common_name"]
        orig_clade = r["clade"]
        
        cname = NAME_FIXES.get(cname, cname)
        
        if cname == "Armadillo":
            clade = "Mammal"
        elif orig_clade in ["Cetartiodactyla", "Carnivora & Perissodactyla", "Chiroptera", "Eulipotyphla"]:
            clade = "Laurasiatheria"
        elif orig_clade == "Afrotheria & Xenarthra":
            clade = "Afrotheria"
        elif orig_clade == "Primates":
            clade = "Primate"
        elif orig_clade == "Glires" or orig_clade == "Tree shrew" or cname == "Chinese tree shrew":
            clade = "Euarchontoglires"
        else:
            clade = orig_clade
            
        rows.append({
            "common_name": cname,
            "clade":       clade,
            "importance":  val,
        })

rows.sort(key=lambda r: r["importance"], reverse=True)

# Symlog threshold: linear region between -linthresh and +linthresh
LINTHRESH = 1e-3

fig = plt.figure(figsize=(20, 8))
ax_m = fig.add_subplot(111)

names_o  = [r["common_name"] for r in rows]
vals_o   = [r["importance"]  for r in rows]

# Color: negative bars get a hatched/lighter version of their clade color
colors_o = []
for r in rows:
    base = CLADE_COLORS.get(r["clade"], "#adb5bd")
    colors_o.append(base)

bars = ax_m.bar(range(len(names_o)), vals_o, color=colors_o, zorder=3,
                edgecolor="white", linewidth=0.4)

# Hatch negative bars to make them visually distinct
for bar, val in zip(bars, vals_o):
    if val < 0:
        bar.set_hatch("///")
        bar.set_edgecolor("#333333")
        bar.set_linewidth(0.6)

# Symlog scale
ax_m.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.5)
ax_m.axhline(0, color="#333", linewidth=0.8, zorder=4, linestyle="-")

ax_m.set_xticks(range(len(names_o)))
ax_m.set_xticklabels(names_o, rotation=65, ha="right", fontsize=9)
ax_m.set_ylabel(r"Additive Importance (symlog scale, negatives shown)", fontsize=11)
ax_m.grid(axis="y", alpha=0.3, zorder=0, linestyle="--")
ax_m.spines["right"].set_visible(False)
ax_m.spines["top"].set_visible(False)
ax_m.set_xlim(-0.8, len(names_o) - 0.2)

# Annotate top 5 positive
top5 = sorted(range(len(vals_o)), key=lambda i: vals_o[i], reverse=True)[:5]
for i in top5:
    ypos = vals_o[i]
    ax_m.text(i, ypos * 1.5, f"{ypos:.2f}",
              ha="center", va="bottom", fontsize=8.5, fontweight="bold",
              color=colors_o[i])

# Annotate negative bars with their actual values
neg_idx = [i for i, v in enumerate(vals_o) if v < 0]
for i in neg_idx:
    ax_m.text(i, vals_o[i] * 1.5 if vals_o[i] < -LINTHRESH else -LINTHRESH * 2.5,
              f"{vals_o[i]:.4f}",
              ha="center", va="top", fontsize=7.5, color="#333",
              fontweight="bold")

# ── Legend ────────────────────────────────────────────────────────────────────
clades_present = {r["clade"] for r in rows}
handles = [mpatches.Patch(facecolor=CLADE_COLORS.get(c, "#adb5bd"), label=c,
                          edgecolor="grey", linewidth=0.5)
           for c in CLADE_COLORS if c in clades_present]
handles.append(mpatches.Patch(facecolor="#adb5bd", label="Negative (hatch = hurts prediction)",
                               hatch="///", edgecolor="#333"))
fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10,
           framealpha=0.92, title="Mammalian clade", title_fontsize=10,
           bbox_to_anchor=(0.5, 1.05))

fig.suptitle(
    "Species Importance: Additive Perturbation  \u00b7  58 Placental Mammals  \u00b7  500k variants",
    fontsize=15, y=1.12, fontweight="bold")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"✓ Wrote Figure 6 → {OUT_PATH}", flush=True)
