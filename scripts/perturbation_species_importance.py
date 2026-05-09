"""
perturbation_species_importance.py : Occlusion-based species importance for GraphyloVar.

Refreshes the canonical species-importance figures from a current V3 checkpoint.
For each species channel, the script replaces that channel with the GAP/N token
and measures the increase in nucleotide cross-entropy on a held-out sample.

Generates:
  - figures/perturbation_importance.npy
  - figures/perturbation_importance_summary.json
  - figures/gcn_phylo_variance_scatter.png
  - figures/gcn_species_importance.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
import numpy as np
import tensorflow as tf
from scipy.stats import linregress

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graphylovar.model_io import load_hdf5_weights_compat
from graphylovar.models import build_model
from graphylovar.model_io import resolve_model_path
from graphylovar.phylogeny import NAMES, NUM_NODES, build_graph
from graphylovar.topmed import extract_batch_examples_from_encoded, load_alignment_encoded_cache


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V3_ABLATION_DIR = os.path.join(REPO_ROOT, "topmed_models", "full_streaming_runs", "v3_ablation")
OUT_DIR = os.path.join(REPO_ROOT, "figures")
DEFAULT_RUN_TAG = "v3flank16"
DEFAULT_SAMPLE_CHROMOSOME = 12
DEFAULT_SAMPLE_SIZE = 512
DEFAULT_BATCH_SIZE = 64
GAP_TOKEN = 4
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

CLADES = {
    "Primates": ["Human", "Chimp", "Gorilla", "Orangutan", "Gibbon",
                  "Rhesus macaque", "Crab-eating macaque", "Baboon",
                  "Green monkey", "Marmoset", "Squirrel monkey",
                  "Bushbaby"],
    "Tree shrew": ["Tree shrew"],
    "Glires": ["Squirrel", "Jerboa", "Prairie vole", "Chinese hamster",
                "Golden hamster", "Mouse", "Rat", "Naked mole-rat",
                "Guinea pig", "Chinchilla", "Degu", "Rabbit", "Pika"],
    "Cetartiodactyla": ["Alpaca", "Bactrian camel", "Pig", "Dolphin",
                         "Killer whale", "Sheep", "Goat", "Cow", "Tibetan antelope"],
    "Carnivora & Perissodactyla": ["Horse", "White rhinoceros", "Walrus", "Weddell seal",
                                    "Panda", "Ferret", "Dog", "Cat"],
    "Chiroptera": ["Black flying fox", "Large flying fox",
                    "David's myotis", "Little brown bat", "Big brown bat"],
    "Eulipotyphla": ["Shrew", "Star-nosed mole", "Hedgehog"],
    "Afrotheria & Xenarthra": ["Elephant", "Cape elephant shrew", "Manatee",
                                "Cape golden mole", "Tenrec", "Aardvark", "Armadillo"],
}

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


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate GraphyloVar species importance from a V3 checkpoint.")
    parser.add_argument("--run_tag", default=DEFAULT_RUN_TAG, help="V3 run tag, for example v3flank16")
    parser.add_argument("--sample_chromosome", type=int, default=DEFAULT_SAMPLE_CHROMOSOME)
    parser.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--compact_dir", default="", help="Optional override for compact example directory")
    parser.add_argument("--align_cache_dir", default="", help="Optional override for alignment cache directory")
    parser.add_argument("--output_dir", default=OUT_DIR)
    return parser.parse_args()


def get_clade(name: str) -> str:
    for clade, members in CLADES.items():
        if name in members:
            return clade
    return "Other"


def branch_length_to_human_safe(name: str):
    from graphylovar.phylogeny import branch_length_to_human

    try:
        return branch_length_to_human(name)
    except Exception:
        return None


def find_v3_run_paths(run_tag: str) -> tuple[str, str]:
    config_candidates = sorted(glob.glob(os.path.join(V3_ABLATION_DIR, f"*_{run_tag}_config.json")))
    if not config_candidates:
        raise FileNotFoundError(f"No V3 config found for run tag {run_tag} in {V3_ABLATION_DIR}")
    if len(config_candidates) > 1:
        raise RuntimeError(f"Multiple configs matched {run_tag}: {config_candidates}")

    config_path = config_candidates[0]
    run_prefix = config_path[: -len("_config.json")]
    model_path = resolve_model_path(run_prefix)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Resolved model path does not exist: {model_path}")
    return config_path, model_path


def load_run_spec(run_tag: str) -> dict:
    config_path, model_path = find_v3_run_paths(run_tag)
    with open(config_path, encoding="utf-8") as handle:
        cfg = json.load(handle)

    compact_dir = cfg["compact_dir"]
    align_cache = cfg["alignment_cache_dir"]
    if not os.path.isabs(compact_dir):
        compact_dir = os.path.join(REPO_ROOT, compact_dir)
    if not os.path.isabs(align_cache):
        align_cache = os.path.join(REPO_ROOT, align_cache)

    keras_candidate = config_path.replace("_config.json", ".keras")
    if os.path.exists(keras_candidate):
        model_path = keras_candidate

    return {
        "run_tag": run_tag,
        "config_path": config_path,
        "model_path": model_path,
        "config": cfg,
        "compact_dir": compact_dir,
        "align_cache": align_cache,
        "context": int(cfg["context"]),
        "context_flank": int(cfg["context_flank"]),
        "species_masking": cfg.get("species_masking", "unknown"),
        "model_name": cfg.get("model_name", "unknown"),
    }


def load_model(run_spec: dict) -> dict:
    model_path = run_spec["model_path"]
    print(f"Loading model from: {model_path}")

    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "saved_model.pb")):
        loaded = tf.saved_model.load(model_path)
        infer = loaded.signatures.get("serving_default")
        if infer is None:
            raise RuntimeError(f"SavedModel missing serving_default signature: {model_path}")
        signature_kwargs = infer.structured_input_signature[1]
        if len(signature_kwargs) != 1:
            raise RuntimeError(f"Expected one named SavedModel input, got {list(signature_kwargs.keys())}")
        input_key = next(iter(signature_kwargs))
        return {
            "mode": "savedmodel-signature",
            "callable": infer,
            "raw_model": loaded,
            "input_key": input_key,
        }

    _, adjacency = build_graph()
    seq_len = (2 * int(run_spec["context_flank"]) + 1) * 2
    model = build_model(
        name=run_spec["model_name"],
        input_shape=(NUM_NODES, seq_len),
        A=adjacency,
        learning_rate=float(run_spec["config"].get("learning_rate", 3e-4)),
        binary_loss_weight=float(run_spec["config"].get("binary_loss_weight", 1.5)),
        dense_units=128,
        gcn_units=32,
    )
    load_hdf5_weights_compat(model, model_path)
    return {"mode": "keras", "callable": model}


def extract_nucleotide_predictions(predictions) -> np.ndarray:
    if isinstance(predictions, dict):
        if "nucleotide" in predictions:
            value = predictions["nucleotide"]
        else:
            nucleotide_keys = [key for key in predictions if "nucleotide" in key.lower()]
            value = predictions[nucleotide_keys[0]] if nucleotide_keys else next(iter(predictions.values()))
        return value.numpy() if hasattr(value, "numpy") else np.asarray(value)

    if isinstance(predictions, (list, tuple)):
        first = predictions[0]
        return first.numpy() if hasattr(first, "numpy") else np.asarray(first)

    return predictions.numpy() if hasattr(predictions, "numpy") else np.asarray(predictions)


def load_sample_batch(run_spec: dict, sample_chromosome: int, sample_size: int):
    compact_dir = run_spec["compact_dir"]
    align_cache = run_spec["align_cache"]
    context = run_spec["context"]
    context_flank = run_spec["context_flank"]

    pos_path = os.path.join(compact_dir, f"positions_graphylovar_topmed_chr{sample_chromosome}.npy")
    tgt_path = os.path.join(compact_dir, f"y_graphylovar_topmed_chr{sample_chromosome}.npy")
    meta_path = os.path.join(compact_dir, f"metadata_graphylovar_topmed_chr{sample_chromosome}.json")

    if not os.path.exists(pos_path):
        raise FileNotFoundError(f"Missing positions file: {pos_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)

    cache_arr, _ = load_alignment_encoded_cache(
        alignment_path=meta["alignment_path"],
        cache_dir=align_cache,
        chromosome=sample_chromosome,
    )

    positions = np.load(pos_path, mmap_mode="r")
    targets = np.load(tgt_path, mmap_mode="r")

    chosen = min(sample_size, len(positions))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(positions), size=chosen, replace=False)
    idx.sort()
    pos_batch = positions[idx].astype(np.int64) - 1
    tgt_batch = targets[idx].astype(np.float32)

    x_batch, valid = extract_batch_examples_from_encoded(
        encoded_alignment=cache_arr,
        positions_zero_based=pos_batch,
        context=context,
        context_flank=context_flank,
        mask_indices=None,
    )
    y_batch = tgt_batch[valid][:, :5]
    print(f"Loaded {x_batch.shape[0]} examples from chr{sample_chromosome}")
    return x_batch, y_batch


def cross_entropy_loss(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.clip(pred, 1e-7, 1.0)
    return float(-np.mean(np.sum(target * np.log(pred), axis=-1)))


def run_perturbation(model_wrapper: dict, X: np.ndarray, Y: np.ndarray, batch_size: int) -> np.ndarray:
    importance = np.zeros(len(NAMES), dtype=np.float32)

    def infer(batch_np):
        tensor = tf.constant(batch_np, dtype=tf.uint8)
        if model_wrapper["mode"] == "savedmodel-signature":
            predictions = model_wrapper["callable"](**{model_wrapper["input_key"]: tensor})
        else:
            predictions = model_wrapper["callable"](tensor, training=False)
        return extract_nucleotide_predictions(predictions)

    baseline_pred = []
    for start in range(0, len(X), batch_size):
        baseline_pred.append(infer(X[start:start + batch_size]))
    baseline_pred = np.concatenate(baseline_pred, axis=0)
    baseline_loss = cross_entropy_loss(baseline_pred, Y[:len(baseline_pred)])
    print(f"Baseline nucleotide loss: {baseline_loss:.6f}")

    for species_idx, name in enumerate(NAMES):
        if name.startswith("_"):
            continue

        X_perturbed = X.copy()
        X_perturbed[:, species_idx, :] = GAP_TOKEN

        perturbed_pred = []
        for start in range(0, len(X_perturbed), batch_size):
            perturbed_pred.append(infer(X_perturbed[start:start + batch_size]))
        perturbed_pred = np.concatenate(perturbed_pred, axis=0)
        perturbed_loss = cross_entropy_loss(perturbed_pred, Y[:len(perturbed_pred)])

        delta = perturbed_loss - baseline_loss
        importance[species_idx] = float(delta)
        common = COMMON_NAMES.get(name, name)
        print(f"  [{species_idx:3d}] {common:30s}  Δloss = {delta:+.6f}", flush=True)

    return importance


def make_plots(importance: np.ndarray, output_dir: str) -> float:
    plot_names, branch_lens, imp_vals = [], [], []
    for i, name in enumerate(NAMES):
        if name.startswith("_"):
            continue
        branch_length = branch_length_to_human_safe(name)
        if branch_length is None:
            continue
        plot_names.append(COMMON_NAMES.get(name, name))
        branch_lens.append(branch_length)
        imp_vals.append(float(importance[i]))

    x = np.array(branch_lens)
    y = np.array(imp_vals)
    clades = [get_clade(name) for name in plot_names]
    colors = [CLADE_COLORS.get(clade, "#aaaaaa") for clade in clades]

    slope, intercept, r, *_ = linregress(x, y)
    r2 = float(r ** 2)
    print(f"\nPerturbation R² vs UCSC branch length: {r2:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for clade, color in CLADE_COLORS.items():
        idx = [i for i, value in enumerate(clades) if value == clade]
        if idx:
            ax.scatter(x[idx], y[idx], color=color, edgecolors="k", linewidths=0.5, s=60, alpha=0.85, label=clade, zorder=3)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, slope * xs + intercept, color="#333333", linestyle="--", linewidth=1.8, label=f"$R^2 = {r2:.2f}$", zorder=4)
    ax.set_xlabel("Phylogenetic Branch Length from Human\n(UCSC expected substitutions / site)", fontsize=12)
    ax.set_ylabel("Species Importance\n(Δ nucleotide cross-entropy when occluded)", fontsize=12)
    ax.set_title("GraphyloVar Learns Phylogenetic Distance via Perturbation", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    scatter_path = os.path.join(output_dir, "gcn_phylo_variance_scatter.png")
    legacy_scatter_alias = os.path.join(output_dir, "species_importance_vs_ucsc_branch_length_scatter.png")
    fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
    fig.savefig(legacy_scatter_alias, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter → {scatter_path}")
    print(f"Saved scatter alias → {legacy_scatter_alias}")

    order = np.argsort(x)
    sorted_names = [plot_names[i] for i in order]
    sorted_colors = [colors[i] for i in order]
    sorted_vals = y[order]

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.bar(range(len(sorted_names)), sorted_vals, color=sorted_colors, edgecolor="k", linewidth=0.4, width=0.8)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=90, fontsize=9)
    ax.set_ylabel("Perturbation Importance (Δ loss)", fontsize=12)
    ax.set_title(
        "Species Perturbation Importance : GraphyloVar GCN\n"
        "(ordered by phylogenetic distance from Human, left→right)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(-0.6, len(sorted_names) - 0.4)
    legend_patches = [mpatches.Patch(facecolor=color, edgecolor="k", label=clade) for clade, color in CLADE_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9, framealpha=0.85)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    bar_path = os.path.join(output_dir, "gcn_species_importance.png")
    fig.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bar chart → {bar_path}")

    return r2


def build_summary(run_spec: dict, importance: np.ndarray, r2: float, sample_chromosome: int, sample_size: int) -> dict:
    ranked = []
    for i, name in enumerate(NAMES):
        if name.startswith("_"):
            continue
        ranked.append({
            "ucsc_name": name,
            "common_name": COMMON_NAMES.get(name, name),
            "importance": float(importance[i]),
            "branch_length_to_human": branch_length_to_human_safe(name),
        })
    ranked.sort(key=lambda row: row["importance"], reverse=True)

    explicit = {}
    for name in ["hg38", "panTro4", "gorGor3", "ponAbe2", "nomLeu3", "rheMac3"]:
        explicit[name] = float(importance[NAMES.index(name)])

    return {
        "run_tag": run_spec["run_tag"],
        "model_name": run_spec["model_name"],
        "checkpoint_path": run_spec["model_path"],
        "config_path": run_spec["config_path"],
        "species_masking": run_spec["species_masking"],
        "context": run_spec["context"],
        "context_flank": run_spec["context_flank"],
        "sample_chromosome": sample_chromosome,
        "sample_size_requested": sample_size,
        "perturbation_method": "species row ablation to GAP/N token (4); score = positive delta nucleotide cross-entropy",
        "importance_min": float(np.min(importance)),
        "importance_max": float(np.max(importance)),
        "phylogeny_distance_r2": float(r2),
        "explicit_species_scores": explicit,
        "top_species": ranked[:15],
    }


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    os.makedirs(args.output_dir, exist_ok=True)

    run_spec = load_run_spec(args.run_tag)
    if args.compact_dir:
        run_spec["compact_dir"] = args.compact_dir
    if args.align_cache_dir:
        run_spec["align_cache"] = args.align_cache_dir
    print(f"Using run tag: {run_spec['run_tag']}")
    print(f"Config path: {run_spec['config_path']}")
    print(f"Model path: {run_spec['model_path']}")
    print(f"species_masking: {run_spec['species_masking']}")

    model_wrapper = load_model(run_spec)
    print(f"Loaded via: {model_wrapper['mode']}")

    X, Y = load_sample_batch(run_spec, args.sample_chromosome, args.sample_size)
    print(f"\nRunning perturbation on {len(X)} examples × {len(NAMES)} species...")
    importance = run_perturbation(model_wrapper, X, Y, args.batch_size)

    npy_path = os.path.join(args.output_dir, "perturbation_importance.npy")
    np.save(npy_path, importance)
    print(f"Saved raw importance → {npy_path}")

    r2 = make_plots(importance, args.output_dir)
    summary = build_summary(run_spec, importance, r2, args.sample_chromosome, args.sample_size)
    summary_path = os.path.join(args.output_dir, "perturbation_importance_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved summary → {summary_path}")
    print(f"\nDone. R² = {r2:.4f}")


if __name__ == "__main__":
    main()
