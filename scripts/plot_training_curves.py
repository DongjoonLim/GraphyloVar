#!/usr/bin/env python3
"""Plot GraphyloVar training curves with explicit V3 vs legacy run separation."""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RUN_COLORS = {
    "flank0": "#9467bd",
    "flank1": "#8c564b",
    "flank8": "#1f77b4",
    "flank16": "#ff7f0e",
    "flank32": "#2ca02c",
    "flank100": "#d62728",
    "serverflank8": "#1f77b4",
    "serverflank16": "#ff7f0e",
    "servermain": "#2ca02c",
    "serverflank100": "#d62728",
}

V3_ORDER = ["flank0", "flank1", "flank8", "flank16", "flank32", "flank100"]
LEGACY_ORDER = ["serverflank8", "serverflank16", "servermain", "serverflank100"]

_HISTORY_COLS = [
    "epoch",
    "binary_accuracy",
    "binary_loss",
    "loss",
    "lr",
    "nucleotide_accuracy",
    "nucleotide_loss",
    "val_binary_accuracy",
    "val_binary_loss",
    "val_loss",
    "val_nucleotide_accuracy",
    "val_nucleotide_loss",
]


def has_exact_flank_token(name: str, flank: int) -> bool:
    return bool(re.search(rf"(^|_)flank{flank}(_|[^0-9]|$)", name))


def has_exact_v3_run_tag(name: str, flank: int) -> bool:
    return bool(re.search(rf"(^|_)v3flank{flank}(_|[^0-9]|$)", name))


def classify_run(path: str) -> tuple[str, str] | None:
    basename = os.path.basename(path).replace("_history.csv", "")
    if "/v3_ablation/" in path:
        is_hybrid_v3 = "multitask_hybrid_v3" in basename
        for tag in V3_ORDER:
            flank = int(tag.replace("flank", ""))
            if is_hybrid_v3 and has_exact_flank_token(basename, flank):
                return tag, "v3"
            if has_exact_v3_run_tag(basename, flank):
                return tag, "v3"
    for tag in LEGACY_ORDER:
        if basename.endswith(tag):
            return tag, "legacy"
    return None


def find_history_csvs(model_root: str, run_set: str) -> dict[str, str]:
    pattern = os.path.join(model_root, "**", "*_history.csv")
    best: dict[str, tuple[str, float]] = {}
    for path in glob.glob(pattern, recursive=True):
        classified = classify_run(path)
        if classified is None:
            continue
        tag, family = classified
        if run_set == "v3" and family != "v3":
            continue
        if run_set == "legacy" and family != "legacy":
            continue
        mtime = os.path.getmtime(path)
        prev = best.get(tag)
        if prev is None or mtime > prev[1]:
            best[tag] = (path, mtime)
    return {tag: path for tag, (path, _) in best.items()}


def load_history(csv_path: str) -> pd.DataFrame | None:
    try:
        with open(csv_path, "r", encoding="utf-8") as handle:
            all_lines = handle.readlines()
    except Exception:
        return None

    good_rows = []
    canonical_ncols = len(_HISTORY_COLS)
    for raw in all_lines:
        stripped = raw.strip()
        if not stripped:
            continue
        parts = stripped.split(",")
        try:
            float(parts[0])
        except ValueError:
            continue
        if len(parts) < canonical_ncols - 1:
            continue
        while len(parts) < canonical_ncols:
            parts.append("")
        if len(parts) > canonical_ncols:
            parts = parts[:canonical_ncols]
        good_rows.append(parts)

    if not good_rows:
        return None

    df = pd.DataFrame(good_rows, columns=_HISTORY_COLS)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["epoch"] = range(len(df))
    return df


def ordered_tags(histories: dict[str, pd.DataFrame]) -> list[str]:
    tags = []
    for tag in V3_ORDER + LEGACY_ORDER:
        if tag in histories:
            tags.append(tag)
    for tag in histories:
        if tag not in tags:
            tags.append(tag)
    return tags


def plot_metric(
    ax: plt.Axes,
    histories: dict[str, pd.DataFrame],
    metric: str,
    val_metric: str,
    title: str,
    ylabel: str,
    log_scale: bool = False,
) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)

    for tag in ordered_tags(histories):
        df = histories[tag]
        color = RUN_COLORS.get(tag)
        label_root = tag.replace("server", "")
        if metric in df.columns:
            ax.plot(df["epoch"], df[metric], color=color, linewidth=1.5, label=f"{label_root} train", alpha=0.9)
        if val_metric in df.columns:
            ax.plot(
                df["epoch"],
                df[val_metric],
                color=color,
                linewidth=1.5,
                linestyle="--",
                label=f"{label_root} val",
                alpha=0.7,
            )

    if log_scale:
        ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)


def summarize(histories: dict[str, pd.DataFrame], csv_map: dict[str, str]) -> pd.DataFrame:
    rows = []
    for tag in ordered_tags(histories):
        df = histories[tag]
        last = df.iloc[-1]
        best_idx = df["val_loss"].idxmin() if "val_loss" in df.columns else None
        family = "v3" if tag in V3_ORDER else "legacy"
        rows.append(
            {
                "run": tag,
                "family": family,
                "source": csv_map.get(tag, ""),
                "epochs": len(df),
                "last_train_loss": round(float(last.get("loss", float("nan"))), 5),
                "last_val_loss": round(float(last.get("val_loss", float("nan"))), 5),
                "best_val_loss": round(float(df["val_loss"].min()), 5) if "val_loss" in df.columns else float("nan"),
                "best_val_epoch": int(best_idx) if best_idx is not None else -1,
                "last_nuc_acc": round(float(last.get("nucleotide_accuracy", float("nan"))), 4),
                "last_val_nuc_acc": round(float(last.get("val_nucleotide_accuracy", float("nan"))), 4),
            }
        )
    return pd.DataFrame(rows)


def render_outputs(histories: dict[str, pd.DataFrame], plots_dir: str, summary_name: str, title_suffix: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"GraphyloVar TOPMed Pretraining - Loss Curves ({title_suffix})", fontsize=13)
    plot_metric(axes[0, 0], histories, "loss", "val_loss", "Total Loss", "Loss")
    plot_metric(axes[0, 1], histories, "nucleotide_loss", "val_nucleotide_loss", "Nucleotide Loss", "Loss")
    plot_metric(axes[1, 0], histories, "binary_loss", "val_binary_loss", "Binary Loss", "Binary Cross-Entropy", log_scale=True)
    plot_metric(axes[1, 1], histories, "nucleotide_accuracy", "val_nucleotide_accuracy", "Nucleotide Accuracy", "Accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "training_loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Binary Head Performance ({title_suffix})", fontsize=13)
    plot_metric(axes[0], histories, "binary_accuracy", "val_binary_accuracy", "Binary Accuracy", "Accuracy")
    plot_metric(axes[1], histories, "binary_loss", "val_binary_loss", "Binary Loss (log scale)", "Loss", log_scale=True)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "binary_head_performance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_metric(ax, histories, "val_loss", "val_loss", f"Validation Loss Overview ({title_suffix})", "Validation Loss")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "learning_curves_full.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plots for {title_suffix}")


def make_plots(model_root: str, plots_dir: str, run_set: str) -> None:
    os.makedirs(plots_dir, exist_ok=True)
    csv_map = find_history_csvs(model_root, run_set)
    if not csv_map:
        print(f"No history CSVs found for run_set={run_set} under {model_root}")
        return

    histories: dict[str, pd.DataFrame] = {}
    for tag, path in csv_map.items():
        df = load_history(path)
        if df is not None:
            histories[tag] = df
    if not histories:
        print("All discovered history CSVs were empty.")
        return

    render_outputs(histories, plots_dir, "training_summary.csv", title_suffix=run_set.upper())
    summary_df = summarize(histories, csv_map)
    summary_df.to_csv(os.path.join(plots_dir, "training_summary.csv"), index=False)
    print(f"Saved {os.path.join(plots_dir, 'training_summary.csv')}")

    if run_set == "v3":
        all_csv_map = find_history_csvs(model_root, "all")
        all_histories = {tag: load_history(path) for tag, path in all_csv_map.items()}
        all_histories = {tag: df for tag, df in all_histories.items() if df is not None}
        if all_histories:
            summarize(all_histories, all_csv_map).to_csv(
                os.path.join(plots_dir, "training_summary_all_runs.csv"),
                index=False,
            )
            print(f"Saved {os.path.join(plots_dir, 'training_summary_all_runs.csv')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GraphyloVar training curves")
    parser.add_argument(
        "--model_root",
        default=os.path.expanduser("~/GraphyloVar/topmed_models/full_streaming_runs"),
    )
    parser.add_argument(
        "--plots_dir",
        default=os.path.expanduser("~/GraphyloVar/topmed_models/full_streaming_runs/plots"),
    )
    parser.add_argument(
        "--run_set",
        choices=["v3", "legacy", "all"],
        default="v3",
        help="Which run family to summarize. Default is paper-facing V3 runs only.",
    )
    parser.add_argument("--watch", action="store_true", help="Continuously regenerate plots")
    parser.add_argument("--interval", type=int, default=120, help="Seconds between refreshes in watch mode")
    args = parser.parse_args()

    if args.watch:
        print(f"Watching for updates every {args.interval}s. Press Ctrl+C to stop.")
        while True:
            try:
                make_plots(args.model_root, args.plots_dir, args.run_set)
            except Exception as exc:
                print(f"Plot error (will retry): {exc}", file=sys.stderr)
            time.sleep(args.interval)
    else:
        make_plots(args.model_root, args.plots_dir, args.run_set)


if __name__ == "__main__":
    main()
