#!/usr/bin/env python
# Copyright (c) 2025 Dongjoon Lim, McGill University
# Licensed under the MIT License. See LICENSE file in the project root.
"""
Multi-chromosome flank ablation training for GraphyloVar CNN-GCN v2.

Split strategy:
    Train: chr1-8
    Val:   chr9-10
    Test:  chr11-22

Usage:
    python scripts/train_ablation.py --context_flank 32 --gpu 5
    python scripts/train_ablation.py --context_flank 0 --gpu 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ensure_binary_labels(y: np.ndarray) -> np.ndarray:
    """Convert label tensors to binary one-hot shape (N, 2) for this model."""
    if y.ndim == 3 and y.shape[-1] == 2:
        center = y.shape[1] // 2
        return y[:, center, :]
    return y


def load_multi_chrom(data_dir: str, chroms: list, context: int, context_flank: int):
    """Load and concatenate data from multiple chromosomes."""
    from graphylovar.data import load_cadd_data
    X_all, y_all = [], []
    for c in chroms:
        xpath = os.path.join(data_dir, f"X_graphylo_chr{c}.npy")
        ypath = os.path.join(data_dir, f"y_graphylo_chr{c}.npy")
        if not os.path.exists(xpath) or not os.path.exists(ypath):
            print(f"  WARNING: chr{c} data not found, skipping")
            continue
        print(f"  Loading chr{c}...")
        X, y = load_cadd_data(data_dir, chromosome=c,
                              context=context, context_flank=context_flank)
        y = _ensure_binary_labels(y)
        print(f"    X: {X.shape}, y: {y.shape}")
        X_all.append(X)
        y_all.append(y)
    if not X_all:
        raise RuntimeError(f"No data found in {data_dir} for chroms {chroms}")
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    print(f"  Combined: X={X.shape}, y={y.shape}")
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Flank ablation training")
    parser.add_argument("--context_flank", type=int, required=True,
                        help="Flanking context size (0, 8, 32, 50, 100)")
    parser.add_argument("--gpu", type=str, default="5",
                        help="GPU device ID")
    parser.add_argument("--data_dir", type=str,
                        default="/home/mcb/users/dlim63/research/alignment/data")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--context", type=int, default=100,
                        help="Original preprocessing context window")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=9999)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--model_name", type=str, default="cnn_gcn_v2")
    parser.add_argument("--loss", type=str, default="focal")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument(
        "--allow_val_from_train",
        action="store_true",
        help="If val chromosomes are unavailable, carve out validation split from loaded training data.",
    )
    parser.add_argument(
        "--val_split_fraction",
        type=float,
        default=0.1,
        help="Validation fraction used when --allow_val_from_train is enabled.",
    )
    args = parser.parse_args()

    # GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import tensorflow as tf
    for device in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)

    from graphylovar.data import mask_species
    from graphylovar.losses import get_loss
    from graphylovar.model_io import normalize_model_path
    from graphylovar.models import build_model
    from graphylovar.phylogeny import build_graph
    from graphylovar.training import train_model

    # Build phylogenetic graph
    _, A = build_graph()

    # Define chromosome splits
    train_chroms = [1, 2, 3, 4, 5, 6, 7, 8]
    val_chroms   = [9, 10]
    # test_chroms = [11-22] -- used later for evaluation

    # Load data
    print(f"\n{'='*60}")
    print(f"Flank ablation: context_flank={args.context_flank}")
    print(f"{'='*60}")

    print("\nLoading TRAINING data (chr1-8)...")
    X_train, y_train = load_multi_chrom(
        args.data_dir, train_chroms, args.context, args.context_flank
    )

    print("\nLoading VALIDATION data (chr9-10)...")
    try:
        X_val, y_val = load_multi_chrom(
            args.data_dir, val_chroms, args.context, args.context_flank
        )
    except RuntimeError as exc:
        if not args.allow_val_from_train:
            raise
        frac = float(args.val_split_fraction)
        if not (0.0 < frac < 0.5):
            raise ValueError("--val_split_fraction must be in (0, 0.5)") from exc
        n = X_train.shape[0]
        n_val = max(1, int(n * frac))
        rng = np.random.default_rng(42)
        idx = np.arange(n)
        rng.shuffle(idx)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        X_train, y_train = X_train[train_idx], y_train[train_idx]
        print("  WARNING: validation chromosomes missing; using train/val split fallback")
        print(f"  Fallback split: train={X_train.shape[0]} val={X_val.shape[0]}")

    # Mask species (human, chimp, gorilla, ancestors)
    print("\nMasking species...")
    X_train = mask_species(X_train)
    X_val = mask_species(X_val)

    # Build model
    loss_fn = get_loss(args.loss, gamma=args.focal_gamma, alpha=args.focal_alpha)
    input_shape = X_train.shape[1:]  # (115, seq_len)
    print(f"\nInput shape: {input_shape}")
    model = build_model(args.model_name, input_shape=input_shape, A=A, loss=loss_fn)

    # Train
    ctx_label = args.context_flank * 2 + 1
    save_name = f"{args.model_name}_flank{args.context_flank}_ctx{ctx_label}"
    save_path = normalize_model_path(os.path.join(args.model_dir, save_name))

    print(f"\n{'='*60}")
    print(f"Training {args.model_name} with flank={args.context_flank}")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Save:  {save_path}")
    print(f"{'='*60}\n")

    history, model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        save_path=save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    # Persist a standard history CSV for downstream reviewer/ablation pipelines.
    history_path = os.path.splitext(save_path)[0] + "_history.csv"
    pd.DataFrame(history.history).to_csv(history_path, index=False)

    # Print final results
    best_val_loss = min(history.history["val_loss"])
    best_val_acc = max(history.history["val_accuracy"])
    n_epochs = len(history.history["loss"])
    print(f"\n{'='*60}")
    print(f"DONE: flank={args.context_flank}")
    print(f"  Epochs trained: {n_epochs}")
    print(f"  Best val_loss:  {best_val_loss:.6f}")
    print(f"  Best val_acc:   {best_val_acc:.4f}")
    print(f"  Model saved:    {save_path}")
    print(f"  History saved:  {history_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
