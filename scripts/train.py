#!/usr/bin/env python
# Copyright (c) 2025 Dongjoon Lim, McGill University
# Licensed under the MIT License. See LICENSE file in the project root.
"""
Train a GraphyloVar model.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --model cnn_gcn --loss focal --batch_size 64
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from graphylovar.data import load_cadd_data, prepare_train_val
from graphylovar.losses import get_loss
from graphylovar.models import build_model
from graphylovar.phylogeny import build_graph
from graphylovar.training import train_model


def main():
    parser = argparse.ArgumentParser(description="Train a GraphyloVar model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name: cnn_gcn, lstm_gcn, transformer_gcn, evolstm")
    parser.add_argument("--loss", type=str, default=None)
    parser.add_argument("--chromosome", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    args = parser.parse_args()

    # ── Load config ─────────────────────────────────────────────────
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    # CLI overrides
    model_name = args.model or cfg.get("model_name", "cnn_gcn")
    loss_name = args.loss or cfg.get("loss", "focal")
    chrom = args.chromosome or cfg.get("chromosome", 1)
    batch_size = args.batch_size or cfg.get("batch_size", 64)
    epochs = args.epochs or cfg.get("epochs", 9999)
    gpu_id = args.gpu or cfg.get("gpu_id", "3")
    data_dir = args.data_dir or cfg.get("data_dir", "data")
    model_dir = args.model_dir or cfg.get("model_dir", "models")
    context = cfg.get("context", 100)
    context_flank = cfg.get("context_flank", 10)
    test_size = cfg.get("test_size", 0.2)
    random_state = cfg.get("random_state", 42)
    patience = cfg.get("patience", 7)
    focal_gamma = cfg.get("focal_gamma", 2.0)
    focal_alpha = cfg.get("focal_alpha", 0.25)

    # ── GPU setup ───────────────────────────────────────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for device in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)

    # ── Build phylogenetic graph ────────────────────────────────────
    _, A = build_graph()

    # ── Load data ───────────────────────────────────────────────────
    try:
        chrom = int(chrom)
    except (ValueError, TypeError):
        pass

    print(f"Loading chr{chrom} from {data_dir} ...")
    X, y = load_cadd_data(data_dir, chromosome=chrom,
                          context=context, context_flank=context_flank)
    print(f"  X: {X.shape}, y: {y.shape}")

    splits = prepare_train_val(X, y, test_size=test_size,
                               random_state=random_state)

    # ── Build model ─────────────────────────────────────────────────
    loss_fn = get_loss(loss_name, gamma=focal_gamma, alpha=focal_alpha)
    input_shape = splits["X_train"].shape[1:]  # (115, seq_len)

    if model_name == "evolstm":
        # EvoLSTM baseline uses only single species row
        model = build_model(model_name, input_length=input_shape[-1], loss=loss_fn)
        X_tr = splits["X_evolstm_train"]
        X_vl = splits["X_evolstm_val"]
    else:
        model = build_model(model_name, input_shape=input_shape, A=A, loss=loss_fn)
        X_tr = splits["X_train"]
        X_vl = splits["X_val"]

    # ── Train ───────────────────────────────────────────────────────
    ctx_label = context_flank * 2 + 1
    save_name = f"{model_name}_{loss_name}_context{ctx_label}_chr{chrom}"
    save_path = os.path.join(model_dir, save_name)

    print(f"\nTraining {model_name} → {save_path}")
    history, model = train_model(
        model=model,
        X_train=X_tr,
        y_train=splits["y_train"],
        X_val=X_vl,
        y_val=splits["y_val"],
        save_path=save_path,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )

    print(f"\nDone. Best model saved to: {save_path}")


if __name__ == "__main__":
    main()
