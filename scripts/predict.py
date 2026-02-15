#!/usr/bin/env python
"""
Run inference with a trained GraphyloVar model.

Usage:
    python scripts/predict.py \\
        --model_path models/cnn_gcn_focal_context21_chr1 \\
        --data_dir data --chromosome 22
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from graphylovar.data import load_cadd_data, mask_species


def main():
    parser = argparse.ArgumentParser(description="Predict with a trained GraphyloVar model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--chromosome", type=str, default="22")
    parser.add_argument("--context", type=int, default=100)
    parser.add_argument("--context_flank", type=int, default=10)
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npy file for predictions")
    parser.add_argument("--gpu", type=str, default="3")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    # ── GPU setup ───────────────────────────────────────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for device in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)

    # ── Load data ───────────────────────────────────────────────────
    try:
        chrom = int(args.chromosome)
    except (ValueError, TypeError):
        chrom = args.chromosome

    print(f"Loading chr{chrom} from {args.data_dir} ...")
    X, y = load_cadd_data(
        args.data_dir, chromosome=chrom,
        context=args.context, context_flank=args.context_flank,
    )
    X = mask_species(X)
    print(f"  X: {X.shape}, y: {y.shape}")

    # ── Load model & predict ────────────────────────────────────────
    print(f"Loading model from {args.model_path} ...")
    model = tf.keras.models.load_model(args.model_path, compile=False)
    model.summary()

    preds = model.predict(X, batch_size=args.batch_size)
    print(f"Predictions shape: {preds.shape}")

    # ── Save ────────────────────────────────────────────────────────
    if args.output is None:
        base = os.path.basename(args.model_path)
        args.output = f"predictions_{base}_chr{chrom}.npy"

    np.save(args.output, preds)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
