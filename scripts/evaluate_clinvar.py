#!/usr/bin/env python
"""
Evaluate a trained GraphyloVar model on ClinVar data.

Usage:
    python scripts/evaluate_clinvar.py \\
        --model_path models/graphylo_cadddata_focalloss \\
        --x_clinvar data/X_clinvar.npy \\
        --y_clinvar data/y_clinvar.npy \\
        --output_dir figures/clinvar
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from graphylovar.evaluation import score_clinvar


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GraphyloVar on ClinVar variants"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--x_clinvar", type=str, required=True,
                        help="Path to X_clinvar.npy")
    parser.add_argument("--y_clinvar", type=str, required=True,
                        help="Path to y_clinvar.npy")
    parser.add_argument("--output_dir", type=str, default="figures/clinvar")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_name", type=str, default="GraphyloVar")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(f"Loading model: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path, compile=False)

    X = np.load(args.x_clinvar)
    y = np.load(args.y_clinvar)
    print(f"ClinVar data: X={X.shape}, y={y.shape}")

    results = score_clinvar(
        model, X, y,
        model_name=args.model_name,
        batch_size=args.batch_size,
        save_dir=args.output_dir,
    )
    print(f"\nFinal: AUROC={results['auroc']:.4f}, AUPRC={results['auprc']:.4f}")


if __name__ == "__main__":
    main()
