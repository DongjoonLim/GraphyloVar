#!/usr/bin/env python
# Copyright (c) 2025 Dongjoon Lim, McGill University
# Licensed under the MIT License. See LICENSE file in the project root.
"""
Preprocess multi-species alignment pickles + CADD VCFs into
training-ready .npy arrays for GraphyloVar.

Usage:
    python scripts/preprocess.py --config configs/default.yaml
    python scripts/preprocess.py --chromosomes 1 2 3 --cadd_dir cadd_data
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

# Ensure package is importable when running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphylovar.data import preprocess_cadd_chromosome


def main():
    parser = argparse.ArgumentParser(description="Preprocess CADD data for GraphyloVar")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--chromosomes", nargs="+", default=None,
                        help="Chromosomes to process (e.g. 1 2 3 X Y)")
    parser.add_argument("--cadd_dir", type=str, default=None)
    parser.add_argument("--alignment_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--context", type=int, default=None)
    args = parser.parse_args()

    # Load config
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    # CLI overrides
    cadd_dir = args.cadd_dir or cfg.get("cadd_dir", "cadd_data")
    alignment_dir = args.alignment_dir or cfg.get("alignment_dir", "../../conservation/data")
    output_dir = args.output_dir or cfg.get("data_dir", "data")
    context = args.context or cfg.get("context", 100)

    if args.chromosomes:
        chromosomes = args.chromosomes
    else:
        chromosomes = [cfg.get("chromosome", 1)]

    for chrom in chromosomes:
        # Try to convert to int if possible
        try:
            chrom = int(chrom)
        except (ValueError, TypeError):
            pass
        print(f"\n{'='*60}")
        print(f"Processing chromosome {chrom}")
        print(f"{'='*60}")
        preprocess_cadd_chromosome(
            chromosome=chrom,
            cadd_dir=cadd_dir,
            alignment_dir=alignment_dir,
            output_dir=output_dir,
            context=context,
        )

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
