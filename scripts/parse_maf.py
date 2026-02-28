#!/usr/bin/env python
# Copyright (c) 2025 Dongjoon Lim, McGill University
# Licensed under the MIT License. See LICENSE file in the project root.
"""
Parse raw MAF alignment files into pickle dictionaries.

Usage:
    python scripts/parse_maf.py --maf data/chr22.anc.maf --chrom 22 \\
        --output data/seqDictPad_chr22.pkl
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphylovar.maf_parser import parse_and_save


def main():
    parser = argparse.ArgumentParser(
        description="Parse MAF alignment files to pickle format"
    )
    parser.add_argument("--maf", type=str, required=True,
                        help="Path to the .maf file")
    parser.add_argument("--chrom", type=str, required=True,
                        help="Chromosome label (e.g. 22, X)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .pkl path (default: seqDictPad_chr{chrom}.pkl)")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"seqDictPad_chr{args.chrom}.pkl"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    parse_and_save(args.maf, args.output, chromosome=args.chrom)


if __name__ == "__main__":
    main()
