#!/usr/bin/env python
# Copyright (c) 2025 Dongjoon Lim, McGill University
# Licensed under the MIT License. See LICENSE file in the project root.
"""
Low-RAM TOPMed preprocessing for GraphyloVar pretraining.

This script processes one chromosome at a time and writes mmap-backed `.npy`
files without building giant intermediate arrays in memory.

Outputs per chromosome:
    X_graphylovar_topmed_chr{chrom}.npy           uint8   (N, 115, 4*context+2)
    y_graphylovar_topmed_chr{chrom}.npy           float32 (N, 6)
    positions_graphylovar_topmed_chr{chrom}.npy   int64   (N,)
    metadata_graphylovar_topmed_chr{chrom}.json
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphylovar.topmed import (  # noqa: E402
    extract_example,
    load_topmed_labels,
    make_reference_target,
    parse_chromosome_spec,
    sample_nonvariant_positions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess TOPMed GraphyloVar data")
    parser.add_argument("--chromosomes", default="1-22",
                        help="Chromosomes to process, e.g. 1-10,11,12")
    parser.add_argument("--topmed_tsv", required=True,
                        help="Processed TSV with TOPMed allele_frequency_vector labels")
    parser.add_argument("--alignment_dir", required=True,
                        help="Directory containing seqDictPad_chr*.pkl")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for mmap-backed NumPy arrays")
    parser.add_argument("--context", type=int, default=100,
                        help="Preprocessing flank size per side before reverse complement")
    parser.add_argument("--negative_ratio", type=float, default=1.0,
                        help="Number of sampled non-polymorphic sites per positive TOPMed site")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_positive", type=int, default=None,
                        help="Optional cap for debugging / smoke tests")
    parser.add_argument("--max_negative", type=int, default=None,
                        help="Optional cap for debugging / smoke tests")
    return parser


def _shrink_memmap(path: str, dtype, shape: tuple[int, ...], used: int) -> None:
    if used >= shape[0]:
        return
    mmap = np.load(path, mmap_mode="r")
    trimmed = np.asarray(mmap[:used], dtype=dtype)
    del mmap

    directory = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".npy", delete=False) as handle:
        temp_path = handle.name

    try:
        np.save(temp_path, trimmed)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def process_chromosome(
    chromosome: int,
    topmed_tsv: str,
    alignment_dir: str,
    output_dir: str,
    context: int,
    negative_ratio: float,
    seed: int,
    max_positive: int | None,
    max_negative: int | None,
) -> dict:
    print(f"\n=== chr{chromosome} ===", flush=True)
    labels = load_topmed_labels(topmed_tsv, chromosome)
    if labels.empty:
        print("No TOPMed SNV labels found; skipping", flush=True)
        return {"chromosome": chromosome, "status": "skipped_no_labels"}

    if max_positive is not None:
        if len(labels) > max_positive:
            labels = labels.sample(n=max_positive, random_state=seed + chromosome).copy()
            labels = labels.sort_values("chromEnd").reset_index(drop=True)

    alignment_path = os.path.join(alignment_dir, f"seqDictPad_chr{chromosome}.pkl")
    print(f"Loading alignment: {alignment_path}", flush=True)
    alignment = pd.read_pickle(alignment_path)
    hg38 = alignment["hg38"]

    positive_positions_zero_based = (labels["chromEnd"].astype(np.int64) - 1).tolist()
    negative_count = int(round(len(positive_positions_zero_based) * negative_ratio))
    if max_negative is not None:
        negative_count = min(negative_count, max_negative)

    negative_positions_zero_based = sample_nonvariant_positions(
        hg38_sequence=hg38,
        excluded_positions_zero_based=set(positive_positions_zero_based),
        sample_count=negative_count,
        context=context,
        seed=seed + chromosome,
    )

    total_samples = len(positive_positions_zero_based) + len(negative_positions_zero_based)
    seq_len = context * 4 + 2
    print(
        f"positives={len(positive_positions_zero_based):,} "
        f"negatives={len(negative_positions_zero_based):,} total={total_samples:,}"
    , flush=True)

    os.makedirs(output_dir, exist_ok=True)
    x_path = os.path.join(output_dir, f"X_graphylovar_topmed_chr{chromosome}.npy")
    y_path = os.path.join(output_dir, f"y_graphylovar_topmed_chr{chromosome}.npy")
    pos_path = os.path.join(output_dir, f"positions_graphylovar_topmed_chr{chromosome}.npy")
    meta_path = os.path.join(output_dir, f"metadata_graphylovar_topmed_chr{chromosome}.json")

    X = open_memmap(x_path, mode="w+", dtype=np.uint8, shape=(total_samples, 115, seq_len))
    Y = open_memmap(y_path, mode="w+", dtype=np.float32, shape=(total_samples, 6))
    P = open_memmap(pos_path, mode="w+", dtype=np.int64, shape=(total_samples,))

    cursor = 0
    written_positive = 0
    for position_zero_based, target in zip(
        positive_positions_zero_based,
        labels["allele_frequency_vector"],
    ):
        example = extract_example(alignment, position_zero_based, context)
        if example is None:
            continue
        X[cursor] = example
        Y[cursor] = target
        P[cursor] = position_zero_based + 1
        cursor += 1
        written_positive += 1

    written_negative = 0
    for position_zero_based in negative_positions_zero_based:
        example = extract_example(alignment, position_zero_based, context)
        if example is None:
            continue
        X[cursor] = example
        Y[cursor] = make_reference_target(hg38[position_zero_based])
        P[cursor] = position_zero_based + 1
        cursor += 1
        written_negative += 1

    X.flush()
    Y.flush()
    P.flush()

    _shrink_memmap(x_path, np.uint8, (total_samples, 115, seq_len), cursor)
    _shrink_memmap(y_path, np.float32, (total_samples, 6), cursor)
    _shrink_memmap(pos_path, np.int64, (total_samples,), cursor)

    metadata = {
        "chromosome": chromosome,
        "context": context,
        "sequence_length": seq_len,
        "positive_requested": len(positive_positions_zero_based),
        "negative_requested": len(negative_positions_zero_based),
        "positive_written": written_positive,
        "negative_written": written_negative,
        "total_written": written_positive + written_negative,
        "negative_ratio": negative_ratio,
        "topmed_tsv": str(topmed_tsv),
        "alignment_path": alignment_path,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(
        f"Wrote chr{chromosome}: positives={written_positive:,}, "
        f"negatives={written_negative:,}, total={written_positive + written_negative:,}"
    , flush=True)
    return metadata


def main() -> None:
    args = build_parser().parse_args()
    chromosomes = parse_chromosome_spec(args.chromosomes)
    summary = []
    for chromosome in chromosomes:
        summary.append(
            process_chromosome(
                chromosome=chromosome,
                topmed_tsv=args.topmed_tsv,
                alignment_dir=args.alignment_dir,
                output_dir=args.output_dir,
                context=args.context,
                negative_ratio=args.negative_ratio,
                seed=args.seed,
                max_positive=args.max_positive,
                max_negative=args.max_negative,
            )
        )

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = Path(args.output_dir) / "topmed_preprocess_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nSaved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
