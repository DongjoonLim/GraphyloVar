#!/usr/bin/env python
# Copyright (c) 2025 Dongjoon Lim, McGill University
# Licensed under the MIT License. See LICENSE file in the project root.
"""
Generate genome-wide prediction bedGraph tracks.

Slides a window across a chromosome using a trained GraphyloVar model
and writes a bedGraph file suitable for genome browser visualisation.

Adapted from ``conservation/predictGraphyloScore_newCores.py``.

Usage:
    python scripts/predict_genome.py \\
        --model_path models/graphylo_cadddata_focalloss \\
        --alignment_dir data \\
        --chrom 22 --start 16000000 --end 51000000 \\
        --output graphylo_chr22.bdg
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphylovar.data import label_encode, reverse_complement, mask_species
from graphylovar.phylogeny import NAMES


def predict_window(
    model: tf.keras.Model,
    alignment: dict,
    start: int,
    end: int,
    context: int = 100,
    step: int = 10,
    batch_size: int = 64,
) -> tuple[list[int], list[int], np.ndarray]:
    """
    Generate predictions for a genomic region.

    Parameters
    ----------
    model     : trained model
    alignment : alignment dict (species -> list[str])
    start     : start position (alignment coords)
    end       : end position  (alignment coords)
    context   : flanking bases
    step      : step size between predictions
    batch_size : model batch size

    Returns
    -------
    (starts, ends, predictions) lists
    """
    le = label_encode()
    indices_start = []
    indices_end = []
    examples = []

    for i in range(start, end, step):
        try:
            if alignment["hg38"][i] == "N":
                continue
            example = []
            for spec in NAMES:
                seg = alignment[spec][i - context : i + context + 1]
                rc = reverse_complement(seg)
                encoded = le.transform(list(seg) + rc)
                example.append(encoded)
            arr = np.array(example, dtype=np.uint8)
            if arr.shape[0] != len(NAMES):
                continue
            examples.append(arr)
            indices_start.append(i - step // 2)
            indices_end.append(i + step // 2)
        except Exception:
            continue

    if not examples:
        return [], [], np.array([])

    X = np.array(examples, dtype=np.uint8)
    X = mask_species(X)
    predictions = model.predict(X, batch_size=batch_size)
    if predictions.ndim == 2 and predictions.shape[1] == 2:
        predictions = predictions[:, 1]

    return indices_start, indices_end, predictions


def write_bedgraph(
    output_path: str,
    chromosome: str,
    starts: list[int],
    ends: list[int],
    scores: np.ndarray,
    track_name: str = "GraphyloVar",
) -> None:
    """Write a bedGraph file."""
    color = f"{random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)}"
    with open(output_path, "w") as fh:
        fh.write(
            f'track type=bedGraph name="{track_name}" '
            f'description="{track_name}" visibility=full '
            f"color={color} altColor=0,100,200 priority=20\n"
        )
        for s, e, score in zip(starts, ends, scores):
            fh.write(f"chr{chromosome}\t{s}\t{e}\t{score:.6f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate genome-wide GraphyloVar prediction bedGraph"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--alignment_dir", type=str, default="data",
                        help="Directory containing seqDictPad_chr*.pkl")
    parser.add_argument("--chrom", type=str, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--context", type=int, default=100)
    parser.add_argument("--step", type=int, default=10,
                        help="Step size between predictions")
    parser.add_argument("--stride", type=int, default=1000,
                        help="Chunk size for progress bar")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--track_name", type=str, default="GraphyloVar")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ── Load model ──────────────────────────────────────────────────
    print(f"Loading model: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path, compile=False)

    # ── Load alignment ──────────────────────────────────────────────
    pkl = os.path.join(args.alignment_dir, f"seqDictPad_chr{args.chrom}.pkl")
    print(f"Loading alignment: {pkl}")
    alignment = pd.read_pickle(pkl)
    print(f"  Alignment length: {len(alignment['hg38'])}")

    # ── Predict in chunks ───────────────────────────────────────────
    if args.output is None:
        args.output = f"graphylovar_chr{args.chrom}_{args.start}_{args.end}.bdg"

    all_starts, all_ends, all_scores = [], [], []
    for chunk_start in tqdm(range(args.start, args.end, args.stride)):
        chunk_end = min(chunk_start + args.stride, args.end)
        try:
            s, e, p = predict_window(
                model, alignment, chunk_start, chunk_end,
                context=args.context, step=args.step,
                batch_size=args.batch_size,
            )
            all_starts.extend(s)
            all_ends.extend(e)
            all_scores.extend(p.tolist() if len(p) > 0 else [])
        except Exception:
            continue

    write_bedgraph(
        args.output, args.chrom,
        all_starts, all_ends, np.array(all_scores),
        track_name=args.track_name,
    )
    print(f"Saved {len(all_starts)} predictions to {args.output}")


if __name__ == "__main__":
    main()
