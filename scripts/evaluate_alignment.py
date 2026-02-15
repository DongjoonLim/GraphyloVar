#!/usr/bin/env python
"""
Evaluate alignment quality using a trained GraphyloVar (or EvoLSTM) model.

Loads an alignment pickle, extracts substitution log-probabilities from
a trained model, runs EvolignSubst alignment, and computes F1 vs the
true (reference) alignment.

Usage:
    python scripts/evaluate_alignment.py \\
        --model_path models/graphylo_lstm_mutation \\
        --alignment_pkl ../../conservation/data/seqDictPad_chr22.pkl \\
        --s_species camFer1 --t_species mm10 \\
        --start 32490000 --stop 32590000
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from graphylovar.alignment import (
    evolign_subst,
    needleman_wunsch,
    alignment_f1,
    find_non_gap_indices,
    listify,
    ungap_common,
)
from graphylovar.data import extract_windows, label_encode, reverse_complement
from graphylovar.phylogeny import NAMES


def main():
    parser = argparse.ArgumentParser(description="Evaluate context-dependent alignment")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--alignment_pkl", type=str, required=True)
    parser.add_argument("--s_species", type=str, default="camFer1")
    parser.add_argument("--t_species", type=str, default="mm10")
    parser.add_argument("--start", type=int, default=32490000)
    parser.add_argument("--stop", type=int, default=32590000)
    parser.add_argument("--context", type=int, default=100)
    parser.add_argument("--gpu", type=str, default="3")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for device in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)

    # ── Load alignment ──────────────────────────────────────────────
    print(f"Loading alignment from {args.alignment_pkl} ...")
    alignment = pd.read_pickle(args.alignment_pkl)

    S_raw = "".join(alignment[args.s_species][args.start:args.stop])
    T_raw = "".join(alignment[args.t_species][args.start:args.stop])
    S = S_raw.replace("-", "").replace("N", "")
    T = T_raw.replace("-", "").replace("N", "")
    print(f"S ({args.s_species}): {len(S_raw)} raw → {len(S)} ungapped")
    print(f"T ({args.t_species}): {len(T_raw)} raw → {len(T)} ungapped")

    # ── Extract windows for model scoring ───────────────────────────
    indices = [i + args.start for i in find_non_gap_indices(S_raw)]
    examples = extract_windows(alignment, indices, context=args.context)
    print(f"Extracted {len(examples)} windows")

    # Mask human/chimp
    masked = examples.copy()
    masked[:, 0, :] = 0
    masked[:, 1, :] = 0

    # ── Get model predictions ───────────────────────────────────────
    print(f"Loading model from {args.model_path} ...")
    model = tf.keras.models.load_model(args.model_path, compile=False)
    tableM = np.log(np.clip(model.predict(masked, batch_size=256), 1e-10, 1.0))

    # ── Run alignments ──────────────────────────────────────────────
    print("\nRunning EvolignSubst alignment ...")
    aligned_S_model, aligned_T_model, score_model = evolign_subst(
        S, T, tableM,
        match_score=1, mismatch_score=-1,
        gap_open=-5, gap_extend=-2,
        seq_length=1, verbose=True,
    )

    print("\nRunning classical NW alignment ...")
    aligned_S_nw, aligned_T_nw, score_nw = needleman_wunsch(
        S, T,
        gap_open=-400, gap_extend=-30,
        use_lastz=True, verbose=True,
    )

    # ── Evaluate ────────────────────────────────────────────────────
    S_true, T_true = ungap_common(S_raw, T_raw)
    true_pairs = listify(S_true, T_true)
    model_pairs = listify(aligned_S_model, aligned_T_model)
    nw_pairs = listify(aligned_S_nw, aligned_T_nw)

    f1_model = alignment_f1(true_pairs, model_pairs)
    f1_nw = alignment_f1(true_pairs, nw_pairs)

    print(f"\n{'='*50}")
    print(f"F1 (model-guided):  {f1_model:.4f}")
    print(f"F1 (classical NW):  {f1_nw:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
