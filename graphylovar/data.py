"""
Data loading, windowing, masking, and train/val splitting utilities.
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from graphylovar.phylogeny import NAMES, MASK_INDICES

# ── Label encoder for nucleotide characters ─────────────────────────
_LE = LabelEncoder()
_LE.fit(["A", "C", "G", "T", "N", "-"])


def label_encode() -> LabelEncoder:
    """Return the shared nucleotide LabelEncoder."""
    return _LE


def reverse_complement(seq: list[str] | str) -> list[str]:
    """Return the reverse complement of a DNA sequence (list or str)."""
    _comp = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N", "-": "-"}
    return [_comp.get(b, "N") for b in reversed(seq)]


def reverse_complement_str(seq: str) -> str:
    """Return reverse complement as a string."""
    return "".join(reverse_complement(seq))


# ── Window extraction from alignment pickle ─────────────────────────

def extract_windows(
    alignment: dict[str, list[str]],
    indices: list[int],
    context: int = 100,
    species_names: list[str] | None = None,
) -> np.ndarray:
    """
    Extract (N, 115, context*4+2) uint8 feature matrices from alignment.

    Each window is [forward_segment | reverse_complement_segment]
    for every species, label-encoded.

    Parameters
    ----------
    alignment : dict mapping species name -> list of characters
    indices   : genomic positions (alignment coordinates)
    context   : flanking bases on each side (total window = 2*context+1)
    species_names : override default NAMES order

    Returns
    -------
    np.ndarray of shape (N_valid, 115, context*4+2), dtype uint8
    """
    le = label_encode()
    names = species_names or NAMES
    examples = []
    for i in tqdm(indices, desc="Extracting windows"):
        try:
            if alignment["hg38"][i] == "N":
                continue
            row = []
            for key in names:
                seg = alignment[key][i - context : i + context + 1]
                rc = reverse_complement(seg)
                encoded = le.transform(list(seg) + rc)
                row.append(encoded)
            arr = np.array(row, dtype=np.uint8)
            expected_cols = context * 4 + 2
            if arr.shape == (len(names), expected_cols):
                examples.append(arr)
        except Exception as exc:  # noqa: BLE001
            continue
    return np.array(examples, dtype=np.uint8) if examples else np.empty((0,))


# ── CADD-style data loading ─────────────────────────────────────────

def load_cadd_data(
    data_dir: str,
    chromosome: str | int = 1,
    context: int = 100,
    context_flank: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load preprocessed GraphyloVar arrays for one chromosome.

    Parameters
    ----------
    data_dir      : directory containing X_graphylo_chr*.npy / y_graphylo_chr*.npy
    chromosome    : e.g. 1 or "X"
    context       : original flanking context used during preprocessing
    context_flank : if set, slice to a narrower window (e.g. 10 for context21)

    Returns
    -------
    X : np.ndarray (N, 115, seq_len)
    y : np.ndarray (N, 2) one-hot
    """
    X = np.load(os.path.join(data_dir, f"X_graphylo_chr{chromosome}.npy"))
    y = np.load(os.path.join(data_dir, f"y_graphylo_chr{chromosome}.npy"))

    if context_flank is not None:
        left = X[:, :, context - context_flank : context + context_flank + 1]
        right = X[:, :, -context - context_flank - 1 : -context + context_flank]
        X = np.concatenate([left, right], axis=-1)

    y = np.array(tf.one_hot(y, 2))
    return X, y


def mask_species(X: np.ndarray, indices: list[int] | None = None) -> np.ndarray:
    """
    Zero out specific species channels (e.g. human, chimp, gorilla, ancestors).

    Parameters
    ----------
    X       : (N, 115, L) array
    indices : species indices to zero; defaults to MASK_INDICES

    Returns
    -------
    X with masked rows zeroed out (modifies in-place and returns).
    """
    if indices is None:
        indices = MASK_INDICES
    for idx in indices:
        X[:, idx, :] = 0
    return X


def prepare_train_val(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    mask: bool = True,
) -> dict[str, np.ndarray]:
    """
    Mask species, copy EvoLSTM baseline row, and split into train/val.

    Returns dict with keys:
        X_train, X_val, y_train, y_val,
        X_evolstm_train, X_evolstm_val
    """
    # Copy human row before masking (for EvoLSTM baseline)
    X_evolstm = X[:, 0, :].copy()

    if mask:
        X = mask_species(X)

    (X_train, X_val,
     y_train, y_val,
     X_ev_train, X_ev_val) = train_test_split(
        X, y, X_evolstm, test_size=test_size, random_state=random_state
    )

    return {
        "X_train": X_train, "X_val": X_val,
        "y_train": y_train, "y_val": y_val,
        "X_evolstm_train": X_ev_train, "X_evolstm_val": X_ev_val,
    }


# ── Preprocessing from raw CADD VCFs ──────────────────────────────

def preprocess_cadd_chromosome(
    chromosome: str | int,
    cadd_dir: str,
    alignment_dir: str,
    output_dir: str,
    context: int = 100,
    species_names: list[str] | None = None,
) -> None:
    """
    Build X/y arrays from CADD simulation VCFs + alignment pickle.

    Saves:
        data/X_graphylo_chr{chrom}.npy
        data/y_graphylo_chr{chrom}.npy
    """
    names = species_names or NAMES
    le = label_encode()

    # Load VCFs
    snv = pd.read_csv(os.path.join(cadd_dir, "simulation_SNVs.vcf"), sep="\t", header=None)
    snv = snv.loc[snv[0] == chromosome]
    indel = pd.read_csv(os.path.join(cadd_dir, "simulation_InDels.vcf"), sep="\t", header=None)
    indel = indel.loc[indel[0] == chromosome]
    snv_neg = pd.read_csv(os.path.join(cadd_dir, "humanDerived_SNVs.vcf"), sep="\t", header=None)
    snv_neg = snv_neg.loc[snv_neg[0] == chromosome]
    indel_neg = pd.read_csv(os.path.join(cadd_dir, "humanDerived_InDels.vcf"), sep="\t", header=None)
    indel_neg = indel_neg.loc[indel_neg[0] == chromosome]

    mutated_idx = list(snv_neg[1]) + list(indel_neg[1])
    conserved_idx = list(snv[1]) + list(indel[1])
    all_idx = mutated_idx + conserved_idx
    labels_raw = [0] * len(mutated_idx) + [1] * len(conserved_idx)

    # Load alignment
    pkl_path = os.path.join(alignment_dir, f"seqDictPad_chr{chromosome}.pkl")
    alignment = pd.read_pickle(pkl_path)

    examples, labels = [], []
    for pos, lab in tqdm(zip(all_idx, labels_raw), total=len(all_idx),
                         desc=f"chr{chromosome}"):
        try:
            if alignment["hg38"][pos] == "N":
                continue
            row = []
            for key in names:
                seg = alignment[key][pos - context : pos + context + 1]
                rc = reverse_complement(seg)
                encoded = le.transform(list(seg) + rc)
                row.append(encoded)
            arr = np.array(row, dtype=np.uint8)
            if arr.shape == (len(names), context * 4 + 2):
                examples.append(arr)
                labels.append(lab)
        except Exception:  # noqa: BLE001
            continue

    X = np.array(examples, dtype=np.uint8)
    y = np.array(labels)
    print(f"chr{chromosome}: X {X.shape}, y {y.shape}")

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"X_graphylo_chr{chromosome}.npy"), X)
    np.save(os.path.join(output_dir, f"y_graphylo_chr{chromosome}.npy"), y)
