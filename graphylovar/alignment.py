# Copyright (c) 2025 Dongjoon Lim, McGill University
# Licensed under the MIT License. See LICENSE file in the project root.
"""
Sequence alignment utilities for GraphyloVar.

Implements affine-gap Needleman-Wunsch alignment with two scoring modes:

1. **Classical DP** – fixed match/mismatch scores (LASTZ-style).
2. **EvolignSubst** – learned substitution log-probabilities from a
   GraphyloVar or EvoLSTM model, enabling context-dependent scoring.

Also provides:
    - Alignment evaluation helpers (listify, ungap, F1 score)
    - Substitution score tables from LASTZ
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


# ── LASTZ substitution scores ───────────────────────────────────────

LASTZ_SCORES: dict[tuple[str, str], int] = {
    ("A", "A"): 91,  ("T", "T"): 91,
    ("C", "C"): 100, ("G", "G"): 100,
    ("A", "G"): -31,  ("G", "A"): -31,
    ("A", "C"): -114, ("C", "A"): -114,
    ("A", "T"): -123, ("T", "A"): -123,
    ("C", "G"): -125, ("G", "C"): -125,
    ("C", "T"): -31,  ("T", "C"): -31,
    ("G", "T"): -114, ("T", "G"): -114,
}


def _match_bool(nuc1: str, nuc2: str, match: float, mismatch: float) -> float:
    return match if nuc1 == nuc2 else mismatch


# =====================================================================
# Classical affine-gap Needleman-Wunsch
# =====================================================================

def needleman_wunsch(
    S: str,
    T: str,
    match_score: float = 1.0,
    mismatch_score: float = -1.0,
    gap_open: float = -400.0,
    gap_extend: float = -30.0,
    use_lastz: bool = True,
    verbose: bool = False,
) -> tuple[str, str, float]:
    """
    Affine-gap Needleman-Wunsch global alignment.

    Parameters
    ----------
    S, T           : ungapped sequences
    match_score    : used when use_lastz=False
    mismatch_score : used when use_lastz=False
    gap_open       : gap opening penalty
    gap_extend     : gap extension penalty
    use_lastz      : if True, use LASTZ substitution matrix instead of simple match/mismatch
    verbose        : print alignment to stdout

    Returns
    -------
    (aligned_S, aligned_T, score)
    """
    m, n = len(S), len(T)
    M = np.zeros((m + 1, n + 1))
    Ix = np.zeros((m + 1, n + 1))
    Iy = np.zeros((m + 1, n + 1))

    NEG_INF = float("-inf")

    for i in range(m + 1):
        M[i, 0] = NEG_INF
        Ix[i, 0] = (gap_open + gap_extend * (i - 1)) if i > 0 else 0
        Iy[i, 0] = NEG_INF if i > 0 else 0

    for j in range(n + 1):
        M[0, j] = NEG_INF
        Ix[0, j] = NEG_INF if j > 0 else 0
        Iy[0, j] = (gap_open + gap_extend * (j - 1)) if j > 0 else 0

    M[0, 0] = 0

    for i in tqdm(range(1, m + 1), desc="NW fill", disable=not verbose):
        for j in range(1, n + 1):
            if use_lastz:
                sub = LASTZ_SCORES.get((S[i - 1], T[j - 1]), mismatch_score)
            else:
                sub = _match_bool(S[i - 1], T[j - 1], match_score, mismatch_score)

            Ix[i, j] = max(M[i - 1, j] + gap_open, Ix[i - 1, j] + gap_extend)
            Iy[i, j] = max(M[i, j - 1] + gap_open, Iy[i, j - 1] + gap_extend)
            M[i, j] = max(
                M[i - 1, j - 1] + sub,
                Ix[i - 1, j - 1] + sub,
                Iy[i - 1, j - 1] + sub,
            )

    # ── Traceback ───────────────────────────────────────────────────
    score = max(M[m, n], Ix[m, n], Iy[m, n])
    aligned_S, aligned_T = _traceback_nw(
        S, T, M, Ix, Iy, m, n,
        gap_open, gap_extend,
        use_lastz, match_score, mismatch_score,
    )

    if verbose:
        print(f"Optimal alignment score: {score}")
        print("S:", aligned_S)
        print("T:", aligned_T)

    return aligned_S, aligned_T, score


def _traceback_nw(S, T, M, Ix, Iy, m, n,
                  gap_open, gap_extend,
                  use_lastz, match_score, mismatch_score):
    """Traceback for standard NW."""
    aligned_S, aligned_T = "", ""
    i, j = m, n

    best = max(M[i, j], Ix[i, j], Iy[i, j])
    if best == M[i, j]:
        ptr = "match"
    elif best == Ix[i, j]:
        ptr = "up"
    else:
        ptr = "left"

    while i > 0 or j > 0:
        if ptr == "match":
            aligned_S = S[i - 1] + aligned_S
            aligned_T = T[j - 1] + aligned_T
            if use_lastz:
                sub = LASTZ_SCORES.get((S[i - 1], T[j - 1]), mismatch_score)
            else:
                sub = _match_bool(S[i - 1], T[j - 1], match_score, mismatch_score)
            if M[i, j] == M[i - 1, j - 1] + sub:
                ptr = "match"
            elif M[i, j] == Ix[i - 1, j - 1] + sub:
                ptr = "up"
            else:
                ptr = "left"
            i -= 1
            j -= 1
        elif ptr == "up":
            aligned_S = S[i - 1] + aligned_S
            aligned_T = "-" + aligned_T
            if Ix[i, j] == M[i - 1, j] + gap_open:
                ptr = "match"
            else:
                ptr = "up"
            i -= 1
        else:  # left
            aligned_S = "-" + aligned_S
            aligned_T = T[j - 1] + aligned_T
            if Iy[i, j] == M[i, j - 1] + gap_open:
                ptr = "match"
            else:
                ptr = "left"
            j -= 1

    return aligned_S, aligned_T


# =====================================================================
# Context-dependent alignment (EvolignSubst)
# =====================================================================

def evolign_subst(
    S: str,
    T: str,
    tableM: pd.DataFrame | np.ndarray,
    match_score: float = 1.0,
    mismatch_score: float = -1.0,
    gap_open: float = -5.0,
    gap_extend: float = -2.0,
    seq_length: int = 1,
    verbose: bool = False,
) -> tuple[str, str, float]:
    """
    Affine-gap alignment using learned substitution log-probabilities.

    For positions >= seq_length, substitution cost comes from `tableM`
    (precomputed model log-probabilities). For earlier positions, falls
    back to simple match/mismatch scoring.

    Parameters
    ----------
    S, T         : ungapped sequences
    tableM       : DataFrame or array of shape (len(S)+1, 5) with columns [A,C,G,T,-]
                   Row i holds log-prob scores for position i of S.
    match_score  : fallback match score
    mismatch_score : fallback mismatch score
    gap_open     : gap opening penalty
    gap_extend   : gap extension penalty
    seq_length   : positions < seq_length use classical scoring
    verbose      : print alignment

    Returns
    -------
    (aligned_S, aligned_T, score)
    """
    m, n = len(S), len(T)
    NEG_INF = float("-inf")

    # Ensure tableM is a DataFrame with proper columns
    if isinstance(tableM, np.ndarray):
        tableM = np.concatenate([np.zeros(5).reshape(1, 5), tableM], axis=0)
        tableM = pd.DataFrame(tableM, columns=["A", "C", "G", "T", "-"])
    elif isinstance(tableM, pd.DataFrame):
        if len(tableM) == m:
            pad = pd.DataFrame(np.zeros((1, tableM.shape[1])), columns=tableM.columns)
            tableM = pd.concat([pad, tableM], ignore_index=True)

    M = np.zeros((m + 1, n + 1))
    Ix = np.zeros((m + 1, n + 1))
    Iy = np.zeros((m + 1, n + 1))

    for i in range(m + 1):
        M[i, 0] = NEG_INF
        Ix[i, 0] = (gap_open + gap_extend * (i - 1)) if i > 0 else 0
        Iy[i, 0] = NEG_INF if i > 0 else 0

    for j in range(n + 1):
        M[0, j] = NEG_INF
        Ix[0, j] = NEG_INF if j > 0 else 0
        Iy[0, j] = (gap_open + gap_extend * (j - 1)) if j > 0 else 0

    M[0, 0] = 0

    for i in tqdm(range(1, m + 1), desc="Evolign fill", disable=not verbose):
        for j in range(1, n + 1):
            if i < seq_length or j < seq_length:
                sub = _match_bool(S[i - 1], T[j - 1], match_score, mismatch_score)
                Ix[i, j] = max(M[i - 1, j] + gap_open, Ix[i - 1, j] + gap_extend)
                Iy[i, j] = max(M[i, j - 1] + gap_open, Iy[i, j - 1] + gap_extend)
                M[i, j] = max(
                    M[i - 1, j - 1] + sub,
                    Ix[i - 1, j - 1] + sub,
                    Iy[i - 1, j - 1] + sub,
                )
            else:
                sub = tableM[T[j - 1]][i]
                M[i, j] = max(
                    M[i - 1, j - 1] + sub,
                    Ix[i - 1, j - 1] + sub,
                    Iy[i - 1, j - 1] + sub,
                )
                Ix[i, j] = max(M[i - 1, j] + gap_open, Ix[i - 1, j] + gap_extend)
                Iy[i, j] = max(M[i, j - 1] + gap_open, Iy[i, j - 1] + gap_extend)

    score = max(M[m, n], Ix[m, n], Iy[m, n])

    # ── Traceback ───────────────────────────────────────────────────
    aligned_S, aligned_T = _traceback_evolign(
        S, T, M, Ix, Iy, m, n, tableM,
        gap_open, gap_extend, match_score, mismatch_score, seq_length,
    )

    if verbose:
        print(f"Optimal alignment score: {score}")
        print("S:", aligned_S)
        print("T:", aligned_T)

    return aligned_S, aligned_T, score


def _traceback_evolign(S, T, M, Ix, Iy, m, n, tableM,
                       gap_open, gap_extend,
                       match_score, mismatch_score, seq_length):
    """Traceback for EvolignSubst."""
    aligned_S, aligned_T = "", ""
    i, j = m, n

    best = max(M[i, j], Ix[i, j], Iy[i, j])
    if best == M[i, j]:
        ptr = "match"
    elif best == Ix[i, j]:
        ptr = "up"
    else:
        ptr = "left"

    while i > 0 or j > 0:
        if i >= seq_length and j >= seq_length:
            sub_fn = lambda ii, jj: tableM[T[jj - 1]][ii]
        else:
            sub_fn = lambda ii, jj: _match_bool(
                S[ii - 1], T[jj - 1], match_score, mismatch_score
            )

        if ptr == "match":
            aligned_S = S[i - 1] + aligned_S
            aligned_T = T[j - 1] + aligned_T
            sub = sub_fn(i, j)
            if M[i, j] == M[i - 1, j - 1] + sub:
                ptr = "match"
            elif M[i, j] == Ix[i - 1, j - 1] + sub:
                ptr = "up"
            else:
                ptr = "left"
            i -= 1
            j -= 1
        elif ptr == "up":
            aligned_S = S[i - 1] + aligned_S
            aligned_T = "-" + aligned_T
            if Ix[i, j] == M[i - 1, j] + gap_open:
                ptr = "match"
            else:
                ptr = "up"
            i -= 1
        else:  # left
            aligned_S = "-" + aligned_S
            aligned_T = T[j - 1] + aligned_T
            if Iy[i, j] == M[i, j - 1] + gap_open:
                ptr = "match"
            else:
                ptr = "left"
            j -= 1

    return aligned_S, aligned_T


# =====================================================================
# Evaluation helpers
# =====================================================================

def ungap_common(S_aligned: str, T_aligned: str) -> tuple[str, str]:
    """Remove columns where both sequences have a gap."""
    a, d = [], []
    for s_char, t_char in zip(S_aligned, T_aligned):
        if s_char == "-" and t_char == "-":
            continue
        a.append(s_char)
        d.append(t_char)
    return "".join(a), "".join(d)


def listify(S_aligned: str, T_aligned: str) -> list[tuple[int, int]]:
    """
    Convert aligned sequences to a list of (S_pos, T_pos) matched pairs.
    Gaps are skipped.
    """
    result = []
    count_s, count_t = 0, 0
    for s_ch, t_ch in zip(S_aligned, T_aligned):
        if s_ch != "-" and t_ch != "-":
            result.append((count_s, count_t))
            count_s += 1
            count_t += 1
        elif s_ch == "-":
            count_t += 1
        else:
            count_s += 1
    return result


def alignment_f1(
    true_pairs: list[tuple[int, int]],
    pred_pairs: list[tuple[int, int]],
) -> float:
    """
    Compute F1 score between true and predicted alignment pairs.
    """
    true_set = set(true_pairs)
    pred_set = set(pred_pairs)
    if len(pred_set) == 0 or len(true_set) == 0:
        return 0.0
    overlap = len(true_set & pred_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(true_set)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def find_non_gap_indices(sequence: str, gap_char: str = "-") -> list[int]:
    """Return indices of non-gap characters in a sequence."""
    return [i for i, c in enumerate(sequence) if c != gap_char]
