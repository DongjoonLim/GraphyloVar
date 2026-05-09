# Copyright (c) 2025 Dongjoon Lim, McGill University
# Licensed under the MIT License. See LICENSE file in the project root.
"""
Utilities for TOPMed-based GraphyloVar pretraining.

This module provides helpers for:
    * loading chromosome-specific TOPMed labels from the processed TSV
    * extracting GraphyloVar alignment windows for one genomic position
    * building positive / negative multitask labels
    * sampling non-polymorphic reference positions safely

The implementation is intentionally chromosome-local and mmap-friendly so it can
be used on very large alignments without loading multiple chromosomes into RAM.
"""

from __future__ import annotations

import ast
import fcntl
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from graphylovar.data import label_encode, reverse_complement
from graphylovar.phylogeny import NAMES


ALLELE_TO_INDEX = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "-": 4,
    "N": 4,
}

COMPLEMENT_TO_INDEX = {
    "A": 3,
    "C": 2,
    "G": 1,
    "T": 0,
    "N": 4,
    "-": 4,
}

COMPLEMENT_INDEX_ARRAY = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
ASCII_TO_INDEX = np.full(256, 4, dtype=np.uint8)
for _base, _index in {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
    "-": 4,
    "a": 0,
    "c": 1,
    "g": 2,
    "t": 3,
    "n": 4,
}.items():
    ASCII_TO_INDEX[ord(_base)] = _index


def allele_to_index(allele: str) -> int:
    """Map an allele to the 5-way GraphyloVar nucleotide index."""
    return ALLELE_TO_INDEX.get(str(allele).upper(), 4)


def normalize_allele_frequency_vector(value: str | list[float] | np.ndarray) -> np.ndarray:
    """
    Parse and normalize a TOPMed allele-frequency vector.

    Expected layout is `[A, C, G, T, gap, polymorphic_flag]`. The first five
    entries are clipped to non-negative values and renormalized if needed.
    """
    if isinstance(value, str):
        value = ast.literal_eval(value)

    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[0] != 6:
        raise ValueError(f"Expected 6 values, got shape {arr.shape}")

    arr[:5] = np.clip(arr[:5], 0.0, None)
    total = float(arr[:5].sum())
    if total > 0:
        arr[:5] /= total
    arr[5] = 1.0
    return arr


def make_reference_target(reference_base: str) -> np.ndarray:
    """Return the 6-dim target for a non-polymorphic reference position."""
    target = np.zeros(6, dtype=np.float32)
    target[allele_to_index(reference_base)] = 1.0
    target[5] = 0.0
    return target


def load_topmed_labels(
    topmed_tsv: str,
    chromosome: int | str,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """
    Load processed TOPMed labels for a single chromosome from a TSV.

    The TSV is expected to contain at least:
        `chrom`, `chromEnd`, `ref`, `class`, `allele_frequency_vector`
    """
    chrom_label = f"chr{chromosome}"
    cols = ["chrom", "chromEnd", "ref", "class", "allele_frequency_vector"]

    if os.path.isdir(topmed_tsv):
        candidates = [
            os.path.join(topmed_tsv, f"topmed_chr{chromosome}.tsv.gz"),
            os.path.join(topmed_tsv, f"topmed_chr{chromosome}.tsv"),
        ]
        chosen = None
        for candidate in candidates:
            if os.path.exists(candidate):
                chosen = candidate
                break
        if chosen is None:
            return pd.DataFrame(columns=["chromEnd", "ref", "allele_frequency_vector"])

        frames: list[pd.DataFrame] = []
        reader = pd.read_csv(chosen, sep="\t", chunksize=chunksize, low_memory=True)
        for chunk in reader:
            if "class" in chunk.columns:
                chunk = chunk.loc[chunk["class"].fillna("").str.lower() == "snv"].copy()
            if "chrom" in chunk.columns:
                chunk = chunk.loc[chunk["chrom"] == chrom_label].copy()
            if chunk.empty:
                continue
            chunk["chromEnd"] = pd.to_numeric(chunk["chromEnd"], errors="coerce")
            chunk = chunk.dropna(subset=["chromEnd", "allele_frequency_vector"])
            if chunk.empty:
                continue
            chunk["chromEnd"] = chunk["chromEnd"].astype(np.int64)
            chunk["allele_frequency_vector"] = chunk["allele_frequency_vector"].apply(
                normalize_allele_frequency_vector
            )
            frames.append(chunk[["chromEnd", "ref", "allele_frequency_vector"]])

        if not frames:
            return pd.DataFrame(columns=["chromEnd", "ref", "allele_frequency_vector"])
        df = pd.concat(frames, ignore_index=True)
        return df.drop_duplicates(subset=["chromEnd"], keep="first").sort_values(
            "chromEnd"
        ).reset_index(drop=True)

    frames: list[pd.DataFrame] = []

    reader = pd.read_csv(
        topmed_tsv,
        sep="\t",
        usecols=cols,
        chunksize=chunksize,
        low_memory=True,
    )

    for chunk in reader:
        chunk = chunk.loc[chunk["chrom"] == chrom_label].copy()
        if chunk.empty:
            continue
        chunk = chunk.loc[chunk["class"].fillna("").str.lower() == "snv"].copy()
        if chunk.empty:
            continue
        chunk["chromEnd"] = pd.to_numeric(chunk["chromEnd"], errors="coerce")
        chunk = chunk.dropna(subset=["chromEnd", "allele_frequency_vector"])
        if chunk.empty:
            continue
        chunk["chromEnd"] = chunk["chromEnd"].astype(np.int64)
        chunk["allele_frequency_vector"] = chunk["allele_frequency_vector"].apply(
            normalize_allele_frequency_vector
        )
        frames.append(chunk[["chromEnd", "ref", "allele_frequency_vector"]])

    if not frames:
        return pd.DataFrame(columns=["chromEnd", "ref", "allele_frequency_vector"])

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["chromEnd"], keep="first")
    df = df.sort_values("chromEnd").reset_index(drop=True)
    return df


def extract_example(
    alignment: dict[str, list[str] | str],
    position_zero_based: int,
    context: int,
    species_names: list[str] | None = None,
) -> np.ndarray | None:
    """
    Extract one GraphyloVar example of shape `(115, context*4+2)`.

    The center coordinate is expected to be 0-based.
    """
    names = species_names or NAMES
    hg38 = alignment["hg38"]

    if position_zero_based < context or position_zero_based + context >= len(hg38):
        return None
    if hg38[position_zero_based] == "N":
        return None

    rows = []
    for key in names:
        sequence_raw = alignment[key]
        segment = sequence_raw[position_zero_based - context : position_zero_based + context + 1]
        if len(segment) != 2 * context + 1:
            return None
        encoded = np.fromiter(
            (ALLELE_TO_INDEX.get(base, 4) for base in segment),
            dtype=np.uint8,
            count=2 * context + 1,
        )
        encoded_rc = np.fromiter(
            (COMPLEMENT_TO_INDEX.get(base, 4) for base in reversed(segment)),
            dtype=np.uint8,
            count=2 * context + 1,
        )
        encoded = np.concatenate([encoded, encoded_rc])
        rows.append(encoded)

    arr = np.asarray(rows, dtype=np.uint8)
    expected_shape = (len(names), context * 4 + 2)
    if arr.shape != expected_shape:
        return None
    return arr


def _sequence_to_ascii_bytes(sequence_raw: list[str] | str) -> bytes:
    if isinstance(sequence_raw, str):
        return sequence_raw.encode("ascii", errors="ignore")
    return "".join(sequence_raw).encode("ascii", errors="ignore")


def alignment_cache_paths(cache_dir: str, chromosome: int) -> tuple[Path, Path, Path]:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    return (
        root / f"alignment_encoded_chr{chromosome}.npy",
        root / f"alignment_encoded_chr{chromosome}.json",
        root / f"alignment_encoded_chr{chromosome}.lock",
    )


def ensure_alignment_encoded_cache(
    alignment_path: str,
    cache_dir: str,
    chromosome: int,
    species_names: list[str] | None = None,
) -> tuple[str, dict]:
    names = species_names or NAMES
    cache_path, meta_path, lock_path = alignment_cache_paths(cache_dir, chromosome)
    if cache_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as handle:
            return str(cache_path), json.load(handle)

    with open(lock_path, "w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if cache_path.exists() and meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as handle:
                return str(cache_path), json.load(handle)

        alignment = pd.read_pickle(alignment_path)
        hg38 = alignment["hg38"]
        sequence_length = len(hg38)
        encoded = np.lib.format.open_memmap(
            cache_path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(names), sequence_length),
        )

        for row_index, key in enumerate(names):
            ascii_bytes = _sequence_to_ascii_bytes(alignment[key])
            encoded[row_index, :] = ASCII_TO_INDEX[np.frombuffer(ascii_bytes, dtype=np.uint8)]

        encoded.flush()
        del encoded
        meta = {
            "alignment_path": alignment_path,
            "chromosome": chromosome,
            "num_nodes": len(names),
            "sequence_length": sequence_length,
        }
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        return str(cache_path), meta


def load_alignment_encoded_cache(
    alignment_path: str,
    cache_dir: str,
    chromosome: int,
    species_names: list[str] | None = None,
) -> tuple[np.memmap, dict]:
    cache_path, meta = ensure_alignment_encoded_cache(
        alignment_path=alignment_path,
        cache_dir=cache_dir,
        chromosome=chromosome,
        species_names=species_names,
    )
    return np.load(cache_path, mmap_mode="r"), meta


def extract_batch_examples_from_encoded(
    encoded_alignment: np.ndarray,
    positions_zero_based: np.ndarray,
    context: int,
    context_flank: int | None = None,
    mask_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions_zero_based, dtype=np.int64)
    if positions.size == 0:
        return np.empty((0, encoded_alignment.shape[0], 0), dtype=np.uint8), np.empty((0,), dtype=bool)

    sequence_length = encoded_alignment.shape[1]
    valid = (positions >= context) & (positions + context < sequence_length)
    if valid.any():
        # Update only currently valid rows to avoid boolean broadcasting mismatch
        # when filtering very large batches.
        valid_idx = np.flatnonzero(valid)
        keep_idx = encoded_alignment[0, positions[valid_idx]] != 4
        valid[valid_idx] &= keep_idx
    valid_positions = positions[valid]
    if valid_positions.size == 0:
        return np.empty((0, encoded_alignment.shape[0], 0), dtype=np.uint8), valid

    offsets = np.arange(-context, context + 1, dtype=np.int64)
    window_indices = valid_positions[:, None] + offsets[None, :]
    forward = encoded_alignment[:, window_indices].transpose(1, 0, 2)
    reverse = COMPLEMENT_INDEX_ARRAY[forward[:, :, ::-1]]
    batch = np.concatenate([forward, reverse], axis=-1)

    if mask_indices:
        batch[:, mask_indices, :] = 0

    if context_flank is not None:
        left = batch[:, :, context - context_flank : context + context_flank + 1]
        right_start = batch.shape[-1] - context - 1 - context_flank
        right_end = batch.shape[-1] - context - 1 + context_flank + 1
        right = batch[:, :, right_start:right_end]
        if right.shape[-1] == 0:
            right = batch[:, :, -context - context_flank - 1 : -context + context_flank]
        batch = np.concatenate([left, right], axis=-1)

    return batch, valid


def sample_nonvariant_positions(
    hg38_sequence: list[str] | str,
    excluded_positions_zero_based: set[int],
    sample_count: int,
    context: int,
    seed: int = 42,
) -> list[int]:
    """
    Sample non-overlapping non-`N` reference positions.

    Sampling is reproducible and chromosome-local. If random attempts are not
    sufficient, the function falls back to a deterministic linear scan.
    """
    rng = np.random.default_rng(seed)
    length = len(hg38_sequence)
    lower = context
    upper = length - context - 1
    if upper <= lower:
        return []

    chosen: set[int] = set()
    max_attempts = max(sample_count * 20, 10_000)
    attempts = 0

    while len(chosen) < sample_count and attempts < max_attempts:
        pos = int(rng.integers(lower, upper + 1))
        attempts += 1
        if pos in excluded_positions_zero_based or pos in chosen:
            continue
        if hg38_sequence[pos] == "N":
            continue
        chosen.add(pos)

    if len(chosen) < sample_count:
        for pos in range(lower, upper + 1):
            if pos in excluded_positions_zero_based or pos in chosen:
                continue
            if hg38_sequence[pos] == "N":
                continue
            chosen.add(pos)
            if len(chosen) >= sample_count:
                break

    return sorted(chosen)


def parse_chromosome_spec(spec: str) -> list[int]:
    """Parse chromosome specs such as `1-10,12,22`."""
    chroms: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            chroms.extend(range(int(start), int(end) + 1))
        else:
            chroms.append(int(token))
    return sorted(set(chroms))


def iter_existing_chromosomes(chromosomes: Iterable[int], data_dir: str) -> list[int]:
    """Return only chromosomes that have both X and y TOPMed arrays."""
    found = []
    for chrom in chromosomes:
        x_path = f"{data_dir}/X_graphylovar_topmed_chr{chrom}.npy"
        y_path = f"{data_dir}/y_graphylovar_topmed_chr{chrom}.npy"
        try:
            with open(x_path, "rb"), open(y_path, "rb"):
                found.append(chrom)
        except OSError:
            continue
    return found
