#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap


ALLELE_TO_INDEX = {
    b"A": 0,
    b"C": 1,
    b"G": 2,
    b"T": 3,
    b"-": 4,
    b"N": 4,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build compact full TOPMed targets without materializing X arrays")
    parser.add_argument("--chromosomes", default="1-22")
    parser.add_argument("--topmed_tsv", required=True)
    parser.add_argument("--alignment_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--context", type=int, default=100)
    parser.add_argument("--negative_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--skip_existing", action="store_true")
    return parser


def parse_chromosome_spec(spec: str) -> list[int]:
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


def normalize_target(value: str) -> np.ndarray:
    stripped = value.strip()
    arr = np.fromstring(stripped.strip("[]"), sep=",", dtype=np.float32)
    if arr.size != 6:
        arr = np.asarray(ast.literal_eval(value), dtype=np.float32)
    arr[:5] = np.clip(arr[:5], 0.0, None)
    total = float(arr[:5].sum())
    if total > 0:
        arr[:5] /= total
    arr[5] = 1.0
    return arr


def topmed_file_path(topmed_tsv: str, chromosome: int) -> str:
    if os.path.isdir(topmed_tsv):
        for candidate in (
            os.path.join(topmed_tsv, f"topmed_chr{chromosome}.tsv.gz"),
            os.path.join(topmed_tsv, f"topmed_chr{chromosome}.tsv"),
        ):
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"Missing TOPMed label file for chr{chromosome} in {topmed_tsv}")
    return topmed_tsv


def load_summary_counts(topmed_tsv: str) -> dict[int, int]:
    counts: dict[int, int] = {}

    summary_path = os.path.join(topmed_tsv, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            entries = json.load(handle)
        for entry in entries:
            chromosome = int(entry["chromosome"])
            counts[chromosome] = int(entry.get("topmed_rows", 0))

    for summary_file in sorted(glob.glob(os.path.join(topmed_tsv, "topmed_chr*.summary.json"))):
        with open(summary_file, "r", encoding="utf-8") as handle:
            entry = json.load(handle)
        chromosome = int(entry["chromosome"])
        counts[chromosome] = int(entry.get("topmed_rows", 0))

    return counts


def sequence_to_bytes(sequence: str | list[str]) -> np.ndarray:
    if isinstance(sequence, str):
        seq_text = sequence
    else:
        seq_text = "".join(sequence)
    return np.frombuffer(seq_text.encode("ascii"), dtype="S1")


def read_alignment_hg38(alignment_path: str) -> np.ndarray:
    alignment = pd.read_pickle(alignment_path)
    hg38 = sequence_to_bytes(alignment["hg38"])
    del alignment
    return hg38


def iter_topmed_rows(topmed_tsv: str, chromosome: int, chunksize: int):
    chrom_label = f"chr{chromosome}"
    cols = ["chrom", "chromEnd", "ref", "class", "allele_frequency_vector"]
    reader = pd.read_csv(
        topmed_file_path(topmed_tsv, chromosome),
        sep="\t",
        usecols=cols,
        chunksize=chunksize,
        low_memory=True,
        compression="infer",
    )
    for chunk in reader:
        if "class" in chunk.columns:
            chunk = chunk.loc[chunk["class"].fillna("").str.lower() == "snv"]
        if "chrom" in chunk.columns:
            chunk = chunk.loc[chunk["chrom"] == chrom_label]
        if chunk.empty:
            continue
        chunk["chromEnd"] = pd.to_numeric(chunk["chromEnd"], errors="coerce")
        chunk = chunk.dropna(subset=["chromEnd", "allele_frequency_vector"])
        if chunk.empty:
            continue
        yield chunk[["chromEnd", "allele_frequency_vector"]]


def build_reference_targets(hg38: np.ndarray, sampled_positions: np.ndarray) -> np.ndarray:
    bases = hg38[sampled_positions]
    targets = np.zeros((sampled_positions.shape[0], 6), dtype=np.float32)
    for allele, index in ALLELE_TO_INDEX.items():
        mask = bases == allele
        if np.any(mask):
            targets[mask, index] = 1.0
    return targets


def process_chromosome(
    chromosome: int,
    topmed_tsv: str,
    alignment_dir: str,
    output_dir: str,
    context: int,
    negative_ratio: float,
    seed: int,
    chunksize: int,
    positive_capacity: int,
) -> dict:
    alignment_path = os.path.join(alignment_dir, f"seqDictPad_chr{chromosome}.pkl")
    print(f"\n=== chr{chromosome} ===", flush=True)
    print(f"Loading hg38 from {alignment_path}", flush=True)
    hg38 = read_alignment_hg38(alignment_path)
    chrom_length = int(hg38.shape[0])

    valid_mask = hg38 != b"N"
    valid_mask[:context] = False
    valid_mask[chrom_length - context :] = False

    negative_capacity = int(math.ceil(positive_capacity * negative_ratio))
    total_capacity = positive_capacity + negative_capacity

    os.makedirs(output_dir, exist_ok=True)
    pos_path = os.path.join(output_dir, f"positions_graphylovar_topmed_chr{chromosome}.npy")
    y_path = os.path.join(output_dir, f"y_graphylovar_topmed_chr{chromosome}.npy")
    meta_path = os.path.join(output_dir, f"metadata_graphylovar_topmed_chr{chromosome}.json")

    positions = open_memmap(pos_path, mode="w+", dtype=np.int64, shape=(total_capacity,))
    targets = open_memmap(y_path, mode="w+", dtype=np.float32, shape=(total_capacity, 6))

    cursor = 0
    positive_written = 0
    duplicate_or_invalid = 0
    for chunk in iter_topmed_rows(topmed_tsv, chromosome, chunksize=chunksize):
        for chrom_end, allele_frequency_vector in chunk.itertuples(index=False, name=None):
            position_zero_based = int(chrom_end) - 1
            if position_zero_based < 0 or position_zero_based >= chrom_length:
                duplicate_or_invalid += 1
                continue
            if not valid_mask[position_zero_based]:
                duplicate_or_invalid += 1
                continue
            positions[cursor] = position_zero_based + 1
            targets[cursor] = normalize_target(allele_frequency_vector)
            valid_mask[position_zero_based] = False
            cursor += 1
            positive_written += 1

    candidate_positions = np.flatnonzero(valid_mask)
    negative_requested = min(int(round(positive_written * negative_ratio)), int(candidate_positions.shape[0]))
    rng = np.random.default_rng(seed + chromosome)
    sampled_positions = rng.choice(candidate_positions, size=negative_requested, replace=False)
    sampled_positions.sort()

    write_cursor = cursor
    block_size = 1_000_000
    for start in range(0, sampled_positions.shape[0], block_size):
        stop = min(start + block_size, sampled_positions.shape[0])
        block = sampled_positions[start:stop]
        positions[write_cursor : write_cursor + block.shape[0]] = block + 1
        targets[write_cursor : write_cursor + block.shape[0]] = build_reference_targets(hg38, block)
        write_cursor += block.shape[0]

    positions.flush()
    targets.flush()

    _shrink_memmap(pos_path, np.int64, (total_capacity,), write_cursor)
    _shrink_memmap(y_path, np.float32, (total_capacity, 6), write_cursor)

    metadata = {
        "chromosome": chromosome,
        "mode": "compact_full",
        "capped": False,
        "context": context,
        "positive_capacity": positive_capacity,
        "positive_written": positive_written,
        "negative_requested": negative_requested,
        "negative_written": negative_requested,
        "total_written": write_cursor,
        "negative_ratio": negative_ratio,
        "duplicate_or_invalid_skipped": duplicate_or_invalid,
        "topmed_tsv": str(topmed_tsv),
        "alignment_path": alignment_path,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(
        f"Wrote chr{chromosome}: positives={positive_written:,}, negatives={negative_requested:,}, total={write_cursor:,}",
        flush=True,
    )
    return metadata


def main() -> None:
    args = build_parser().parse_args()
    chromosomes = parse_chromosome_spec(args.chromosomes)
    counts = load_summary_counts(args.topmed_tsv)
    summary = []
    for chromosome in chromosomes:
        meta_path = os.path.join(args.output_dir, f"metadata_graphylovar_topmed_chr{chromosome}.json")
        pos_path = os.path.join(args.output_dir, f"positions_graphylovar_topmed_chr{chromosome}.npy")
        y_path = os.path.join(args.output_dir, f"y_graphylovar_topmed_chr{chromosome}.npy")
        if args.skip_existing and os.path.exists(meta_path) and os.path.exists(pos_path) and os.path.exists(y_path):
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            if meta.get("mode") == "compact_full" and not meta.get("capped", True):
                print(f"Skipping chr{chromosome}; full compact outputs already exist", flush=True)
                summary.append(meta)
                continue

        positive_capacity = counts.get(chromosome)
        if positive_capacity is None:
            raise RuntimeError(f"Missing topmed_rows count for chr{chromosome}; summary.json is required")
        summary.append(
            process_chromosome(
                chromosome=chromosome,
                topmed_tsv=args.topmed_tsv,
                alignment_dir=args.alignment_dir,
                output_dir=args.output_dir,
                context=args.context,
                negative_ratio=args.negative_ratio,
                seed=args.seed,
                chunksize=args.chunksize,
                positive_capacity=positive_capacity,
            )
        )

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = Path(args.output_dir) / "topmed_preprocess_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()