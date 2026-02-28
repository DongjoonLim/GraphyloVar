# Copyright (c) 2025 Dongjoon Lim, McGill University
# Licensed under the MIT License. See LICENSE file in the project root.
"""
MAF alignment parser.

Reads raw Multi-Alignment Format (MAF) files from Boreoeutherian / UCSC 100-way
alignments, ungaps relative to the human reference, and serialises to a pickle
dictionary suitable for GraphyloVar's preprocessing pipeline.

Original implementation: ``parserPreprocess.py`` in ``graphylo/`` and
``conservation/`` directories.

Usage (as library)::

    from graphylovar.maf_parser import parse_maf_file
    seq_dict = parse_maf_file("data/chr22.anc.maf", chromosome=22)

Usage (via CLI)::

    python scripts/parse_maf.py --maf data/chr22.anc.maf --chrom 22 \\
        --output data/seqDictPad_chr22.pkl
"""

from __future__ import annotations

import pickle
from typing import Optional

import numpy as np
from tqdm import tqdm

from graphylovar.phylogeny import NAMES


# ─── Internal: ungap two aligned sequences ──────────────────────────

def _ungap(anc: str, des: str) -> tuple[str, str]:
    """Remove positions where both sequences are gaps."""
    a_out, d_out = [], []
    for a, d in zip(anc, des):
        if a == "-" and d == "-":
            continue
        a_out.append(a)
        d_out.append(d)
    return "".join(a_out), "".join(d_out)


# ─── Read MAF blocks ────────────────────────────────────────────────

def _read_maf_blocks(maf_path: str) -> list[list[tuple]]:
    """
    Parse a ``.maf`` file into a list of alignment blocks.

    Each block is a list of ``(species_name, start, length, sequence)``
    tuples corresponding to the ``s`` lines.
    """
    blocks: list[list[tuple]] = []
    current: list[tuple] = []
    with open(maf_path, "rb") as fh:
        for raw_line in fh:
            line = raw_line.decode("utf-8", errors="replace")
            tokens = line.split()
            if not tokens:
                if current:
                    blocks.append(current)
                    current = []
            elif tokens[0] == "s":
                try:
                    name = tokens[1]
                    start = int(tokens[2])
                    length = int(tokens[3])
                    seq = tokens[6]
                    current.append((name, start, length, seq))
                except (IndexError, ValueError):
                    continue
    if current:
        blocks.append(current)
    return blocks


# ─── Build ungapped alignment dict ──────────────────────────────────

def _build_alignment(
    blocks: list[list[tuple]],
    species_names: list[str],
) -> dict[str, list[str]]:
    """
    Assemble a full-length alignment dictionary from parsed MAF blocks.

    The resulting dictionary maps each species name to a list of
    characters (one per human-reference-aligned position), padded with
    ``'N'`` in regions without alignment data.

    Steps:
        1. Walk blocks in order; for each block, append species
           sequences and pad unrepresented species with ``'-'``.
        2. Fill inter-block gaps with ``'N'``.
        3. Ungap relative to human (remove positions where hg38 is ``'-'``).
    """
    seq_dict: dict[str, list[str]] = {s: [] for s in species_names}

    # Pad before first block if needed
    if blocks and blocks[0]:
        first_start = blocks[0][0][1]
        for s in species_names:
            seq_dict[s].extend(["N"] * first_start)

    prev_end: int | None = None
    for block in tqdm(blocks, desc="Building alignment"):
        if not block:
            continue

        # Determine the block's reference start from the first entry
        block_start = block[0][1]
        block_len = block[0][2]

        # Fill inter-block gaps with 'N'
        if prev_end is not None and block_start > prev_end:
            gap_len = block_start - prev_end
            for s in species_names:
                seq_dict[s].extend(["N"] * gap_len)

        # Temporary per-species sequences for this block
        block_seqs: dict[str, list[str]] = {}
        max_len = 0
        for name_raw, _start, _length, seq_str in block:
            name = name_raw.split(".")[0]
            chars = list(seq_str.upper())
            block_seqs[name] = chars
            max_len = max(max_len, len(chars))

        # Append block sequences (pad missing species with '-')
        for s in species_names:
            if s in block_seqs:
                seq_dict[s].extend(block_seqs[s])
            else:
                seq_dict[s].extend(["-"] * max_len)

        # Ensure all species are the same length (pad shorter)
        lengths = [len(seq_dict[s]) for s in species_names]
        target = max(lengths)
        for s in species_names:
            if len(seq_dict[s]) < target:
                seq_dict[s].extend(["-"] * (target - len(seq_dict[s])))

        prev_end = block_start + block_len

    # ── Ungap relative to hg38 ──────────────────────────────────────
    human_key = "hg38"
    keep_idx = [i for i, c in enumerate(seq_dict[human_key]) if c != "-"]
    for s in tqdm(species_names, desc="Ungapping"):
        seq_dict[s] = [seq_dict[s][i] for i in keep_idx]

    return seq_dict


# ─── Public API ─────────────────────────────────────────────────────

def parse_maf_file(
    maf_path: str,
    chromosome: int | str | None = None,
    species_names: list[str] | None = None,
) -> dict[str, list[str]]:
    """
    Parse a MAF file and return an ungapped alignment dictionary.

    Parameters
    ----------
    maf_path      : path to the ``.maf`` file
    chromosome    : chromosome label (used only for logging)
    species_names : override default NAMES from :mod:`graphylovar.phylogeny`

    Returns
    -------
    dict mapping species name → list[str] of characters, ungapped to hg38.
    """
    names = species_names or NAMES
    print(f"Parsing MAF: {maf_path} ({len(names)} species) ...")
    blocks = _read_maf_blocks(maf_path)
    print(f"  {len(blocks)} alignment blocks read.")
    alignment = _build_alignment(blocks, names)
    print(f"  Alignment length (hg38): {len(alignment['hg38'])}")
    return alignment


def parse_and_save(
    maf_path: str,
    output_path: str,
    chromosome: int | str | None = None,
    species_names: list[str] | None = None,
) -> None:
    """
    Parse a MAF file and save the alignment as a pickle.

    Parameters
    ----------
    maf_path    : path to the ``.maf`` file
    output_path : path for the output ``.pkl``
    chromosome  : chromosome label (for logging)
    species_names : optional species list override
    """
    alignment = parse_maf_file(maf_path, chromosome, species_names)
    with open(output_path, "wb") as fh:
        pickle.dump(alignment, fh)
    print(f"  Saved to {output_path}")
