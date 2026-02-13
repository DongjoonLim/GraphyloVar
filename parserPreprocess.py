"""
Parse and preprocess MAF (Multiple Alignment Format) files.

This script converts MAF alignment files to serialized Python dictionaries (pickle/npy)
for efficient downstream processing. It handles gap removal, sequence padding, and
creates aligned sequence dictionaries for all species.
"""

import argparse
import logging
import pickle
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from config import SPECIES_LIST
from utils import create_output_directory, validate_file_path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess MAF alignments to serialized NumPy/Pickle format.")
    parser.add_argument("--chrom", type=str, required=True, help="Chromosome number (e.g., '21').")
    parser.add_argument(
        "--maf_path", type=str, default="../research/data/chr{chrom}.anc.maf", help="Path to MAF file (use {chrom} placeholder)."
    )
    parser.add_argument(
        "--output", type=str, default="seqDictPad_chr{chrom}.pkl", help="Output file path (use {chrom} placeholder)."
    )
    return parser.parse_args()


def ungap(anc: str, des: str) -> Tuple[str, str]:
    """
    Remove positions where both ancestor and descendant are gaps.

    Args:
        anc: Ancestor sequence string
        des: Descendant sequence string

    Returns:
        Tuple of (ungapped_ancestor, ungapped_descendant)
    """
    a, d = "", ""
    for anc_char, des_char in zip(anc, des):
        if anc_char == "-" and des_char == "-":
            continue
        a += anc_char
        d += des_char
    return a, d


def parse_maf_file(maf_path: str) -> List[List[List]]:
    """
    Parse MAF file into alignment blocks.

    Args:
        maf_path: Path to MAF file

    Returns:
        List of alignment blocks, each containing sequence information

    Format of each block entry: [species_name, start_position, length, sequence]
    """
    validate_file_path(maf_path, "file")

    logger.info(f"Parsing MAF file: {maf_path}")

    with open(maf_path, "rb") as file:
        lines = file.readlines()

    seq_list = []
    temp_list = []
    skipped_lines = 0

    for line in lines:
        line_str = str(line, "utf-8").strip()

        if not line_str:
            # Empty line marks end of block
            if temp_list:
                seq_list.append(temp_list)
                temp_list = []
        elif line_str.startswith("s"):
            # Sequence line
            parts = line_str.split()
            try:
                # Format: [species, start, length, sequence]
                temp_list.append([parts[1], int(parts[2]), int(parts[3]), parts[6]])
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping invalid line: {line_str[:50]}... Error: {e}")
                skipped_lines += 1
                continue

    # Add last block if exists
    if temp_list:
        seq_list.append(temp_list)

    if skipped_lines > 0:
        logger.warning(f"Skipped {skipped_lines} invalid lines during parsing")

    logger.info(f"Parsed {len(seq_list)} alignment blocks")
    return seq_list


def build_aligned_sequences(alignment_blocks: List[List[List]], species_list: List[str]) -> Dict[str, List[str]]:
    """
    Build aligned sequences dictionary from MAF alignment blocks.

    Args:
        alignment_blocks: List of parsed alignment blocks
        species_list: List of species/ancestor names to include

    Returns:
        Dictionary mapping species names to aligned sequences
    """
    if not alignment_blocks:
        raise ValueError("No alignment blocks provided")

    # Initialize sequences with N's for the first block start position
    first_block_start = alignment_blocks[0][0][1]
    seq_dict = {species: ["N"] * first_block_start for species in species_list}

    logger.info("Building aligned sequences...")

    for block_idx in tqdm(range(len(alignment_blocks)), desc="Processing blocks"):
        block = alignment_blocks[block_idx]

        # Add sequences from current block
        for entry in block:
            species_full = entry[0].split(".")[0]  # Remove any suffix
            if species_full in seq_dict:
                seq_dict[species_full].extend(list(entry[3].upper()))

        # Add gap padding between blocks if needed
        if block_idx > 0:
            prev_block = alignment_blocks[block_idx - 1]
            current_start = block[0][1]
            prev_end = prev_block[0][1] + prev_block[0][2]

            if current_start != prev_end:
                gap_size = current_start - prev_end
                for species in species_list:
                    seq_dict[species].extend(["N"] * gap_size)

        # Pad all sequences to same length (handle missing species in block)
        lengths = [len(seq_dict[species]) for species in species_list]
        max_len = max(lengths)

        for species in species_list:
            if len(seq_dict[species]) < max_len:
                seq_dict[species].extend(["-"] * (max_len - len(seq_dict[species])))

    logger.info(f"Alignment complete. Sequence length: {max_len}")
    return seq_dict


def remove_human_gaps(seq_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Remove positions where human sequence has gaps.

    Args:
        seq_dict: Dictionary of aligned sequences

    Returns:
        Dictionary with human-gapped positions removed
    """
    logger.info("Removing positions with human gaps...")

    human_seq = seq_dict["hg38"]
    non_gap_indices = [i for i in range(len(human_seq)) if human_seq[i] != "-"]

    logger.info(f"Original length: {len(human_seq)}, After ungapping: {len(non_gap_indices)}")

    ungapped_dict = {}
    for species in tqdm(seq_dict.keys(), desc="Ungapping sequences"):
        ungapped_dict[species] = [seq_dict[species][i] for i in non_gap_indices]

    return ungapped_dict


def main():
    """Main MAF parsing and preprocessing pipeline."""
    args = parse_arguments()

    logger.info(f"Processing chromosome {args.chrom}")
    logger.info(f"Number of species/ancestors to process: {len(SPECIES_LIST)}")

    # Parse MAF file
    maf_path = args.maf_path.format(chrom=args.chrom)
    alignment_blocks = parse_maf_file(maf_path)

    # Build aligned sequences
    seq_dict_raw = build_aligned_sequences(alignment_blocks, SPECIES_LIST)

    logger.info(f"Human sequence length before ungapping: {len(seq_dict_raw['hg38'])}")

    # Remove human gaps
    seq_dict_ungapped = remove_human_gaps(seq_dict_raw)

    logger.info(f"Final human sequence length: {len(seq_dict_ungapped['hg38'])}")

    # Save outputs
    output_path = args.output.format(chrom=args.chrom)
    output_dir = "/".join(output_path.split("/")[:-1])
    if output_dir:
        create_output_directory(output_dir)

    # Save as pickle
    logger.info(f"Saving pickle to {output_path}")
    with open(output_path, "wb") as handle:
        pickle.dump(seq_dict_ungapped, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save as numpy (for compatibility)
    npy_path = output_path.replace(".pkl", ".npy")
    logger.info(f"Saving numpy to {npy_path}")
    np.save(npy_path, seq_dict_ungapped)

    logger.info("MAF preprocessing complete!")


if __name__ == "__main__":
    main()
