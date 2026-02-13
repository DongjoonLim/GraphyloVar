"""
Preprocess BED regions to graph inputs for GraphyloVar.

This script extracts sequence windows from multi-species alignments centered on
positions defined in a BED file, applies reverse complement augmentation, and
encodes sequences for input to graph neural networks.
"""

import argparse
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from config import SPECIES_LIST, DNA_BASES, DEFAULT_CONTEXT_WINDOW
from utils import reverse_complement, validate_file_path, create_output_directory, save_numpy_array

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess BED regions to graph inputs.")
    parser.add_argument("--bed", type=str, required=True, help="BED file path.")
    parser.add_argument("--chrom", type=str, required=True, help="Chromosome (e.g., '20').")
    parser.add_argument("--output_x", type=str, required=True, help="Output X .npy path.")
    parser.add_argument("--output_y", type=str, required=True, help="Output y .npy path.")
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT_WINDOW, help="Context window size.")
    parser.add_argument(
        "--pkl_dir",
        type=str,
        default="/home/mcb/users/dlim63/conservation/data/",
        help="Directory with .pkl alignments.",
    )
    return parser.parse_args()

def extract_sequence_window(
    alignment: dict, position: int, context_window: int, species_name: str, label_encoder: LabelEncoder
) -> np.ndarray:
    """
    Extract sequence window from alignment for a given species.

    Args:
        alignment: Dictionary of species alignments
        position: Center position for extraction
        context_window: Size of window on each side of position
        species_name: Name of species to extract from
        label_encoder: Fitted LabelEncoder for DNA bases

    Returns:
        Encoded sequence with reverse complement augmentation
    """
    sequence_raw = alignment[species_name]
    segment = sequence_raw[position - context_window : position + context_window + 1]
    rc_segment = reverse_complement(list(segment))
    full_sequence = list(segment) + rc_segment
    return label_encoder.transform(full_sequence)


def process_bed_file(
    bed_path: str, chromosome: str, alignment: dict, context_window: int, label_encoder: LabelEncoder
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process BED file to extract training examples and targets.

    Args:
        bed_path: Path to BED file
        chromosome: Chromosome to process
        alignment: Multi-species alignment dictionary
        context_window: Size of context window
        label_encoder: Fitted LabelEncoder for DNA bases

    Returns:
        Tuple of (examples array, targets array)
    """
    validate_file_path(bed_path, "file")

    # Load and filter BED file
    input_df = pd.read_csv(bed_path, delimiter=r"\s+")
    input_df = input_df[input_df.iloc[:, 0] == f"chr{chromosome}"]
    indices = input_df.iloc[:, 1]
    y_true = input_df.iloc[:, 3]

    logger.info(f"Processing {len(indices)} positions from chromosome {chromosome}")

    examples, targets = [], []
    skipped_count = 0
    expected_shape = (len(SPECIES_LIST), context_window * 4 + 2)

    for position, target in tqdm(zip(indices, y_true), total=len(indices), desc="Processing positions"):
        position = int(position)
        try:
            # Skip positions with N in reference
            if alignment["hg38"][position] == "N":
                skipped_count += 1
                continue

            # Extract sequences for all species
            example = []
            for species_name in SPECIES_LIST:
                sequence = extract_sequence_window(alignment, position, context_window, species_name, label_encoder)
                example.append(sequence)

            example = np.array(example).astype("uint8")

            # Validate shape
            if example.shape != expected_shape:
                logger.warning(f"Position {position}: unexpected shape {example.shape}, expected {expected_shape}")
                skipped_count += 1
                continue

            examples.append(example)
            targets.append(target)

        except Exception as e:
            logger.warning(f"Skipping position {position}: {e}")
            skipped_count += 1

    logger.info(f"Processed {len(examples)} examples, skipped {skipped_count} positions")

    return np.array(examples), np.array(targets)


def main():
    """Main preprocessing pipeline."""
    args = parse_arguments()

    # Validate inputs
    validate_file_path(args.bed, "file")
    validate_file_path(args.pkl_dir, "directory")

    # Setup label encoder
    le = LabelEncoder().fit(DNA_BASES)

    # Load alignment
    logger.info(f"Loading alignment for chromosome {args.chrom}")
    alignment_path = f"{args.pkl_dir}/seqDictPad_chr{args.chrom}.pkl"
    validate_file_path(alignment_path, "file")
    alignment = pd.read_pickle(alignment_path)
    logger.info(f"Alignment loaded successfully")

    # Process BED file
    examples, targets = process_bed_file(args.bed, args.chrom, alignment, args.context, le)

    logger.info(f"Final shapes: examples {examples.shape}, targets {targets.shape}")

    # Save outputs
    create_output_directory("/".join(args.output_x.split("/")[:-1]) or ".")
    save_numpy_array(examples, args.output_x)
    save_numpy_array(targets, args.output_y)

    logger.info("Preprocessing complete!")

if __name__ == "__main__":
    main()
