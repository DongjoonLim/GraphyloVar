"""
Augment training data with reverse complement sequences.

This script augments existing training data by adding reverse complement versions
of sequences, effectively doubling the training set size and improving model
robustness to sequence orientation.
"""

import argparse
import logging
import os

import numpy as np
from tqdm import tqdm

from utils import create_output_directory, load_numpy_array, save_numpy_array

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Augment data with reverse complements.")
    parser.add_argument("--tf", type=str, required=True, help="Transcription factor (e.g., 'CTCF').")
    parser.add_argument("--celltype", type=str, required=True, help="Cell type (e.g., 'K562').")
    parser.add_argument("--data_dir", type=str, default="graphs/{tf}", help="Data directory (use {tf} placeholder).")
    parser.add_argument("--length", type=int, default=1001, help="Sequence length.")
    parser.add_argument(
        "--chromosomes",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22],
        help="List of training chromosomes.",
    )
    return parser.parse_args()


def reverse_complement_encoding(input_val: int) -> int:
    """
    Map nucleotide encoding to its reverse complement.

    Args:
        input_val: Integer encoding of nucleotide

    Returns:
        Integer encoding of reverse complement

    Note:
        Mapping assumes specific integer encoding:
        1 <-> 5 (typically A <-> T)
        2 <-> 3 (typically C <-> G)
    """
    mapping = {1: 5, 5: 1, 2: 3, 3: 2}
    return mapping.get(int(input_val), input_val)


def load_chromosome_data(data_dir: str, length: int, chrom: int, tf: str, celltype: str):
    """
    Load data for a specific chromosome.

    Args:
        data_dir: Data directory path
        length: Sequence length
        chrom: Chromosome number
        tf: Transcription factor
        celltype: Cell type

    Returns:
        Tuple of (X, y) arrays
    """
    x_path = os.path.join(data_dir, f"dataset_{length}_chr{chrom}_{tf}_{celltype}_X_train.npy")
    y_path = os.path.join(data_dir, f"dataset_{length}_chr{chrom}_{tf}_{celltype}_y_train.npy")

    try:
        X = load_numpy_array(x_path).astype(np.uint8)
        y = load_numpy_array(y_path).astype(np.uint8)
        return X, y
    except Exception as e:
        logger.warning(f"Error loading chromosome {chrom}: {e}")
        return None, None


def augment_with_reverse_complement(X: np.ndarray) -> np.ndarray:
    """
    Augment data with reverse complement sequences.

    Args:
        X: Input data array

    Returns:
        Augmented array with reverse complement concatenated

    Note:
        The function reverses both the nucleotide encoding and sequence order.
    """
    logger.info(f"Creating reverse complement for data of shape {X.shape}")

    # Apply reverse complement mapping to all elements
    myfunc_vec = np.vectorize(reverse_complement_encoding)
    result = myfunc_vec(X)

    # Reverse the sequence order (axis=2 for sequence dimension)
    result = np.flip(result, axis=2)

    # Concatenate original and reverse complement along sequence dimension
    X_augmented = np.concatenate((X, result), axis=2).astype(np.uint8)

    logger.info(f"Augmented shape: {X_augmented.shape}")
    return X_augmented


def main():
    """Main data augmentation pipeline."""
    args = parse_arguments()

    # Setup data directory
    data_dir = args.data_dir.format(tf=args.tf)
    create_output_directory(data_dir)

    logger.info(f"Processing {args.tf} in {args.celltype} cell type")
    logger.info(f"Loading data from {len(args.chromosomes)} chromosomes")

    # Load first chromosome as base
    first_chrom = args.chromosomes[0]
    logger.info(f"Loading base chromosome {first_chrom}...")
    X_train, y_train = load_chromosome_data(data_dir, args.length, first_chrom, args.tf, args.celltype)

    if X_train is None:
        raise FileNotFoundError(f"Could not load base chromosome {first_chrom}")

    # Load remaining chromosomes
    for chrom in tqdm(args.chromosomes[1:], desc="Loading chromosomes"):
        X_chrom, y_chrom = load_chromosome_data(data_dir, args.length, chrom, args.tf, args.celltype)

        if X_chrom is not None:
            X_train = np.concatenate((X_train, X_chrom), axis=0)
            y_train = np.concatenate((y_train, y_chrom), axis=0)
        else:
            logger.warning(f"Skipping chromosome {chrom}")

    logger.info(f"Combined training data: X shape {X_train.shape}, y shape {y_train.shape}")

    # Augment with reverse complement
    logger.info("Generating reverse complement augmentation...")
    X_train_augmented = augment_with_reverse_complement(X_train)

    logger.info(f"Final shapes: X {X_train_augmented.shape}, y {y_train.shape}")

    # Save augmented data
    output_prefix = f"{data_dir}/X_revCompConcatenatedTrue{args.length}_{args.celltype}"
    save_numpy_array(X_train_augmented, f"{output_prefix}.npy")

    # Save labels (y doesn't change with augmentation)
    y_output_prefix = f"{data_dir}/y_revCompConcatenatedTrue{args.length}_{args.celltype}"
    save_numpy_array(y_train, f"{y_output_prefix}.npy")

    # Also save with alternative naming for compatibility
    y_alt_output = f"{data_dir}/y_revCompConcatenated{args.length}_{args.celltype}.npy"
    save_numpy_array(y_train, y_alt_output)

    logger.info("Data augmentation complete!")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
