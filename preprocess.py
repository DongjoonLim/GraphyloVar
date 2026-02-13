"""
Preprocess chromosome alignments for training GraphyloVar.

This script samples mutated and conserved sites from multi-species alignments,
extracts sequence windows with reverse complement augmentation, and encodes
them for training. The output includes both sequence features and target labels.
"""

import argparse
import logging
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from config import DNA_BASES, SPECIES_LIST
from utils import create_output_directory, reverse_complement, save_numpy_array, validate_file_path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess chromosome alignments for training.")
    parser.add_argument("--chrom", type=str, required=True, help="Chromosome number (e.g., '21').")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with .pkl alignments.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .npy files.")
    parser.add_argument("--flank_size", type=int, default=100, help="Flank size around center.")
    parser.add_argument("--num_mutated", type=int, default=100000, help="Number of mutated sites to sample.")
    parser.add_argument("--num_conserved", type=int, default=400000, help="Number of conserved sites to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def one_hot_encode_targets(base: str) -> np.ndarray:
    """
    One-hot encode target base (A,C,G,T,-/N).

    Args:
        base: DNA base character

    Returns:
        One-hot encoded vector of length 5
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "-": 4, "N": 4}
    vec = np.zeros(5, dtype=np.float32)
    idx = mapping.get(base, 4)
    vec[idx] = 1
    return vec


def identify_variant_sites(human_seq: str, ancestor_seq: str) -> Tuple[list, list]:
    """
    Identify mutated and conserved sites between human and ancestor sequences.

    Args:
        human_seq: Human reference sequence
        ancestor_seq: Ancestral sequence

    Returns:
        Tuple of (mutated_indices, conserved_indices)
    """
    seq_len = len(human_seq)

    # Find sites where human differs from ancestor (and not N)
    mutated_indices = [i for i in range(seq_len) if human_seq[i] != ancestor_seq[i] and human_seq[i] != "N"]

    # Find conserved sites
    conserved_indices = list(set(range(seq_len)) - set(mutated_indices))

    logger.info(f"Identified {len(mutated_indices)} mutated sites and {len(conserved_indices)} conserved sites")

    return mutated_indices, conserved_indices


def extract_window_features(
    alignment: dict, position: int, flank_size: int, species_list: list, label_encoder: LabelEncoder
) -> np.ndarray:
    """
    Extract sequence window features for all species at a given position.

    Args:
        alignment: Dictionary of species alignments
        position: Center position for extraction
        flank_size: Number of bases on each side of position
        species_list: List of species names to extract
        label_encoder: Fitted LabelEncoder for DNA bases

    Returns:
        Encoded feature matrix of shape (num_species, window_size)
    """
    example_matrix = []

    for species in species_list:
        seq_raw = alignment[species]
        segment = seq_raw[position - flank_size : position + flank_size + 1]

        # Add reverse complement augmentation
        rc_segment = reverse_complement(list(segment))
        combined_seq = list(segment) + rc_segment

        # Encode sequence
        encoded_seq = label_encoder.transform(combined_seq)
        example_matrix.append(encoded_seq)

    return np.array(example_matrix, dtype="uint8")


def preprocess_chromosome(args):
    """
    Main preprocessing pipeline for a chromosome.

    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Validate input
    pkl_path = os.path.join(args.input_dir, f"seqDictPad_chr{args.chrom}.pkl")
    validate_file_path(pkl_path, "file")

    # Load alignment
    logger.info(f"Loading alignment from {pkl_path}...")
    alignment = pd.read_pickle(pkl_path)
    logger.info("Alignment loaded successfully")

    # Get human and ancestor sequences
    human = alignment["hg38"]
    ancestor = alignment["_HP"]  # Human-Chimp Ancestor

    # Identify variant sites
    mutated_indices, conserved_indices = identify_variant_sites(human, ancestor)

    # Sample sites
    num_mutated_to_sample = min(len(mutated_indices), args.num_mutated)
    num_conserved_to_sample = min(len(conserved_indices), args.num_conserved)

    logger.info(f"Sampling {num_mutated_to_sample} mutated and {num_conserved_to_sample} conserved sites...")
    mutated_sampled = random.sample(mutated_indices, num_mutated_to_sample)
    conserved_sampled = random.sample(conserved_indices, num_conserved_to_sample)

    # Combine and shuffle
    all_indices = mutated_sampled + conserved_sampled
    random.shuffle(all_indices)
    logger.info(f"Total {len(all_indices)} sites to process")

    # Setup label encoder
    le = LabelEncoder().fit(DNA_BASES)

    # Extract features and targets
    X, Y = [], []
    skipped_count = 0
    seq_len = len(human)

    logger.info("Extracting windows and encoding...")
    for position in tqdm(all_indices, desc="Processing sites"):
        # Skip positions too close to chromosome ends
        if position < args.flank_size or position >= seq_len - args.flank_size:
            skipped_count += 1
            continue

        try:
            # Get current base and mutation status
            current_base = human[position]
            is_mutated = 1 if position in mutated_indices else 0

            # Create target vector (one-hot base + binary mutation status)
            target_vec = one_hot_encode_targets(current_base)
            y_sample = np.append(target_vec, is_mutated)

            # Extract features for all species
            example_matrix = extract_window_features(alignment, position, args.flank_size, SPECIES_LIST, le)

            X.append(example_matrix)
            Y.append(y_sample)

        except Exception as e:
            logger.warning(f"Skipping position {position}: {e}")
            skipped_count += 1
            continue

    # Convert to arrays
    X = np.array(X)
    Y = np.array(Y)

    logger.info(f"Processed {len(X)} examples, skipped {skipped_count} positions")
    logger.info(f"Final shapes: X {X.shape}, Y {Y.shape}")

    # Save outputs
    create_output_directory(args.output_dir)
    x_path = os.path.join(args.output_dir, f"X_train_chr{args.chrom}.npy")
    y_path = os.path.join(args.output_dir, f"y_train_chr{args.chrom}.npy")

    save_numpy_array(X, x_path)
    save_numpy_array(Y, y_path)

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    args = parse_arguments()
    preprocess_chromosome(args)
