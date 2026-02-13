"""
Utility functions for GraphyloVar preprocessing and data handling.

This module contains reusable utility functions for DNA sequence processing,
data validation, and common operations used across the codebase.
"""

import logging
from typing import List, Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder

from config import DNA_BASES, DNA_COMPLEMENT

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def reverse_complement(dna_sequence: List[str]) -> List[str]:
    """
    Compute reverse complement of a DNA sequence.

    Args:
        dna_sequence: List of DNA bases (A, C, G, T, N, or -)

    Returns:
        Reverse complement of the input sequence

    Example:
        >>> reverse_complement(['A', 'T', 'C', 'G'])
        ['C', 'G', 'A', 'T']
    """
    return [DNA_COMPLEMENT.get(base, "N") for base in reversed(dna_sequence)]


def encode_dna_sequence(sequence: List[str], label_encoder: Optional[LabelEncoder] = None) -> np.ndarray:
    """
    Encode DNA sequence using label encoding.

    Args:
        sequence: List of DNA bases
        label_encoder: Pre-fitted LabelEncoder. If None, creates a new one.

    Returns:
        Encoded sequence as numpy array

    Raises:
        ValueError: If sequence contains invalid bases
    """
    if label_encoder is None:
        label_encoder = LabelEncoder().fit(DNA_BASES)

    try:
        encoded = label_encoder.transform(sequence)
        return encoded.astype("uint8")
    except ValueError as e:
        logger.error(f"Error encoding sequence: {e}")
        raise ValueError(f"Sequence contains invalid DNA bases: {e}")


def one_hot_encode_dna(sequence: np.ndarray, num_classes: int = 6) -> np.ndarray:
    """
    Convert label-encoded DNA sequence to one-hot encoding.

    Args:
        sequence: Label-encoded DNA sequence
        num_classes: Number of DNA base classes (default: 6 for A,C,G,T,N,-)

    Returns:
        One-hot encoded sequence of shape (len(sequence), num_classes)
    """
    one_hot = np.zeros((len(sequence), num_classes), dtype=np.float32)
    one_hot[np.arange(len(sequence)), sequence] = 1
    return one_hot


def validate_sequence_length(sequence: np.ndarray, expected_length: int) -> bool:
    """
    Validate that a sequence has the expected length.

    Args:
        sequence: DNA sequence array
        expected_length: Expected sequence length

    Returns:
        True if sequence length matches expected length
    """
    return len(sequence) == expected_length


def load_alignment_safely(file_path: str, chromosome: str) -> Optional[dict]:
    """
    Safely load alignment data from pickle file with error handling.

    Args:
        file_path: Path to pickle file
        chromosome: Chromosome identifier

    Returns:
        Alignment dictionary or None if loading fails

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If alignment data is invalid
    """
    import pandas as pd
    import os

    full_path = f"{file_path}/seqDictPad_chr{chromosome}.pkl"

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Alignment file not found: {full_path}")

    try:
        alignment = pd.read_pickle(full_path)
        logger.info(f"Successfully loaded alignment for chromosome {chromosome}")
        return alignment
    except Exception as e:
        logger.error(f"Error loading alignment from {full_path}: {e}")
        raise ValueError(f"Failed to load alignment: {e}")


def validate_file_path(file_path: str, file_type: str = "file") -> None:
    """
    Validate that a file or directory exists.

    Args:
        file_path: Path to validate
        file_type: Type of path ('file' or 'directory')

    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If path type is invalid
    """
    import os

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_type.capitalize()} not found: {file_path}")

    if file_type == "file" and not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")
    elif file_type == "directory" and not os.path.isdir(file_path):
        raise ValueError(f"Path is not a directory: {file_path}")


def create_output_directory(output_path: str) -> None:
    """
    Create output directory if it doesn't exist.

    Args:
        output_path: Path to output directory
    """
    import os

    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Output directory ready: {output_path}")


def save_numpy_array(array: np.ndarray, file_path: str) -> None:
    """
    Save numpy array with error handling and logging.

    Args:
        array: Numpy array to save
        file_path: Output file path

    Raises:
        IOError: If save operation fails
    """
    try:
        np.save(file_path, array)
        logger.info(f"Saved array of shape {array.shape} to {file_path}")
    except Exception as e:
        logger.error(f"Error saving array to {file_path}: {e}")
        raise IOError(f"Failed to save numpy array: {e}")


def load_numpy_array(file_path: str, mmap_mode: Optional[str] = None, allow_pickle: bool = False) -> np.ndarray:
    """
    Load numpy array with error handling and logging.

    Args:
        file_path: Path to numpy array file
        mmap_mode: Memory mapping mode ('r', 'r+', 'w+', 'c')
        allow_pickle: Allow loading pickled object arrays (SECURITY WARNING: only use with trusted data)

    Returns:
        Loaded numpy array

    Raises:
        FileNotFoundError: If file does not exist
        IOError: If load operation fails

    Security Note:
        Setting allow_pickle=True can execute arbitrary code if the file is malicious.
        Only enable this for data from trusted sources.
    """
    validate_file_path(file_path, "file")

    try:
        array = np.load(file_path, mmap_mode=mmap_mode, allow_pickle=allow_pickle)
        logger.info(f"Loaded array from {file_path} with shape {array.shape}")
        return array
    except Exception as e:
        logger.error(f"Error loading array from {file_path}: {e}")
        raise IOError(f"Failed to load numpy array: {e}")
