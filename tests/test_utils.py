"""
Unit tests for GraphyloVar utility functions.

Run tests with: python -m pytest tests/
"""

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path to import modules
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import reverse_complement, encode_dna_sequence, one_hot_encode_dna, validate_sequence_length
from config import DNA_BASES, DNA_COMPLEMENT


class TestReverseComplement:
    """Test reverse complement function."""

    def test_simple_sequence(self):
        """Test reverse complement of simple sequence."""
        sequence = ["A", "T", "C", "G"]
        expected = ["C", "G", "A", "T"]
        assert reverse_complement(sequence) == expected

    def test_with_gaps(self):
        """Test reverse complement with gaps."""
        sequence = ["A", "T", "-", "G"]
        expected = ["C", "-", "A", "T"]
        assert reverse_complement(sequence) == expected

    def test_with_n(self):
        """Test reverse complement with N (unknown base)."""
        sequence = ["A", "N", "C", "G"]
        expected = ["C", "G", "N", "T"]
        assert reverse_complement(sequence) == expected

    def test_empty_sequence(self):
        """Test reverse complement of empty sequence."""
        assert reverse_complement([]) == []

    def test_palindrome(self):
        """Test reverse complement of palindromic sequence."""
        sequence = ["A", "T", "A", "T"]
        expected = ["A", "T", "A", "T"]
        assert reverse_complement(sequence) == expected


class TestEncodeDNASequence:
    """Test DNA sequence encoding."""

    def test_basic_encoding(self):
        """Test basic DNA sequence encoding."""
        sequence = ["A", "C", "G", "T"]
        le = LabelEncoder().fit(DNA_BASES)
        encoded = encode_dna_sequence(sequence, le)

        assert encoded.dtype == np.uint8
        assert len(encoded) == len(sequence)

    def test_with_gaps_and_n(self):
        """Test encoding with gaps and N."""
        sequence = ["A", "C", "-", "N"]
        le = LabelEncoder().fit(DNA_BASES)
        encoded = encode_dna_sequence(sequence, le)

        assert len(encoded) == len(sequence)

    def test_invalid_base_raises_error(self):
        """Test that invalid bases raise ValueError."""
        sequence = ["A", "X", "C"]  # X is invalid
        le = LabelEncoder().fit(DNA_BASES)

        with pytest.raises(ValueError):
            encode_dna_sequence(sequence, le)


class TestOneHotEncode:
    """Test one-hot encoding."""

    def test_basic_one_hot(self):
        """Test basic one-hot encoding."""
        sequence = np.array([0, 1, 2, 3])  # A, C, G, T
        one_hot = one_hot_encode_dna(sequence, num_classes=6)

        assert one_hot.shape == (4, 6)
        assert one_hot.dtype == np.float32
        assert np.sum(one_hot) == 4  # Each position has one 1

    def test_one_hot_values(self):
        """Test one-hot encoding values."""
        sequence = np.array([0, 1])
        one_hot = one_hot_encode_dna(sequence, num_classes=6)

        assert one_hot[0, 0] == 1
        assert np.sum(one_hot[0, 1:]) == 0
        assert one_hot[1, 1] == 1
        assert np.sum(one_hot[1, [0, 2, 3, 4, 5]]) == 0


class TestValidateSequenceLength:
    """Test sequence length validation."""

    def test_valid_length(self):
        """Test validation with correct length."""
        sequence = np.array([1, 2, 3, 4, 5])
        assert validate_sequence_length(sequence, 5) is True

    def test_invalid_length(self):
        """Test validation with incorrect length."""
        sequence = np.array([1, 2, 3])
        assert validate_sequence_length(sequence, 5) is False

    def test_empty_sequence(self):
        """Test validation with empty sequence."""
        sequence = np.array([])
        assert validate_sequence_length(sequence, 0) is True
        assert validate_sequence_length(sequence, 5) is False


class TestConfig:
    """Test configuration values."""

    def test_dna_complement_completeness(self):
        """Test that DNA complement mapping is complete."""
        for base in DNA_BASES:
            assert base in DNA_COMPLEMENT

    def test_complement_symmetry(self):
        """Test that complement mapping is symmetric for Watson-Crick pairs."""
        assert DNA_COMPLEMENT["A"] == "T"
        assert DNA_COMPLEMENT["T"] == "A"
        assert DNA_COMPLEMENT["C"] == "G"
        assert DNA_COMPLEMENT["G"] == "C"

    def test_self_complement(self):
        """Test that N and - are self-complementary."""
        assert DNA_COMPLEMENT["N"] == "N"
        assert DNA_COMPLEMENT["-"] == "-"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
