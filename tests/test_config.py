"""
Unit tests for GraphyloVar configuration module.
"""

import pytest

# Add parent directory to path to import modules
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    SPECIES_LIST,
    NUM_SPECIES,
    DNA_BASES,
    DNA_COMPLEMENT,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    get_phylogenetic_adjacency_matrix,
)


class TestSpeciesConfig:
    """Test species configuration."""

    def test_species_list_not_empty(self):
        """Test that species list is not empty."""
        assert len(SPECIES_LIST) > 0

    def test_num_species_matches(self):
        """Test that NUM_SPECIES matches list length."""
        assert NUM_SPECIES == len(SPECIES_LIST)

    def test_species_list_expected_size(self):
        """Test that species list has expected size (115 species)."""
        assert NUM_SPECIES == 115

    def test_species_unique(self):
        """Test that all species names are unique."""
        assert len(SPECIES_LIST) == len(set(SPECIES_LIST))

    def test_reference_species_present(self):
        """Test that reference species (hg38) is present."""
        assert "hg38" in SPECIES_LIST


class TestDNAConfig:
    """Test DNA configuration."""

    def test_dna_bases_complete(self):
        """Test that DNA bases list is complete."""
        expected_bases = ["A", "C", "G", "T", "N", "-"]
        assert DNA_BASES == expected_bases

    def test_dna_complement_keys(self):
        """Test that complement has all bases as keys."""
        for base in DNA_BASES:
            assert base in DNA_COMPLEMENT


class TestDefaultHyperparameters:
    """Test default hyperparameters."""

    def test_context_window_positive(self):
        """Test that context window is positive."""
        assert DEFAULT_CONTEXT_WINDOW > 0

    def test_batch_size_positive(self):
        """Test that batch size is positive."""
        assert DEFAULT_BATCH_SIZE > 0

    def test_epochs_positive(self):
        """Test that epochs is positive."""
        assert DEFAULT_EPOCHS > 0

    def test_reasonable_values(self):
        """Test that hyperparameters have reasonable values."""
        assert 10 <= DEFAULT_CONTEXT_WINDOW <= 1000
        assert 1 <= DEFAULT_BATCH_SIZE <= 256
        assert 1 <= DEFAULT_EPOCHS <= 1000


class TestPhylogeneticFunctions:
    """Test phylogenetic utility functions."""

    def test_adjacency_matrix_shape(self):
        """Test that adjacency matrix has correct shape."""
        import numpy as np

        A = get_phylogenetic_adjacency_matrix()
        assert A.shape == (NUM_SPECIES, NUM_SPECIES)

    def test_adjacency_matrix_type(self):
        """Test that adjacency matrix is numpy array."""
        import numpy as np

        A = get_phylogenetic_adjacency_matrix()
        assert isinstance(A, np.ndarray)

    def test_adjacency_matrix_symmetric(self):
        """Test that adjacency matrix is symmetric."""
        import numpy as np

        A = get_phylogenetic_adjacency_matrix()
        assert np.allclose(A, A.T)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
