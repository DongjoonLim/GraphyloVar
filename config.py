"""
Configuration constants for GraphyloVar.

This module contains all shared configuration values used across the codebase,
including species lists, phylogenetic relationships, and hyperparameters.
"""

from typing import List, Tuple

# Species list - 115 species including ancestral nodes
SPECIES_LIST: List[str] = [
    # Primates
    "hg38",
    "panTro4",
    "gorGor3",
    "ponAbe2",
    "nomLeu3",
    "rheMac3",
    "macFas5",
    "papAnu2",
    "chlSab2",
    "calJac3",
    "saiBol1",
    "otoGar3",
    # Rodents & Lagomorphs
    "tupChi1",
    "speTri2",
    "jacJac1",
    "micOch1",
    "criGri1",
    "mesAur1",
    "mm10",
    "rn6",
    "hetGla2",
    "cavPor3",
    "chiLan1",
    "octDeg1",
    "oryCun2",
    "ochPri3",
    # Artiodactyls & Carnivores
    "susScr3",
    "vicPac2",
    "camFer1",
    "turTru2",
    "orcOrc1",
    "panHod1",
    "bosTau8",
    "oviAri3",
    "capHir1",
    "equCab2",
    "cerSim1",
    "felCat8",
    "canFam3",
    "musFur1",
    "ailMel1",
    "odoRosDiv1",
    "lepWed1",
    # Chiroptera & Other Mammals
    "pteAle1",
    "pteVam1",
    "eptFus1",
    "myoDav1",
    "myoLuc2",
    "eriEur2",
    "sorAra2",
    "conCri1",
    "loxAfr3",
    "eleEdw1",
    "triMan1",
    "chrAsi1",
    "echTel2",
    "oryAfe1",
    "dasNov3",
    # Ancestral nodes (phylogenetic tree internal nodes)
    "_HP",
    "_HPG",
    "_HPGP",
    "_HPGPN",
    "_RM",
    "_RMP",
    "_RMPC",
    "_HPGPNRMPC",
    "_CS",
    "_HPGPNRMPCCS",
    "_HPGPNRMPCCSO",
    "_HPGPNRMPCCSOT",
    "_CM",
    "_MR",
    "_MCM",
    "_MCMMR",
    "_JMCMMR",
    "_SJMCMMR",
    "_CO",
    "_CCO",
    "_HCCO",
    "_SJMCMMRHCCO",
    "_OO",
    "_SJMCMMRHCCOOO",
    "_HPGPNRMPCCSOTSJMCMMRHCCOOO",
    "_VC",
    "_TO",
    "_OC",
    "_BOC",
    "_PBOC",
    "_TOPBOC",
    "_VCTOPBOC",
    "_SVCTOPBOC",
    "_EC",
    "_OL",
    "_AOL",
    "_MAOL",
    "_CMAOL",
    "_FCMAOL",
    "_ECFCMAOL",
    "_PP",
    "_MM",
    "_EMM",
    "_PPEMM",
    "_ECFCMAOLPPEMM",
    "_SVCTOPBOCECFCMAOLPPEMM",
    "_SC",
    "_ESC",
    "_SVCTOPBOCECFCMAOLPPEMMESC",
    "_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESC",
    "_LE",
    "_LET",
    "_CE",
    "_LETCE",
    "_LETCEO",
    "_LETCEOD",
    "_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESCLETCEOD",
]

# Number of species/nodes in the phylogenetic graph
NUM_SPECIES: int = len(SPECIES_LIST)

# DNA bases for encoding/decoding
DNA_BASES: List[str] = ["A", "C", "G", "T", "N", "-"]

# Complement mapping for reverse complement calculation
DNA_COMPLEMENT: dict = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N", "-": "-"}

# Default hyperparameters
DEFAULT_CONTEXT_WINDOW: int = 100
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_EPOCHS: int = 10
DEFAULT_LEARNING_RATE: float = 0.001

# CNN hyperparameters
DEFAULT_CNN_FILTERS: int = 64
DEFAULT_CNN_KERNEL_SIZE: int = 5

# LSTM/RNN hyperparameters
DEFAULT_LSTM_UNITS: int = 128

# GCN hyperparameters
DEFAULT_GCN_UNITS: int = 64

# Focal loss parameters
DEFAULT_FOCAL_LOSS_GAMMA: float = 2.0


def get_phylogenetic_edges() -> List[Tuple[str, str]]:
    """
    Get phylogenetic tree edges representing evolutionary relationships.

    Returns:
        List of tuples representing parent-child relationships in the phylogenetic tree.

    Note:
        This is a placeholder. In production, these edges should be loaded from
        a phylogenetic tree file (e.g., Newick format) or database.
    """
    # TODO: Load actual phylogenetic relationships from tree file
    # For now, return a simple hierarchical structure as an example
    edges = [
        ("hg38", "_HP"),
        ("panTro4", "_HP"),
        ("_HP", "_HPG"),
        ("gorGor3", "_HPG"),
        # Add more edges based on your actual phylogenetic tree
    ]
    return edges


def get_phylogenetic_adjacency_matrix():
    """
    Generate adjacency matrix from phylogenetic tree structure.

    Returns:
        numpy array of shape (NUM_SPECIES, NUM_SPECIES) representing the adjacency matrix.

    Note:
        This is currently a placeholder returning an identity matrix.
        In production, this should be computed from the actual phylogenetic tree.
    """
    import numpy as np

    # TODO: Compute adjacency from phylogenetic edges
    # For now, return identity matrix as placeholder
    return np.eye(NUM_SPECIES)
