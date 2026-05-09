# Copyright (c) 2025 Dongjoon Lim, McGill University
# Licensed under the MIT License. See LICENSE file in the project root.
"""
Single source of truth for the 115-node mammalian phylogenetic graph
used across all GraphyloVar / Graphylo models.

Provides:
    SPECIES       – ordered list of 58 extant species
    ANCESTORS     – ordered list of 57 internal / ancestor nodes
    NAMES         – SPECIES + ANCESTORS (len 115)
    EDGES         – list of (parent, child) tuples
    build_graph() – returns (networkx.Graph, adjacency np.ndarray)
"""

from __future__ import annotations

import networkx as nx
import numpy as np

# ── Extant species (leaves) ─────────────────────────────────────────
SPECIES: list[str] = [
    "hg38", "panTro4", "gorGor3", "ponAbe2", "nomLeu3",
    "rheMac3", "macFas5", "papAnu2", "chlSab2",
    "calJac3", "saiBol1", "otoGar3", "tupChi1",
    "speTri2", "jacJac1", "micOch1", "criGri1", "mesAur1",
    "mm10", "rn6", "hetGla2", "cavPor3", "chiLan1", "octDeg1",
    "oryCun2", "ochPri3",
    "susScr3", "vicPac2", "camFer1",
    "turTru2", "orcOrc1", "panHod1", "bosTau8", "oviAri3", "capHir1",
    "equCab2", "cerSim1",
    "felCat8", "canFam3", "musFur1", "ailMel1", "odoRosDiv1", "lepWed1",
    "pteAle1", "pteVam1", "eptFus1", "myoDav1", "myoLuc2",
    "eriEur2", "sorAra2", "conCri1",
    "loxAfr3", "eleEdw1", "triMan1", "chrAsi1", "echTel2",
    "oryAfe1", "dasNov3",
]

# ── Ancestral / internal nodes ──────────────────────────────────────
ANCESTORS: list[str] = [
    "_HP", "_HPG", "_HPGP", "_HPGPN",
    "_RM", "_RMP", "_RMPC", "_HPGPNRMPC",
    "_CS", "_HPGPNRMPCCS", "_HPGPNRMPCCSO", "_HPGPNRMPCCSOT",
    "_CM", "_MR", "_MCM", "_MCMMR", "_JMCMMR", "_SJMCMMR",
    "_CO", "_CCO", "_HCCO", "_SJMCMMRHCCO",
    "_OO", "_SJMCMMRHCCOOO", "_HPGPNRMPCCSOTSJMCMMRHCCOOO",
    "_VC", "_TO", "_OC", "_BOC", "_PBOC", "_TOPBOC", "_VCTOPBOC", "_SVCTOPBOC",
    "_EC", "_OL", "_AOL", "_MAOL", "_CMAOL", "_FCMAOL", "_ECFCMAOL",
    "_PP", "_MM", "_EMM", "_PPEMM", "_ECFCMAOLPPEMM", "_SVCTOPBOCECFCMAOLPPEMM",
    "_SC", "_ESC", "_SVCTOPBOCECFCMAOLPPEMMESC",
    "_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESC",
    "_LE", "_LET", "_CE", "_LETCE", "_LETCEO", "_LETCEOD",
    "_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESCLETCEOD",
]

# Full ordered list – 115 nodes
NAMES: list[str] = SPECIES + ANCESTORS
NUM_NODES: int = len(NAMES)  # 115

# ── Edges (undirected) ──────────────────────────────────────────────
EDGES: list[tuple[str, str]] = [
    # Great apes
    ("hg38", "_HP"), ("panTro4", "_HP"),
    ("gorGor3", "_HPG"), ("_HP", "_HPG"),
    ("ponAbe2", "_HPGP"), ("_HPG", "_HPGP"),
    ("nomLeu3", "_HPGPN"), ("_HPGP", "_HPGPN"),
    # Old-world monkeys
    ("rheMac3", "_RM"), ("macFas5", "_RM"),
    ("_RM", "_RMP"), ("papAnu2", "_RMP"),
    ("_RMP", "_RMPC"), ("chlSab2", "_RMPC"),
    ("_RMPC", "_HPGPNRMPC"), ("_HPGPN", "_HPGPNRMPC"),
    # New-world monkeys
    ("calJac3", "_CS"), ("saiBol1", "_CS"),
    ("_CS", "_HPGPNRMPCCS"), ("_HPGPNRMPC", "_HPGPNRMPCCS"),
    # Strepsirrhini + Scandentia
    ("otoGar3", "_HPGPNRMPCCSO"), ("_HPGPNRMPCCS", "_HPGPNRMPCCSO"),
    ("tupChi1", "_HPGPNRMPCCSOT"), ("_HPGPNRMPCCSO", "_HPGPNRMPCCSOT"),
    # Rodents
    ("speTri2", "_SJMCMMR"), ("_SJMCMMR", "_JMCMMR"),
    ("jacJac1", "_JMCMMR"),
    ("micOch1", "_MCM"), ("_MCM", "_MCMMR"), ("_MCMMR", "_JMCMMR"),
    ("criGri1", "_CM"), ("mesAur1", "_CM"), ("_CM", "_MCM"),
    ("mm10", "_MR"), ("rn6", "_MR"), ("_MR", "_MCMMR"),
    ("_SJMCMMRHCCO", "_HCCO"), ("_SJMCMMRHCCO", "_SJMCMMR"),
    ("_SJMCMMRHCCO", "_SJMCMMRHCCOOO"),
    ("_HPGPNRMPCCSOTSJMCMMRHCCOOO", "_HPGPNRMPCCSOT"),
    ("_HPGPNRMPCCSOTSJMCMMRHCCOOO", "_SJMCMMRHCCOOO"),
    ("_CCO", "_HCCO"), ("_CO", "_CCO"), ("_OO", "_SJMCMMRHCCOOO"),
    ("hetGla2", "_HCCO"), ("cavPor3", "_CCO"),
    ("chiLan1", "_CO"), ("octDeg1", "_CO"),
    ("oryCun2", "_OO"), ("ochPri3", "_OO"),
    # Laurasiatheria – Cetartiodactyla
    ("vicPac2", "_VC"), ("camFer1", "_VC"),
    ("turTru2", "_TO"), ("orcOrc1", "_TO"),
    ("oviAri3", "_OC"), ("capHir1", "_OC"),
    ("bosTau8", "_BOC"), ("_OC", "_BOC"),
    ("panHod1", "_PBOC"), ("_BOC", "_PBOC"),
    ("_PBOC", "_TOPBOC"), ("_TO", "_TOPBOC"),
    ("_TOPBOC", "_VCTOPBOC"), ("_VC", "_VCTOPBOC"),
    ("_VCTOPBOC", "_SVCTOPBOC"), ("susScr3", "_SVCTOPBOC"),
    # Perissodactyla + Carnivora
    ("equCab2", "_EC"), ("cerSim1", "_EC"),
    ("odoRosDiv1", "_OL"), ("lepWed1", "_OL"), ("_OL", "_AOL"),
    ("ailMel1", "_AOL"), ("_AOL", "_MAOL"), ("musFur1", "_MAOL"),
    ("_MAOL", "_CMAOL"), ("canFam3", "_CMAOL"),
    ("_CMAOL", "_FCMAOL"), ("felCat8", "_FCMAOL"),
    ("_FCMAOL", "_ECFCMAOL"), ("_EC", "_ECFCMAOL"),
    # Chiroptera
    ("pteAle1", "_PP"), ("pteVam1", "_PP"),
    ("myoDav1", "_MM"), ("myoLuc2", "_MM"),
    ("eptFus1", "_EMM"), ("_MM", "_EMM"),
    ("_EMM", "_PPEMM"), ("_PP", "_PPEMM"),
    ("_PPEMM", "_ECFCMAOLPPEMM"), ("_ECFCMAOL", "_ECFCMAOLPPEMM"),
    ("_ECFCMAOLPPEMM", "_SVCTOPBOCECFCMAOLPPEMM"),
    ("_SVCTOPBOC", "_SVCTOPBOCECFCMAOLPPEMM"),
    # Eulipotyphla
    ("sorAra2", "_SC"), ("conCri1", "_SC"),
    ("_SC", "_ESC"), ("eriEur2", "_ESC"),
    ("_ESC", "_SVCTOPBOCECFCMAOLPPEMMESC"),
    ("_SVCTOPBOCECFCMAOLPPEMM", "_SVCTOPBOCECFCMAOLPPEMMESC"),
    # Boreoeutherian root
    ("_SVCTOPBOCECFCMAOLPPEMM",
     "_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESC"),
    ("_HPGPNRMPCCSOTSJMCMMRHCCOOO",
     "_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESC"),
    # Afrotheria + Xenarthra
    ("loxAfr3", "_LE"), ("eleEdw1", "_LE"),
    ("triMan1", "_LET"), ("_LE", "_LET"),
    ("chrAsi1", "_CE"), ("echTel2", "_CE"),
    ("_LET", "_LETCE"), ("_CE", "_LETCE"),
    ("_LETCE", "_LETCEO"), ("oryAfe1", "_LETCEO"),
    ("_LETCEO", "_LETCEOD"), ("dasNov3", "_LETCEOD"),
    # Root
    ("_LETCEOD",
     "_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESCLETCEOD"),
    ("_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESC",
     "_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESCLETCEOD"),
]


def build_graph() -> tuple[nx.Graph, np.ndarray]:
    """
    Build the phylogenetic graph and its adjacency matrix.

    Returns
    -------
    G : nx.Graph
        NetworkX graph with 115 nodes (name attr = integer index).
    A : np.ndarray, shape (115, 115)
        Adjacency matrix ordered by NAMES.
    """
    G = nx.Graph(name="phylogeny")
    for idx, name in enumerate(NAMES):
        G.add_node(name, name=idx)
    G.add_edges_from(EDGES)
    A = np.array(nx.attr_matrix(G, node_attr="name")[0])
    return G, A


# ── Convenience: indices of species to mask during training ─────────
# (human, chimp, gorilla, and two specific ancestor nodes)
MASK_INDICES: list[int] = [
    NAMES.index("hg38"),       # 0
    NAMES.index("panTro4"),    # 1
    NAMES.index("gorGor3"),    # 2
    NAMES.index("_HP"),        # 58
    NAMES.index("_HPG"),       # 59
]


# ── Branch-length distances from hg38 (substitutions / site) ────────
# Computed from the UCSC hg38 100-way Multiz alignment Newick tree.
# Values represent the total branch length along the path from hg38
# to each species (sum of edge lengths, measured in expected
# substitutions per neutral site).
BRANCH_LENGTH_FROM_HUMAN: dict[str, float] = {
    "hg38": 0.000000,
    "panTro4": 0.013390,
    "gorGor3": 0.019734,
    "ponAbe2": 0.039403,
    "nomLeu3": 0.046204,
    "rheMac3": 0.079575,
    "macFas5": 0.079575,
    "papAnu2": 0.079626,
    "chlSab2": 0.087974,
    "calJac3": 0.107454,
    "saiBol1": 0.087804,
    "otoGar3": 0.270334,
    "tupChi1": 0.318845,
    "speTri2": 0.335427,
    "jacJac1": 0.409959,
    "micOch1": 0.510109,
    "criGri1": 0.510109,
    "mesAur1": 0.510109,
    "mm10": 0.502391,
    "rn6": 0.509471,
    "hetGla2": 0.347117,
    "cavPor3": 0.362746,
    "chiLan1": 0.417117,
    "octDeg1": 0.457117,
    "oryCun2": 0.376911,
    "ochPri3": 0.463753,
    "susScr3": 0.339399,
    "vicPac2": 0.327009,
    "camFer1": 0.319734,
    "turTru2": 0.329575,
    "orcOrc1": 0.334575,
    "panHod1": 0.378479,
    "bosTau8": 0.388479,
    "oviAri3": 0.388479,
    "capHir1": 0.388479,
    "equCab2": 0.319523,
    "cerSim1": 0.285126,
    "felCat8": 0.358583,
    "canFam3": 0.332429,
    "musFur1": 0.359971,
    "ailMel1": 0.359971,
    "odoRosDiv1": 0.379971,
    "lepWed1": 0.379971,
    "pteAle1": 0.337613,
    "pteVam1": 0.351012,
    "eptFus1": 0.317613,
    "myoDav1": 0.387613,
    "myoLuc2": 0.390153,
    "eriEur2": 0.465906,
    "sorAra2": 0.513683,
    "conCri1": 0.444121,
    "loxAfr3": 0.345811,
    "eleEdw1": 0.393569,
    "triMan1": 0.396579,
    "chrAsi1": 0.286882,
    "echTel2": 0.492818,
    "oryAfe1": 0.246882,
    "dasNov3": 0.366691,
}


def branch_length_to_human(species_name: str) -> float:
    """
    Return the total branch-length distance from hg38 to *species_name*.

    Units are expected substitutions per neutral site, computed from the
    UCSC hg38 100-way Multiz alignment phylogenetic tree.
    """
    if species_name not in BRANCH_LENGTH_FROM_HUMAN:
        raise KeyError(
            f"Species '{species_name}' not in BRANCH_LENGTH_FROM_HUMAN. "
            f"Available: {sorted(BRANCH_LENGTH_FROM_HUMAN)}"
        )
    return BRANCH_LENGTH_FROM_HUMAN[species_name]
