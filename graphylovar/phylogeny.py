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
    ("susScr3", "_SVCTOPBOC"),
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
