"""
GraphyloVar: Deep Learning for Predicting Functional Impact of
Non-Coding Variants Using Multi-Species Evolutionary Graphs.

Submodules:
    phylogeny  – species tree, adjacency matrix
    data       – dataset loading, windowing, masking, splitting
    models     – CNN-GCN, LSTM-GCN, Transformer-GCN architectures
    losses     – focal loss wrapper
    training   – fit loop, callbacks, plotting
    alignment  – Needleman-Wunsch with learned substitution matrices
"""

__version__ = "0.2.0"
