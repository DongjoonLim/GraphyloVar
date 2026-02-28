# Copyright (c) 2025 Dongjoon Lim, McGill University
# Licensed under the MIT License. See LICENSE file in the project root.
"""
GraphyloVar: Deep Learning for Predicting Functional Impact of
Non-Coding Variants Using Multi-Species Evolutionary Graphs.

Submodules:
    phylogeny   – species tree, adjacency matrix
    data        – dataset loading, windowing, masking, splitting
    models      – CNN-GCN, LSTM-GCN, Transformer-GCN, Conv2D-GCN, Bahdanau-GCN
    losses      – focal loss wrapper
    training    – fit loop, callbacks, plotting
    alignment   – Needleman-Wunsch with learned substitution matrices
    maf_parser  – raw MAF file → ungapped alignment dict
    evaluation  – ROC/PRC curves, AUC metrics, calibration
"""

__version__ = "0.3.0"
