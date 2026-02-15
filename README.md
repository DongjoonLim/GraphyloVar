# GraphyloVar

**Predicting Functional Impact of Non-Coding Variants Using Multi-Species Evolutionary Graphs**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow 2.6+](https://img.shields.io/badge/tensorflow-2.6+-orange.svg)](https://www.tensorflow.org/)

## Data Availability Note

The full-scale Multi-Alignment Format (MAF) files and pre-split training/validation/test datasets are very large (terabyte scale). We are working to host them publicly. In the meantime, Boreoeutherian MAF alignments are available from [the Boreoeutherian Repository](http://repo.cs.mcgill.ca/PUB/blanchem/Boreoeutherian/).

## Overview

GraphyloVar is a deep learning framework that uses phylogenetic graph neural networks to predict the functional impact of non-coding genetic variants. It leverages multi-species whole-genome alignments (58 mammals + 57 inferred ancestors = 115 nodes) encoded as a graph, combining convolutional / recurrent / transformer encoders with Graph Convolutional Networks (GCN). The model is pretrained on allele frequency data from TOPMed.

### Distinction: Graphylo vs GraphyloVar

| Project | Purpose | Loss |
|---------|---------|------|
| **Graphylo** | CNN-GCN conservation / protein binding site prediction | Binary cross-entropy |
| **GraphyloVar** | Variant functional impact, pretrained on allele frequencies | Binary focal loss |

## Repository Structure

```
GraphyloVar/
├── graphylovar/              # Python package
│   ├── __init__.py
│   ├── phylogeny.py          # 115-node species tree: species, edges, adjacency matrix
│   ├── data.py               # Data loading, windowing, masking, train/val splitting
│   ├── models.py             # CNN-GCN, LSTM-GCN, Transformer-GCN, EvoLSTM baseline
│   ├── losses.py             # Binary focal loss for imbalanced classification
│   ├── training.py           # Training loop with early stopping & checkpointing
│   └── alignment.py          # Needleman-Wunsch & context-dependent alignment (EvolignSubst)
├── scripts/
│   ├── preprocess.py         # Build .npy arrays from CADD VCFs + alignment pickles
│   ├── train.py              # Train any model variant via CLI
│   ├── predict.py            # Run inference on held-out chromosomes
│   └── evaluate_alignment.py # Benchmark model-guided vs classical alignment
├── configs/
│   └── default.yaml          # All hyperparameters in one place
├── environment.yml           # Conda environment
├── setup.py                  # pip-installable package
└── README.md
```

## Model Architectures

All three graph-based models share the same pipeline:

1. **Input**: `(batch, 115, seq_len)` uint8 → one-hot to 6 channels `[A, C, G, T, N, -]`
2. **Siamese split**: forward strand + reverse complement halves
3. **Encoder**: CNN, BiLSTM, or Transformer (per-species)
4. **Species attention**: Squeeze-and-Excitation on the species axis
5. **GCN layers**: 2× GCNConv on the phylogenetic adjacency matrix
6. **Classifier**: Dense → softmax (2 classes: conserved vs mutated)

### CNN-GCN (`cnn_gcn`)
Siamese 1-D convolutions with channel attention → species attention → GCN.

### LSTM-GCN (`lstm_gcn`)
Bidirectional LSTM with multi-head attention per half → species attention → GCN.

### Transformer-GCN (`transformer_gcn`)
Embedding + positional encoding → stacked transformer encoder → species attention → GCN.

### EvoLSTM Baseline (`evolstm`)
Single-species BiLSTM + self-attention (no phylogenetic graph).

## Quick Start

### 1. Install

```bash
git clone https://github.com/DongjoonLim/GraphyloVar.git
cd GraphyloVar
conda env create -f environment.yml
conda activate graphylo
pip install -e .
```

### 2. Preprocess

```bash
python scripts/preprocess.py \
    --chromosomes 1 2 3 \
    --cadd_dir cadd_data \
    --alignment_dir ../../conservation/data \
    --output_dir data
```

### 3. Train

```bash
# Train CNN-GCN with focal loss (default config)
python scripts/train.py --config configs/default.yaml

# Train LSTM-GCN on a specific chromosome
python scripts/train.py --model lstm_gcn --chromosome 1 --gpu 0

# Train Transformer-GCN
python scripts/train.py --model transformer_gcn --batch_size 128
```

### 4. Predict

```bash
python scripts/predict.py \
    --model_path models/cnn_gcn_focal_context21_chr1 \
    --chromosome 22
```

### 5. Evaluate Alignment

```bash
python scripts/evaluate_alignment.py \
    --model_path models/graphylo_lstm_mutation \
    --alignment_pkl ../../conservation/data/seqDictPad_chr22.pkl \
    --s_species camFer1 --t_species mm10
```

## Configuration

All hyperparameters live in `configs/default.yaml`. CLI arguments override config values:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `cnn_gcn` | Architecture: `cnn_gcn`, `lstm_gcn`, `transformer_gcn`, `evolstm` |
| `loss` | `focal` | Loss function: `focal`, `bce`, `cce` |
| `context` | 100 | Flanking bases in preprocessing |
| `context_flank` | 10 | Narrow window (21 bases total) |
| `focal_gamma` | 2.0 | Focal loss focusing parameter |
| `batch_size` | 64 | Training batch size |
| `patience` | 7 | Early stopping patience |

## Species Tree

The model operates on a 115-node phylogenetic graph covering 58 extant mammalian species (from human to armadillo) plus 57 inferred ancestral nodes. During training, human (`hg38`), chimpanzee (`panTro4`), gorilla (`gorGor3`), and two ancestral nodes (`_HP`, `_HPG`) are masked to prevent information leakage.

## License

MIT License — see [LICENSE](LICENSE) for details.
