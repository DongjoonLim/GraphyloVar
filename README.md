# GraphyloVar

**Predicting Functional Impact of Non-Coding Variants Using Multi-Species Evolutionary Graphs**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow 2.6+](https://img.shields.io/badge/tensorflow-2.6+-orange.svg)](https://www.tensorflow.org/)

## Data Availability Note

The full-scale Multi-Alignment Format (MAF) files and pre-split training/validation/test datasets are very large (terabyte scale). We are working to host them publicly. In the meantime, Boreoeutherian MAF alignments are available from [the Boreoeutherian Repository](http://repo.cs.mcgill.ca/PUB/blanchem/Boreoeutherian/).

## Overview

GraphyloVar is a deep learning framework that uses phylogenetic graph neural networks to predict the functional impact of non-coding genetic variants. It leverages multi-species whole-genome alignments (58 mammals + 57 inferred ancestors = 115 nodes) encoded as a graph, combining convolutional / recurrent / transformer encoders with Graph Convolutional Networks (GCN). The model is pretrained on allele frequency data from **TOPMed** (Trans-Omics for Precision Medicine).

### Distinction: Graphylo vs GraphyloVar

| Project | Purpose | Input | Task |
|---------|---------|-------|------|
| **Graphylo** | CNN-GCN for protein binding site prediction | Alignment graph | Binary classification |
| **GraphyloVar** | Variant functional impact scoring | Alignment graph + variant | Allele frequency regression / variant effect prediction |

## Repository Structure

```
GraphyloVar/
├── graphylovar/              # Python package (pip-installable)
│   ├── __init__.py
│   ├── phylogeny.py          # 115-node species tree: species, edges, adjacency matrix
│   ├── data.py               # Data loading, windowing, masking, train/val splitting
│   ├── models.py             # CNN-GCN, LSTM-GCN, Transformer-GCN, Conv2D-GCN, Bahdanau-GCN, EvoLSTM
│   ├── losses.py             # Binary focal loss for imbalanced classification
│   ├── training.py           # Training loop with early stopping & checkpointing
│   ├── alignment.py          # Needleman-Wunsch & context-dependent alignment (EvolignSubst)
│   ├── maf_parser.py         # Raw MAF file → ungapped alignment pickle
│   └── evaluation.py         # ROC/PRC curves, AUC metrics, probability calibration
├── scripts/
│   ├── parse_maf.py          # Parse raw MAF alignments per chromosome
│   ├── preprocess.py         # Build .npy arrays from TOPMed VCFs + alignment pickles
│   ├── train.py              # Train any model variant via CLI
│   ├── predict.py            # Run inference on held-out chromosomes
│   ├── predict_genome.py     # Generate genome-wide bedGraph prediction tracks
│   ├── evaluate_alignment.py # Benchmark model-guided vs classical alignment
│   └── benchmarks/           # External benchmark comparison scripts
│       └── score_evo2_api.py # Score variants via NVIDIA Evo2 API
├── configs/
│   └── default.yaml          # All hyperparameters in one place
├── environment.yml           # Conda environment
├── setup.py                  # pip-installable package
└── README.md
```

## Model Architectures

All graph-based models share the same pipeline:

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

### Conv2D-GCN (`conv2d_gcn`)
2-D convolutions treating (species, position, one-hot) as an image → GCN with L2 regularization.

### Bahdanau-GCN (`bahdanau_gcn`)
Conv2D encoder with additive (Bahdanau) attention between query/value branches → GCN.

### EvoLSTM Baseline (`evolstm`)
Single-species BiLSTM + self-attention (no phylogenetic graph).

## Benchmark Results

GraphyloVar has been benchmarked against a comprehensive set of variant effect prediction tools. All variants are sourced from **TOPMed** (UCSC `snp151` track, filtered to TOPMED-submitted SNPs with allele frequency data). The evaluation task is distinguishing **common TOPMed variants** from **rare/non-SNP variants** across multiple chromosomes.

### TOPMed Variant Evaluation (Genome-Wide)

Multi-chromosome evaluation on TOPMed variants with MAF-stratified AUC:

| Model | AUC | Category |
|-------|-----|----------|
| GPN-MSA | 0.742 | Alignment-aware DNA language model |
| CADD | 0.715 | Ensemble integrator |
| PhyloP (100-way) | 0.700 | Conservation |
| PhastCons (100-way) | 0.680 | Conservation |
| **GraphyloVar (Transformer, flank=8)** | **0.695** | **Ours** |
| **GraphyloVar (Conditional, MLP)** | **0.688** | **Ours** |
| **GraphyloVar (MLP, flank=32)** | **0.671** | **Ours** |
| **GraphyloVar (EvoLSTM)** | **0.634** | **Ours (no graph)** |
| GPN-Star (V100) | 0.635 | DNA language model |
| Roulette | 0.578 | Mutation rate |

*Results are based on TOPMed allele frequency labels. Higher AUC indicates better discrimination between common and rare/non-SNP variants.*

## Comparison Tools Referenced

| Tool | Type | Reference |
|------|------|-----------|
| **Evo2** (7B, 40B) | DNA foundation model (7B/40B params) | [Genome biology with 11,000 genomes](https://arcinstitute.org/news/evo2) |
| **GPN-Star** (V100/M447/P243/P36) | Alignment-aware DNA language model | [songlab/gpn-msa](https://huggingface.co/songlab) |
| **GPN-MSA** | Alignment-conditioned DNA language model | [Benegas et al. 2023](https://www.biorxiv.org/content/10.1101/2023.10.10.561776) |
| **CADD** | Ensemble deleteriousness score | [Kircher et al. 2014](https://doi.org/10.1038/ng.2892) |
| **PhyloP** | Conservation (phyloP scores) | [Pollard et al. 2010](https://doi.org/10.1101/gr.097857.109) |
| **PhastCons** | Conservation (phastCons scores) | [Siepel et al. 2005](https://doi.org/10.1101/gr.3715005) |
| **Nucleotide Transformer** (2.5B) | DNA foundation model | [Dalla-Torre et al. 2023](https://doi.org/10.1101/2023.01.11.523679) |
| **HyenaDNA** | Long-range DNA model | [Nguyen et al. 2023](https://arxiv.org/abs/2306.15794) |
| **Roulette** | Mutation rate estimation | [Samocha et al. 2017](https://doi.org/10.1038/ng.3831) |

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

### 5. Parse Raw MAF Alignments

```bash
python scripts/parse_maf.py \
    --maf_path data/chr22.anc.maf \
    --chromosome 22 \
    --output_path data/seqDictPad_chr22.pkl
```

### 6. Genome-Wide Predictions (bedGraph)

```bash
python scripts/predict_genome.py \
    --model_path models/graphylo_cadddata \
    --alignment_pkl data/seqDictPad_chr22.pkl \
    --chromosome 22 --start 1000000 --end 2000000 \
    --output predictions_chr22.bed
```

### 7. Evaluate Alignment

```bash
python scripts/evaluate_alignment.py \
    --model_path models/graphylo_lstm_mutation \
    --alignment_pkl ../../conservation/data/seqDictPad_chr22.pkl \
    --s_species camFer1 --t_species mm10
```

### 8. Score Variants with Evo2 via NVIDIA API

```bash
export NVCF_RUN_KEY=your_api_key
python scripts/benchmarks/score_evo2_api.py --input variants.csv
```

## Configuration

All hyperparameters live in `configs/default.yaml`. CLI arguments override config values:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `cnn_gcn` | Architecture: `cnn_gcn`, `lstm_gcn`, `transformer_gcn`, `conv2d_gcn`, `bahdanau_gcn`, `evolstm` |
| `loss` | `focal` | Loss function: `focal`, `bce`, `cce` |
| `context` | 100 | Flanking bases in preprocessing |
| `context_flank` | 10 | Narrow window (21 bases total) |
| `focal_gamma` | 2.0 | Focal loss focusing parameter |
| `batch_size` | 64 | Training batch size |
| `patience` | 7 | Early stopping patience |

## Species Tree

The model operates on a 115-node phylogenetic graph covering 58 extant mammalian species (from human to armadillo) plus 57 inferred ancestral nodes. During training, human (`hg38`), chimpanzee (`panTro4`), gorilla (`gorGor3`), and two ancestral nodes (`_HP`, `_HPG`) are masked to prevent information leakage.

## Variant Data Source

All variants used for training and evaluation are sourced from **TOPMed** (Trans-Omics for Precision Medicine) via the UCSC Genome Browser `snp151` track. Variants are filtered to:

- **Single nucleotide polymorphisms** only (`class == 'single'`)
- **TOPMed-submitted** (submitters field contains `TOPMED`)
- **With allele frequency data** (`alleleFreqs` is not null)

The evaluation task is to distinguish common TOPMed variants (MAF >= 0.01) from rare variants (MAF < 0.01), providing a biologically meaningful assessment of variant functional impact prediction.

## License

MIT License — see [LICENSE](LICENSE) for details.
