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
│   ├── preprocess.py         # Build .npy arrays from CADD VCFs + alignment pickles
│   ├── train.py              # Train any model variant via CLI
│   ├── predict.py            # Run inference on held-out chromosomes
│   ├── predict_genome.py     # Generate genome-wide bedGraph prediction tracks
│   ├── evaluate_alignment.py # Benchmark model-guided vs classical alignment
│   ├── evaluate_clinvar.py   # Evaluate model on ClinVar pathogenic/benign variants
│   └── benchmarks/           # External benchmark comparison scripts
│       ├── visualize_all_comparisons.py  # Comprehensive multi-benchmark visualization
│       ├── benchmark_gnomad_balanced.py  # gnomAD common vs rare benchmark
│       ├── add_gpnstar_comparison.py     # GPN-Star integration into MAF comparison
│       └── score_evo2_api.py             # Score variants via NVIDIA Evo2 API
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

GraphyloVar has been benchmarked against a comprehensive set of variant effect prediction tools across multiple evaluation settings. All results are reproducible using the scripts in `scripts/benchmarks/`.

### ClinVar Pathogenic vs Benign (songlab/clinvar_vs_benign)

Evaluation on 50,164 ClinVar variants (22,254 Pathogenic / 27,910 Benign) from the [songlab/clinvar_vs_benign](https://huggingface.co/datasets/songlab/clinvar_vs_benign) benchmark:

| Model | AUC | Category |
|-------|-----|----------|
| GPN-Star (V100, AF adj.) | 0.963 | DNA language model (alignment-aware) |
| AlphaMissense | 0.955 | Protein structure (dbNSFP) |
| GPN-Star (V100) | 0.925 | DNA language model (alignment-aware) |
| ESM-1b | 0.914 | Protein language model (dbNSFP) |
| CADD | 0.909 | Ensemble (dbNSFP) |
| GPN-Star (M447) | 0.907 | DNA language model (alignment-aware) |
| GPN-MSA | 0.880 | DNA language model (alignment-aware) |
| Evo2 (40B) | 0.865 | DNA language model |
| GPN-Star (P243) | 0.857 | DNA language model (alignment-aware) |
| Evo2 (7B) | 0.856 | DNA language model |
| PhyloP (V100) | 0.851 | Conservation |
| PhyloP (M447) | 0.825 | Conservation |
| PhastCons (V100) | 0.774 | Conservation |
| NT (2.5B) | 0.603 | DNA language model |
| Roulette | 0.582 | Mutation rate |

### GraphyloVar ClinVar Analysis (35,613 variants)

Evaluation on our curated ClinVar dataset using TOPMed allele frequency labels:

| Model | AUC | Notes |
|-------|-----|-------|
| GPN-MSA | 0.970 | Alignment-aware DNA LM |
| CADD | 0.966 | Ensemble integrator |
| ESM-1b | 0.944 | Protein LM |
| PhyloP (V100) | 0.926 | Conservation |
| PhastCons (V100) | 0.883 | Conservation |
| **GraphyloVar (Transformer, flank=0)** | **0.820** | **Ours** |
| **GraphyloVar (Conditional, MLP)** | **0.804** | **Ours** |
| **GraphyloVar (Transformer, flank=32)** | **0.789** | **Ours** |
| **GraphyloVar (MLP)** | **0.797** | **Ours** |
| **GraphyloVar (EvoLSTM)** | **0.694** | **Ours (no graph)** |
| NT | 0.606 | DNA language model |
| HyenaDNA | 0.502 | DNA language model |

### gnomAD Balanced Common vs Rare (chr2, ~1M variants)

Evaluation on the [songlab/gnomad_balanced](https://huggingface.co/datasets/songlab/gnomad_balanced) benchmark (distinguishing common from rare gnomAD variants):

| Model | AUC |
|-------|-----|
| GPN-Star (M447) | 0.678 |
| GPN-Star (P243) | 0.669 |
| GPN-MSA | 0.665 |
| GPN-Star (V100) | 0.635 |
| PhyloP (P243) | 0.622 |
| PhyloP (V100) | 0.616 |
| Roulette | 0.578 |
| CADD | 0.570 |

## Comparison Tools Referenced

The following external tools are compared in our benchmarks. Models marked **(dbNSFP)** are included in the [dbNSFP database](http://dbnsfp.org/) (v5.3.1, 36 prediction algorithms):

| Tool | Type | Reference |
|------|------|-----------|
| **Evo2** (7B, 40B) | DNA foundation model (7B/40B params) | [Genome biology with 11,000 genomes](https://arcinstitute.org/news/evo2) |
| **GPN-Star** (V100/M447/P243/P36) | Alignment-aware DNA language model | [songlab/gpn-msa](https://huggingface.co/songlab) |
| **GPN-MSA** | Alignment-conditioned DNA language model | [Benegas et al. 2023](https://www.biorxiv.org/content/10.1101/2023.10.10.561776) |
| **CADD** | Ensemble deleteriousness score **(dbNSFP)** | [Kircher et al. 2014](https://doi.org/10.1038/ng.2892) |
| **AlphaMissense** | Protein structure-based **(dbNSFP)** | [Cheng et al. 2023](https://doi.org/10.1126/science.adg7492) |
| **ESM-1b** | Protein language model **(dbNSFP)** | [Rives et al. 2021](https://doi.org/10.1073/pnas.2016239118) |
| **PhyloP** | Conservation (phyloP scores) **(dbNSFP)** | [Pollard et al. 2010](https://doi.org/10.1101/gr.097857.109) |
| **PhastCons** | Conservation (phastCons scores) **(dbNSFP)** | [Siepel et al. 2005](https://doi.org/10.1101/gr.3715005) |
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

### 7. Evaluate on ClinVar

```bash
python scripts/evaluate_clinvar.py \
    --model_path models/graphylo_cadddata_focalloss \
    --x_path data/X_clinvar.npy \
    --y_path data/y_clinvar.npy
```

### 8. Evaluate Alignment

```bash
python scripts/evaluate_alignment.py \
    --model_path models/graphylo_lstm_mutation \
    --alignment_pkl ../../conservation/data/seqDictPad_chr22.pkl \
    --s_species camFer1 --t_species mm10
```

### 9. Run Comprehensive Benchmarks

```bash
# Full visualization: ClinVar + gnomAD + GraphyloVar comparisons
python scripts/benchmarks/visualize_all_comparisons.py

# gnomAD balanced benchmark only
python scripts/benchmarks/benchmark_gnomad_balanced.py --download --chroms 2

# Add GPN-Star to MAF comparison analysis
python scripts/benchmarks/add_gpnstar_comparison.py

# Score variants with Evo2 via NVIDIA API
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

## External Benchmark Datasets

Our benchmark comparisons use standardized datasets from the [Song Lab](https://huggingface.co/songlab):

- **[songlab/clinvar_vs_benign](https://huggingface.co/datasets/songlab/clinvar_vs_benign)**: 50,164 ClinVar variants (Pathogenic vs Benign) with precomputed scores for 19 methods including Evo2, GPN-Star, AlphaMissense, CADD, ESM-1b, and more.
- **[songlab/gnomad_balanced](https://huggingface.co/datasets/songlab/gnomad_balanced)**: ~12M gnomAD variants (balanced common vs rare) with precomputed scores for GPN-Star, GPN-MSA, CADD, PhyloP, PhastCons, and Roulette.

These datasets provide a fair, reproducible comparison framework where all methods are evaluated on the exact same variant set.

## License

MIT License — see [LICENSE](LICENSE) for details.
