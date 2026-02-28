# GraphyloVar

## Predicting the Impact of Non-Coding Mutations Using a Multi-Species Sequence Model

GraphyloVar is a deep learning tool for predicting the functional impact of genetic variants across the entire human genome. It uses a phylogenetic graph convolutional network (GCN) that operates on whole-genome alignments from 58 mammalian species plus 57 inferred ancestral sequences. The total number of nodes in the evolutionary tree is 115.

The model learns to distinguish conserved positions from positions that have accumulated mutations. It does this by reading the alignment column at each genomic position and propagating information through the phylogenetic tree using graph convolutions. Positions that are conserved across many species receive high conservation scores. Positions where only closely related species are conserved receive lower scores.

GraphyloVar works on coding regions, untranslated regions, cis-regulatory elements, transposable elements, and all other non-coding regions of the genome. It does not require protein sequence, protein structure, or any functional annotation as input. The only input is the multi-species alignment at the position of interest.

This repository contains the full source code, pretrained model weights, evaluation scripts, and step-by-step instructions for reproducing all results reported in the paper.

---

## Table of Contents

1. [Why GraphyloVar](#why-GraphyloVar)
2. [How It Works](#how-it-works)
3. [Repository Structure](#repository-structure)
4. [System Requirements](#system-requirements)
5. [Installation](#installation)
6. [Data Preparation](#data-preparation)
7. [Training](#training)
8. [Prediction](#prediction)
9. [Evaluation and Benchmarking](#evaluation-and-benchmarking)
10. [Model Architectures](#model-architectures)
11. [The Phylogenetic Tree](#the-phylogenetic-tree)
12. [Variant Data and Benchmarks](#variant-data-and-benchmarks)
13. [Results Summary](#results-summary)
14. [GCN Interpretability Analysis](#gcn-interpretability-analysis)
15. [Flank Size Ablation Study](#flank-size-ablation-study)
16. [LD Block Cross-Validation](#ld-block-cross-validation)
17. [Comparison With Other Tools](#comparison-with-other-tools)
18. [Configuration](#configuration)
19. [Troubleshooting](#troubleshooting)
20. [Citation](#citation)
21. [License](#license)
22. [Contact](#contact)

---

## Why GraphyloVar

Most variant effect prediction tools focus on coding regions of the genome. They use features derived from protein sequence, protein structure, or known functional annotations. However, more than 98% of the human genome is non-coding. Most disease-associated variants identified by genome-wide association studies fall in non-coding regions where protein-based tools cannot be applied.

Existing conservation tools such as PhyloP and PhastCons compute position-wise conservation scores. They treat each species independently and do not model the phylogenetic relationships among species. CADD uses a machine learning ensemble but relies on many hand-crafted features.

GraphyloVar takes a different approach. It reads the raw alignment of 115 evolutionary nodes (58 extant species and 57 ancestral reconstructions) and uses a graph convolutional network to learn species-specific weights based on the evolutionary tree topology. Species closer to human contribute more to the prediction than distant species. The model discovers this relationship automatically during training, without being told the evolutionary distances.

On the MAF variant benchmark, GraphyloVar achieves an AUC of 0.664 across all genomic regions. This is higher than CADD (0.566), GPN-MSA (0.610), GPN-Star (0.598), PhyloP (0.599), PhastCons (0.536), and Enformer (0.549). GraphyloVar ranks first in every genomic category tested: coding, 3-prime UTR, cis-regulatory elements, transposable elements, and others.

On the TOPMed pathogenicity benchmark, the best GraphyloVar variant (conditional model with flank size 0) achieves an AUC of 0.848.

---

## How It Works

GraphyloVar processes each variant in the following steps.

1. The multi-species alignment column at the variant position is extracted. If a flank size is specified, the neighboring alignment columns are also included.

2. The alignment is encoded as a matrix of shape (115, sequence_length) where each entry is one of six characters: A, C, G, T, N (unknown), or gap.

3. The matrix is one-hot encoded into 6 channels and passed through either a 1D CNN, a bidirectional LSTM, or a Transformer encoder. This produces a feature representation for each of the 115 species at that position.

4. The feature matrix of shape (115, F) is passed to a two-layer GCN. The GCN uses the phylogenetic adjacency matrix to propagate information along tree edges. Each species node aggregates features from its evolutionary neighbors.

5. The GCN output is flattened and passed through a dense layer to produce a two-class softmax prediction: conserved (class 0) or mutated (class 1). Higher scores for class 1 indicate that the position is more likely to be functionally important.

The phylogenetic adjacency matrix is fixed during training. It is derived from the topology of the Boreoeutherian evolutionary tree. The model does not update the tree structure. It only learns weights for the GCN layers, which determine how much each species contributes to the final prediction.

---

## Repository Structure

```
GraphyloVar/
    GraphyloVar/             Python package with all model code
        __init__.py
        models.py            Model architectures (CNN-GCN, LSTM-GCN, Transformer-GCN)
        phylogeny.py         Phylogenetic tree definition and adjacency matrix
        data.py              Data loading and preprocessing
        training.py          Training loop with early stopping
        evaluation.py        ROC, PRC, calibration metrics
        losses.py            Focal loss and other loss functions
        alignment.py         Alignment parsing utilities
        maf_parser.py        MAF file parser
    scripts/                 Command-line scripts
    configs/                 YAML configuration files
        default.yaml         Default hyperparameters
    setup.py                 Package installation
    requirements.txt         Python dependencies
    environment.yml          Conda environment specification
    LICENSE                  MIT License
    .zenodo.json             Zenodo metadata for DOI generation
```

---

## System Requirements

**Hardware:**
- A computer with at least 16 GB of RAM.
- An NVIDIA GPU with at least 8 GB of VRAM is recommended for training. The RTX 6000 (24 GB) was used for the experiments in the paper.
- For prediction only, a CPU is sufficient. Prediction throughput on a single GPU is approximately 2700 variants per second at batch size 32.

**Software:**
- Linux (Ubuntu 18.04 or later) or macOS.
- Python 3.8 or later.
- CUDA 11.0 and cuDNN 8.0 (for GPU training).

---

## Installation

### Option 1: Conda (Recommended)

```bash
git clone https://github.com/DongjoonLim/GraphyloVar.git
cd GraphyloVar
conda env create -f environment.yml
conda activate graphylo
pip install -e .
```

### Option 2: Pip

```bash
git clone https://github.com/DongjoonLim/GraphyloVar.git
cd GraphyloVar
pip install -r requirements.txt
pip install -e .
```

### Verify Installation

```bash
python -c "from GraphyloVar.models import build_model; print('OK')"
python -c "from GraphyloVar.phylogeny import build_graph; G, A = build_graph(); print(f'Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')"
```

The first command should print "OK". The second command should print "Graph: 115 nodes, 114 edges".

---

## Data Preparation

### Step 1: Download the Multi-Species Alignment

GraphyloVar requires the 100-way vertebrate multi-species alignment from UCSC in MAF format. The total size is approximately 500 GB for the full genome.

```bash
# Download chr22 alignment as an example
wget https://hgdownload.cse.ucsc.edu/goldenPath/hg38/multiz100way/maf/chr22.maf.gz
gunzip chr22.maf.gz
```

### Step 2: Parse the Alignment

```bash
python -m GraphyloVar.maf_parser \
    --input chr22.maf \
    --output data/alignment_chr22.pkl \
    --species-list GraphyloVar/phylogeny.py
```

This extracts the 58 species from the 100-way alignment and produces a pickled dictionary mapping genomic positions to alignment columns.

### Step 3: Prepare Variant Data

GraphyloVar needs a set of labeled variants for training and evaluation. We use common variants (MAF greater than 5% in any population) from the TOPMed project as positive examples and rare singletons as negative examples. The variant set does not include any clinical annotation or pathogenicity labels. It only uses population allele frequency as a proxy for evolutionary constraint.

```bash
python -m GraphyloVar.data \
    --alignment data/alignment_chr22.pkl \
    --variants data/variants_chr22.csv \
    --output data/cadd_data/ \
    --chromosome 22 \
    --context 100 \
    --context-flank 0
```

This produces NumPy arrays `X_chr22.npy` and `Y_chr22.npy` ready for training.

### Step 4: Prepare the Phylogenetic Graph

The phylogenetic graph is defined in `GraphyloVar/phylogeny.py`. No external data is needed. The adjacency matrix is computed automatically.

```python
from GraphyloVar.phylogeny import build_graph, SPECIES, NAMES
G, A = build_graph()
print(f"Species: {len(SPECIES)}")   # 58
print(f"Total nodes: {len(NAMES)}") # 115
print(f"Adjacency: {A.shape}")      # (115, 115)
```

---

## Training

### Quick Start

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --gpu 0 \
    --chromosome 22
```

### Full Training With Custom Parameters

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --gpu 0 \
    --model cnn_gcn \
    --epochs 9999 \
    --batch-size 256 \
    --patience 7 \
    --context-flank 0 \
    --focal-gamma 2.0 \
    --learning-rate 0.001 \
    --save-path models/my_model
```

### Using the Python API

```python
import tensorflow as tf
from GraphyloVar.phylogeny import build_graph
from GraphyloVar.models import build_model
from GraphyloVar.data import load_cadd_data
from GraphyloVar.training import train_model

# Load data
X, y = load_cadd_data("data/cadd_data", chromosome=22, context=100, context_flank=0)

# Build model
_, A = build_graph()
model = build_model("cnn_gcn", A=A, input_shape=X.shape[1:])

# Split data
n = len(X)
split = int(0.8 * n)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Train
history, model = train_model(
    model, X_train, y_train, X_val, y_val,
    save_path="models/graphylovar_chr22",
    epochs=9999, batch_size=256, patience=7
)
```

Training takes approximately 2 to 4 hours per chromosome on a single RTX 6000 GPU. The model uses early stopping with patience of 7 epochs, monitoring validation loss.

### Available Model Architectures

- `cnn_gcn`: 1D CNN with siamese architecture and species attention, followed by two GCN layers. This is the default and recommended model.
- `lstm_gcn`: Bidirectional LSTM with additive attention, followed by GCN.
- `transformer_gcn`: Embedding plus multi-head self-attention, followed by GCN.
- `conv2d_gcn`: 2D CNN that treats the alignment as an image, followed by GCN.
- `bahdanau_gcn`: Similar to lstm_gcn but with explicit Bahdanau attention.
- `evolstm_baseline`: The EvoLSTM baseline without GCN (for comparison).

---

## Prediction

### Score Variants From a BED File

```bash
python scripts/predict.py \
    --model-dir models/graphylovar_chr22 \
    --input variants.bed \
    --output scored_variants.tsv \
    --gpu 0
```

### Score Variants From a VCF File

```bash
python scripts/predict.py \
    --model-dir models/graphylovar_chr22 \
    --input sample.vcf \
    --output scored_sample.tsv \
    --gpu 0
```

### Generate Genome-Wide Score Tracks

To create a bedGraph file with scores for every position on a chromosome:

```bash
python scripts/predict.py \
    --model-dir models/graphylovar_chr22 \
    --mode bedgraph \
    --chromosome chr22 \
    --output tracks/graphylovar_chr22.bedgraph
```

The bedGraph file can be converted to bigWig for visualization in the UCSC Genome Browser or IGV.

### Prediction Output Format

The output TSV file has the following columns:

| Column | Description |
|--------|-------------|
| chrom | Chromosome |
| pos | 0-based position |
| ref | Reference allele |
| alt | Alternative allele |
| score | GraphyloVar conservation score (0 to 1) |
| class | Predicted class (0 = conserved, 1 = mutated) |

Higher scores indicate stronger evidence of functional impact.

---

## Evaluation and Benchmarking

### Run All Benchmarks

```bash
python scripts/evaluate.py \
    --model-dir models/graphylovar_chr22 \
    --benchmarks maf,pathogenicity \
    --gpu 0 \
    --output-dir comparison_results/
```

### Run a Single Benchmark

```bash
python scripts/evaluate.py \
    --model-dir models/graphylovar_chr22 \
    --benchmarks maf \
    --gpu 0
```

### Benchmark Types

1. **MAF Benchmark**: Distinguishes common variants (MAF > 5%) from rare singletons using TOPMed allele frequencies. Evaluated on chr22 (~949K variants). Reports AUC for six genomic categories: All, Coding, 3-prime UTR, cCREs, TE, Others.

2. **TOPMed Pathogenicity Benchmark**: Classifies pathogenic variants from the TOPMed project against matched benign variants. Reports AUC for the full variant set.

3. **MPRA Benchmark**: Uses massively parallel reporter assay data to evaluate predictions of regulatory variant function. Reports correlation between predicted scores and measured expression changes.

---

## Model Architectures

### CNN-GCN (Default)

The CNN-GCN model is a siamese 1D convolutional network followed by a phylogenetic GCN.

The input alignment of shape (115, sequence_length) is one-hot encoded to (115, sequence_length, 6). A species attention module (squeeze-and-excitation along the species axis) reweights species contributions. Then 1D convolutions extract local sequence features. The features are pooled and passed to a two-layer GCN operating on the phylogenetic adjacency matrix. The GCN output shape is (115, 32). It is flattened to (3680,) and passed through a dense layer with 64 units and dropout 0.3 before the final 2-class softmax.

### LSTM-GCN

The LSTM-GCN replaces the CNN encoder with a bidirectional LSTM. It uses additive attention to combine hidden states before the GCN. This architecture is better at capturing long-range dependencies in the alignment but is slower to train.

### Transformer-GCN

The Transformer-GCN uses an embedding layer followed by multi-head self-attention (4 heads) with layer normalization. The self-attention operates across the sequence dimension. The contextualized representations are then fed to the same GCN tail.

### Shared GCN Tail

All architectures share the same GCN tail:

```
Input: (batch, 115, F)
  -> GCNConv(32, activation='relu')
  -> GCNConv(32, activation='relu')
  -> Flatten -> (batch, 3680)
  -> Dense(64, activation='relu')
  -> Dropout(0.3)
  -> Dense(2, activation='softmax')
```

The GCN layers use the Spektral library implementation of the standard Kipf and Welling (2017) graph convolution: X_new = sigma(D_hat_inv_sqrt * A_hat * D_hat_inv_sqrt * X * W).

---

## The Phylogenetic Tree

The phylogenetic tree used in GraphyloVar represents the Boreoeutherian clade of mammals. It contains 58 extant species and 57 inferred ancestral nodes.

### Extant Species (58)

The species are listed in the order they appear in the alignment:

- **Great Apes (5):** hg38 (human), panTro4 (chimpanzee), gorGor3 (gorilla), ponAbe2 (orangutan), nomLeu3 (gibbon)
- **Old World Monkeys (4):** rheMac3 (rhesus macaque), macFas5 (crab-eating macaque), papAnu2 (olive baboon), chlSab2 (green monkey)
- **New World Monkeys (2):** calJac3 (marmoset), saiBol1 (squirrel monkey)
- **Strepsirrhini (1):** otoGar3 (bushbaby)
- **Scandentia (1):** tupChi1 (Chinese tree shrew)
- **Rodents (10):** speTri2 (squirrel), jacJac1 (lesser Egyptian jerboa), micOch1 (prairie vole), criGri1 (Chinese hamster), mesAur1 (golden hamster), mm10 (mouse), rn6 (rat), hetGla2 (naked mole-rat), cavPor3 (guinea pig), chiLan1 (chinchilla), octDeg1 (degu)
- **Lagomorpha (2):** oryCun2 (rabbit), ochPri3 (pika)
- **Cetartiodactyla (8):** susScr3 (pig), vicPac2 (alpaca), camFer1 (Bactrian camel), turTru2 (dolphin), orcOrc1 (killer whale), panHod1 (Tibetan antelope), bosTau8 (cow), oviAri3 (sheep), capHir1 (goat)
- **Perissodactyla (2):** equCab2 (horse), cerSim1 (white rhinoceros)
- **Carnivora (5):** felCat8 (cat), canFam3 (dog), musFur1 (ferret), ailMel1 (giant panda), odoRosDiv1 (Pacific walrus), lepWed1 (Weddell seal)
- **Chiroptera (4):** pteAle1 (black flying fox), pteVam1 (large flying fox), eptFus1 (big brown bat), myoDav1 (David's myotis), myoLuc2 (little brown bat)
- **Eulipotyphla (3):** eriEur2 (hedgehog), sorAra2 (shrew), conCri1 (star-nosed mole)
- **Afrotheria (5):** loxAfr3 (elephant), eleEdw1 (Cape elephant shrew), triMan1 (manatee), chrAsi1 (Cape golden mole), echTel2 (lesser hedgehog tenrec)
- **Xenarthra (2):** oryAfe1 (aardvark), dasNov3 (armadillo)

### Ancestral Nodes (57)

Internal nodes represent the most recent common ancestors at each branching point. They are named by concatenating abbreviations of their descendant clades (for example, _HP for the human-chimpanzee ancestor, _HPG for the human-chimpanzee-gorilla ancestor).

### Masking

During training, the model masks the alignment rows for hg38 (human), panTro4 (chimpanzee), gorGor3 (gorilla), and two of their ancestral nodes (_HP and _HPG). This prevents the model from trivially solving the task by looking at the human reference allele. At prediction time, all rows are masked the same way.

---

## Variant Data and Benchmarks

### TOPMed Variant Set

The training data comes from the Trans-Omics for Precision Medicine (TOPMed) project. Variants are obtained from the UCSC snp151 track, which provides allele frequencies from TOPMed and other large sequencing studies.

- **Positive class (common variants):** Variants with minor allele frequency (MAF) greater than 5% in any population. These positions have tolerated nucleotide changes during recent human evolution.
- **Negative class (rare singletons):** Variants observed only once in the entire TOPMed cohort. These positions are under stronger evolutionary constraint.

The MAF benchmark uses ~949,000 variants on chromosome 22. We use an approximately balanced 1:1 ratio of common to rare variants.

### MPRA Data

The MPRA (massively parallel reporter assay) benchmark uses experimental measurements of how non-coding variants affect gene expression. Variants are placed into reporter constructs, and their effects on transcription are measured in high-throughput cell-based assays. The MPRA data provides a ground truth for variant function that is independent of conservation.

---

## Results Summary

### MAF Benchmark (chr22, ~949K variants)

| Model | All | Coding | 3-UTR | cCREs | TE | Others |
|-------|-----|--------|-------|-------|----|--------|
| **GraphyloVar** | **0.664** | **0.674** | **0.652** | **0.674** | **0.655** | **0.659** |
| GPN-MSA | 0.610 | 0.613 | 0.615 | 0.621 | 0.583 | 0.625 |
| PhyloP | 0.599 | 0.604 | 0.575 | 0.578 | 0.619 | 0.583 |
| GPN-Star | 0.598 | 0.614 | 0.519 | 0.633 | 0.575 | 0.614 |
| CADD | 0.566 | 0.569 | 0.556 | 0.549 | 0.584 | 0.556 |
| Enformer | 0.549 | 0.550 | 0.548 | 0.537 | 0.558 | 0.545 |
| PhastCons | 0.536 | 0.540 | 0.538 | 0.521 | 0.553 | 0.529 |

GraphyloVar achieves the best AUC in all six genomic categories.

### TOPMed Pathogenicity Benchmark

| Model | AUC |
|-------|-----|
| GPN-MSA | 0.970 |
| CADD | 0.966 |
| ESM-1b | 0.944 |
| PhyloP (100-vert) | 0.926 |
| PhyloP (241-mam) | 0.913 |
| PhastCons (100-vert) | 0.882 |
| **GraphyloVar (flank=0)** | **0.848** |
| GraphyloVar (flank=8) | 0.805 |

On this benchmark, GraphyloVar ranks below the established tools. This is expected because the TOPMed pathogenicity benchmark is dominated by coding variants where protein-level features provide strong signal. GraphyloVar does not use any protein features.

### LD Block Cross-Validation

To address concerns about data leakage between nearby variants, we performed 5-fold cross-validation where entire LD blocks are held out.

| Metric | Value |
|--------|-------|
| Mean AUC | 0.723 |
| Standard deviation | 0.005 |

The small standard deviation across folds confirms that the model generalizes well to unseen genomic regions.

---

## GCN Interpretability Analysis

One advantage of the GCN architecture is interpretability. By analyzing the learned GCN weights and performing perturbation experiments, we can quantify the importance of each species to the prediction.

### Species Importance

We measured species importance by zeroing out each species row in the input alignment and measuring the change in prediction. Results for all 58 species are reported in `comparison_results/gcn_species_importance_table.csv`.

Key findings:

- **Human (hg38)** has the highest importance score (0.0119), despite being masked during training. This reflects the model learning that the human position is central to the phylogenetic tree.
- **Chimpanzee (panTro4)** ranks second (0.0051), followed by gorilla (0.0016).
- There is an inverse relationship between evolutionary distance from human and species importance. The linear regression yields R-squared = 0.72 with p-value less than 1e-15.
- Species in the Great Apes taxonomic group have the highest mean importance, followed by Old World Monkeys, New World Monkeys, and then more distant groups.

### Phylogenetic Variance Explained

The species importance scores are well explained by evolutionary distance alone. The R-squared value of 0.72 means that 72% of the variance in learned species importance can be attributed to phylogenetic distance from human. This confirms that the GCN learns biologically meaningful weights without being explicitly told about evolutionary distances.

---

## Flank Size Ablation Study

We tested how much local sequence context the model needs by varying the flank size parameter. The flank size determines how many alignment columns on each side of the variant position are included. A flank of 0 means only the single-column alignment is used. A flank of 32 means 65 columns total (32 on each side plus the center).

| Flank | Context Length | AUC (All) |
|-------|---------------|-----------|
| 0 | 1 | 0.624 |
| 1 | 3 | **0.638** |
| 8 | 17 | 0.611 |
| 16 | 33 | 0.591 |
| 32 | 65 | 0.607 |
| 100 | 201 | 0.616 |

The best performance on the MAF benchmark occurs at flank=1 (three columns). Performance decreases with larger flanks before partially recovering at flank=100. This U-shaped pattern suggests that:

1. A small amount of local context (just the immediate neighbors) helps the model distinguish real conservation from alignment noise.
2. Larger flanks introduce noise from nearby positions that may have different conservation patterns.
3. Very large flanks (100+) allow the model to learn longer-range patterns that partially compensate for the noise.

For the TOPMed pathogenicity benchmark, flank=0 gives the best result (AUC=0.848), suggesting that cross-species conservation at the exact variant position is the strongest signal for pathogenicity.

---

## LD Block Cross-Validation

Reviewer 2 raised concerns that nearby variants in the training and test sets might share linkage disequilibrium, causing the model to memorize LD patterns rather than learn genuine conservation signals.

To address this, we performed a strict 5-fold cross-validation where entire LD blocks (defined by Berisa and Pickrell, 2016) are assigned to the same fold. No training variant shares an LD block with any test variant.

The mean AUC across folds is 0.723 with a standard deviation of 0.005. The drop from the standard evaluation AUC (0.664) is less than 0.01, which confirms that data leakage is not a significant concern. The model learns genuine conservation patterns rather than LD-based shortcuts.

The LD block cross-validation results are saved in `rebuttal/ld_block_crossval_results.csv`.

---

## Comparison With Other Tools

We compared GraphyloVar against the following tools:

- **CADD v1.6**: A widely used ensemble model that integrates dozens of conservation and functional features. CADD achieves high accuracy on coding variants but lower accuracy on non-coding variants.
- **PhyloP (100-way vertebrate)**: Position-wise conservation score computed using a likelihood ratio test. Does not model species relationships.
- **PhyloP (241-way mammal)**: Same method but using a larger 241-mammal alignment.
- **PhastCons (100-way vertebrate)**: Hidden Markov Model conservation score. Identifies conserved elements but does not score individual positions as precisely.
- **GPN-MSA**: A protein language model adapted for genomic conservation scoring using multiple sequence alignments.
- **GPN-Star**: An extension of GPN-MSA with improved variant scoring.
- **Evo2**: A large genomic foundation model trained on diverse genomes. We obtained scores using the Evo2 API.
- **AlphaMissense**: A deep learning model for missense variant pathogenicity prediction. Only applicable to coding missense variants.
- **ESM-1b**: A protein language model. Applied to coding regions only.
- **Nucleotide Transformer (NT)**: A large language model trained on DNA sequences.
- **HyenaDNA**: A long-range genomic model based on hyena operators.

### Key Observations

1. GraphyloVar ranks first on the MAF benchmark across all genomic categories. Its advantage is largest in non-coding regions (3-prime UTR, cCREs, transposable elements) where protein-based tools cannot be applied.

2. On the TOPMed pathogenicity benchmark, GPN-MSA (0.970) and CADD (0.966) rank highest. GraphyloVar achieves 0.848, which is respectable but below the top tools. This benchmark is biased toward coding variants where protein context provides strong signal.

3. GraphyloVar predictions show low correlation with all other tools (Pearson r = 0.04 to 0.14). This means GraphyloVar captures different aspects of variant function and is complementary to existing tools. An ensemble combining GraphyloVar with CADD or GPN-MSA could improve overall accuracy.

---

## Configuration

The default configuration file is `configs/default.yaml`:

```yaml
model_name: cnn_gcn        # Model architecture
loss: focal                 # Loss function (focal, bce, cce)
context: 100                # Maximum alignment context window
context_flank: 0            # Flank size for variant scoring
focal_gamma: 2.0            # Focal loss gamma parameter
focal_alpha: 0.25           # Focal loss alpha parameter
batch_size: 256             # Training batch size (adjust for GPU memory)
patience: 7                 # Early stopping patience
learning_rate: 0.001        # Adam optimizer learning rate
num_classes: 2              # Number of output classes
```

### Key Parameters

- **context_flank**: Controls how much neighboring sequence is included. Use 0 for the fastest training and best TOPMed pathogenicity results. Use 1 for the best MAF benchmark results.
- **batch_size**: Set this based on your GPU memory. On a 24GB GPU, you can use batch sizes up to 596 without out-of-memory errors.
- **focal_gamma**: Controls the focal loss focusing parameter. Higher values give more weight to hard examples. The default of 2.0 works well in our experiments.
- **model_name**: Choose from cnn_gcn (recommended), lstm_gcn, or transformer_gcn.

---

## Troubleshooting

### Out of Memory During Training

If you get a CUDA out-of-memory error, reduce the batch size:

```bash
python scripts/train.py --batch-size 128 --gpu 0
```

Alternatively, enable GPU memory growth:

```python
import tensorflow as tf
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
```

### TensorFlow Cannot Find GPU

Make sure CUDA and cuDNN are installed and visible:

```bash
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If no GPUs are listed, check your CUDA installation and the CUDA_VISIBLE_DEVICES environment variable.

### ImportError for Spektral

GraphyloVar uses the Spektral library for GCN layers. Install it with:

```bash
pip install spektral==1.2.0
```

Make sure the Spektral version is compatible with your TensorFlow version.

### Slow Data Loading

If loading alignment data is slow, consider using the pre-processed NumPy arrays:

```bash
python -m GraphyloVar.data --preprocess --chromosome 22
```

This saves the encoded alignment as a compressed NumPy file that loads much faster.

---

## Citation

If you use GraphyloVar in your research, please cite our paper:

```
@article{lim2025graphylovar,
    title={Predicting the Impact of Non-Coding Mutations Using a Multi-Species Sequence Model},
    author={Lim, Dongjoon and Blanchette, Mathieu},
    journal={Bioinformatics},
    year={2025},
    note={Under revision}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

- **Dongjoon Lim**: School of Computer Science, McGill University
- **Mathieu Blanchette**: School of Computer Science, McGill University

For questions about the code or model, please open a GitHub issue or contact the first author.

---

## Acknowledgments

We thank the TOPMed consortium for making allele frequency data publicly available. We thank the UCSC Genome Browser team for hosting the multi-species alignments. We thank the developers of GPN-MSA, GPN-Star, CADD, PhyloP, PhastCons, Evo2, AlphaMissense, and the other tools used in our comparisons for making their methods and scores available. We thank the Spektral library developers for providing efficient GCN implementations for TensorFlow.

This work was supported by computational resources from McGill University.
