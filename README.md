# GraphyloVar

**GraphyloVar: Predicting the impact of non-coding variants using a multi-species sequence model**

GraphyloVar is a deep learning model that predicts whether a DNA mutation is likely to be harmful, specifically for mutations outside of protein-coding genes. It does this by looking at the same short DNA window simultaneously across 58 mammalian species and using the evolutionary relationships between those species to help interpret the variant.

---

## Table of Contents

- [Background: What problem does this solve?](#background-what-problem-does-this-solve)
- [Key concepts you need to know](#key-concepts-you-need-to-know)
- [How GraphyloVar works (overview)](#how-graphylovar-works-overview)
- [Key results](#key-results)
- [Comparison with other tools](#comparison-with-other-tools)
- [Repository layout](#repository-layout)
- [System requirements](#system-requirements)
- [Installation](#installation)
- [Data sources and download](#data-sources-and-download)
- [Data preprocessing](#data-preprocessing)
- [Training the model from scratch](#training-the-model-from-scratch)
- [Monitoring training progress](#monitoring-training-progress)
- [Scoring variants with a trained model](#scoring-variants-with-a-trained-model)
- [Evaluating performance](#evaluating-performance)
- [Fine-tuning on MPRA data](#fine-tuning-on-mpra-data)
- [Ensembling with CADD](#ensembling-with-cadd)
- [Interpretability analysis](#interpretability-analysis)
- [Model architecture (detailed)](#model-architecture-detailed)
- [Understanding the loss functions](#understanding-the-loss-functions)
- [The phylogenetic tree and GCN adjacency matrix](#the-phylogenetic-tree-and-gcn-adjacency-matrix)
- [The 58 species used](#the-58-species-used)
- [Context window ablation study](#context-window-ablation-study)
- [Troubleshooting](#troubleshooting)
- [Frequently asked questions](#frequently-asked-questions)
- [Data availability](#data-availability)
- [Citation](#citation)
- [License](#license)

---

## Background: What problem does this solve?

### The scale of the problem

Every human genome contains roughly 4 to 5 million positions that differ from the reference genome. These differences are called single nucleotide variants (SNVs) or single nucleotide polymorphisms (SNPs). When a rare variant is found in a patient with a disease, one of the first questions researchers ask is: could this variant be causing the disease, or is it just a harmless difference?

The challenge is that the vast majority of these 4 to 5 million variants are completely harmless. A very small fraction, perhaps a few hundred to a few thousand per genome, may have functional consequences. Finding those needles in the haystack requires computational prioritization.

### Why non-coding variants are so hard

About 98% of the human genome does not code for proteins. Non-coding regions include:

- **Promoters**: Short DNA sequences just upstream of genes that control when and how much the gene is expressed.
- **Enhancers**: Regulatory elements that can be thousands of base pairs away from the genes they control, and that boost gene expression in specific cell types.
- **Insulators**: Elements that prevent enhancers from activating the wrong genes.
- **Non-coding RNA genes**: Genes that produce functional RNA molecules rather than proteins (e.g., microRNAs, lncRNAs).
- **Splice sites**: Short sequences at the boundaries of exons and introns that tell the cell's machinery where to splice.
- **Untranslated regions (UTRs)**: Portions of mRNA that are not translated but control mRNA stability and translation efficiency.

Mutations in any of these elements can cause disease just as surely as coding mutations. But predicting their effects is much harder because we do not have a simple code (like the genetic code) to translate the sequence into function.

### How existing tools approach the problem

**PhyloP and PhastCons** are conservation-based tools. They compute a score for each genomic position based on how conserved that position is across a multi-species alignment. If a position has been the same base in every mammal for 100 million years, it is probably important. These tools are simple and fast but they look at each position in isolation and do not consider local sequence context.

**CADD (Combined Annotation Dependent Depletion)** trains a support vector machine on features extracted from many annotation sources, including conservation, regulatory annotations, and sequence features. It is powerful but does not directly model the multi-species DNA sequences.

**Enformer** is a large deep learning model trained to predict gene expression and epigenetic signals from DNA sequence in a single species (human). It captures local sequence features extremely well but does not use multi-species alignments.

**GPN-MSA** uses a masked language model trained on multi-species alignments, similar to BERT for DNA. It models cross-species sequence patterns but does not incorporate the phylogenetic tree structure explicitly.

### What GraphyloVar adds

GraphyloVar is designed to do all of the following simultaneously:

1. Process local sequence context (65 bp window) using Transformer encoders that capture motif patterns and local dependencies.
2. Use sequences from 58 species simultaneously, so the model sees the full evolutionary picture at each position.
3. Explicitly incorporate the phylogenetic tree structure via a Graph Convolutional Network, so the model knows which species are closely related and which are distant.
4. Learn from a massive training signal: population-level allele frequencies from the TOPMed cohort, covering approximately 149 million SNVs across the human genome.

The result is a model that captures complementary information to all of the tools listed above. When combined with CADD in a simple z-score ensemble, the AUROC improves by +0.020 over GraphyloVar alone (p < 10^-15), which is a statistically significant and practically meaningful improvement.

---

## Key concepts you need to know

If you are new to genomics or deep learning, this section explains the essential vocabulary used throughout this documentation.

### DNA and the genome

DNA is a molecule made of four building blocks called nucleotides, abbreviated A (adenine), C (cytosine), G (guanine), and T (thymine). The human genome is approximately 3.2 billion base pairs long. Each position in the genome has a specific nucleotide in the reference genome (hg38 for human). A variant is a position where some individuals have a different nucleotide than the reference.

### Single nucleotide variants (SNVs) and SNPs

An SNV is a single-base difference between an individual's genome and the reference. An SNP (single nucleotide polymorphism) is an SNV that is common enough in the population that we consider it a normal variant (typically with a minor allele frequency above 1%). In practice, the terms are often used interchangeably, and GraphyloVar uses "SNP" loosely to mean any single-base variant.

### Allele frequency and minor allele frequency (MAF)

An allele is one of the possible nucleotides at a given position. The allele frequency is how common that allele is in the population. For a biallelic SNV (where position X can be either A or G in the population), the minor allele is the less common one. If 95% of people have A and 5% have G, the minor allele frequency is 0.05 (5%).

GraphyloVar uses allele frequency as a proxy for variant impact during pre-training. Very rare variants (MAF below 0.01%) are more likely to be under purifying selection, meaning the variant is slightly harmful and evolution has been removing it from the population. Common variants (MAF above 1%) are much more likely to be neutral.

### Multiple sequence alignment (MSA)

A multiple sequence alignment lines up DNA sequences from multiple species at orthologous (evolutionarily corresponding) positions. For example, in the UCSC 100-way alignment, position 16,050,075 on human chromosome 22 is aligned to the corresponding position in chimpanzee, gorilla, mouse, cow, and so on. Conserved columns (where most or all species have the same nucleotide) indicate functionally important positions.

### Phylogenetic tree

A phylogenetic tree is a diagram showing the evolutionary relationships between species. Branch lengths represent evolutionary distance (the expected number of substitutions per site since the two lineages diverged). Human and chimpanzee are very closely related (branch length approximately 0.013 substitutions per site). Human and mouse diverged much longer ago (branch length approximately 0.18). The phylogenetic tree is crucial for GraphyloVar because it tells the GCN which species to cluster together when aggregating information.

### Transformer encoder

The Transformer architecture was introduced for natural language processing and has been widely adapted for biology. A Transformer encoder takes a sequence of tokens (in our case, nucleotides at each position) and uses self-attention to let each position look at all other positions in the window. This allows the model to capture long-range dependencies within the 65 bp window, such as the distance between two transcription factor binding sites.

### Graph Convolutional Network (GCN)

A GCN extends convolutional neural networks to work on graph-structured data. In a GCN, each node (in our case, each species node in the phylogenetic tree) updates its feature vector by aggregating information from its neighbors. After two layers of message passing, each node's representation incorporates information from nodes up to two edges away. This is how the model propagates evolutionary information along the phylogenetic tree.

### AUROC

AUROC stands for Area Under the Receiver Operating Characteristic curve. It measures how well a binary classifier separates two classes. An AUROC of 0.5 is no better than random; 1.0 is perfect. For variant impact prediction, the task is to separate common variants (treated as benign, positive class) from rare variants (treated as potentially harmful, negative class). An AUROC of 0.6246 means GraphyloVar correctly ranks a randomly drawn benign variant above a randomly drawn potentially harmful variant 62.46% of the time on a dataset of 149 million variants. This is competitive with the best existing tools.

### Spearman correlation

Spearman correlation measures the monotonic relationship between two variables. Here it measures how well the model's score tracks the actual minor allele frequency across held-out variants. A higher Spearman correlation means the model's scores more faithfully reflect the observed population frequencies.

---

## How GraphyloVar works (overview)

Here is a step-by-step description of what happens when GraphyloVar scores a variant.

**Step 1: Sequence extraction**

For a variant at position p on chromosome c, the pipeline extracts a 65 bp window centered on p from the UCSC 100-way multiple sequence alignment. This window is extracted for all 58 placental mammalian species in the alignment. If a species has a gap or insertion/deletion at that region, the gap character is used. The result is a matrix of shape 58 x 65 x 5 (species by positions by nucleotide states).

**Step 2: Reverse complement augmentation**

Each species' 65 bp sequence is processed twice: once as-is (the forward strand) and once as the reverse complement. This is important because regulatory elements can be on either strand, and a model that only sees the forward strand would miss enhancer motifs that happen to be encoded on the reverse strand.

**Step 3: Per-species Transformer encoding**

Each of the 58 species' sequences passes independently through the same Transformer encoder. The encoder has a dense embedding layer (projecting from 5-dimensional one-hot to 32 dimensions), sinusoidal positional encodings, and two Transformer encoder layers. The encoder produces a 32-dimensional feature vector for every position in the 65 bp window. Only the feature vector at the center position (position 32, the actual variant site) is kept. This gives 32 dimensions per strand, or 64 dimensions per species after concatenating forward and reverse complement.

**Step 4: Squeeze-and-Excitation gating**

A Squeeze-and-Excitation (SE) module computes a single attention weight for each of the 58 species based on its 64-dimensional feature vector. This is a learned, data-driven way of asking "how informative is this species' sequence for this particular position?" On average, human gets the highest weight (approximately 1.00), and chimpanzee gets a very low weight (approximately 0.011) because chimp's near-identical sequence to human provides essentially no new information.

**Step 5: Graph Convolutional Network**

The 64-dimensional feature vectors for all 58 extant species are placed as nodes in the mammalian phylogenetic tree. The tree has 115 nodes total (58 extant + 57 reconstructed ancestral nodes). The 57 ancestral nodes are initialized to zero vectors since we do not directly observe ancestral sequences. Two GCN layers propagate information along the tree edges. After the GCN, each node has a 32-dimensional representation. The 115 x 32 matrix is flattened to 3680 dimensions.

**Step 6: Prediction**

The 3680-dimensional vector is passed through a shared fully connected layer (128 units, ReLU) and then through two separate prediction heads:

- The nucleotide frequency head predicts the distribution of alleles at this position in the human population (softmax over 5 states: A, C, G, T, gap).
- The SNP probability head predicts whether this position is a polymorphic SNP (sigmoid, output between 0 and 1).

**Step 7: Variant impact score**

The SNP probability score is used as the variant impact score. A score close to 0 means the model predicts this position is not polymorphic, which implies it is functionally constrained and that a mutation there is more likely to be harmful. A score close to 1 means the model predicts the position is tolerant of variation, implying the variant is likely benign.

---

## Key results

### Zero-shot performance on TOPMed held-out variants

GraphyloVar was pre-trained on chromosomes 1 to 10 (training) and 11 to 12 (validation) of the TOPMed dataset. All results below are on the held-out test set of chromosomes 13 to 22, containing approximately 149 million SNVs, which were never seen during training or hyperparameter selection.

**Overall AUROC (all variant types):** 0.6246

**Region-stratified AUROC (flank=32, full 149M-variant holdout):**

| Genomic region | AUROC |
|----------------|-------|
| All variants | 0.625 |
| Coding exons | 0.616 |
| 3-prime UTR | 0.621 |
| cCREs (regulatory elements) | 0.617 |
| Transposable elements | 0.626 |

**Spearman correlation with minor allele frequency:**

| Method | Spearman rho |
|--------|-------------|
| GraphyloVar | 0.164 |
| GPN-MSA | 0.143 |
| Enformer | 0.131 |
| PhyloP | 0.099 |
| CADD | 0.081 |
| PhastCons | 0.058 |

### Ensemble with CADD

GraphyloVar's scores are not redundant with CADD. Combining the two with a simple z-score ensemble yields:

- GraphyloVar alone: AUROC 0.6246
- CADD alone (sign-aligned to common vs. rare convention): AUROC 0.5546
- GraphyloVar + CADD z-score ensemble: AUROC 0.6442

This is an improvement of +0.020 over GraphyloVar alone. A DeLong test on 500,000 held-out variants gives z = 12.24 and p < 10^-15. Bootstrap 95% confidence intervals (B=1000, n=200,000) are 0.6496 [0.6395, 0.6587] for the ensemble and 0.6271 [0.6174, 0.6368] for GraphyloVar alone, with non-overlapping intervals confirming the improvement is statistically significant.

### Fine-tuned performance on MPRA benchmarks

Massively Parallel Reporter Assays (MPRAs) are experiments that measure the regulatory activity of hundreds of thousands of variant sequences simultaneously. GraphyloVar, when fine-tuned on each MPRA dataset, achieves the highest AUROC on all 13 MPRA benchmark datasets tested, outperforming CADD, PhyloP, PhastCons, GPN-MSA, and Enformer across all 13 datasets.

---

## Comparison with other tools

| Tool | Architecture | Species | Tree-aware | Trained on |
|------|-------------|---------|------------|------------|
| PhyloP | Phylogenetic HMM | 100 vertebrates | Yes (implicitly) | Evolutionary model |
| PhastCons | Phylogenetic HMM | 100 vertebrates | Yes (implicitly) | Evolutionary model |
| CADD | SVM on features | Single (human) | No | Simulated variants vs. observed |
| Enformer | CNN + Transformer | Single (human) | No | Regulatory genomics tracks |
| GPN-MSA | Masked language model | Multiple | No (flat MSA) | MSA sequences |
| GraphyloVar | Transformer + GCN | 58 placental mammals | Yes (explicitly) | TOPMed allele frequencies |

The key distinction of GraphyloVar is that it is the only method that explicitly feeds the phylogenetic tree structure into the neural network (via the GCN) while also using a learned sequence encoder (Transformer) rather than a fixed alignment score. This combination captures both local sequence context and deep evolutionary signal simultaneously.

---

## Repository layout

The repository is organized as follows:

```
GraphyloVar/
|
+-- graphylovar/                    The core Python package
|   +-- __init__.py
|   +-- models.py                   All model architectures
|   |                               (build_multitask_hybrid_v3,
|   |                                build_multitask_hybrid_v4, etc.)
|   +-- phylogeny.py                Phylogenetic tree parsing,
|   |                               GCN adjacency matrix construction,
|   |                               NEWICK tree reading
|   +-- training.py                 Training loop, loss functions,
|   |                               callback definitions
|   +-- topmed.py                   TOPMed data loading and streaming
|   +-- model_io.py                 Checkpoint loading and saving utilities
|
+-- scripts/                        Executable Python scripts
|   +-- preprocess_topmed_full_compact.py
|   |                               Converts raw UCSC MAF + TOPMed VCF
|   |                               to compact per-chromosome .npz format
|   +-- train_topmed_full_streaming.py
|   |                               Main training script with streaming
|   |                               data loading, early stopping, logging
|   +-- predict.py                  Score variants on a single chromosome
|   +-- predict_genome.py           Score variants across all chromosomes
|   +-- additive_species_importance.py
|   |                               Additive per-species interpretability
|   |                               analysis (each species alone vs gap)
|   +-- per_species_all58_perturbation.py
|   |                               LOO perturbation analysis
|   |                               (mask one species, keep 57 others)
|   +-- recompute_region_flank_auc_from_ucsc.py
|   |                               Compute region-stratified AUROC
|   |                               using UCSC annotation tracks
|   +-- bootstrap_ensemble_auc_ci.py
|   |                               Bootstrap confidence intervals
|   |                               for AUROC estimates
|   +-- delong_pairwise_significance.py
|   |                               DeLong test for comparing AUROCs
|   +-- compute_raw_ensemble_auc.py
|   |                               Compute AUROC for z-score ensemble
|   +-- extract_se_attention_scores.py
|   |                               Extract SE gate weights per species
|   +-- evaluate_alignment.py       Evaluate alignment quality metrics
|   +-- train_ablation.py           Train ablation variants of model
|   +-- plot_*.py                   Various figure generation scripts
|
+-- configs/
|   +-- default.yaml                Default hyperparameter configuration
|
+-- latex/
|   +-- graphylovar_submission/     Final submission files
|   |   +-- oup-authoring-template.tex
|   |   |                           Main manuscript
|   |   +-- figures/                All figures used in the manuscript
|   |   +-- reference_curated.bib   Bibliography
|   +-- auc_macros.tex              AUC value macros for manuscript
|   +-- gcn_shape_paragraph.tex     GCN interpretability section fragment
|   +-- phact_paragraph.tex         PHACT comparison discussion fragment
|
+-- figures/                        Generated analysis figures and CSVs
+-- requirements.txt                Python package requirements
+-- environment.yml                 Conda environment specification
+-- setup.py                        Package installation configuration
```

---

## System requirements

### Minimum requirements (inference only)

- Python 3.9 or higher
- 16 GB RAM
- 8 GB GPU VRAM (or CPU-only, but much slower)
- 100 GB disk space (for a single chromosome's compact data)

### Recommended requirements (full training)

- Python 3.9 or higher
- TensorFlow 2.10 or higher with GPU support
- CUDA 11.2 or higher
- cuDNN 8.1 or higher
- 64 GB RAM (for streaming all chromosomes efficiently)
- At least one GPU with 16 GB VRAM (24 GB recommended for batch size 64)
- 2 TB disk space for the full compact dataset (chromosomes 1 to 22)
- 14 TB disk space if you plan to keep the raw UCSC MAF files

### Hardware used in this project

All training and evaluation was done on servers with NVIDIA Quadro RTX 6000 GPUs (24 GB VRAM each, compute capability 7.5). Training a single model takes approximately 2 to 7 hours. Preprocessing the full genome takes approximately 3 to 5 days on a single CPU core (this is embarrassingly parallelizable across chromosomes).

### Operating system

All scripts were developed and tested on Linux (Ubuntu 20.04). They should work on any Unix-like system. Windows is not tested.

---

## Installation

### Option 1: Install using Conda (recommended)

Conda manages the entire software environment including TensorFlow, CUDA, and all dependencies. This is the most reliable option for reproducibility.

```bash
# Step 1: Clone the repository
git clone https://github.com/DongjoonLim/GraphyloVar.git
cd GraphyloVar

# Step 2: Create the conda environment from the provided file
#         This installs Python, TensorFlow, NumPy, SciPy, and all other
#         dependencies with the exact versions used in the paper
conda env create -f environment.yml

# Step 3: Activate the environment
conda activate graphylo

# Step 4: Install the graphylovar package in development mode
#         The -e flag means "editable" -- changes to the source code
#         take effect immediately without reinstalling
pip install -e .
```

### Option 2: Install using pip

If you already have a Python environment with TensorFlow installed:

```bash
# Step 1: Clone the repository
git clone https://github.com/DongjoonLim/GraphyloVar.git
cd GraphyloVar

# Step 2: (Strongly recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate

# Step 3: Upgrade pip to the latest version
pip install --upgrade pip

# Step 4: Install all dependencies
pip install -r requirements.txt

# Step 5: Install the graphylovar package
pip install -e .
```

### Option 3: Manual dependency installation

If you prefer to manage dependencies yourself, the key packages are:

```bash
pip install tensorflow>=2.10
pip install numpy>=1.21
pip install scipy>=1.7
pip install pandas>=1.3
pip install scikit-learn>=1.0
pip install matplotlib>=3.4
pip install seaborn>=0.11
pip install networkx>=2.6
pip install biopython>=1.79
pip install tqdm>=4.62
pip install pysam>=0.18
pip install pyBigWig>=0.3.18
pip install wandb>=0.12    # optional, for experiment tracking
pip install -e .
```

### Verifying your installation

After installation, run the following to confirm everything is working:

```bash
# Check that the graphylovar package is importable
python -c "import graphylovar; print('graphylovar package: OK')"

# Check TensorFlow version and GPU availability
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available:', len(gpus))
for g in gpus:
    print('  ', g.name)
"

# Check that the phylogenetic tree loads correctly
python -c "
from graphylovar.phylogeny import load_phylogenetic_tree
tree = load_phylogenetic_tree()
print('Tree loaded. Nodes:', len(tree.nodes), 'Edges:', len(tree.edges))
"
```

If the tree loading step fails, make sure you ran `pip install -e .` from the GraphyloVar root directory so that the data files bundled in the `graphylovar/` package are accessible.

### Troubleshooting installation

**"ImportError: libcudnn.so.8: cannot open shared object file"**

This means cuDNN is not installed or not on your library path. If using Conda: `conda install cudnn=8.1`. If using system CUDA, follow NVIDIA's installation guide for your Linux distribution.

**"tensorflow.python.framework.errors_impl.NotFoundError: ... cannot allocate memory"**

Your GPU does not have enough memory for the default batch size. Try adding `--batch_size 32` or `--batch_size 16` to the training command.

**"ModuleNotFoundError: No module named 'graphylovar'"**

You need to run `pip install -e .` from the repository root. Make sure your conda environment or virtual environment is activated first.

---

## Data sources and download

### Source 1: UCSC 100-way vertebrate multiple sequence alignment

The alignment data comes from the UCSC Genome Browser's 100-way vertebrate alignment, lifted to the human reference genome hg38. You only need the placental mammalian portion (58 species), but the files contain all 100 species -- the preprocessing script automatically selects the 58 species GraphyloVar uses.

**Where to download:**

```
https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz100way/
```

The files are organized by chromosome: `chr1.maf.gz`, `chr2.maf.gz`, etc.

**How to download all autosomes (approximately 500 GB compressed):**

```bash
mkdir -p /path/to/ucsc_maf
cd /path/to/ucsc_maf

# Download chromosomes 1 through 22
for chr in $(seq 1 22); do
    echo "Downloading chr${chr}..."
    wget -q "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz100way/chr${chr}.maf.gz"
done
```

This download may take several hours depending on your internet connection. Each file ranges from approximately 5 GB (small chromosomes) to 100 GB (chromosome 1) when decompressed.

**What is a MAF file?**

A MAF (Multiple Alignment Format) file stores aligned sequences from multiple genomes. Each alignment block looks like this:

```
a score=12345.0
s hg38.chr22      16050075  65  +  51304566  ATCG...
s panTro4.chr22   16050075  65  +  51304560  ATCG...
s gorGor3.chr22A  16000123  65  +  50000000  ATCG...
...

a score=9876.0
s hg38.chr22      16050150  30  +  51304566  GCTA...
...
```

Each block is a local alignment of multiple species at a specific genomic region. The `s` lines give the species, coordinates, strand, chromosome length, and the aligned sequence (with gaps indicated by `-`).

### Source 2: TOPMed allele frequency data

TOPMed (Trans-Omics for Precision Medicine) is a large whole-genome sequencing program with data from approximately 140,000 individuals. The allele frequency annotations for common and rare variants are available through dbGaP.

**dbGaP accession:** phs000964

**How to access:**

1. Go to https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000964
2. Apply for access through dbGaP (requires institutional affiliation and IRB approval)
3. Once approved, download the VCF files with AF (allele frequency) annotations
4. You need the multi-sample VCF or the per-chromosome AF annotation files

**What information is used from TOPMed:**

For each SNV, GraphyloVar uses only the allele frequency (a number between 0 and 1 indicating how common the non-reference allele is in the TOPMed cohort). Variants with AF below 0.001 (0.1%) are labeled as "rare" (SNP label = 0). Variants with AF above 0.01 (1%) are labeled as "common" (SNP label = 1).

### Source 3: Ancestral sequence reconstruction

Ancestral sequences (representing what the sequence of the common ancestor of each pair of species likely was) come from the Ancestors1.0 reconstruction by Blanchette et al. (2004). These are already incorporated into the UCSC 100-way alignment files in the form of special species rows named after ancestral nodes (e.g., `Ancestor_4` for the common ancestor of human and chimpanzee).

You do not need to download ancestral sequences separately. The preprocessing script automatically extracts them from the MAF files.

---

## Data preprocessing

### Why preprocessing is needed

The raw UCSC MAF files are not suitable for training directly. They are very large (hundreds of GB), not indexed for random access by variant, and contain all 100 species (not just the 58 placental mammals GraphyloVar uses). The preprocessing step converts them into compact `.npz` files that store, for each SNV in the TOPMed dataset, the 65 bp aligned window across all 58 species as a numpy array. This allows the training script to efficiently stream variants without loading entire chromosomes into memory.

### Step-by-step preprocessing

**Step 1: Organize your data directories**

```bash
# Set up your directory structure
export MAF_DIR=/data/ucsc_maf            # Where you downloaded the .maf.gz files
export TOPMED_DIR=/data/topmed_vcfs      # Where the TOPMed VCF files are
export COMPACT_DIR=/data/topmed_compact  # Where the output .npz files will go

mkdir -p $COMPACT_DIR
```

**Step 2: Run the preprocessing script**

```bash
python scripts/preprocess_topmed_full_compact.py \
  --alignment_dir $MAF_DIR \
  --topmed_vcf_dir $TOPMED_DIR \
  --output_dir $COMPACT_DIR \
  --chromosomes 1-22 \
  --context_flank 32
```

**Full list of arguments:**

```
--alignment_dir DIR        Directory containing chr1.maf.gz, chr2.maf.gz, etc.
--topmed_vcf_dir DIR       Directory containing TOPMed VCF files
--output_dir DIR           Where to write the output .npz files
--chromosomes RANGE        Chromosomes to process. Can be:
                             1-22    (all autosomes)
                             1,2,3   (specific chromosomes)
                             22      (a single chromosome)
--context_flank INT        Flanking bases on each side of the variant.
                           32 gives a 65 bp window (default, used for
                           the main published model).
                           16 gives a 33 bp window (interpretability model).
                           100 gives a 201 bp window (ablation model).
--n_workers INT            Number of parallel CPU workers per chromosome.
                           Default is 4. Use more if you have many cores.
--chunk_size INT           Number of variants to process per chunk.
                           Reduce if you run out of RAM. Default is 100000.
```

**Step 3: Verify the output**

After preprocessing, each chromosome should have a file like `chr22_flank32.npz`. Check that it looks reasonable:

```bash
python -c "
import numpy as np
data = np.load('/data/topmed_compact/chr22_flank32.npz')
print('Arrays in file:', list(data.keys()))
print('Sequences shape:', data['sequences'].shape)
print('  (expected: [N_variants, 115_species, 65_positions, 5_nucleotides])')
print('SNP labels shape:', data['snp_labels'].shape)
print('  (expected: [N_variants])')
print('Allele freq labels shape:', data['af_labels'].shape)
print('  (expected: [N_variants, 5])')
print('Number of variants:', data['sequences'].shape[0])
print('Example SNP label:', data['snp_labels'][:5])
print('Example AF label:', data['af_labels'][:5])
"
```

A typical chromosome 22 file contains approximately 6 to 8 million variants and the `.npz` file is approximately 50 to 100 GB.

**Step 4: Parallel preprocessing (optional, strongly recommended)**

Preprocessing is independent per chromosome. You can preprocess all 22 chromosomes in parallel if you have multiple CPU cores or multiple machines:

```bash
# On a multi-core machine: process 4 chromosomes at a time
for i in 1 5 9 13 17; do
    for chr in $i $((i+1)) $((i+2)) $((i+3)); do
        if [ $chr -le 22 ]; then
            python scripts/preprocess_topmed_full_compact.py \
              --alignment_dir $MAF_DIR \
              --topmed_vcf_dir $TOPMED_DIR \
              --output_dir $COMPACT_DIR \
              --chromosomes $chr \
              --context_flank 32 \
              >> /tmp/preprocess_chr${chr}.log 2>&1 &
        fi
    done
    wait
done
```

### Expected disk usage

| Flank setting | Window size | Approx size per chromosome | Total for chr1-22 |
|--------------|-------------|---------------------------|-------------------|
| flank=16 | 33 bp | 25 to 50 GB | ~800 GB |
| flank=32 | 65 bp | 50 to 100 GB | ~1.6 TB |
| flank=100 | 201 bp | 150 to 300 GB | ~5 TB |

---

## Training the model from scratch

### The chromosome split

GraphyloVar uses a strict chromosome-level data split:

- **Training chromosomes (1 to 10)**: Used for gradient updates. The model never evaluates on these chromosomes.
- **Validation chromosomes (11 to 12)**: Used for early stopping. Training stops when validation loss does not improve for 30 consecutive epochs.
- **Test chromosomes (13 to 22)**: Never touched during training or hyperparameter selection. All reported performance numbers come from this set, which contains approximately 149 million SNVs.

This chromosome-level split prevents any data leakage from nearby variants sharing local sequence context or evolutionary conservation patterns.

### Basic training command

```bash
python scripts/train_topmed_full_streaming.py \
  --compact_dir /data/topmed_compact \
  --model_dir /data/topmed_models \
  --context_flank 32 \
  --model_name multitask_hybrid_v3 \
  --mixed_precision \
  --train_chromosomes 1-10 \
  --val_chromosomes 11-12
```

### Full list of training arguments

```
--compact_dir DIR           Directory with preprocessed .npz files.
                            Must contain chrN_flankF.npz for each
                            chromosome N and flank F.

--model_dir DIR             Where to save model checkpoints, logs,
                            and training summaries.

--context_flank INT         Must match the flank used during preprocessing.
                            Default: 32 (65 bp window).

--model_name NAME           Architecture to use. Options:
                              multitask_hybrid_v3   Main model (published)
                              multitask_hybrid_v4   Higher-capacity ablation

--run_tag TAG               A short string appended to the checkpoint
                            filename to identify this run.
                            Example: v3main, v3flank16, ablation_noGCN

--mixed_precision           Use float16 mixed precision on GPU.
                            Roughly 2x faster and uses half the memory
                            on NVIDIA RTX/Ampere GPUs. Strongly recommended.

--train_chromosomes RANGE   Chromosomes for training. Default: 1-10.

--val_chromosomes RANGE     Chromosomes for validation. Default: 11-12.

--batch_size INT            Number of variants per gradient step.
                            Default: 64. Reduce to 32 or 16 if you
                            run out of GPU memory.

--steps_per_epoch INT       Number of gradient steps per epoch.
                            Default: 5000 (processes 320,000 variants
                            per epoch at batch_size=64).

--validation_steps INT      Number of steps for validation loss estimate.
                            Default: 1000.

--epochs INT                Maximum number of epochs. Default: 9999.
                            Early stopping (--patience) controls actual
                            stopping in practice.

--patience INT              Stop training if validation loss does not
                            improve for this many consecutive epochs.
                            Default: 30.

--learning_rate FLOAT       Initial learning rate for Adam optimizer.
                            Default: 0.0003.

--lr_reduce_factor FLOAT    Factor by which to reduce LR when validation
                            loss plateaus. Default: 0.5.

--lr_reduce_patience INT    Epochs without improvement before reducing LR.
                            Default: 5.

--lr_min FLOAT              Minimum learning rate (floor for ReduceLROnPlateau).
                            Default: 1e-6.

--binary_loss_weight FLOAT  Weight on the SNP probability head loss
                            relative to the nucleotide frequency head.
                            Default: 1.5.

--skip_test_eval            Do not evaluate on test set after training.
                            Useful for training ablation models where you
                            will evaluate manually later.

--resume_from PATH          Path to a checkpoint .keras file to resume
                            training from. The training history and
                            optimizer state are restored.
```

### Selecting a specific GPU

On a multi-GPU server, use `CUDA_VISIBLE_DEVICES` to specify which GPU to use:

```bash
# Use GPU 2 only
CUDA_VISIBLE_DEVICES=2 python scripts/train_topmed_full_streaming.py \
  --compact_dir /data/topmed_compact \
  --model_dir /data/topmed_models \
  --context_flank 32 \
  --model_name multitask_hybrid_v3 \
  --mixed_precision \
  --train_chromosomes 1-10 \
  --val_chromosomes 11-12 \
  --run_tag v3flank32

# Use GPUs 2 and 3 together (data-parallel training)
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_topmed_full_streaming.py ...
```

Running in the background with logging:

```bash
CUDA_VISIBLE_DEVICES=2 nohup python scripts/train_topmed_full_streaming.py \
  --compact_dir /data/topmed_compact \
  --model_dir /data/topmed_models \
  --context_flank 32 \
  --model_name multitask_hybrid_v3 \
  --mixed_precision \
  --train_chromosomes 1-10 \
  --val_chromosomes 11-12 \
  --run_tag v3flank32 \
  >> ~/training_v3flank32.log 2>&1 &

echo "Training PID: $!"
```

### What to expect during training

Each epoch takes approximately 5 minutes on a single Quadro RTX 6000 with flank=32. You will see output like:

```
Epoch 1/9999
5000/5000 [==============================] - 285s 57ms/step
  loss: 0.4231 - nuc_loss: 0.2891 - binary_loss: 0.2103
  val_loss: 0.3812 - val_nuc_loss: 0.2512 - val_binary_loss: 0.1921
  val_binary_accuracy: 0.5124
  LR: 0.0003000

Epoch 2/9999
...
```

The key metric to watch is `val_nuc_loss` (nucleotide prediction loss on the validation set). This is the most reliable indicator of model quality because the binary label (SNP/not SNP) has severe class imbalance. A typical training run converges in 20 to 80 epochs.

**Warning about val_binary_accuracy**: This metric can appear stuck near 0.5 or even near 0 during training, especially for short flanks. This is a known artifact of class imbalance in the binary SNP labels, not a training failure. The AUROC metric computed after training is the correct evaluation, not accuracy.

### Checkpoint files

The training script saves two types of checkpoints in `--model_dir`:

1. **Best checkpoint**: Saved whenever validation loss improves. Named like:
   `multitask_hybrid_v3_train1-10_val11-12_flank32_v3flank32.keras`
2. **Last checkpoint**: Saved at the end of every epoch. Named with `_last` suffix.

The `.keras` format is TensorFlow's native format and includes both the model weights and the architecture. You can load it with:

```python
import tensorflow as tf
model = tf.keras.models.load_model('path/to/checkpoint.keras')
```

---

## Monitoring training progress

### Weights and Biases (W&B) integration

If you have a W&B account, training metrics are automatically logged when the `wandb` package is installed:

```bash
pip install wandb
wandb login   # Enter your API key when prompted
```

Then add `--wandb_project your_project_name` to the training command. You can then track loss curves, compare runs, and set alerts in the W&B dashboard.

### Checking GPU utilization

While training is running, use `nvidia-smi` in a separate terminal to check that the GPU is being used:

```bash
# Check current GPU status
nvidia-smi

# Watch GPU usage every 2 seconds
watch -n2 nvidia-smi
```

You should see memory usage near the maximum for your GPU and GPU utilization above 80%. If GPU utilization is low (below 30%), the data loading pipeline may be a bottleneck. Try increasing `--n_workers` in the preprocessing step or pre-loading data into RAM.

### Reading training logs

The training log file (if you used `nohup` and redirected output) contains the full epoch-by-epoch history. To get a quick summary:

```bash
# Show last 20 lines of the log
tail -20 ~/training_v3flank32.log

# Extract just the validation loss per epoch
grep "val_nuc_loss" ~/training_v3flank32.log

# Find the epoch with the best validation loss
grep "val_nuc_loss" ~/training_v3flank32.log | sort -t: -k2 -n | head -5
```

---

## Scoring variants with a trained model

### Scoring a single chromosome

Once you have a trained model checkpoint, you can score variants on any chromosome in the test set:

```bash
python scripts/predict.py \
  --model_path /data/topmed_models/multitask_hybrid_v3_train1-10_val11-12_flank32_v3flank32.keras \
  --data_dir /data/topmed_compact \
  --chromosome 22 \
  --output_file scores_chr22.tsv
```

This script loads the compact data for chromosome 22, runs the model on all variants in batches, and writes the scores to a TSV file.

**Arguments:**

```
--model_path PATH     Path to the .keras checkpoint file.
--data_dir DIR        Directory with preprocessed .npz compact files.
--chromosome INT      Which chromosome to score (1 to 22).
--output_file PATH    Where to write the output TSV.
--batch_size INT      Inference batch size. Default: 256.
                      Can use larger batches at inference than training
                      since no gradients are stored.
--gpu INT             Which GPU to use. Default: 0.
```

### Output file format

The output TSV has the following columns:

```
chrom    pos        ref  alt  snp_prob   af_A    af_C    af_G    af_T    af_gap
chr22    16050075   A    G    0.0234     0.7821  0.0012  0.2054  0.0103  0.0010
chr22    16050115   C    T    0.7891     0.0023  0.8932  0.0912  0.0021  0.0112
```

Column descriptions:

- `chrom`: Chromosome name
- `pos`: 1-based genomic position (hg38 coordinates)
- `ref`: Reference allele
- `alt`: Alternative allele
- `snp_prob`: The main variant impact score. This is the model's predicted probability that this position is polymorphic in the population. Higher values mean the model predicts the position tolerates variation (likely benign). Lower values mean the model predicts the position is functionally constrained (more likely harmful). To use as a deleteriousness score (higher = more harmful), compute 1.0 minus snp_prob.
- `af_A`, `af_C`, `af_G`, `af_T`, `af_gap`: The model's predicted allele frequency distribution over the 5 nucleotide states. These sum to 1.0 for each variant. The state with the highest predicted probability is the model's prediction for the most common allele at this position.

### Genome-wide scoring

To score all chromosomes:

```bash
for CHR in $(seq 1 22); do
    echo "Scoring chromosome $CHR..."
    python scripts/predict.py \
      --model_path /data/topmed_models/checkpoint.keras \
      --data_dir /data/topmed_compact \
      --chromosome $CHR \
      --output_file /data/scores/scores_chr${CHR}.tsv \
      --batch_size 256
done
```

Or use the genome-wide scoring script directly:

```bash
python scripts/predict_genome.py \
  --model_path /data/topmed_models/checkpoint.keras \
  --data_dir /data/topmed_compact \
  --output_dir /data/scores \
  --chromosomes 1-22 \
  --batch_size 256
```

---

## Evaluating performance

### Computing AUROC on the test set

The primary evaluation is AUROC for discriminating common variants (label=1) from rare variants (label=0) on the held-out test set (chromosomes 13 to 22):

```bash
python scripts/recompute_region_flank_auc_from_ucsc.py \
  --compact_dir /data/topmed_compact \
  --model_path /data/topmed_models/checkpoint.keras \
  --output_dir /data/evaluation_results \
  --chromosomes 13-22 \
  --flanks 32
```

This script computes AUROC for all variants together and stratified by genomic region (coding, 3'UTR, cCREs, transposable elements, etc.) using UCSC annotation tracks.

**Expected output** (`measured_region_flank_auc_values.csv`):

```
flank,region,auroc,n_variants
32,all,0.6246,149283471
32,coding,0.6162,3847291
32,3utr,0.6213,2194830
32,ccres,0.6173,7382019
32,te,0.6263,48291847
```

### Computing Spearman correlation with MAF

```bash
python -c "
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Load scores and true allele frequencies
scores = pd.read_csv('/data/scores/scores_chr22.tsv', sep='\t')
compact = np.load('/data/topmed_compact/chr22_flank32.npz')

# The true AF for the alt allele
true_af = compact['af_labels'][:, 1]   # index depends on alt allele encoding

# Spearman correlation between snp_prob and true AF
rho, pval = spearmanr(scores['snp_prob'], true_af)
print(f'Spearman rho: {rho:.4f}, p-value: {pval:.2e}')
"
```

---

## Fine-tuning on MPRA data

### What is MPRA?

Massively Parallel Reporter Assay (MPRA) is an experimental technique that measures the regulatory activity of large numbers of DNA sequences simultaneously. A typical MPRA experiment tests tens of thousands of variant sequences (each with a common and a rare allele) and measures how much each sequence drives expression of a reporter gene. The ratio of activity between the two alleles gives a direct functional measurement of the variant's regulatory effect.

GraphyloVar was fine-tuned on 13 published MPRA datasets covering enhancer activity in various cell types and tissues.

### Fine-tuning procedure

Fine-tuning adapts the pre-trained model to a specific regulatory context by continuing training on the MPRA data with a small learning rate:

```bash
python scripts/train_ablation.py \
  --pretrained_model /data/topmed_models/checkpoint.keras \
  --mpra_data /data/mpra/dataset_name.csv \
  --output_dir /data/finetuned_models \
  --learning_rate 0.00001 \
  --epochs 20 \
  --patience 5 \
  --task mpra
```

The MPRA data CSV should have columns: `sequence_human` (65 bp reference sequence), `sequence_alt` (65 bp alternative allele sequence), `label` (1 for active, 0 for inactive, or continuous activity score).

---

## Ensembling with CADD

CADD (Combined Annotation Dependent Depletion) scores are available as precomputed genome-wide files from the CADD website. The CADD score is in a different direction from GraphyloVar: higher CADD means more deletrious, while higher GraphyloVar snp_prob means more benign. Before ensembling, you need to align the conventions.

### Downloading CADD scores

```bash
# CADD v1.7 pre-scored SNVs for hg38 can be downloaded from:
# https://cadd.gs.washington.edu/download
# File: whole_genome_SNVs.tsv.gz (approximately 80 GB)

wget https://krishna.gs.washington.edu/download/CADD/v1.7/GRCh38/whole_genome_SNVs.tsv.gz
```

### Computing the z-score ensemble

The ensemble works by converting each model's scores to z-scores (subtracting the mean and dividing by the standard deviation across all variants on a chromosome) and then averaging:

```python
import numpy as np
import pandas as pd
from scipy.stats import zscore

# Load GraphyloVar scores for chromosome 22
gv_scores = pd.read_csv('scores_chr22.tsv', sep='\t')

# Load CADD scores for chromosome 22
# CADD is higher = more deleterious; we need to align direction
# GraphyloVar snp_prob is higher = more benign
# So we negate GraphyloVar (or negate CADD) before ensembling
cadd_scores = pd.read_csv('cadd_chr22.tsv', sep='\t',
    names=['chrom', 'pos', 'ref', 'alt', 'rawscore', 'phred'])

# Merge on position
merged = gv_scores.merge(cadd_scores, on=['chrom', 'pos', 'ref', 'alt'])

# Compute z-scores
# GraphyloVar: negate so higher z-score = more deleterious
gv_z = -zscore(merged['snp_prob'])
cadd_z = zscore(merged['phred'])

# Ensemble: average z-scores
merged['ensemble_score'] = (gv_z + cadd_z) / 2
```

The ensemble achieves AUROC 0.6442 on the full 149M variant test set, an improvement of +0.020 over GraphyloVar alone (0.6246).

---

## Interpretability analysis

GraphyloVar provides two complementary analyses to understand which species contribute to its predictions.

### Analysis 1: Additive species importance

In this analysis, each species is evaluated in isolation. The model is run with all 58 species set to gap tokens, then with each species' real sequence restored one at a time. The "additive importance" of a species is the decrease in cross-entropy loss when that species' sequence is added to an otherwise all-gap alignment.

This measures: "How much information does each species' sequence alone provide?"

**Running the analysis:**

```bash
python scripts/additive_species_importance.py \
  --run_tag v3flank16 \
  --samples_per_chrom 50000 \
  --chromosomes 13-22 \
  --batch_size 16 \
  --output_dir outputs/interpretability/additive_species
```

**Arguments:**

```
--run_tag TAG         Which model to use (must match a training run tag
                      in the model directory).
                      Use v3flank16 for the interpretability model.
--samples_per_chrom N Number of variants to sample from each chromosome.
                      Total variants = N times number of chromosomes.
                      50000 per chrom x 10 chroms = 500,000 total.
--chromosomes RANGE   Test set chromosomes. Use 13-22.
--batch_size INT      Batch size for inference. 16 is conservative;
                      increase if GPU memory allows.
--output_dir DIR      Where to save the output CSV.
```

**Why use v3flank16 (not v3flank32) for interpretability?**

The interpretability analyses (additive importance, SE attention, LOO perturbation) are all run using the flank=16 (33 bp) model, not the flank=32 (65 bp) main model. This is a deliberate choice: the smaller window is easier to interpret because there are fewer positions that could be driving the species' importance. The ranking of species is similar across both models, but the absolute values differ. The interpretability results reported in the paper always specify "v3flank16" explicitly.

**Output file:**

The output is a CSV file with one row per species:

```
ucsc_name,common_name,delta_ce_additive,rank
hg38,human,7.1234,1
ponAbe2,orangutan,1.5012,2
chlSab2,green monkey,1.4987,3
gorGor3,gorilla,1.2401,4
nomLeu3,gibbon,0.6189,5
...
panTro4,chimpanzee,0.0621,11
...
```

**Interpreting the results:**

| Rank | Species (UCSC name) | Common name | Importance | Interpretation |
|------|---------------------|-------------|------------|----------------|
| 1 | hg38 | Human | ~7.1 | Human sequence alone is by far the most informative |
| 2 | ponAbe2 | Orangutan | ~1.50 | Diverged enough from human to add useful signal |
| 3 | chlSab2 | Green monkey | ~1.50 | Old World monkey, good evolutionary distance |
| 4 | gorGor3 | Gorilla | ~1.24 | Intermediate distance to human |
| 5 | nomLeu3 | Gibbon | ~0.62 | Furthest great ape from human |
| 11 | panTro4 | Chimpanzee | ~0.062 | Very close to human, adds almost no new info |

**Why is chimpanzee ranked so low?**

Human and chimpanzee share approximately 98.7% sequence identity. When the model already has the human sequence, chimpanzee's sequence tells it almost nothing new: every position that is conserved between human and chimp is already captured by the human row. In contrast, orangutan (approximately 96% identity) and green monkey (approximately 93% identity) have diverged enough that they carry independent evolutionary signals that the human row does not capture.

This is a principle called information redundancy: the value of adding a second source of information depends on how much of that information is already available from the first source. Chimpanzee and human are so similar that chimp is essentially redundant given human.

This finding is consistent with the SE gate values: the SE module assigns chimpanzee a mean gate value of approximately 0.011, far below human (1.00) and even below many distant species.

### Analysis 2: Leave-one-out (LOO) species perturbation

In this analysis, each species is removed from the complete alignment one at a time, and the increase in loss is measured. This answers a different question: "How much does the model rely on this species given that all 57 other species are present?"

**Running the analysis:**

```bash
python scripts/per_species_all58_perturbation.py \
  --run_tag v3flank16 \
  --n_samples 2000000 \
  --batch_size 128 \
  --output_dir outputs/interpretability/loo_perturbation
```

**Output file** (`all58_species_perturbation_scores.csv`):

Same format as the additive analysis, but the `delta_ce` values are usually much smaller because removing any single species has a small effect when 57 others are present.

**LOO vs. additive: a concrete example**

Imagine you are writing a report and you have 10 sources all saying essentially the same thing. If you remove any one source, the report is barely affected (low LOO importance for each). But if you evaluate each source standalone, source 1 (the most comprehensive one) has high standalone importance and sources 2 to 10 have lower but positive standalone importance. GraphyloVar's species are like these sources: individually useful (positive additive importance), but collectively redundant (low LOO importance for most).

### Analysis 3: SE attention gate values

The Squeeze-and-Excitation module produces a scalar attention weight for each species. These weights can be extracted and compared across species to get a model-internal measure of species importance:

```bash
python scripts/extract_se_attention_scores.py \
  --run_tag v3flank16 \
  --n_samples 1000000 \
  --output_dir outputs/se_attention
```

**Expected output** (`actual_se_attention_scores.csv`):

```
ucsc_name,mean_gate_value,std_gate_value
hg38,1.0000,0.0000
ponAbe2,0.5412,0.1234
chlSab2,0.5289,0.1198
...
panTro4,0.0112,0.0043
```

The SE gate values and the additive importance values tell a consistent story. Human has the highest gate (1.00) and the highest additive importance (~7.1). Chimpanzee has the lowest gate among great apes (0.011) and the lowest additive importance among primates (~0.062).

---

## Model architecture (detailed)

### Overview

GraphyloVar is called a Transformer + GCN model. The Transformer encodes local sequence context per species, and the GCN aggregates that information across species using the phylogenetic tree structure. Both components are trained end-to-end jointly.

### Input tensor

For each variant, the input is a tensor of shape:

```
[N_species, N_positions, N_nucleotides] = [115, 65, 5]
```

- 115 species: 58 extant species with real observed sequences, plus 57 ancestral nodes initialized to gap vectors (all zeros or gap one-hot).
- 65 positions: 32 flanking positions on each side of the variant, plus the variant position itself.
- 5 nucleotide states: one-hot encoding of A, C, G, T, and gap (-).

### Per-species Transformer encoder

The same Transformer encoder is applied independently to each of the 115 species. For each species:

**Sub-step 1: Dense embedding**

The 5-dimensional one-hot input at each position is mapped to a 32-dimensional embedding using a dense layer with GELU activation and batch normalization:

```
input: [65, 5]
Dense(32, activation='gelu') + BatchNorm
output: [65, 32]
```

GELU (Gaussian Error Linear Unit) is used here rather than ReLU because GELU has been shown to work better for sequence models. It smoothly gates the activations rather than hard-clipping them.

**Sub-step 2: Sinusoidal positional encoding**

The model adds a fixed sinusoidal positional encoding to the embedding to tell the Transformer which position in the 65 bp window it is looking at. Without positional encoding, the Transformer would treat the input as a bag of features with no positional information. The encoding is:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

where pos is the position (0 to 64), i is the dimension index, and d_model is 32.

**Sub-step 3: Two Transformer encoder layers**

Each layer has:
- Multi-head self-attention: 4 heads, key dimension 8 (total attention dimension = 4 * 8 = 32)
- Feedforward sublayer: Dense(128, GELU) followed by Dense(32)
- Layer normalization (pre-norm architecture)
- Dropout (rate 0.1 during training)

The self-attention allows every position in the 65 bp window to attend to every other position, capturing long-range dependencies within the window.

**Sub-step 4: Center position extraction**

Only the feature vector at position 32 (the center of the 65 bp window, which is the variant position) is retained:

```
input: [65, 32]
Extract position 32
output: [32]
```

This is done independently for the forward strand and the reverse complement strand, then concatenated:

```
forward:  [32]
revcomp:  [32]
concat:   [64]
```

The result is a 64-dimensional representation for each of the 115 species.

### Squeeze-and-Excitation attention gate

After the Transformer, a Squeeze-and-Excitation module computes a scalar weight for each of the 58 extant species (not the ancestral nodes):

```
input for each species: [64]
Dense(16, ReLU)        -> [16]
Dense(1, sigmoid)      -> scalar in [0, 1]
```

The scalar weight multiplies the 64-dim feature vector:

```
gated_feature = scalar * feature_vector
```

The SE weights are learned during training. They capture a data-driven measure of how informative each species' sequence is on average. Note that the SE gate is a global measure (it does not vary per variant), while the Transformer self-attention is a local measure (it does vary per variant).

### Graph Convolutional Network (GCN)

The 64-dimensional feature vectors for all 115 species are arranged on the nodes of the phylogenetic tree. The GCN updates each node's feature vector by aggregating from its neighbors:

```
H^(l+1) = sigma(D^(-1/2) * A * D^(-1/2) * H^(l) * W^(l))
```

where:
- H^(l) is the node feature matrix at layer l (shape [115, d_l])
- A is the normalized adjacency matrix of the phylogenetic tree (shape [115, 115])
- D is the degree matrix (diagonal matrix of node degrees)
- W^(l) is the learnable weight matrix at layer l
- sigma is a ReLU activation

**Layer dimensions:**
- GCN Layer 1: input 64 dims, output 32 dims, ReLU activation
- GCN Layer 2: input 32 dims, output 32 dims, ReLU activation

After two GCN layers, each node has a 32-dimensional feature vector that incorporates information from its immediate neighbors and its neighbors' neighbors. For a leaf node (extant species), this means information from its sister species and from ancestral nodes.

The 115 x 32 GCN output is flattened to a single vector of 3680 dimensions.

### Prediction heads

**Shared fully connected layer:**
```
Dense(128, activation='relu')
Dropout(0.3)
```

**Nucleotide frequency head:**
```
Dense(64, activation='relu')
Dense(5, activation='softmax')
Loss: categorical cross-entropy
```

This head predicts the probability distribution over the 5 nucleotide states {A, C, G, T, gap} at the variant position in the human population. The target is the observed allele frequency vector from TOPMed, e.g., [0.95, 0.0, 0.05, 0.0, 0.0] for a position where A has frequency 0.95 and G has frequency 0.05.

**SNP probability head:**
```
Dense(64, activation='relu')
Dense(1, activation='sigmoid')
Loss: binary cross-entropy
```

This head predicts whether the position is polymorphic in the population (label=1 for common variants with AF > 0.01, label=0 for rare variants with AF < 0.001). The sigmoid output is the main variant impact score.

### Total parameter count

```
Transformer encoders (x115 species, but shared weights):
  Per-species embedding:   5*32 + 32 = 192 params
  Positional encoding:     fixed, no params
  Attention Layer 1:       ~4*(32*8 + 32*8 + 32*32) = ~4800 params
  Attention Layer 2:       ~4800 params
  Total per-encoder:       ~10,000 params

SE module (x58 species, but shared weights):
  Dense(64->16):           64*16 + 16 = 1040 params
  Dense(16->1):            16 + 1 = 17 params

GCN Layer 1:               64*32 = 2048 params
GCN Layer 2:               32*32 = 1024 params

Shared FC:                 3680*128 = 471,040 params
Nuc head:                  128*64 + 64*5 = 8512 params
SNP head:                  128*64 + 64*1 = 8256 params

Total: approximately 500,000 to 600,000 parameters
```

This is a relatively small model by modern deep learning standards, which contributes to its fast training time and resistance to overfitting even on the limited MPRA fine-tuning data.

### Architecture diagram

```
Input: [115 species x 65 bp x 5 nucleotides]
|
+--[For each species independently]------------------+
|  Dense(32, GELU) + BatchNorm                        |
|  + Sinusoidal Positional Encoding                   |
|  TransformerLayer(4 heads, key_dim=8)               |
|  TransformerLayer(4 heads, key_dim=8)               |
|  Extract center position 32 -> [32-dim]             |
|    do this for forward AND reverse complement       |
|  Concatenate forward + revcomp -> [64-dim]          |
+----------------------------------------------------+
|
v   [115 x 64 tensor]
|
SE gate: [64-dim] -> Dense(16, ReLU) -> Dense(1, sigmoid) -> scalar
  (for each of 58 extant species; ancestral nodes pass through unchanged)
|
v   [115 x 64 tensor, with extant species gated]
|
GCN Layer 1: [115, 64] -> [115, 32] on the phylogenetic tree (ReLU)
GCN Layer 2: [115, 32] -> [115, 32] (ReLU)
|
Flatten: [115, 32] -> [3680]
|
Shared Dense(128, ReLU) + Dropout(0.3)
|
+---[Nucleotide head]---+   +---[SNP head]----------+
|  Dense(64, ReLU)      |   |  Dense(64, ReLU)       |
|  Dense(5, softmax)    |   |  Dense(1, sigmoid)     |
|  CatCrossEntropy loss |   |  BinCrossEntropy loss  |
+-----------------------+   +------------------------+
```

---

## Understanding the loss functions

### Why two loss functions?

GraphyloVar uses multitask learning with two prediction targets simultaneously. Using both targets together is important because:

1. The nucleotide frequency task provides a richer training signal. Predicting the full distribution over 5 nucleotide states (rather than a binary label) gives the model more gradient information per training example.
2. The SNP probability task directly optimizes for the variant impact discrimination goal.
3. The two tasks share all layers up to the final heads, so information useful for nucleotide frequency prediction also benefits SNP probability prediction and vice versa.

### Nucleotide frequency loss

The allele frequency target is a probability vector over 5 states (A, C, G, T, gap), derived from the TOPMed variant frequencies. For example, for a position where 95% of people have reference allele A and 5% have alternative allele G, the target is [0.95, 0.0, 0.05, 0.0, 0.0].

The loss is categorical cross-entropy:
```
L_nuc = -sum_i (y_i * log(p_i))
```
where y_i is the target frequency and p_i is the model's predicted probability for state i.

### SNP probability loss

The binary target is 1 (common, presumably benign) for variants with allele frequency above 0.01, and 0 (rare, potentially harmful) for variants with allele frequency below 0.001. Variants between 0.001 and 0.01 are excluded from the binary classification task to reduce label noise near the threshold.

The loss is binary cross-entropy:
```
L_snp = -(y * log(p) + (1-y) * log(1-p))
```

### Total loss and weighting

The final loss is a weighted sum:
```
L_total = L_nuc + w * L_snp
```

where w is the `--binary_loss_weight` argument (default 1.5). Setting w higher places more emphasis on the variant discrimination task; setting it lower makes the model focus more on nucleotide frequency prediction.

---

## The phylogenetic tree and GCN adjacency matrix

### The phylogenetic tree

GraphyloVar uses the mammalian phylogenetic tree from the UCSC 100-way alignment. The tree is a binary rooted tree in NEWICK format. It has 115 nodes: 58 extant leaf nodes (the species in the alignment) and 57 internal nodes (the reconstructed ancestral species). The root represents the last common ancestor of all 58 species.

The tree is stored in `graphylovar/data/mammalian_tree.nwk` and is loaded automatically when you import the `graphylovar.phylogeny` module.

### Branch lengths

Branch lengths are measured in expected substitutions per site. Short branches connect closely related species (e.g., human and chimpanzee, branch length approximately 0.013). Long branches connect distantly related species (e.g., human and armadillo, which separated approximately 90 million years ago).

### How the GCN adjacency matrix is built

The GCN uses a normalized adjacency matrix derived from the phylogenetic tree:

1. Start with the tree as an undirected graph. Each branch in the tree becomes an undirected edge between two nodes (e.g., an edge between the human node and the human-chimp ancestor node).

2. Assign edge weights inversely proportional to branch length. Short branches (closely related species) get higher weights; long branches get lower weights. This means the GCN propagates more information between closely related species than between distant ones.

3. Normalize the adjacency matrix using the graph Laplacian normalization:
   ```
   A_norm = D^(-1/2) * A * D^(-1/2)
   ```
   where D is the diagonal degree matrix. This normalization ensures that node features are not artificially amplified or diminished based on the number of connections.

### Why ancestral nodes?

Including ancestral nodes allows the GCN to aggregate information through the tree structure more naturally. Without ancestral nodes, the GCN would have to propagate information between two species by going through many intermediate edges. With ancestral nodes, information can flow directly from a leaf node to its parent (ancestor) and then to sibling leaves. This makes the GCN message passing more efficient and biologically meaningful.

---

## The 58 species used

GraphyloVar uses 58 extant placental mammalian species from the UCSC 100-way vertebrate alignment. All species are identified by their UCSC genome browser assembly accession.

### Why 58 species?

The UCSC 100-way alignment contains 100 vertebrate species, but many are birds, fish, and reptiles. GraphyloVar focuses exclusively on placental mammals because:

1. They are the most closely related to human and thus provide the most relevant evolutionary signal for human variant interpretation.
2. The alignment quality for placental mammals in human-centered regions is high.
3. Birds and fish have very different regulatory grammars and may introduce noise rather than signal for regulatory variant prediction.

The 58 species span approximately 90 million years of evolution and provide dense sampling of the mammalian phylogeny.

### Species list by clade

**Primates**

Great Apes (5 species):
- hg38: Homo sapiens (human)
- panTro4: Pan troglodytes (chimpanzee)
- gorGor3: Gorilla gorilla (western lowland gorilla)
- ponAbe2: Pongo abelii (Sumatran orangutan)
- nomLeu3: Nomascus leucogenys (northern white-cheeked gibbon)

Old World Monkeys (4 species):
- rheMac3: Macaca mulatta (rhesus macaque)
- macFas5: Macaca fascicularis (crab-eating macaque)
- papAnu2: Papio anubis (olive baboon)
- chlSab2: Chlorocebus sabaeus (African green monkey)

New World Monkeys (2 species):
- calJac3: Callithrix jacchus (common marmoset)
- saiBol1: Saimiri boliviensis (Bolivian squirrel monkey)

Strepsirrhini (1 species):
- otoGar3: Otolemur garnettii (small-eared greater bushbaby)

**Euarchontoglires** (superorder including primates, rodents, and rabbits)

Scandentia (1 species):
- tupChi1: Tupaia chinensis (Chinese tree shrew)

Rodentia (11 species):
- speTri2: Spermophilus tridecemlineatus (thirteen-lined ground squirrel)
- jacJac1: Jaculus jaculus (lesser Egyptian jerboa)
- micOch1: Microtus ochrogaster (prairie vole)
- criGri1: Cricetulus griseus (Chinese hamster)
- mesAur1: Mesocricetus auratus (golden hamster)
- mm10: Mus musculus (house mouse)
- rn6: Rattus norvegicus (brown rat)
- hetGla2: Heterocephalus glaber (naked mole-rat)
- cavPor3: Cavia porcellus (domestic guinea pig)
- chiLan1: Chinchilla lanigera (long-tailed chinchilla)
- octDeg1: Octodon degus (degu)

Lagomorpha (2 species):
- oryCun2: Oryctolagus cuniculus (European rabbit)
- ochPri3: Ochotona princeps (American pika)

**Laurasiatheria** (superorder including carnivores, bats, whales, and horses)

Cetartiodactyla (9 species):
- susScr3: Sus scrofa (domestic pig)
- vicPac2: Vicugna pacos (alpaca)
- camFer1: Camelus ferus (Bactrian camel)
- turTru2: Tursiops truncatus (common bottlenose dolphin)
- orcOrc1: Orcinus orca (killer whale)
- panHod1: Pantholops hodgsonii (Tibetan antelope)
- bosTau8: Bos taurus (domestic cattle)
- oviAri3: Ovis aries (domestic sheep)
- capHir1: Capra hircus (domestic goat)

Perissodactyla (2 species):
- equCab2: Equus caballus (domestic horse)
- cerSim1: Ceratotherium simum (white rhinoceros)

Carnivora (6 species):
- felCat8: Felis catus (domestic cat)
- canFam3: Canis lupus familiaris (domestic dog)
- musFur1: Mustela putorius furo (domestic ferret)
- ailMel1: Ailuropoda melanoleuca (giant panda)
- odoRosDiv1: Odobenus rosmarus divergens (Pacific walrus)
- lepWed1: Leptonychotes weddellii (Weddell seal)

Chiroptera (5 species):
- pteAle1: Pteropus alecto (black flying fox)
- pteVam1: Pteropus vampyrus (large flying fox)
- eptFus1: Eptesicus fuscus (big brown bat)
- myoDav1: Myotis davidii (David's myotis)
- myoLuc2: Myotis lucifugus (little brown bat)

Eulipotyphla (3 species):
- eriEur2: Erinaceus europaeus (western European hedgehog)
- sorAra2: Sorex araneus (common shrew)
- conCri1: Condylura cristata (star-nosed mole)

**Atlantogenata** (superorder including elephants and South American mammals)

Afrotheria (5 species):
- loxAfr3: Loxodonta africana (African savanna elephant)
- eleEdw1: Elephantulus edwardii (Cape elephant shrew)
- triMan1: Trichechus manatus latirostris (Florida manatee)
- chrAsi1: Chrysochloris asiatica (Cape golden mole)
- echTel2: Echinops telfairi (lesser hedgehog tenrec)

Xenarthra (2 species):
- oryAfe1: Orycteropus afer (aardvark)
- dasNov3: Dasypus novemcinctus (nine-banded armadillo)

**Total: 58 extant species + 57 reconstructed ancestral nodes = 115 nodes in the phylogenetic tree**

---

## Context window ablation study

We trained three versions of GraphyloVar with different context window sizes to determine the optimal flank setting. All models use the v3 architecture (multitask_hybrid_v3) and the same training procedure (chromosomes 1 to 10 for training, 11 to 12 for validation, 13 to 22 for testing).

### Flank ablation results (region-stratified AUROC on full 149M-variant holdout)

| Flank | Window size | All | Coding | 3-prime UTR | cCREs | Transposable elements |
|-------|-------------|-----|--------|-------------|-------|-----------------------|
| 16 | 33 bp | 0.622 | **0.625** | 0.619 | **0.620** | 0.621 |
| **32** | **65 bp** | **0.625** | 0.616 | **0.621** | 0.617 | **0.626** |
| 100 | 201 bp | 0.617 | 0.610 | 0.616 | 0.617 | 0.616 |

The flank=32 (65 bp) model achieves the best overall AUROC (0.625) and is the main published model. The flank=16 model performs best for coding regions specifically. The flank=100 model performs worst overall, suggesting that the extra context beyond 65 bp adds noise rather than signal for allele frequency discrimination.

This non-monotonic pattern (flank=8 achieves even higher AUROC at 0.650 on some evaluations) is discussed in detail in the paper's Section 3.5. The fine-tuned flank=32 model achieves the highest AUROC on all 13 MPRA benchmark datasets, confirming that the 65 bp window provides the best balance of zero-shot performance and fine-tuning capability.

### Reproducing the ablation

To train and evaluate all three flank sizes:

```bash
# Train flank=16
CUDA_VISIBLE_DEVICES=0 python scripts/train_topmed_full_streaming.py \
  --compact_dir /data/topmed_compact_flank16 \
  --model_dir /data/topmed_models \
  --context_flank 16 \
  --model_name multitask_hybrid_v3 \
  --mixed_precision \
  --train_chromosomes 1-10 \
  --val_chromosomes 11-12 \
  --run_tag v3flank16

# Train flank=32 (main model)
CUDA_VISIBLE_DEVICES=1 python scripts/train_topmed_full_streaming.py \
  --compact_dir /data/topmed_compact_flank32 \
  --model_dir /data/topmed_models \
  --context_flank 32 \
  --model_name multitask_hybrid_v3 \
  --mixed_precision \
  --train_chromosomes 1-10 \
  --val_chromosomes 11-12 \
  --run_tag v3flank32

# Train flank=100
CUDA_VISIBLE_DEVICES=2 python scripts/train_topmed_full_streaming.py \
  --compact_dir /data/topmed_compact_flank100 \
  --model_dir /data/topmed_models \
  --context_flank 100 \
  --model_name multitask_hybrid_v3 \
  --mixed_precision \
  --train_chromosomes 1-10 \
  --val_chromosomes 11-12 \
  --run_tag v3flank100

# Evaluate all three on the test set
python scripts/recompute_region_flank_auc_from_ucsc.py \
  --compact_dir /data/topmed_compact \
  --flanks 16,32,100 \
  --output_dir /data/evaluation_results/flank_ablation
```

---

## Troubleshooting

### Training crashes with "OOM" (out of memory)

Reduce the batch size:

```bash
# Add this flag to your training command:
--batch_size 32
# Or even:
--batch_size 16
```

If the crash happens at the start of training (before any epoch completes), it is likely the model or a single large batch does not fit in GPU memory. If it happens mid-training, it might be a transient allocation spike; try setting `TF_GPU_ALLOCATOR=cuda_malloc_async` in your environment.

### Training loss is NaN from the first epoch

This usually means learning rate is too high or there is an issue with the input data. Try:

```bash
# Lower learning rate
--learning_rate 0.0001
# Or even smaller:
--learning_rate 0.00003
```

Also verify that your input data does not contain NaN or Inf values:

```python
import numpy as np
data = np.load('chr22_flank32.npz')
print('NaN in sequences:', np.any(np.isnan(data['sequences'])))
print('NaN in labels:', np.any(np.isnan(data['snp_labels'])))
```

### Preprocessing is very slow

The preprocessing script is CPU-bound and single-threaded per chromosome. To speed it up:

1. Run multiple chromosomes in parallel (each on a separate process or machine).
2. Use a faster disk (SSD rather than spinning HDD).
3. Pre-decompress the MAF files to avoid gzip overhead during preprocessing.

### The model checkpoint file does not exist after training

If training exits early (e.g., due to a crash before the first validation checkpoint is saved), there may be no checkpoint file. Check whether the training ran for at least 2 epochs by looking at the log file. If it exited after epoch 1, there is no validation checkpoint yet and only the last-epoch checkpoint (if that was enabled) may exist.

### git push fails with "Permission denied (publickey)"

If you are on a headless server and your SSH key is not registered on GitHub, use HTTPS for the remote URL instead:

```bash
git remote set-url origin https://github.com/DongjoonLim/GraphyloVar.git
git push origin main
```

This will prompt for your GitHub username and password (or personal access token if you have 2FA enabled).

### The phylogenetic tree fails to load

If you see an error like "FileNotFoundError: mammalian_tree.nwk", it means the package data was not installed correctly. Run:

```bash
pip install -e .
```

from the repository root. The `-e` flag ensures that the package data files (including the NEWICK tree) are accessible from the source directory without copying them.

---

## Frequently asked questions

**Q: Can I use GraphyloVar on a single variant without preprocessing the whole genome?**

Yes, but you need to write a small wrapper. The key steps are:

1. Extract a 65 bp window from the UCSC MAF file for your variant's position, for all 58 species.
2. Encode each species' sequence as a one-hot numpy array of shape [65, 5].
3. Stack all 58 arrays to get shape [115, 65, 5] (pad ancestral nodes with zeros).
4. Load the model and call `model.predict(input[np.newaxis, ...])[1]` to get the SNP probability.

A convenience function for this workflow is planned for a future release.

**Q: How do I interpret the snp_prob score for clinical variant prioritization?**

A high snp_prob means the model thinks this is a tolerant position that commonly varies in the population, so the variant is likely benign. A low snp_prob means the model thinks this is a constrained position, so the variant may be harmful.

To use GraphyloVar as a deleteriousness score (higher = more harmful, matching the convention of CADD and PhyloP), compute `1 - snp_prob`.

GraphyloVar is designed as a complementary tool, not a standalone clinical predictor. In practice, you should combine it with other scores (e.g., using the z-score ensemble with CADD as described above) for better performance.

**Q: What is the difference between the additive importance and the LOO perturbation analysis?**

Additive importance evaluates each species in isolation: "How much does this species alone contribute?" LOO (leave-one-out) evaluates each species in the context of all others: "How much worse does the model get if we remove this species while keeping all 57 others?"

These are complementary measures. Additive importance is dominated by human (which has by far the most standalone signal). LOO importance tends to be near zero for most species because they are redundant given the full alignment. Additive importance is more interpretable for understanding which species carry the most evolutionary information.

**Q: Why does the model use 65 bp (flank=32) and not a larger window?**

This is justified by the ablation study described above. The flank=32 model achieves higher zero-shot AUROC than flank=100 on all genomic region categories. Non-coding regulatory variants have their functional effects determined primarily by their immediate local sequence context (transcription factor binding sites are typically 6 to 20 bp long), so a 65 bp window captures the relevant context without introducing noise from distant positions.

**Q: What GPU do I need to run inference only?**

For inference (scoring variants with a trained model), you need a GPU with at least 8 GB VRAM. The model can also run on CPU but will be approximately 50 to 100 times slower. For genome-wide scoring (~149 million variants), GPU is strongly recommended.

**Q: How do I get dbGaP access to the TOPMed data?**

Apply at https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000964. You need an institutional affiliation, an IRB protocol number (or exemption), and a data access request describing your intended use. Approval typically takes a few weeks to a few months. The allele frequency annotations are in the per-chromosome VCF files in the `Freeze10` release.

**Q: Can I add more species to the model?**

Yes, but it requires retraining. You would need to:

1. Obtain aligned sequences for the new species (from the UCSC alignment or from your own alignment).
2. Add the new species to the phylogenetic tree in `graphylovar/data/mammalian_tree.nwk`.
3. Rebuild the GCN adjacency matrix in `graphylovar/phylogeny.py`.
4. Retrain the model from scratch.

The architecture naturally scales to any number of species because the Transformer encoder and GCN are parameterized by the embedding dimension and the tree structure, not by the number of species directly.

**Q: How do I reproduce the DeLong significance test results?**

```bash
python scripts/delong_pairwise_significance.py \
  --scores_a scores_graphylovar_chr13-22.tsv \
  --scores_b scores_cadd_chr13-22.tsv \
  --labels topmed_snp_labels_chr13-22.tsv \
  --method delong
```

This computes the z-statistic and p-value for the DeLong test comparing two AUROCs on the same set of variants.

**Q: What does "common vs. rare variant discrimination" mean exactly?**

The TOPMed dataset contains SNVs with a range of allele frequencies. We create a binary classification task: variants with minor allele frequency above 1% are labeled "common" (likely benign), and variants with minor allele frequency below 0.1% are labeled "rare" (potentially harmful). The model is evaluated on its ability to rank the common variants above the rare variants. This setup is motivated by the fact that harmful variants are more likely to be rare (because natural selection removes them from the population over generations).

**Q: How were the MPRA benchmark datasets selected?**

We used all publicly available MPRA datasets with sufficient variant coverage to compute a meaningful AUROC. The 13 datasets come from the Kircher et al. (2019), Griesemer et al. (2021), and Abell et al. (2022) studies, covering enhancer activity in various cell types. GraphyloVar achieves the highest AUROC on all 13 datasets after fine-tuning.

---

## Data availability

- **TOPMed whole-genome sequencing data**: dbGaP accession phs000964
- **UCSC 100-way vertebrate alignments**: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz100way/
- **Ancestral sequence reconstruction (Ancestors1.0)**: https://ancestors1.cs.mcgill.ca/
- **Trained model weights and code**: https://github.com/DongjoonLim/GraphyloVar

---

## Citation

Lim D. and Blanchette M. (2025). Predicting the impact of non-coding mutations using a multi-species sequence model. Under review at *Bioinformatics* (manuscript ID: BIOINF-2025-2871).

---

## License

MIT License.
