# GraphyloVar

**GraphyloVar: Predicting the impact of non-coding variants using a multi-species sequence model**

---

## Table of Contents

- [What is GraphyloVar?](#what-is-graphylovar)
- [Why does this matter?](#why-does-this-matter)
- [How does it work?](#how-does-it-work)
- [Key results](#key-results)
- [Repository layout](#repository-layout)
- [Requirements and installation](#requirements-and-installation)
- [Data preparation](#data-preparation)
- [Training the model](#training-the-model)
- [Scoring variants](#scoring-variants)
- [Interpretability analysis](#interpretability-analysis)
- [Model architecture (detailed)](#model-architecture-detailed)
- [The 58 species used](#the-58-species-used)
- [Frequently asked questions](#frequently-asked-questions)
- [Data availability](#data-availability)
- [Citation](#citation)
- [License](#license)

---

## What is GraphyloVar?

GraphyloVar is a deep learning model that predicts **how harmful a DNA mutation is likely to be**, specifically for mutations that fall outside of protein-coding genes (so-called *non-coding variants*). These variants make up the vast majority of genetic differences between people but are much harder to interpret than coding mutations.

The key insight behind GraphyloVar is that evolution is a powerful filter. If a DNA position has been conserved across many mammalian species over tens of millions of years, it is probably functionally important — and a mutation at that position is more likely to be harmful. GraphyloVar formalizes this intuition by:

1. Looking at a short stretch of DNA (~65 base pairs) centered on the variant, **simultaneously in 58 different mammalian species**.
2. Using the **phylogenetic tree** (the evolutionary family tree of those 58 species) to weight contributions from different species intelligently.
3. Pre-training the model to predict **population-level allele frequencies** from the TOPMed whole-genome sequencing cohort — the model learns that rare variants (found in very few people) are more likely to be harmful.

---

## Why does this matter?

Genome-wide association studies and clinical sequencing produce millions of candidate variants. The vast majority are harmless; a small fraction cause disease. Prioritizing which variants to study further is a central bottleneck in genomics. Most existing tools either:

- Only look at a single species' DNA (missing the evolutionary context), or
- Use conservation scores computed independently per position (missing local sequence patterns).

GraphyloVar is the first model to **jointly** process multi-species alignments through a phylogenetic graph neural network and transformer-based sequence encoders, allowing it to capture both local sequence features and deep evolutionary signals at once.

---

## How does it work?

At a high level:

1. **Input**: A 65 bp window centered on the variant of interest, retrieved for each of the 58 placental mammalian species from the UCSC 100-way vertebrate multiple sequence alignment.

2. **Sequence encoding**: Each species' sequence is independently processed by a small Transformer encoder (2 attention layers per DNA strand, forward and reverse complement processed separately). This extracts a 64-dimensional feature vector for each species summarizing the local sequence context.

3. **Phylogenetic aggregation**: The per-species feature vectors are fed into a two-layer Graph Convolutional Network (GCN). The graph is the mammalian phylogenetic tree (58 extant species + 57 reconstructed ancestral nodes = 115 nodes total). The GCN propagates information between species according to their evolutionary relationships, so closely related species share information while distantly related ones influence each other less.

4. **Prediction**: The GCN output is flattened and passed through a shared fully connected layer, then split into two prediction heads:
   - **Allele frequency head**: Predicts which nucleotide is most likely at this position in the human population (softmax over A, C, G, T, or gap). Trained with categorical cross-entropy loss.
   - **SNP probability head**: Predicts whether this position is a polymorphic SNP (sigmoid). Trained with binary cross-entropy loss.

5. **Variant scoring**: At inference time, the SNP probability score is used as the variant impact score. A higher score means the model predicts the position is more tolerant of variation (i.e., likely benign); a lower score indicates a likely deleterious variant.

---

## Key results

- **AUROC 0.6246** on ~149 million held-out TOPMed SNVs (chromosomes 13-22), discriminating common (benign) from rare (potentially harmful) variants
- **Z-score ensemble** with CADD reaches **AUROC 0.6442** (+0.020 improvement, p < 10⁻¹⁵), showing that GraphyloVar captures complementary information to existing tools
- **Fine-tuned GraphyloVar achieves the highest AUROC on all 13 MPRA benchmark datasets** tested, outperforming CADD, PhyloP, PhastCons, GPN-MSA, and Enformer
- **Spearman correlation with minor allele frequency**: GraphyloVar 0.164, GPN-MSA 0.143, Enformer 0.131, PhyloP 0.099, CADD 0.081, PhastCons 0.058
- **Interpretability**: Human sequence provides the largest signal, followed by four diverged primates (Orangutan, Green monkey, Gorilla, Gibbon). Chimpanzee ranks 11th despite being the closest relative — because its near-identical sequence (~98.7% identity with human) provides almost no additional information beyond the human row alone.

---

## Repository layout

```
graphylovar/            Python package: model builders, phylogeny utilities,
│                       data loading, training helpers
│   models.py           All model architectures (multitask_hybrid_v3, v4, etc.)
│   phylogeny.py        Phylogenetic tree parsing and GCN adjacency matrix
│   training.py         Training loop and loss utilities
│   topmed.py           TOPMed data loading
│   model_io.py         Checkpoint loading utilities
│
scripts/                Entry-point scripts for training, preprocessing, inference
│   preprocess_topmed_full_compact.py   Convert raw UCSC MAF → compact per-chrom format
│   train_topmed_full_streaming.py      Main training script (streaming, multi-GPU)
│   predict.py                          Score variants chromosome by chromosome
│   additive_species_importance.py      Interpretability: additive per-species importance
│   per_species_all58_perturbation.py   LOO perturbation analysis (all 58 species)
│   recompute_region_flank_auc_from_ucsc.py   Region-level AUC evaluation
│   ...                                 Additional analysis and figure scripts
│
configs/                Default YAML configuration file for training
latex/                  Manuscript source (LaTeX) and submission figures
│   graphylovar_submission/   Final camera-ready submission directory
│       oup-authoring-template.tex   Main manuscript
│       figures/                     All figures referenced in the paper
│
figures/                Intermediate analysis figures and CSV result files
```

---

## Requirements and installation

### Prerequisites

- **Python 3.9 or higher**
- **TensorFlow 2.10 or higher** (GPU support strongly recommended)
- **CUDA 11.2+** and **cuDNN 8.1+** (for GPU training)
- **~20 GB RAM** minimum for data loading; 64+ GB recommended for streaming the full TOPMed dataset
- **~2 TB disk space** for the preprocessed compact alignment data (or ~14 TB for raw MAF files)

### Installation via pip

```bash
# 1. Clone the repository
git clone https://github.com/DongjoonLim/GraphyloVar.git
cd GraphyloVar

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the graphylovar package in development mode
pip install -e .
```

### Installation via Conda (recommended for reproducibility)

```bash
# 1. Clone the repository
git clone https://github.com/DongjoonLim/GraphyloVar.git
cd GraphyloVar

# 2. Create and activate the conda environment (installs all dependencies)
conda env create -f environment.yml
conda activate graphylo

# 3. Install the graphylovar package
pip install -e .
```

### Verifying your installation

```bash
python -c "import graphylovar; print('GraphyloVar installed successfully')"
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import tensorflow as tf; print('GPUs available:', tf.config.list_physical_devices('GPU'))"
```

---

## Data preparation

GraphyloVar requires two input data sources:

### 1. UCSC 100-way vertebrate multiple sequence alignment (MAF files)

Download the per-chromosome MAF files from UCSC:

```
https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz100way/
```

Each file is named `chr1.maf.gz`, `chr2.maf.gz`, etc. You need chromosomes 1-22 at minimum (X/Y optional). These are large files (~20-100 GB each uncompressed).

### 2. TOPMed allele frequency labels

TOPMed whole-genome sequencing variant data is available via dbGaP (accession **phs000964**). Access requires dbGaP approval. Once you have the VCF files with allele frequency annotations, the preprocessing script handles the rest.

### Preprocessing: MAF → compact format

The raw MAF files are large and slow to read. We preprocess them into a compact per-variant format:

```bash
python scripts/preprocess_topmed_full_compact.py \
  --alignment_dir /path/to/ucsc_maf_files \
  --topmed_vcf_dir /path/to/topmed_vcfs \
  --output_dir topmed_compact_full \
  --chromosomes 1-22 \
  --context_flank 32
```

**Arguments:**
- `--alignment_dir`: Directory containing the UCSC MAF files (one per chromosome)
- `--topmed_vcf_dir`: Directory containing TOPMed VCF files with AF annotations
- `--output_dir`: Where to write the compact `.npz` files (one per chromosome)
- `--chromosomes`: Which chromosomes to process (e.g., `1-22` or `1,2,3`)
- `--context_flank`: Number of flanking base pairs on each side of the variant (32 = 65 bp total window; default for the main model)

This step takes several hours per chromosome on a single CPU core. It is embarrassingly parallel — you can run multiple chromosomes simultaneously on different machines.

**Output structure:**

```
topmed_compact_full/
    chr1_flank32.npz        Per-variant arrays: sequences for all 58 species,
    chr2_flank32.npz        allele frequency labels, SNP labels, positions
    ...
    chr22_flank32.npz
```

---

## Training the model

### Basic training (recommended defaults)

```bash
python scripts/train_topmed_full_streaming.py \
  --compact_dir topmed_compact_full \
  --model_dir topmed_models \
  --context_flank 32 \
  --model_name multitask_hybrid_v3 \
  --mixed_precision \
  --train_chromosomes 1-10 \
  --val_chromosomes 11-12
```

**What the arguments mean:**
- `--compact_dir`: Directory with preprocessed `.npz` files (output of preprocessing)
- `--model_dir`: Where to save checkpoints and training logs
- `--context_flank 32`: Use a 65 bp window (32 bp on each side of the variant + 1 center)
- `--model_name multitask_hybrid_v3`: The main published architecture (Transformer + GCN)
- `--mixed_precision`: Use float16 on GPU for faster training and lower memory usage (highly recommended if your GPU supports it)
- `--train_chromosomes 1-10`: Use chromosomes 1-10 for training
- `--val_chromosomes 11-12`: Use chromosomes 11-12 for validation / early stopping

The test set (chromosomes 13-22) is never touched during training.

### Additional training options

```bash
python scripts/train_topmed_full_streaming.py \
  --compact_dir topmed_compact_full \
  --model_dir topmed_models \
  --context_flank 32 \
  --model_name multitask_hybrid_v3 \
  --mixed_precision \
  --train_chromosomes 1-10 \
  --val_chromosomes 11-12 \
  --batch_size 64 \              # Variants per gradient step (default: 64)
  --steps_per_epoch 5000 \       # Gradient steps per epoch
  --validation_steps 1000 \      # Steps for validation loss estimate
  --epochs 9999 \                # Max epochs (early stopping controls actual stopping)
  --patience 30 \                # Stop if no improvement for 30 epochs
  --learning_rate 0.0003 \       # Initial learning rate
  --binary_loss_weight 1.5 \     # Weight on the SNP probability head vs. allele frequency head
  --run_tag v3main               # Tag appended to checkpoint filename for easy identification
```

### Training time estimates

On a single NVIDIA Quadro RTX 6000 (24 GB VRAM):
- ~5 minutes per epoch (5000 steps × 64 batch = 320,000 variants/epoch)
- Typically converges in 20-80 epochs
- Total training time: ~2-7 hours for the main flank=32 model

### Multiple GPU support

Set `CUDA_VISIBLE_DEVICES` before launching:

```bash
# Single GPU (GPU 2)
CUDA_VISIBLE_DEVICES=2 python scripts/train_topmed_full_streaming.py ...

# Multi-GPU (GPUs 2 and 3 — uses TensorFlow MirroredStrategy automatically)
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_topmed_full_streaming.py ...
```

---

## Scoring variants

Once you have a trained model checkpoint, score variants on any chromosome:

```bash
python scripts/predict.py \
  --model_path topmed_models/checkpoint.keras \
  --data_dir topmed_compact_full \
  --chromosome 22 \
  --output_file scores_chr22.tsv
```

**Output format** (`scores_chr22.tsv`):

```
chrom   pos         ref  alt  snp_prob    af_pred_A   af_pred_C   af_pred_G   af_pred_T   af_pred_gap
chr22   16050075    A    G    0.0234      0.7821      0.0012      0.2054      0.0103      0.0010
chr22   16050115    C    T    0.7891      0.0023      0.8932      0.0912      0.0021      0.0112
...
```

- `snp_prob`: Main variant impact score. **Lower = more likely harmful** (the model predicts fewer people carry this variant). Higher = more likely benign (common polymorphism).
- `af_pred_*`: Predicted allele frequency distribution over the 5 states.

### Genome-wide scoring

To score all chromosomes in one run:

```bash
for CHR in {1..22}; do
  python scripts/predict.py \
    --model_path topmed_models/checkpoint.keras \
    --data_dir topmed_compact_full \
    --chromosome $CHR \
    --output_file scores_chr${CHR}.tsv
done
```

---

## Interpretability analysis

### Per-species additive importance

This analysis measures how much each of the 58 species contributes to the model's predictions. For each species, we present that species' sequence alone (all others set to gap) and measure the reduction in loss compared to an all-gap baseline:

```bash
python scripts/additive_species_importance.py \
  --run_tag v3flank16 \
  --samples_per_chrom 50000 \
  --chromosomes 13-22 \
  --output_dir outputs/interpretability/additive_species
```

**Interpretation of results:**

The output CSV ranks species by their standalone contribution. Key findings from the published model (v3flank16, 500,000 held-out variants):

| Rank | Species | Delta CE (additive importance) |
|------|---------|-------------------------------|
| 1 | hg38 (human) | ~7.1 |
| 2 | ponAbe2 (orangutan) | ~1.50 |
| 3 | chlSab2 (green monkey) | ~1.50 |
| 4 | gorGor3 (gorilla) | ~1.24 |
| 5 | nomLeu3 (gibbon) | ~0.62 |
| ... | ... | ... |
| 11 | panTro4 (chimpanzee) | ~0.062 |

**Why is chimpanzee ranked 11th?** Despite being the closest living relative of humans (~98.7% sequence identity), chimp provides almost no additional information to the model beyond what the human sequence already encodes. The model learns to down-weight species whose sequences are near-identical to human. More diverged primates like orangutan (~96% identity) carry distinct evolutionary signals that genuinely add new information.

### Leave-one-out (LOO) species perturbation

This is different from the additive analysis: for each species, we *remove* that one species while keeping all 57 others, and measure the increase in loss:

```bash
python scripts/per_species_all58_perturbation.py \
  --run_tag v3flank16 \
  --n_samples 2000000 \
  --output_dir outputs/interpretability/loo_perturbation
```

In the LOO analysis, most species have near-zero importance (because the other 57 species compensate). The LOO analysis is best interpreted as "which species are least redundant given all others."

---

## Model architecture (detailed)

GraphyloVar uses a Transformer + GCN architecture. Here is a detailed walkthrough:

### Step 1: Input

For a variant at position *p* on chromosome *c*, the model receives a tensor of shape **(115 species × 65 bp × 5 nucleotide states)**. The 5 states are one-hot encodings of {A, C, G, T, gap}. Only 58 rows correspond to extant species (observed sequences); the other 57 rows correspond to ancestral nodes (reconstructed by the Ancestors1.0 method), which are initialized to the gap token at inference time.

### Step 2: Per-species sequence encoding (Transformer)

Each species' 65 bp sequence is encoded independently. The encoding is done **twice**: once for the forward strand and once for the reverse complement strand. For each strand:

1. **Dense projection + GELU + BatchNorm**: The 5-dimensional one-hot input at each position is projected to a 32-dimensional embedding.
2. **Sinusoidal positional encoding**: Standard sinusoidal position embeddings are added so the Transformer knows which position in the sequence it is looking at.
3. **Two Transformer encoder layers**: Each layer has 4 attention heads with key dimension 8, followed by a feedforward sublayer. Standard pre-norm architecture.
4. **Center extraction**: Only the feature vector at position 32 (the center position, corresponding to the variant itself) is kept. This gives a 32-dim vector per strand.

The forward and reverse complement center vectors are concatenated to give a **64-dimensional representation per species**.

### Step 3: Squeeze-and-Excitation attention gate

A small Squeeze-and-Excitation (SE) block computes a scalar attention weight for each of the 58 extant species, based on its 64-dim feature vector. This allows the model to learn which species are more informative for a given variant. On average across held-out variants:
- Human (hg38): attention weight ≈ 1.00
- Orangutan (ponAbe2): ≈ 0.54
- Chimpanzee (panTro4): ≈ 0.011

### Step 4: Graph Convolutional Network (GCN)

The 64-dim per-species features (after SE weighting) are arranged as nodes in a graph. The graph structure is the mammalian phylogenetic tree with 115 nodes (58 extant + 57 ancestral). Edge weights are derived from branch lengths (evolutionary distances) in the tree.

Two GCN layers propagate information between nodes:
- **GCN Layer 1**: 64 → 32 dimensions, ReLU activation
- **GCN Layer 2**: 32 → 32 dimensions, ReLU activation

After the GCN, the 115 × 32 node feature matrix is flattened to a single 3,680-dimensional vector.

### Step 5: Prediction heads

A shared fully connected layer (128 units, ReLU) processes the 3,680-dim GCN output, then splits into two heads:
- **Nucleotide frequency head**: FC(64, ReLU) → FC(5, softmax) — predicts the population allele frequency distribution over {A, C, G, T, gap}
- **SNP probability head**: FC(64, ReLU) → FC(1, sigmoid) — predicts whether the position is polymorphic

### Architecture summary

```
Input: [115 species × 65 bp × 5 nucleotides]
         ↓  (for each of 58 extant species)
 Dense(32) + GELU + BN → SinusPE → 2× TransformerLayer(4 heads, key_dim=8)
         ↓  center position extraction
 Forward strand [32-dim] + RevComp strand [32-dim] → concat [64-dim]
         ↓  SE attention gate (per-species scalar)
 GCN Layer 1: 64→32 (ReLU) on 115-node phylogenetic tree
 GCN Layer 2: 32→32 (ReLU)
         ↓  flatten 115×32 = 3680-dim
 Shared FC(128, ReLU)
         ↓
 ┌──────────────────┐    ┌────────────────────┐
 │ Nuc head          │    │ SNP head            │
 │ FC(64,ReLU)       │    │ FC(64,ReLU)         │
 │ FC(5,softmax)     │    │ FC(1,sigmoid)       │
 │ CatCrossEntropy   │    │ BinCrossEntropy      │
 └──────────────────┘    └────────────────────┘
```

---

## The 58 species used

All species are identified by their UCSC genome browser assembly accession. The alignment covers 58 extant placental mammals (plus 57 computationally reconstructed ancestral sequences).

### Primates
- **Great Apes (5):** hg38 (human), panTro4 (chimpanzee), gorGor3 (gorilla), ponAbe2 (orangutan), nomLeu3 (gibbon)
- **Old World Monkeys (4):** rheMac3 (rhesus macaque), macFas5 (crab-eating macaque), papAnu2 (olive baboon), chlSab2 (green monkey)
- **New World Monkeys (2):** calJac3 (marmoset), saiBol1 (squirrel monkey)
- **Strepsirrhini (1):** otoGar3 (bushbaby)

### Euarchontoglires
- **Scandentia (1):** tupChi1 (Chinese tree shrew)
- **Rodentia (11):** speTri2 (squirrel), jacJac1 (lesser Egyptian jerboa), micOch1 (prairie vole), criGri1 (Chinese hamster), mesAur1 (golden hamster), mm10 (mouse), rn6 (rat), hetGla2 (naked mole-rat), cavPor3 (guinea pig), chiLan1 (chinchilla), octDeg1 (degu)
- **Lagomorpha (2):** oryCun2 (rabbit), ochPri3 (pika)

### Laurasiatheria
- **Cetartiodactyla (9):** susScr3 (pig), vicPac2 (alpaca), camFer1 (Bactrian camel), turTru2 (dolphin), orcOrc1 (killer whale), panHod1 (Tibetan antelope), bosTau8 (cow), oviAri3 (sheep), capHir1 (goat)
- **Perissodactyla (2):** equCab2 (horse), cerSim1 (white rhinoceros)
- **Carnivora (6):** felCat8 (cat), canFam3 (dog), musFur1 (ferret), ailMel1 (giant panda), odoRosDiv1 (Pacific walrus), lepWed1 (Weddell seal)
- **Chiroptera (5):** pteAle1 (black flying fox), pteVam1 (large flying fox), eptFus1 (big brown bat), myoDav1 (David's myotis), myoLuc2 (little brown bat)
- **Eulipotyphla (3):** eriEur2 (hedgehog), sorAra2 (shrew), conCri1 (star-nosed mole)

### Atlantogenata
- **Afrotheria (5):** loxAfr3 (elephant), eleEdw1 (Cape elephant shrew), triMan1 (manatee), chrAsi1 (Cape golden mole), echTel2 (lesser hedgehog tenrec)
- **Xenarthra (2):** oryAfe1 (aardvark), dasNov3 (armadillo)

Total: **58 extant species** + 57 reconstructed ancestral nodes = **115 tree nodes**

---

## Frequently asked questions

**Q: Can I use GraphyloVar on a single variant without preprocessing the whole genome?**

Yes, but it requires some scripting. You need to extract the 65 bp window for your variant from the UCSC MAF files for all 58 species, format it as a numpy array matching the compact format, and pass it to the model. A convenience function for single-variant scoring is planned for a future release.

**Q: How do I interpret the snp_prob score?**

The score is the model's predicted probability that the position is polymorphic (i.e., varies in the population). **Higher score = more likely to be a common, benign variant. Lower score = more likely to be a rare, potentially harmful variant.** When comparing to tools like CADD or PhyloP that score in the opposite direction (higher = more harmful), you should negate the snp_prob before ensembling (or use the 1 - snp_prob transformation).

**Q: What is the difference between additive importance and LOO perturbation?**

- **Additive importance**: Each species is evaluated alone (all other species set to gap). This measures the standalone contribution of each species' sequence to the model's loss. Human dominates this analysis because human sequence alone is highly informative.
- **LOO (leave-one-out) perturbation**: Each species is removed while all 57 others are present. This measures how much is lost when one species is removed from a complete alignment. Because species are redundant with their close relatives, most species show near-zero LOO importance.

**Q: Why does the model use only 65 bp? Isn't that very short?**

The 65 bp window is a deliberate design choice. Our ablation study (Supplementary Table S1) shows that flank=32 (65 bp) achieves the highest zero-shot AUROC (0.625) compared to flank=16 (33 bp, AUROC 0.622) and flank=100 (201 bp, AUROC 0.617). Longer windows add noise — the variant's functional impact is mostly determined by its immediate local sequence context and its cross-species conservation, both of which are captured within 65 bp.

**Q: What GPU do I need?**

Training requires a GPU with at least 16 GB VRAM for the default batch size of 64. We used NVIDIA Quadro RTX 6000 (24 GB). Inference can be done on smaller GPUs or even CPU (but will be much slower). For inference only, 8 GB VRAM is sufficient.

**Q: How do I reproduce the AUROC numbers from the paper?**

Use `scripts/recompute_region_flank_auc_from_ucsc.py` with the trained model checkpoint and the preprocessed data for chromosomes 13-22 (the held-out test set). The script handles annotation of variants into genomic regions (coding, 3'UTR, cCREs, TEs) using UCSC annotation tracks.

---

## Data availability

- **TOPMed whole-genome sequencing data**: dbGaP accession phs000964
- **UCSC 100-way vertebrate alignments**: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz100way/
- **Ancestral sequence reconstruction**: Ancestors1.0 (Blanchette et al. 2004)
- **Trained model weights and code**: https://github.com/DongjoonLim/GraphyloVar

---

## Citation

Lim D. and Blanchette M. (2025). Predicting the impact of non-coding mutations using a multi-species sequence model. Under review at *Bioinformatics* (manuscript ID: BIOINF-2025-2871).

---

## License

MIT License.
