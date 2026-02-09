# GraphyloVar: A Deep Learning Framework for Predicting the Functional Impact of Non-Coding Variants Using Multi-Species Evolutionary Graphs

## ⚠️ Data Availability Note

**Please read before cloning:**
We are currently in the process of hosting the full-scale Multi-Alignment Format (MAF) files, as well as the pre-split **Training, Validation, and Test datasets**.

Because these datasets are extremely large (terabyte scale) and we are operating with limited upload bandwidth at our laboratory, this process is taking some time. We are working to make a persistent public link available as soon as possible. 

We apologize all the inconveniences from this problem.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow 2.5](https://img.shields.io/badge/tensorflow-2.5-orange.svg)](https://www.tensorflow.org/)
[![GitHub issues](https://img.shields.io/github/issues/DongjoonLim/GraphyloVar.svg)](https://github.com/DongjoonLim/GraphyloVar/issues)
[![GitHub stars](https://img.shields.io/github/stars/DongjoonLim/GraphyloVar.svg)](https://github.com/DongjoonLim/GraphyloVar/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/DongjoonLim/GraphyloVar.svg)](https://github.com/DongjoonLim/GraphyloVar/network)

## Introduction and Motivation

**GraphyloVar** is an advanced deep learning framework specifically designed for extracting deep insights from evolutionary genomic data, with a primary focus on predicting the functional impacts of non-coding mutations. Non-coding regions of the genome, which constitute the vast majority of DNA in eukaryotes, have historically been understudied compared to coding regions. However, recent advances in genomics have highlighted their crucial roles in regulating gene expression, chromatin structure, and overall cellular function. Mutations in these regions can lead to diseases, evolutionary adaptations, or neutral variations, but predicting their effects remains challenging due to the lack of clear functional annotations.

This tool addresses this gap by leveraging multiple alignment format (MAF) files from multi-species genome alignments and ancestral sequence reconstructions. By transforming linear sequence alignments into graph-structured data that incorporates phylogenetic relationships, GraphyloVar enables the modeling of evolutionary histories in a more nuanced way. The core neural network architecture combines Transformer layers for capturing long-range sequential dependencies within alignments and Graph Convolutional Networks (GCNs) for propagating information across the phylogenetic tree topology. This hybrid approach allows the model to learn complex patterns of conservation, divergence, and functional constraint across species.

The motivation behind GraphyloVar stems from the limitations of existing tools like PhyloP or GERP, which rely on statistical measures of conservation but do not incorporate deep learning to model non-linear interactions or context-specific effects. By using phylogeny aware deep learning model for variant comparison (reference vs. alternate alleles), GraphyloVar provides log-likelihood ratios that quantify the potential disruptiveness of mutations, making it valuable for applications in precision medicine, evolutionary biology, and variant prioritization in genome-wide association studies (GWAS).

Key features include:
- **Graph-Based Representation**: Converts MAF alignments into graphs where nodes represent species/ancestors and edges reflect phylogenetic distances.
- **Siamese Architecture**: Compares reference and alternate sequences to predict variant impacts.
- **Scalability**: Handles large-scale genomic data with efficient preprocessing and training pipelines.
- **Interpretability**: Outputs include attention maps from Transformers and node activations from GCNs for post-hoc analysis.
- **Extensibility**: Modular design allows integration with other omics data (e.g., epigenomics).

This repository provides all the necessary code, documentation, and examples to get started with GraphyloVar, from data preparation to model deployment.

## Table of Contents

- [Introduction and Motivation](#introduction-and-motivation)
- [Data Availability Note](#data-availability-note)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Data Preparation](#data-preparation)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Training the Model](#training-the-model)
- [Prediction and Inference](#prediction-and-inference)
- [Example Workflow](#example-workflow)
- [Architecture Overview](#architecture-overview)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Performance Evaluation](#performance-evaluation)
- [Troubleshooting](#troubleshooting)
- [Contributing Guidelines](#contributing-guidelines)
- [Related Work and Comparisons](#related-work-and-comparisons)
- [Future Plans](#future-plans)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Data Availability Note

Due to the enormous size of full-scale genomic datasets (often in the terabyte range), we are in the process of hosting the complete Multi-Alignment Format (MAF) files and pre-split Training/Validation/Test datasets on a dedicated server. These include 100-way vertebrate alignments lifted over to hg38 coordinates, with ancestral reconstructions inferred using tools like Ortheus or similar. Bandwidth and storage constraints mean that the upload is ongoing—please check the [releases page](https://github.com/DongjoonLim/GraphyloVar/releases) periodically for download links once they become available.

In the meantime, for testing and small-scale experiments, you can download the Boreoeutherian MAF alignments directly from the [Boreoeutherian Repository](http://repo.cs.mcgill.ca/PUB/blanchem/Boreoeutherian/). These files cover a subset of mammalian species and are sufficient for chromosomes like chr21 or chr22. Note that these alignments may require liftover to hg38 if not already in that assembly. For ancestral sequences, if not provided, you can use tools like EPA-ng or RAxML for reconstruction, but we recommend using the pre-computed ones when available.

If you encounter issues with data access, feel free to open an issue on GitHub, and we'll provide guidance or smaller sample datasets. Sample BED files with regions of interest (e.g., non-coding variants from ClinVar or ENCODE functional elements) are included in the `data/` directory for immediate use.

## Features

GraphyloVar offers a comprehensive set of features tailored for genomics researchers:

- **Multi-Species Alignment Processing**: Supports parsing of MAF files with up to 100 species, including ancestral nodes.
- **Graph Construction**: Automatically builds phylogenetic graphs from alignments, with nodes encoding one-hot sequences and edges weighted by branch lengths (if available).
- **Deep Learning Models**:
  - Base model: CNN for local features, GCN for graph propagation, LSTM/Transformer for sequence modeling.
  - Siamese variants: For pairwise comparison of alleles, outputting delta scores.
- **Data Augmentation**: Includes reverse complement augmentation to handle strand-specific effects.
- **Prediction Outputs**: Log-likelihood ratios, probability scores for functionality, and variant classification (e.g., benign vs. pathogenic).
- **Visualization Tools**: Scripts to generate phylogenetic trees, attention heatmaps, and variant impact plots using libraries like NetworkX and Matplotlib.
- **Modular Scripts**: Separate modules for parsing, preprocessing, training, and inference, allowing easy customization.
- **GPU Acceleration**: Fully compatible with TensorFlow GPU for faster training on large datasets.
- **Error Handling and Logging**: Built-in tqdm progress bars, logging to files, and robust error checking for large-scale runs.

These features make GraphyloVar suitable for both academic research and integration into bioinformatics pipelines.

## Prerequisites

To run GraphyloVar effectively, ensure your system meets the following requirements:

- **Operating System**: Linux or macOS (Windows supported via WSL, but GPU acceleration may require additional setup).
- **Hardware**:
  - CPU: Multi-core processor (at least 4 cores recommended for preprocessing).
  - RAM: Minimum 16GB; 64GB+ for full chromosome training.
  - GPU: NVIDIA GPU with CUDA 11.0+ and at least 8GB VRAM (e.g., RTX 2080) for efficient training. CPU fallback is available but significantly slower.
- **Software**:
  - Python 3.8 or higher.
  - Conda (Anaconda or Miniconda) for environment management.
  - Git for cloning the repository.
- **Data Dependencies**:
  - Genome alignments in .maf format (e.g., from UCSC or Boreoeutherian repo).
  - BED files specifying regions of interest (e.g., variants or functional elements).
  - Optional: BigWig files for conservation scores (e.g., PhyloP) for comparison.
- **Network Access**: Required for downloading dependencies and data; no internet needed during runtime.

If you're new to genomics tools, familiarize yourself with concepts like MAF files, phylogenetic trees, and variant calling formats (VCF/BED).

## Installation

Follow these detailed steps to set up GraphyloVar on your local machine:

1. **Clone the Repository**:
   Open a terminal and run:
   ```
   git clone https://github.com/DongjoonLim/GraphyloVar.git
   cd GraphyloVar
   ```
   This will download all source code, example data, and documentation.

2. **Create Directories**:
   Create folders for data and models to keep things organized:
   ```
   mkdir -p data Models src processed
   ```
   - `data/`: For raw MAF and BED files.
   - `Models/`: For saved trained models.
   - `src/`: For Python scripts (already included).
   - `processed/`: For preprocessed .npy files.

3. **Set Up the Conda Environment**:
   Use the provided `environment.yml` to create a reproducible environment:
   ```
   conda env create -f environment.yml
   conda activate graphylovar
   ```
   This installs Python 3.8, TensorFlow 2.5.0, Spektral for GCNs, and other dependencies like NumPy, Pandas, Scikit-learn, Biopython, tqdm, pyBigWig, NetworkX, and Matplotlib.

4. **Install Additional Pip Packages (if needed)**:
   Some packages might require manual installation due to Conda channel issues:
   ```
   pip install focal_loss spektral tensorflow==2.5.0 numpy==1.20.3 pandas==1.3.4 pyBigWig
   ```
   Note: Stick to TensorFlow 2.5.0 for compatibility; upgrading may require code changes for deprecated APIs.

5. **Verify Installation**:
   Run a simple test:
   ```
   python -c "import tensorflow as tf; print(tf.__version__); print('GPU available:' if tf.test.is_gpu_available() else 'No GPU')"
   ```
   This should output "2.5.0" and confirm GPU if set up correctly.

6. **Optional: CUDA Setup**:
   If using GPU, ensure CUDA and cuDNN are installed. Refer to [TensorFlow GPU guide](https://www.tensorflow.org/install/gpu) for details.

Installation should take 5-10 minutes. If issues arise, check the [Troubleshooting](#troubleshooting) section.


## Data Preparation

Preparing data is crucial for GraphyloVar, as it involves handling large alignment files.

1. **Download Alignments**:
   - Visit the [Boreoeutherian Repository](http://repo.cs.mcgill.ca/PUB/blanchem/Boreoeutherian/).
   - Download .maf files for desired chromosomes (e.g., chr21.anc.maf) to `data/`.
   - These files include sequences from ~30 vertebrate species plus ancestors.

2. **Prepare BED Files**:
   - BED files define regions (chr start end label), where label could be 0/1 for benign/pathogenic or conservation scores.
   - Use provided `data/example.bed` or create your own, e.g., from ClinVar variants or ENCODE enhancers.
   - Format: Tab-separated, no header; ensure coordinates are 0-based start, 1-based end.

3. **Ancestral Reconstructions**:
   - If MAFs lack ancestors, use tools like PRANK or MAFFT with ancestral inference.
   - GraphyloVar expects keys like '_HP' (Human-Primate ancestor) in the alignment dictionary.

4. **Storage Considerations**:
   - MAFs can be >100GB per chromosome; use SSD for faster I/O.
   - Compress unused files with gzip to save space.

Once data is in place, proceed to preprocessing.

## Preprocessing Pipeline

Preprocessing converts raw alignments into model-ready graph inputs. This step is computationally intensive but can be parallelized per chromosome.

1. **Parse MAF to Serialized Format**:
   - Script: `src/parserPreprocess.py`
   - Purpose: Reads MAF, ungaps alignments, pads sequences, and saves as Pickle/NPY.
   - Example Command:
     ```
     python src/parserPreprocess.py --chrom 21 --maf_path data/chr21.anc.maf --output processed/seqDictPad_chr21.pkl
     ```
   - Details: Handles 115 species/ancestors (listed in code). Ungaps to align with human reference. Output is a dict of lists for each species.

2. **Generate Graphs from BED Regions**:
   - Script: `src/preprocess_graphs.py`
   - Purpose: Extracts windows around BED positions, one-hot encodes (A=0, C=1, G=2, T=3, N/-=4), includes reverse complements, saves as NPY arrays.
   - Example:
     ```
     python src/preprocess_graphs.py --bed data/example_chr20.bed --chrom 20 --output_x processed/example_X_chr20.npy --output_y processed/example_y_chr20.npy --context 100
     ```
   - Details: Context=100 means 100bp flanks. Shape: (samples, 115 species, 201*2 bases).

3. **Augment with Reverse Complements**:
   - Script: `src/preprocessRevComp.py`
   - Purpose: Doubles data by adding RC, useful for strand-agnostic modeling.
   - Example:
     ```
     python src/preprocessRevComp.py --tf CTCF --celltype K562
     ```
   - Details: Concatenates original and RC along sequence dimension.

4. **General Preprocessing for Training Data**:
   - Script: `src/preprocess.py`
   - Purpose: Samples mutated/conserved sites, creates balanced dataset.
   - Example:
     ```
     python src/preprocess.py --chrom 21 --input_dir data/ --output_dir processed/ --flank_size 100 --num_mutated 100000 --num_conserved 400000
     ```
   - Details: Identifies differences between human and ancestor, encodes targets as one-hot + mutation flag.

Tips: Run on small chromosomes first. Use `mmap_mode='r'` for large NPY to save memory.

## Training the Model

Training involves merging data and fitting models. Use GPUs for speed.

1. **Merge Chromosomal Data**:
   - Manually or via script: Concatenate NPY files across chromosomes.
   - Example Python snippet:
     ```python
     import numpy as np
     chromosomes = list(range(1, 23)) + ['X', 'Y']
     X_train = np.concatenate([np.load(f"processed/X_train_chr{c}.npy") for c in chromosomes], axis=0)
     y_train = np.concatenate([np.load(f"processed/y_train_chr{c}.npy") for c in chromosomes], axis=0)
     np.save("processed/full_X_train.npy", X_train)
     np.save("processed/full_y_train.npy", y_train)
     ```

2. **Train Base Model**:
   - Script: `src/train.py`
   - Example:
     ```
     python src/train.py --x_train processed/X_train_chr21.npy --y_train processed/y_train_chr21.npy --x_val processed/X_val_chr21.npy --y_val processed/y_val_chr21.npy --output_dir Models/graphylo_chr21 --epochs 50 --batch_size 64
     ```
   - Details: Uses multi-output loss (categorical for allele, binary for polymorphism). Saves best model via checkpoint.

3. **Train Siamese Models**:
   - For Graphylo:
     ```
     python src/train_graphylo_siamese.py processed/full_X.npy Models/graphylo processed/full_y.npy 0 32 128 64 --epochs 50 --batch_size 32
     ```
   - For GraphyloVar (advanced):
     ```
     python src/train_graphylovar_siamese.py processed/full_X.npy Models/graphylovar processed/full_y.npy 0 32 128 64 --epochs 50 --batch_size 32
     ```
   - Details: Siamese compares ref/alt, uses focal loss for imbalance. Hyperparams: filters, hidden, graph hidden.

Monitor with TensorBoard: Add `--log_dir logs/` to commands.

## Prediction and Inference

Once trained, use models for predictions on new variants.

1. **Load Model**:
   ```python
   import tensorflow as tf
   from focal_loss import BinaryFocalLoss  # If used

   model = tf.keras.models.load_model('Models/graphylo_chr21', custom_objects={'BinaryFocalLoss': BinaryFocalLoss})
   ```

2. **Prepare Query Data**:
   - Use `preprocess_graphs.py` on a BED of query positions.

3. **Predict**:
   ```python
   import numpy as np

   X_query = np.load('processed/query_X.npy')
   predictions = model.predict(X_query)  # [allele_probs, poly_prob] or log-LLR for siamese
   print("Log-Likelihood Ratio (alt vs ref):", predictions)
   ```

4. **Interpretation**:
   - High LLR: Likely disruptive variant.
   - Threshold: Use ROC from validation for classification.

Batch predictions for efficiency.

## Example Workflow

Let's walk through a complete example: Predicting impacts on Chromosome 20 non-coding regions.

1. **Download Data**:
   - Get chr20.anc.maf from Boreoeutherian.
   - Create `data/query_regions.bed`: e.g.,
     ```
     chr20	1000000	1000001	0
     chr20	2000000	2000001	1
     ```

2. **Preprocess Alignment**:
   ```
   python src/parserPreprocess.py --chrom 20 --maf_path data/chr20.anc.maf --output processed/seqDictPad_chr20.pkl
   ```

3. **Generate Graphs**:
   ```
   python src/preprocess_graphs.py --bed data/query_regions.bed --chrom 20 --output_x processed/query_X.npy --output_y processed/query_y.npy
   ```

4. **Train (or Use Pre-trained)**:
   - Train on sample data or load pre-trained model.

5. **Predict**:
   Use the code above.

6. **Visualize**:
   - Plot predictions with Matplotlib.

This workflow can be scripted for automation.

## Architecture Overview

GraphyloVar's architecture is a hybrid of sequence and graph models:

```
Raw MAF Alignments
↓ (Preprocessing)
Graph Inputs: Nodes (Species Sequences: One-hot encoded with flanks + RC)
              Edges (Phylogenetic Adjacency: Binary or weighted tree structure)

Input Layer: (Batch, 115 nodes, 402 features)  # 201 bases * 2 (RC)

CNN Branch: Conv1D (filters=32, kernel=5) → MaxPool → Flatten per node
            → Captures local motifs like TF binding sites

GCN Branch: GCNConv (channels=128) on adjacency A
            → Mixes features across phylogeny, e.g., propagates conservation signals

Sequence Modeling: LSTM (units=64) or Transformer (heads=4, layers=2)
                   → Handles dependencies along the alignment

Siamese: Twin networks for ref/alt → Dense → Concat → FC → Sigmoid (impact score)

Output: Log-Likelihood Ratio (LLR) = log(P(alt|context) / P(ref|context))
```

This design allows learning from evolutionary context, outperforming linear models.

For visual: Imagine a tree where each node has a sequence vector, convolved locally, then diffused via GCN edges.

## Hyperparameter Tuning

Tune for best performance:
- Learning Rate: 0.001 default; try 1e-4 to 1e-2.
- Batch Size: 32-128; larger for GPUs.
- Epochs: 50+; use early stopping.
- Filters/Hidden: Start with 32/128, grid search.
- Use tools like Keras Tuner or Optuna.

Validate on held-out chromosomes.

## Performance Evaluation

Evaluate with metrics:
- Accuracy/AUC for classification.
- MSE for LLR regression.
- Compare to baselines (PhyloP, CADD) on datasets like HGMD or 1000 Genomes.

Example: On sample data, achieves 0.85 AUC.

## Troubleshooting

Common issues and fixes:
- **Memory Errors**: Use smaller batches or mmap for NPY. Split chromosomes.
- **TensorFlow Version Mismatch**: Pin to 2.5.0; check custom objects on load.
- **Missing Dependencies**: Rerun pip installs.
- **Data Parsing Failures**: Check MAF format; skip invalid lines.
- **GPU Not Used**: Verify with `tf.test.is_gpu_available()`.
- **Slow Preprocessing**: Parallelize with joblib or run on cluster.
- **Prediction NaNs**: Check input shapes; normalize if needed.

For more, see issues or StackOverflow.

## Contributing Guidelines

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details:
- Fork and PR.
- Follow PEP8.
- Add tests (pytest).
- Document changes.
- For major features, open issue first.

## Related Work and Comparisons

- **PhyloP/GERP**: Statistical conservation; GraphyloVar adds DL for better prediction.
- **CADD/PrimateAI**: Similar goals but focus on coding; ours is non-coding specific.
- **Graph Nets in Genomics**: Inspired by GPN-MSA but phylogeny-focused.

GraphyloVar excels in incorporating tree structure.

## Future Plans

- Integrate more species (100+).
- Add epigenomic features.
- Web interface for predictions.
- Pre-trained models on full genomes.
- Support for other assemblies (e.g., hg19).



Update with actual publication.

## License

MIT License - see [LICENSE](LICENSE) for details. Free for academic/commercial use, with attribution.
