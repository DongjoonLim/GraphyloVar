# Graphylo: Deep Learning for Evolutionary Genome Analysis

Graphylo is a deep learning model designed to extract meaningful insights from evolutionary genome data using aligned sequences and ancestral reconstructions. This guide provides step-by-step instructions for setup, data preparation, training, and prediction.

![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.5-orange)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Prediction](#prediction)
- [Example Workflow](#example-workflow)
- [License](#license)

## Introduction
Graphylo leverages evolutionary sequence alignments (in .maf format) and ancestral genome reconstructions to identify functional genomic elements. It processes alignment data into graph-structured inputs for a Siamese neural network architecture combining CNNs, GCNs, and LSTMs.

## Prerequisites
- Anaconda3 for package management
- NVIDIA GPU (recommended for training)
- Genome alignment data (.maf files) from [Boreoeutherian Repository](http://repo.cs.mcgill.ca/PUB/blanchem/Boreoeutherian/)
- BED files defining regions of interest in hg38 coordinates (see [example](data/example.bed))

## Installation
### 1. Directory Setup
```bash
git clone https://github.com/DongjoonLim/graphylo.git
mkdir data Models
```

### 2. Create Environment (ANACONDA HAS TO BE INSTALLED FIRST!)
```bash
$ source .bashrc
conda env create -f environment.yml
conda activate graphylo
```

### 3. Install Additional Packages
```bash
pip install focal_loss pandas==1.3.4 spektral tensorflow==2.5.0 numpy==1.20.3 pyBigWig
```

## Data Preparation

### 1. Download Alignment Data
Download .maf files to the `data` directory from the [Boreoeutherian Repository](http://repo.cs.mcgill.ca/PUB/blanchem/Boreoeutherian/).

## Preprocessing

### 1. Convert MAF to NPY
```bash
python3 parserPreprocess.py
```
Output: `.pkl` file containing processed alignment data.

### 2. Generate Training Data
Prepare BED files with hg38 coordinates (format: `chr<number> start end label`). Example:
```bed
chr1 1000 1001 0
chr1 2000 2001 1
```

Process training data:
```bash
python3 preprocess_graphs.py data/example_chr20.bed 20 data/example_X_chr20.npy data/example_y_chr20.npy
```

### 3. (Optional) Reverse Complement for RNA Data
```bash
python3 preprocessRevComp.py
```

## Training

### 1. Merge Chromosomal Data
Combine all chromosomal data into single datasets and save them:
```python
import numpy as np

# For features
X = np.concatenate([np.load(f"data/example_X_chr{i}.npy") for i in range(1,23)], axis=0)

# For labels
y = np.concatenate([np.load(f"data/example_y_chr{i}.npy") for i in range(1,23)], axis=0)
```

### 2. Start Training
Run train_graphylo_siamese.py to train graphylo with the data you have preprocessed previously. python3 train_graphylo_siamese.py data_path output_model_path target_path gpu numberOfCNNFilters FCNNhiddenUnits GCNhiddenUnits
```bash
python3 train_graphylo_siamese.py data/example_X.npy Models/model data/example_y.npy 3 32 32 32       
```

## Prediction
Load the trained model and make predictions:
```python
import tensorflow as tf
from focal_loss import BinaryFocalLoss

# Load model
model = tf.keras.models.load_model('Models/model')

# Predict on new data
predictions = model.predict(new_data, batch_size=64)[:, 1]
```

## Example Workflow
**Scenario:** Predict functional elements in chr20 regions.

1. Prepare BED file (`data/query_regions.bed`)
2. Preprocess data:
   ```bash
   python3 preprocess_graphs.py data/query_regions.bed 20 data/query_X.npy data/query_y.npy
   ```
3. Predict:
   ```python
   X = np.load("data/query_X.npy")
   predictions = model.predict(X)
   ```

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

