
# GraphyloVar: Predicting the impact of non-coding mutations using a multi-species sequence model

**GraphyloVar** is a deep learning framework designed to extract meaningful insights from evolutionary genome data. By leveraging aligned sequences (MAF) and ancestral genome reconstructions, it models the evolutionary history of non-coding regions to predict functional impact.

The model processes linear alignment data into **graph-structured inputs**, feeding them into a neural network architecture that combines **Transformer** (for sequence motifs), **GCNs** (for phylogenetic topology), (for sequential dependencies).

[](https://www.google.com/search?q=LICENSE)

## Table of Contents

  - [Data Availability Note](https://www.google.com/search?q=%23data-availability-note)
  - [Prerequisites](https://www.google.com/search?q=%23prerequisites)
  - [Installation](https://www.google.com/search?q=%23installation)
  - [Data Preparation](https://www.google.com/search?q=%23data-preparation)
  - [Preprocessing](https://www.google.com/search?q=%23preprocessing)
  - [Training](https://www.google.com/search?q=%23training)
  - [Prediction](https://www.google.com/search?q=%23prediction)
  - [Example Workflow](https://www.google.com/search?q=%23example-workflow)
  - [Citation](https://www.google.com/search?q=%23citation)

-----

## ⚠️ Data Availability Note

**Please read before cloning:**
We are currently in the process of hosting the full-scale Multi-Alignment Format (MAF) files, as well as the pre-split **Training, Validation, and Test datasets**.

Because these datasets are extremely large (terabyte scale) and we are operating with limited upload bandwidth at our laboratory, this process is taking some time. We are working to make a persistent public link available as soon as possible. 

-----

## Prerequisites

To run GraphyloVar, you will need the following environment:

  * **Anaconda3** (strongly recommended for package management)
  * **NVIDIA GPU** (required for efficient training; CPU training is possible but very slow)
  * **Genome Alignment Data:** `.maf` files (typically from the [Boreoeutherian Repository](http://repo.cs.mcgill.ca/PUB/blanchem/Boreoeutherian/))
  * **Regions of Interest:** BED files defining genomic coordinates (hg38). See `data/example.bed` for formatting.

## Installation

### 1\. Directory Setup

Clone the repository and create the necessary subdirectories for data and model checkpoints.

```bash
git clone https://github.com/DongjoonLim/GraphyloVar.git
cd GraphyloVar
mkdir data Models
```

### 2\. Environment Setup

> **Note:** Ensure Anaconda is installed on your system before running these commands.

```bash
# Update your bash source if necessary
source .bashrc

# Create the environment from the provided yaml file
conda env create -f environment.yml

# Activate the environment
conda activate graphylovar
```

### 3\. Install Python Dependencies

Install the specific versions of the libraries required for the Siamese architecture and graph processing.

```bash
pip install focal_loss pandas==1.3.4 spektral tensorflow==2.5.0 numpy==1.20.3 pyBigWig
```

## Data Preparation

### Download Alignment Data

We utilize 30-way vertebrate alignments. If you are building your own dataset, please download the corresponding `.maf` files to the `data/` directory from the [Boreoeutherian Repository](http://repo.cs.mcgill.ca/PUB/blanchem/Boreoeutherian/).

*(As mentioned above, pre-processed large-scale datasets will be linked here once upload is complete.)*

## Preprocessing

The preprocessing pipeline converts raw MAF alignments into the graph structures required by the GCN layers.

### 1\. Convert MAF to NPY

Parses the raw alignment files into a serialized numpy format.

```bash
python3 parserPreprocess.py
```

*Output:* A `.pkl` or `.npy` file containing processed alignment data.

### 2\. Generate Graph Inputs

You must provide a BED file containing your regions of interest. The format should be `chr start end label` (tab-separated).

**Example BED (`data/example_chr20.bed`):**

```text
chr20 1000 1001 0
chr20 2000 2001 1
```

Run the graph preprocessor:

```bash
# Syntax: python3 preprocess_graphs.py [BED_FILE] [CHROMOSOME] [OUTPUT_X] [OUTPUT_Y]
python3 preprocess_graphs.py data/example_chr20.bed 20 data/example_X_chr20.npy data/example_y_chr20.npy
```

### 3\. (Optional) Reverse Complement

If you are working with RNA or strand-specific data, you may need to augment the data with reverse complements:

```bash
python3 preprocessRevComp.py
```

## Training

### 1\. Merge Chromosomal Data

Before training, it is often necessary to concatenate data from multiple chromosomes into a single training set. You can do this via a simple Python script:

```python
import numpy as np

# Merge features (X) and labels (y) from chromosomes 1-22
X = np.concatenate([np.load(f"data/example_X_chr{i}.npy") for i in range(1,23)], axis=0)
y = np.concatenate([np.load(f"data/example_y_chr{i}.npy") for i in range(1,23)], axis=0)

# Save the merged files
np.save("data/full_X_train.npy", X)
np.save("data/full_y_train.npy", y)
```

### 2\. Run Training
Train the model using the preprocessed data.


python train.py \
    --x_train data/processed/X_train_chr21.npy \
    --y_train data/processed/y_train_chr21.npy \
    --x_val data/processed/X_val_chr21.npy \
    --y_val data/processed/y_val_chr21.npy \
    --output_dir models/graphylo_chr21 \
    --epochs 50 \
    --batch_size 64
3. Inference
The model outputs the log-likelihood ratio (LLR) of the alternative allele vs. reference allele.

Python

import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('models/graphylo_chr21')
# Load query data (must be preprocessed similarly)
X_query = np.load('data/processed/query_data.npy') 
predictions = model.predict(X_query)

## Prediction

To perform inference using a trained model:

```python
import tensorflow as tf
import numpy as np
from focal_loss import BinaryFocalLoss

# Load the trained model
# Note: Custom objects (like FocalLoss) must be handled if saved in the model graph
model = tf.keras.models.load_model('Models/model')

# Load your new query data (preprocessed into NPY format)
new_data = np.load('data/query_X.npy')

# Predict (outputs probability of class 1)
predictions = model.predict(new_data, batch_size=64)[:, 1]
print(predictions)
```

## Example Workflow

**Scenario:** You want to predict functional elements in specific regions of Chromosome 20.

1.  **Prepare Regions:** Create `data/query_regions.bed` with your coordinates.
2.  **Preprocess:**
    ```bash
    python3 preprocess_graphs.py data/query_regions.bed 20 data/query_X.npy data/query_y.npy
    ```
3.  **Inference:**
    Run the prediction script (see above) loading.

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
