# GraphyloVar

**GraphyloVar: Predicting the impact of non-coding variants using a multi-species sequence model**

GraphyloVar is a deep learning model for variant effect prediction that jointly processes multi-species nucleotide alignments and an explicit phylogenetic tree. It combines Transformer encoders (for local sequence context) with a two-layer Graph Convolutional Network (GCN) (for phylogenetic signal propagation), and is pre-trained on population-level allele frequencies from the TOPMed cohort.

## Key results

- AUROC 0.6246 on ~149M held-out TOPMed SNVs (chromosomes 13-22, common vs. rare variant discrimination)
- z-score ensemble with CADD reaches AUROC 0.6442 (+0.020, p < 1e-15)
- Fine-tuned GraphyloVar achieves the highest AUROC on all 13 MPRA benchmark datasets
- Per-species perturbation analysis shows Human provides the largest signal, followed by four diverged primates (Orangutan, Green monkey, Gorilla, Gibbon); Chimpanzee ranks 11th despite being the closest relative, because its near-identical sequence provides almost no additional signal beyond the Human row

## Repository layout

```
graphylovar/        package: model builders, phylogeny utilities, data loading, training helpers
scripts/            training, preprocessing, inference, evaluation, figure generation
configs/            default training configuration
latex/              manuscript source (LaTeX) and submission figures
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Or with Conda:

```bash
conda env create -f environment.yml
conda activate graphylo
pip install -e .
```

## Workflow

**Preprocess TOPMed alignment data:**

```bash
python scripts/preprocess_topmed_full_compact.py \
  --alignment_dir /path/to/ucsc_maf \
  --output_dir topmed_compact_full
```

**Train the main model (Transformer + GCN, flank=32, 65 bp window):**

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

**Score variants genome-wide:**

```bash
python scripts/predict.py \
  --model_path topmed_models/checkpoint.keras \
  --data_dir topmed_compact_full \
  --chromosome 22
```

**Species perturbation interpretability:**

```bash
python scripts/additive_species_importance.py --run_tag v3flank16
```

## Model architecture

GraphyloVar uses a Transformer + GCN architecture. Each species sequence passes through an initial dense projection layer with GELU activation and batch normalization, followed by sinusoidal positional encoding and two Transformer encoder layers per strand (four total, processing forward and reverse complement independently). The center-position features from both strands are concatenated into a 64-dimensional representation per species. A Squeeze-and-Excitation block produces per-species attention weights. These features feed into a two-layer GCN that propagates information along the phylogenetic tree (58 extant species + 57 inferred ancestral nodes = 115 tree nodes). The GCN output is flattened and passed through a shared fully connected layer before dual prediction heads for allele frequency (softmax over A/C/G/T/gap) and SNP probability (sigmoid).

## Extant species (58)

Species are identified by UCSC genome browser accession. The alignment covers 58 extant placental mammalian species plus 57 computationally reconstructed ancestral sequences.

**Primates**
- Great Apes (5): hg38 (human), panTro4 (chimpanzee), gorGor3 (gorilla), ponAbe2 (orangutan), nomLeu3 (gibbon)
- Old World Monkeys (4): rheMac3 (rhesus macaque), macFas5 (crab-eating macaque), papAnu2 (olive baboon), chlSab2 (green monkey)
- New World Monkeys (2): calJac3 (marmoset), saiBol1 (squirrel monkey)
- Strepsirrhini (1): otoGar3 (bushbaby)

**Euarchontoglires**
- Scandentia (1): tupChi1 (Chinese tree shrew)
- Rodentia (11): speTri2 (squirrel), jacJac1 (lesser Egyptian jerboa), micOch1 (prairie vole), criGri1 (Chinese hamster), mesAur1 (golden hamster), mm10 (mouse), rn6 (rat), hetGla2 (naked mole-rat), cavPor3 (guinea pig), chiLan1 (chinchilla), octDeg1 (degu)
- Lagomorpha (2): oryCun2 (rabbit), ochPri3 (pika)

**Laurasiatheria**
- Cetartiodactyla (9): susScr3 (pig), vicPac2 (alpaca), camFer1 (Bactrian camel), turTru2 (dolphin), orcOrc1 (killer whale), panHod1 (Tibetan antelope), bosTau8 (cow), oviAri3 (sheep), capHir1 (goat)
- Perissodactyla (2): equCab2 (horse), cerSim1 (white rhinoceros)
- Carnivora (6): felCat8 (cat), canFam3 (dog), musFur1 (ferret), ailMel1 (giant panda), odoRosDiv1 (Pacific walrus), lepWed1 (Weddell seal)
- Chiroptera (5): pteAle1 (black flying fox), pteVam1 (large flying fox), eptFus1 (big brown bat), myoDav1 (David's myotis), myoLuc2 (little brown bat)
- Eulipotyphla (3): eriEur2 (hedgehog), sorAra2 (shrew), conCri1 (star-nosed mole)

**Atlantogenata**
- Afrotheria (5): loxAfr3 (elephant), eleEdw1 (Cape elephant shrew), triMan1 (manatee), chrAsi1 (Cape golden mole), echTel2 (lesser hedgehog tenrec)
- Xenarthra (2): oryAfe1 (aardvark), dasNov3 (armadillo)

## Data availability

- TOPMed whole-genome sequencing data: dbGaP accession phs000964
- UCSC 100-way vertebrate alignments: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz100way/
- Ancestral sequence reconstruction: Ancestors1.0 (Blanchette et al. 2004)

## Citation

Lim D. and Blanchette M. (2025). Predicting the impact of non-coding mutations using a multi-species sequence model. Under review at *Bioinformatics* (manuscript ID: BIOINF-2025-2871).

## License

MIT License.
