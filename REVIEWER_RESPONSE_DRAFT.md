# Response to Reviewers

**Manuscript ID:** BIOINF-2025-2871
**Title:** Predicting the impact of non-coding mutations using a multi-species sequence model

Dear Editor and Reviewers,

We thank you for your careful reading of our manuscript and for the helpful suggestions. We have revised the manuscript in response to all comments. In particular, we clarified the chromosome-level holdout strategy and dataset scale, expanded the discussion of related phylogeny-aware methods, added evidence for complementarity with existing tools through correlation analysis and a z-score ensemble evaluation, added a context-window ablation study (flank settings 16, 32, and 100 bp), and added a causal species-importance analysis for interpretability.

Below we provide a point-by-point response.

---

## Changes Since Initial Revision

The following items were removed or consolidated during revision:

- **Main Table 1** (AUC summary table): duplicated information already shown graphically in Figure 1A; removed to avoid redundancy.
- **Supplementary Table S1** (Pearson/Spearman correlation table): replaced by a compact main-text Table 1 in Section 3.2 showing six per-method Spearman correlations.
- **Supplementary Figure S1** (correlation heatmaps): duplicated Figure 3; removed.
- **Supplementary Table S2** (DeLong CI table): confidence intervals now cited inline in Section 3.4.
- All supplementary figures consolidated; species importance (main-text Figure in Section 3.6), phylogenetic tree ablation (Supplementary Figure S1), and model component ablation (Supplementary Figure S2) are now included.
- Context-window ablation table now presented as Supplementary Table S1 in Section 3.5.

---

## Reviewer 1

### 1. GCN interpretability
**Comment:** The manuscript lacks detailed analysis of the GCN component and its learned multi-species representations.

**Response:** We added a dedicated interpretability subsection (Section 3.6) describing two analyses. First, a Squeeze-and-Excitation (SE) gate diagnostic reports the average per-species post-gate feature magnitude across held-out variants, providing an internal view of species weighting. Second, we implemented a causal perturbation analysis: for each species, alignment rows are masked with a gap token and the resulting increase in nucleotide cross-entropy is measured on held-out variants from chromosomes 13-22.

We ran per-species perturbation analysis across all 58 extant placental mammals on 500,000 held-out variants (chromosomes 13-22). The results show a clear phylogenetic gradient (Figure showing species importance): Human provides the largest signal by far, followed by four diverged primates: Orangutan, Green monkey, Gorilla, and Gibbon (in order of decreasing importance). Notably, Chimpanzee ranks 11th despite being the closest relative of humans, because its near-identical sequence (about 98.7% identity) provides almost no additional information beyond what the Human row already gives. Non-primate contributions are positive but much smaller. This shows that the GCN learns to weight species according to their useful evolutionary signal rather than simple phylogenetic proximity.

We also added a phylogenetic tree ablation (Supplementary Figure S1), showing that performance increases as more of the tree is included (Human only < Primates < Entire tree), and a model component ablation (Supplementary Figure S2) showing that each architectural component (Transformer, GCN, species attention) adds to performance.

**Manuscript anchor:** Section 3.6, Supplementary Figures S1 and S2.

---

### 2. Sequence fragment length (65 bp) justification
**Comment:** The fixed 65 bp input length was not sufficiently justified; an ablation study varying sequence length would be informative.

**Response:** We now include a context-window ablation study in Section 3.5. We trained models with three context-window settings: flank=16 (33 bp), flank=32 (65 bp), and flank=100 (201 bp). All models were trained on the same TOPMed data split (chr1-10 train, chr11-12 val) and evaluated on the full ~149M held-out TOPMed SNVs from chromosomes 13-22. The region-resolved held-out AUROC values (Supplementary Table S1) show that flank=32 (65 bp) achieves the highest overall held-out AUROC (0.625), outperforming flank=16 (0.622) and flank=100 (0.617). This directly supports the choice of the 65 bp main model. The fine-tuned flank=32 model also achieves the highest AUROC on all 13 MPRA benchmark datasets, confirming that the 65 bp window gives the best balance of zero-shot and fine-tuned performance.

**Manuscript anchor:** Section 3.5, Supplementary Table S1.

---

### 3. Complementarity evidence
**Comment:** The claim that GraphyloVar captures information complementary to existing methods was not sufficiently supported quantitatively.

**Response:** We added complementarity analysis on two fronts. First, Figure 3 (correlation heatmap) shows that GraphyloVar has only weak-to-moderate Spearman correlation with major baselines (CADD, PhyloP, PhastCons, GPN-MSA), suggesting that it captures different information. Second, we conducted a z-score ensemble analysis on the full approximately 149M held-out variant set (chromosomes 13-22). GraphyloVar achieves an overall AUROC of 0.6246. CADD, sign-aligned to the common-versus-rare label convention, achieves approximately 0.5546. A two-model z-score ensemble of GraphyloVar and CADD yields an AUROC of 0.6442 (+0.020 over GraphyloVar alone).

To test statistical significance, we conducted paired two-sided DeLong tests on a sub-sample of 500,000 held-out variants. The GraphyloVar + CADD ensemble outperforms every individual baseline (vs. GraphyloVar alone: delta AUROC = +0.0207, z = 12.24, p < 1e-15; vs. CADD: delta = +0.0950, z = 30.62, p < 1e-15). Bootstrap 95% confidence intervals (B = 1000, n = 200,000 sub-sample) yield 0.6496 [0.6395, 0.6587] for GraphyloVar + CADD and 0.6271 [0.6174, 0.6368] for GraphyloVar alone, with non-overlapping intervals. The difference between the sub-sample estimate (0.6496) and the full 149M-variant estimate (0.6442) reflects sampling variation at n = 200,000 versus n = 149M.

**Manuscript anchor:** Section 3.4.

---

### 4. Evo2 and GPN-Star comparison
**Comment:** The manuscript should include quantitative comparison with Evo2 and GPN-Star.

**Response:** GPN-Star is included in the baseline description (Section 2.4) and in the benchmark figures. Evo2 is not included in the quantitative benchmark. Evo2 requires compute capability 8.9 or higher (Ada or Hopper class GPUs) for inference, which is not available in our current setup (Quadro RTX 6000, compute capability 7.5). We therefore discuss Evo2 qualitatively in Section 2.4 as an important future comparison.

**Manuscript anchor:** Section 2.4 (qualitative discussion only).

---

### 5. Code and data release
**Comment:** The repository did not contain a sufficiently complete release of implementation details, data procedures, and usage instructions.

**Response:** We revised the repository structure and documentation: updated installation and usage instructions, dependency specifications, preprocessing and training entry points, and clarified the data-processing workflow. We also added a formal Data and Code Availability section to the manuscript specifying that code and trained model weights are available at the GraphyloVar repository, TOPMed data are available via dbGaP (accession phs000964), and UCSC alignments are publicly available.

**Manuscript anchor:** Section 2.6; Data and Code Availability section.

---

### 6. Dataset inconsistencies
**Comment:** The Kircher et al. mutagenesis dataset and GWAS Catalog usage were not described consistently between the text and figures.

**Response:** We revised the Methods section to make sure all datasets referenced in the figures are explicitly described in the text. In particular, we clarified the role of the Kircher et al. saturation mutagenesis/MPRA data and described how GWAS Catalog variants were used.

**Manuscript anchor:** Section 2.5.

---

### 7. MAF correlation figures
**Comment:** The relationship between predicted scores and MAF was mostly described in text; supplementary figures or tables would be valuable.

**Response:** We added a compact main-text table (Table 1, Section 3.2) reporting the per-method Spearman correlations alongside the prose description: GraphyloVar 0.164, GPN-MSA 0.143, Enformer 0.131, PhyloP 0.099, CADD 0.081, PhastCons 0.058. Figure 1B continues to show the relationship visually across all methods.

**Manuscript anchor:** Section 3.2, Table 1.

---

### 8. Figure 1 architecture issues
**Comment:** There are inconsistencies between the architecture figure and the text.

**Response:** We revised the figure caption and associated text to match the actual model architecture, correcting the description of the center-position extraction step and the GCN input.

**Manuscript anchor:** Figure 1 and caption.

---

### 9. Figure 2 caption error
**Comment:** The Figure 2 caption contains redundant or incorrect GraphyloVar labels.

**Response:** Corrected; the updated caption now clearly distinguishes the different GraphyloVar-derived scores and removes duplicate wording.

**Manuscript anchor:** Figure 2 caption.

---

### 10. Naming inconsistency
**Comment:** The manuscript uses "GPN-MSA" and "GPNMSA" inconsistently.

**Response:** Standardized to "GPN-MSA" throughout text, tables, and captions.

**Manuscript anchor:** All occurrences.

---

## Reviewer 2

### 1. Held-out data definition / leakage
**Comment:** It is unclear whether the holdout was variant-level or region-level; the training/evaluation split should be stated explicitly.

**Response:** We now state the splitting procedure explicitly. GraphyloVar uses chromosome-level separation: chromosomes 1-10 for training, 11-12 for validation, and 13-22 for all held-out evaluation. No variants from evaluation chromosomes appear in training or validation. This chromosome-based partition prevents leakage from nearby sites that share local sequence and evolutionary context. The evaluated set is approximately 149 million TOPMed SNVs from chromosomes 13-22.

**Manuscript anchor:** Sections 2.1 and 3.1.

---

### 2. Prior work discussion: PHACT / PHACTboost
**Comment:** The manuscript should discuss PHACT and PHACTboost in the context of phylogeny-aware variant prediction.

**Response:** We added a paragraph in the Discussion (Section 4) situating GraphyloVar relative to PHACT and PHACTboost. These methods use codon and protein-level phylogenetic information, mainly for coding variants. GraphyloVar is designed for non-coding variants, processing multi-species nucleotide alignments through a phylogenetic GCN. We describe PHACT and PHACTboost as related but focused on a different domain.

**Manuscript anchor:** Section 4.

---

### 3. dbNSFP comparison breadth
**Comment:** Since dbNSFP contains many tools, the manuscript should either compare against more methods or justify the current selection.

**Response:** We now explain in Section 2.4 that baselines were selected to represent methodologically different families: conservation-based (PhyloP, PhastCons), integrative ML (CADD), sequence deep learning (Enformer), and MSA-aware models (GPN-MSA, GPN-Star). This covers the major methodological landscape without including every dbNSFP tool. Evo2 is discussed qualitatively as future work given hardware constraints noted above.

**Manuscript anchor:** Section 2.4.

---

### 4. Loss-function clarification
**Comment:** Binary cross-entropy is typically used for binary classification; please clarify its use for the allele-frequency output.

**Response:** Corrected. The allele-frequency head uses categorical cross-entropy (predicting a distribution over five nucleotide states A/C/G/T/gap); the SNP-probability head uses binary cross-entropy (predicting polymorphism probability). The revised wording now matches the actual formulation.

**Manuscript anchor:** Section 2.3.

---

## Summary of result status at submission

| Analysis | Status | Location in manuscript |
|----------|--------|----------------------|
| Overall benchmark AUROC (chromosomes 13-22, 149M variants) | Complete - AUROC = 0.6246 | Section 3.1, Figure 1A |
| Region-specific AUROC (coding, cCREs, TEs) | Complete - reported in Section 3.1 | Section 3.1 |
| Spearman correlation with MAF | Complete | Section 3.2 |
| Ensemble AUC (GraphyloVar + CADD) | Complete - AUROC = 0.6442 | Section 3.4 |
| DeLong significance tests | Complete | Section 3.4 |
| Bootstrap 95% CI | Complete | Section 3.4 |
| Context-window ablation (flanks 16, 32, 100) | Complete - flank=32 achieves highest AUROC (0.625) | Section 3.5, Supplementary Table S1 |
| Per-species perturbation (all 58 mammals, 500,000 variants) | Complete | Section 3.6, Figure in Section 3.6 |
| Phylogenetic tree ablation | Complete | Section 3.6, Supplementary Figure S1 |
| Model component ablation | Complete | Section 3.6, Supplementary Figure S2 |
| SE attention gate diagnostic | Complete | Section 3.6 |
| Fine-tuned MPRA prediction (all 13 datasets) | Complete - highest AUROC on all 13 | Section 3.3, Figure 2 |

---

We appreciate the reviewers' feedback, which improved the manuscript.

Sincerely,
Dongjoon Lim and Mathieu Blanchette
