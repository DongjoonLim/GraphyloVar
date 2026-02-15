#!/usr/bin/env python
"""
Comprehensive Visualization of GraphyloVar Benchmark Results.

Generates publication-quality figures comparing GraphyloVar against a broad
collection of variant-effect prediction tools across multiple benchmarks:

  1. ClinVar Pathogenic vs Benign (songlab/clinvar_vs_benign)
     - 50,164 ClinVar variants (Benign / Pathogenic)
     - Models: Evo2 (40B & 7B), GPN-Star (V100, M447, P243, P36),
       GPN-MSA, CADD, AlphaMissense, ESM-1b, NT, PhyloP, PhastCons, Roulette
     - Addresses Reviewer 1 Point 4 (compare with Evo2, GPN-Star)
       and Reviewer 2 Point 3 (compare with more dbNSFP tools)

  2. GraphyloVar ClinVar Analysis (user's own scored ClinVar data)
     - 35,613 ClinVar variants with full GraphyloVar model predictions
     - Compares GraphyloVar (multiple architectures) vs GPN-MSA, CADD,
       phyloP, phastCons, ESM-1b, NT, HyenaDNA

  3. gnomAD Balanced Common vs Rare (songlab/gnomad_balanced)
     - ~1M chr2 variants (balanced common vs rare)
     - Precomputed scores: GPN-Star, GPN-MSA, CADD, PhyloP, PhastCons, Roulette

Outputs:
  - fig1_clinvar_songlab_auc.png         : AUC bar chart (songlab ClinVar benchmark)
  - fig2_clinvar_songlab_roc.png         : ROC curves (songlab ClinVar benchmark)
  - fig3_clinvar_graphylovar_auc.png     : AUC bar chart (GraphyloVar ClinVar analysis)
  - fig4_gnomad_balanced_auc.png         : AUC heatmap (gnomAD balanced)
  - fig5_summary_comparison.png          : Summary comparison across all benchmarks
  - fig6_clinvar_consequence_heatmap.png : Per-consequence AUC heatmap (songlab ClinVar)
  - benchmark_results_summary.csv        : Full results table

Usage:
    python visualize_all_comparisons.py

Author: Dongjoon Lim
Date: 2025-02
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLINVAR_SONGLAB_DIR = os.path.join(BASE_DIR, 'clinvar_vs_benign')
GNOMAD_DIR = os.path.join(BASE_DIR, 'gnomad_balanced')
GRAPHYLOVAR_CLINVAR = os.path.join(BASE_DIR, 'clinvar_with_graphylo_topmed_common2.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'benchmark_figures')


# ============================================================================
# Color palette — consistent across all figures
# ============================================================================
COLORS = {
    # GraphyloVar family (blues)
    'GraphyloVar (Transformer, flank=32)': '#1565C0',
    'GraphyloVar (Transformer, flank=8)':  '#1E88E5',
    'GraphyloVar (Transformer, flank=0)':  '#42A5F5',
    'GraphyloVar (Transformer)':           '#1976D2',
    'GraphyloVar (MLP)':                   '#64B5F6',
    'GraphyloVar (Conditional, flank=32)': '#0D47A1',
    'GraphyloVar (EvoLSTM)':               '#90CAF9',
    'GraphyloVar (Polymorphic, flank=32)': '#BBDEFB',
    # Evo2 (reds)
    'Evo2 (40B)':          '#C62828',
    'Evo2 (7B)':           '#EF5350',
    # GPN family (greens / teals)
    'GPN-Star (V100)':     '#2E7D32',
    'GPN-Star (M447)':     '#43A047',
    'GPN-Star (P243)':     '#66BB6A',
    'GPN-Star (P36)':      '#A5D6A7',
    'GPN-Star (V100, AF)': '#1B5E20',
    'GPN-MSA':             '#00695C',
    # Conservation (oranges / yellows)
    'CADD':                '#E65100',
    'PhyloP (V100)':       '#F57C00',
    'PhyloP (M447)':       '#FB8C00',
    'PhyloP (P243)':       '#FFA726',
    'PhastCons (V100)':    '#FF6F00',
    'PhastCons (M470)':    '#FFB300',
    'PhastCons (P43)':     '#FFCA28',
    'Roulette':            '#FDD835',
    # Protein / DNA language models (purples / pinks)
    'AlphaMissense':       '#6A1B9A',
    'ESM-1b':              '#8E24AA',
    'NT':                  '#AB47BC',
    'NT (2.5B)':           '#AB47BC',
    'HyenaDNA':            '#CE93D8',
}

def get_color(name):
    """Get color for a model name, with fallback to a hash-derived color."""
    if name in COLORS:
        return COLORS[name]
    # Fallback: generate a color from hash
    h = hash(name) % 360
    return matplotlib.colors.hsv_to_rgb([h / 360, 0.7, 0.8])


# ============================================================================
# Helper: compute AUC safely, handling score direction automatically
# ============================================================================
def safe_auc(labels, scores, higher_is_pathogenic=None):
    """
    Compute AUC, automatically determining score direction if not specified.

    Parameters
    ----------
    labels : array-like
        Binary labels (1 = pathogenic/positive, 0 = benign/negative).
    scores : array-like
        Prediction scores.
    higher_is_pathogenic : bool or None
        If True, higher score → pathogenic. If None, pick max(AUC, 1-AUC).

    Returns
    -------
    float : AUC value (always >= 0.5 unless explicitly forced)
    """
    mask = np.isfinite(scores) & np.isfinite(labels)
    if mask.sum() < 50:
        return np.nan, 0

    s, y = np.array(scores[mask], dtype=float), np.array(labels[mask], dtype=int)
    if len(np.unique(y)) < 2:
        return np.nan, 0

    try:
        auc = roc_auc_score(y, s)
    except ValueError:
        return np.nan, 0

    if higher_is_pathogenic is None:
        # Auto-detect: pick max(AUC, 1-AUC)
        if auc < 0.5:
            auc = 1 - auc
    elif not higher_is_pathogenic:
        auc = 1 - auc if auc > 0.5 else auc
        auc = max(auc, 1 - auc)  # safety

    return auc, mask.sum()


# ============================================================================
# Benchmark 1: songlab/clinvar_vs_benign
# ============================================================================
def load_clinvar_songlab():
    """
    Load the songlab ClinVar Pathogenic vs Benign benchmark with all
    available prediction parquets.

    Returns
    -------
    pd.DataFrame with columns: chrom, pos, ref, alt, label (0/1), consequence,
    and one column per model score.
    """
    print('[Benchmark 1] Loading songlab/clinvar_vs_benign...')
    df = pd.read_parquet(os.path.join(CLINVAR_SONGLAB_DIR, 'test.parquet'))
    df['label_binary'] = (df['label'] == 'Pathogenic').astype(int)
    print(f'  Variants: {len(df):,} ({(df.label_binary==1).sum():,} Pathogenic, '
          f'{(df.label_binary==0).sum():,} Benign)')

    # Load all prediction parquets
    pred_files = sorted([f for f in os.listdir(CLINVAR_SONGLAB_DIR)
                         if f.endswith('.parquet') and f != 'test.parquet'])
    for pf in pred_files:
        name = pf.replace('.parquet', '')
        pred = pd.read_parquet(os.path.join(CLINVAR_SONGLAB_DIR, pf))
        df[name] = pred['score'].values
        n_valid = df[name].notna().sum()
        print(f'  {name}: {n_valid:,} valid scores ({100*n_valid/len(df):.1f}%)')

    return df


def run_clinvar_songlab_benchmark(df):
    """
    Compute AUC for all models on the songlab ClinVar benchmark.

    The task: distinguish Pathogenic from Benign ClinVar variants.
    For most sequence-based models, a more negative score indicates the
    alt allele is less likely (= conserved position = higher pathogenicity).

    Returns
    -------
    dict : model_name → {'auc': float, 'n': int, 'display': str}
    """
    # Model definitions: column_name → (display_name, higher_is_pathogenic)
    # For DNA language models (GPN-MSA, GPN-Star, Evo2, NT, ESM-1b):
    #   Score = log-likelihood(alt) - log-likelihood(ref)
    #   More NEGATIVE = ref is favored = conserved = pathogenic
    #   → higher_is_pathogenic = False (we want to reverse)
    #   Actually, let's auto-detect for all models.

    model_defs = {
        'Evo2_40B':                 'Evo2 (40B)',
        'Evo2_7B':                  'Evo2 (7B)',
        'GPN-Star-V100':            'GPN-Star (V100)',
        'GPN-Star-M447':            'GPN-Star (M447)',
        'GPN-Star-P243':            'GPN-Star (P243)',
        'GPN-Star-P36':             'GPN-Star (P36)',
        'GPN-Star-V100_AF_adjusted':'GPN-Star (V100, AF)',
        'GPN-MSA':                  'GPN-MSA',
        'CADD':                     'CADD',
        'AlphaMissense':            'AlphaMissense',
        'ESM-1b':                   'ESM-1b',
        'NT_2.5B_MS':               'NT (2.5B)',
        'PhyloP-V100':              'PhyloP (V100)',
        'PhyloP-M447':              'PhyloP (M447)',
        'PhyloP-P243':              'PhyloP (P243)',
        'PhastCons-V100':           'PhastCons (V100)',
        'PhastCons-M470':           'PhastCons (M470)',
        'PhastCons-P43':            'PhastCons (P43)',
        'Roulette':                 'Roulette',
    }

    results = {}
    for col, display in model_defs.items():
        if col in df.columns:
            auc, n = safe_auc(df['label_binary'], df[col])
            results[display] = {'auc': auc, 'n': n, 'column': col}

    return results


def run_clinvar_songlab_per_consequence(df, model_defs=None):
    """Compute AUC per consequence type for the songlab ClinVar benchmark."""
    if model_defs is None:
        model_defs = {
            'Evo2_40B': 'Evo2 (40B)', 'Evo2_7B': 'Evo2 (7B)',
            'GPN-Star-V100': 'GPN-Star (V100)', 'GPN-MSA': 'GPN-MSA',
            'CADD': 'CADD', 'AlphaMissense': 'AlphaMissense',
            'ESM-1b': 'ESM-1b', 'NT_2.5B_MS': 'NT (2.5B)',
            'PhyloP-V100': 'PhyloP (V100)', 'PhastCons-V100': 'PhastCons (V100)',
            'Roulette': 'Roulette',
        }

    # Get top consequence types
    consequence_counts = df['consequence'].value_counts()
    top_consequences = list(consequence_counts[consequence_counts >= 100].head(8).index)

    results = {}
    for cons in ['all'] + top_consequences:
        sub = df if cons == 'all' else df[df['consequence'] == cons]
        for col, display in model_defs.items():
            if col in sub.columns:
                auc, n = safe_auc(sub['label_binary'], sub[col])
                results.setdefault(display, {})[cons] = auc

    return results, ['all'] + top_consequences


# ============================================================================
# Benchmark 2: User's ClinVar with GraphyloVar
# ============================================================================
def load_graphylovar_clinvar():
    """
    Load the user's ClinVar predictions file with GraphyloVar scores.

    This file has label=True (pathogenic) / label=False (benign/common).
    Verified by score distribution: True has more negative GPN-MSA/CADD
    (more deleterious) → True = pathogenic.

    Returns
    -------
    pd.DataFrame with label_binary column (1 = pathogenic)
    """
    print('[Benchmark 2] Loading GraphyloVar ClinVar predictions...')
    df = pd.read_csv(GRAPHYLOVAR_CLINVAR)
    df['label_binary'] = df['label'].astype(int)  # True=1=pathogenic
    print(f'  Variants: {len(df):,} (Pathogenic={df.label_binary.sum():,}, '
          f'Benign={(1-df.label_binary).sum():,})')
    return df


def run_graphylovar_clinvar_benchmark(df):
    """
    Compute AUC for all models in the user's ClinVar file.

    Score semantics (verified from mean analysis):
    - For conservation tools (GPN-MSA, CADD, phyloP, phastCons, GraphyloVar):
      MORE NEGATIVE score = position is more conserved = more likely pathogenic.
      So for label=1 (pathogenic), we expect lower scores.
      → AUC should be computed with label=1 for low scores → reverse sign, or
        equivalently use 1-AUC.
    - Auto-detect via safe_auc.
    """
    model_defs = {
        # External baselines
        'GPN-MSA':                                 'GPN-MSA',
        'CADD':                                    'CADD',
        'phyloP-100v':                             'PhyloP (V100)',
        'phyloP-241m':                             'PhyloP (M447)',
        'phastCons-100v':                          'PhastCons (V100)',
        'ESM-1b':                                  'ESM-1b',
        'NT':                                      'NT',
        'HyenaDNA':                                'HyenaDNA',
        # GraphyloVar models
        'graphylo_multitask':                      'GraphyloVar (MLP)',
        'graphylo_transformer_multitask':          'GraphyloVar (Transformer)',
        'graphylo_transformer_multitask_0':        'GraphyloVar (Transformer, flank=0)',
        'graphylo_transformer_multitask_8':        'GraphyloVar (Transformer, flank=8)',
        'graphylo_transformer_multitask_32':       'GraphyloVar (Transformer, flank=32)',
        'graphylo_multitask_conditional':          'GraphyloVar (Conditional, MLP)',
        'graphylo_transformer_multitask_conditional':       'GraphyloVar (Conditional, Transformer)',
        'graphylo_transformer_multitask_32_conditional':    'GraphyloVar (Conditional, flank=32)',
        'graphylo_evolstm':                        'GraphyloVar (EvoLSTM)',
        'graphylo_polymorphic':                    'GraphyloVar (Polymorphic, MLP)',
        'graphylo_transformer_polymorphic':        'GraphyloVar (Polymorphic, Transformer)',
        'graphylo_transformer_polymorphic_32':     'GraphyloVar (Polymorphic, flank=32)',
    }

    results = {}
    for col, display in model_defs.items():
        if col in df.columns:
            auc, n = safe_auc(df['label_binary'], df[col])
            results[display] = {'auc': auc, 'n': n, 'column': col}

    return results


# ============================================================================
# Benchmark 3: gnomAD Balanced Common vs Rare
# ============================================================================
def load_gnomad_balanced(chroms=None):
    """
    Load gnomad_balanced data with all prediction parquets.

    Prediction parquets are row-aligned with the full test_sample (12M rows).
    When filtering to specific chromosomes, we keep the original index so
    we can slice prediction parquets correctly.
    """
    print('[Benchmark 3] Loading gnomad_balanced...')
    test_file = os.path.join(GNOMAD_DIR, 'test_sample.parquet')
    if not os.path.exists(test_file):
        test_file = os.path.join(GNOMAD_DIR, 'test.parquet')
    df_full = pd.read_parquet(test_file)
    print(f'  Full dataset: {len(df_full):,} variants')

    if chroms:
        mask = df_full['chrom'].isin([str(c) for c in chroms])
        idx = np.where(mask.values)[0]
        df = df_full.loc[mask].copy().reset_index(drop=True)
        print(f'  Filtered to chrom(s) {chroms}: {len(df):,} variants')
    else:
        idx = np.arange(len(df_full))
        df = df_full

    pred_files = sorted([f for f in os.listdir(GNOMAD_DIR)
                         if f.endswith('.parquet') and 'test' not in f])
    for pf in pred_files:
        name = pf.replace('.parquet', '')
        pred = pd.read_parquet(os.path.join(GNOMAD_DIR, pf))
        if len(pred) == len(df_full):
            # Slice to the filtered indices
            df[name] = pred['score'].values[idx]
        elif len(pred) == len(df):
            df[name] = pred['score'].values
        else:
            print(f'  {name}: unexpected length ({len(pred)}), skipping')
            continue
        n_valid = df[name].notna().sum()
        print(f'  {name}: {n_valid:,} valid')

    del df_full  # free memory
    return df


def run_gnomad_benchmark(df):
    """
    Compute AUC for gnomAD balanced benchmark.

    Task: distinguish common (label=True) from rare (label=False) variants.
    Label=True=Common=1 (positive class for AUC).

    For conservation-aware tools, common variants should have LESS negative
    scores (less conserved positions). Higher score → more common.
    """
    model_defs = {
        'GPN-Star-V100':   'GPN-Star (V100)',
        'GPN-Star-M447':   'GPN-Star (M447)',
        'GPN-Star-P243':   'GPN-Star (P243)',
        'GPN-MSA':         'GPN-MSA',
        'CADD':            'CADD',
        'PhastCons-V100':  'PhastCons (V100)',
        'PhastCons-M470':  'PhastCons (M470)',
        'PhastCons-P43':   'PhastCons (P43)',
        'PhyloP-V100':     'PhyloP (V100)',
        'PhyloP-M447':     'PhyloP (M447)',
        'PhyloP-P243':     'PhyloP (P243)',
        'Roulette':        'Roulette',
    }

    results = {}
    for col, display in model_defs.items():
        if col in df.columns:
            auc, n = safe_auc(df['label'].astype(int), df[col])
            results[display] = {'auc': auc, 'n': n, 'column': col}

    return results


def run_gnomad_per_consequence(df):
    """Compute AUC per consequence type for gnomAD balanced."""
    model_defs = {
        'GPN-Star-V100': 'GPN-Star (V100)', 'GPN-MSA': 'GPN-MSA',
        'CADD': 'CADD', 'PhyloP-V100': 'PhyloP (V100)',
        'PhastCons-V100': 'PhastCons (V100)', 'Roulette': 'Roulette',
    }
    consequence_counts = df['consequence'].value_counts()
    top_consequences = list(consequence_counts[consequence_counts >= 100].head(6).index)

    results = {}
    for cons in ['all'] + top_consequences:
        sub = df if cons == 'all' else df[df['consequence'] == cons]
        for col, display in model_defs.items():
            if col in sub.columns:
                auc, n = safe_auc(sub['label'].astype(int), sub[col])
                results.setdefault(display, {})[cons] = auc
    return results, ['all'] + top_consequences


# ============================================================================
# Figure 1: ClinVar songlab AUC bar chart
# ============================================================================
def plot_clinvar_songlab_auc(results, outfile='fig1_clinvar_songlab_auc.png'):
    """
    Horizontal bar chart of AUC on songlab ClinVar Pathogenic vs Benign.

    Groups models by category for clarity.
    """
    # Sort by AUC
    sorted_results = sorted(results.items(), key=lambda x: x[1]['auc'],
                            reverse=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    names = [r[0] for r in sorted_results]
    aucs = [r[1]['auc'] for r in sorted_results]
    ns = [r[1]['n'] for r in sorted_results]
    colors = [get_color(n) for n in names]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, aucs, color=colors, edgecolor='black', linewidth=0.5,
                   height=0.7)

    # Add AUC values as text
    for i, (auc, n) in enumerate(zip(aucs, ns)):
        if np.isnan(auc):
            ax.text(0.52, i, 'N/A', va='center', fontsize=9, color='gray')
        else:
            ax.text(auc + 0.003, i, f'{auc:.3f} (n={n:,})', va='center',
                    fontsize=8, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('AUC (Pathogenic vs Benign)', fontsize=12)
    ax.set_title('ClinVar Pathogenic vs Benign Classification\n'
                 '(songlab/clinvar_vs_benign, 50,164 variants)',
                 fontsize=13, fontweight='bold')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlim(0.45, max(aucs) + 0.08)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, outfile), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ============================================================================
# Figure 2: ClinVar songlab ROC curves
# ============================================================================
def plot_clinvar_songlab_roc(df, results, outfile='fig2_clinvar_songlab_roc.png'):
    """ROC curves for top models on songlab ClinVar benchmark."""
    # Select top 10 models by AUC
    top_models = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)[:10]

    fig, ax = plt.subplots(figsize=(9, 8))

    for name, info in top_models:
        col = info['column']
        if col not in df.columns:
            continue
        mask = df[col].notna()
        scores = df.loc[mask, col].values
        labels = df.loc[mask, 'label_binary'].values

        # Auto-detect direction
        auc_raw = roc_auc_score(labels, scores)
        if auc_raw < 0.5:
            scores = -scores

        fpr, tpr, _ = roc_curve(labels, scores)
        auc_val = info['auc']
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})',
                color=get_color(name), linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC: ClinVar Pathogenic vs Benign (Top 10 Models)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, outfile), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ============================================================================
# Figure 3: GraphyloVar ClinVar AUC bar chart
# ============================================================================
def plot_graphylovar_clinvar_auc(results, outfile='fig3_clinvar_graphylovar_auc.png'):
    """
    Bar chart comparing GraphyloVar model variants with baselines
    on the user's ClinVar dataset.
    """
    # Separate GraphyloVar models from baselines
    graphylo_results = {k: v for k, v in results.items() if 'GraphyloVar' in k}
    baseline_results = {k: v for k, v in results.items() if 'GraphyloVar' not in k}

    # Sort each group by AUC
    sorted_baselines = sorted(baseline_results.items(),
                              key=lambda x: x[1]['auc'], reverse=True)
    sorted_graphylo = sorted(graphylo_results.items(),
                             key=lambda x: x[1]['auc'], reverse=True)

    # Combine: baselines first, then GraphyloVar
    all_sorted = sorted_graphylo + sorted_baselines

    fig, ax = plt.subplots(figsize=(11, 10))

    names = [r[0] for r in all_sorted]
    aucs = [r[1]['auc'] for r in all_sorted]
    colors = [get_color(n) for n in names]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, aucs, color=colors, edgecolor='black', linewidth=0.5,
                   height=0.7)

    for i, auc in enumerate(aucs):
        if np.isnan(auc):
            ax.text(0.52, i, 'N/A', va='center', fontsize=9, color='gray')
        else:
            ax.text(auc + 0.003, i, f'{auc:.3f}', va='center', fontsize=9,
                    fontweight='bold')

    # Add separator between GraphyloVar and baselines
    sep_y = len(sorted_graphylo) - 0.5
    ax.axhline(y=sep_y, color='gray', linestyle=':', alpha=0.6)
    ax.text(0.46, sep_y - 0.3, 'GraphyloVar Models', fontsize=9,
            fontstyle='italic', color='#333', ha='left')
    ax.text(0.46, sep_y + 0.7, 'Baselines', fontsize=9,
            fontstyle='italic', color='#333', ha='left')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title('GraphyloVar vs Baselines: ClinVar Pathogenic vs Common\n'
                 f'(35,613 variants with TopMed-based labels)',
                 fontsize=13, fontweight='bold')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(0.45, max(aucs) + 0.06)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, outfile), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ============================================================================
# Figure 4: gnomAD balanced AUC heatmap
# ============================================================================
def plot_gnomad_heatmap(results_per_cons, consequences,
                        outfile='fig4_gnomad_balanced_auc.png'):
    """Heatmap of AUC per model × consequence type on gnomAD balanced."""
    models = list(results_per_cons.keys())
    if not models:
        print('  [SKIP] No gnomAD results to plot.')
        return

    # Build matrix
    matrix = np.full((len(models), len(consequences)), np.nan)
    for i, m in enumerate(models):
        for j, c in enumerate(consequences):
            if c in results_per_cons[m]:
                matrix[i, j] = results_per_cons[m][c]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.5, vmax=0.75, aspect='auto')

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(consequences)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='white' if val > 0.65 or val < 0.52 else 'black')

    cons_display = ['All'] + [c.replace('_', ' ').title() for c in consequences[1:]]
    ax.set_xticks(np.arange(len(consequences)))
    ax.set_xticklabels(cons_display, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_title('gnomAD Balanced: Common vs Rare AUC by Consequence Type\n'
                 f'(chr2, ~1M variants)',
                 fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('AUC', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, outfile), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ============================================================================
# Figure 5: Summary comparison across all benchmarks
# ============================================================================
def plot_summary_comparison(clinvar_songlab, clinvar_graphylo, gnomad,
                            outfile='fig5_summary_comparison.png'):
    """
    Multi-panel summary figure showing AUC across all three benchmarks.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # --- Panel A: songlab ClinVar top 12 ---
    ax = axes[0]
    top = sorted(clinvar_songlab.items(), key=lambda x: x[1]['auc'],
                 reverse=True)[:12]
    names = [r[0] for r in top]
    aucs = [r[1]['auc'] for r in top]
    colors = [get_color(n) for n in names]
    y = np.arange(len(names))
    ax.barh(y, aucs, color=colors, edgecolor='black', linewidth=0.4, height=0.65)
    for i, a in enumerate(aucs):
        ax.text(a + 0.003, i, f'{a:.3f}', va='center', fontsize=8,
                fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('AUC', fontsize=11)
    ax.set_title('A) ClinVar Path. vs Benign\n(songlab, 50K variants)',
                 fontsize=11, fontweight='bold')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(0.45, max(aucs) + 0.06)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # --- Panel B: GraphyloVar ClinVar top 12 ---
    ax = axes[1]
    top = sorted(clinvar_graphylo.items(), key=lambda x: x[1]['auc'],
                 reverse=True)[:12]
    names = [r[0] for r in top]
    aucs = [r[1]['auc'] for r in top]
    colors = [get_color(n) for n in names]
    y = np.arange(len(names))
    ax.barh(y, aucs, color=colors, edgecolor='black', linewidth=0.4, height=0.65)
    for i, a in enumerate(aucs):
        ax.text(a + 0.003, i, f'{a:.3f}', va='center', fontsize=8,
                fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('AUC', fontsize=11)
    ax.set_title('B) ClinVar Path. vs Common\n(GraphyloVar, 35K variants)',
                 fontsize=11, fontweight='bold')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(0.45, max(aucs) + 0.06)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # --- Panel C: gnomAD top 12 ---
    ax = axes[2]
    top = sorted(gnomad.items(), key=lambda x: x[1]['auc'], reverse=True)[:12]
    if top:
        names = [r[0] for r in top]
        aucs = [r[1]['auc'] for r in top]
        colors = [get_color(n) for n in names]
        y = np.arange(len(names))
        ax.barh(y, aucs, color=colors, edgecolor='black', linewidth=0.4, height=0.65)
        for i, a in enumerate(aucs):
            ax.text(a + 0.003, i, f'{a:.3f}', va='center', fontsize=8,
                    fontweight='bold')
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('AUC', fontsize=11)
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim(0.45, max(aucs) + 0.06)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                fontsize=14, transform=ax.transAxes)
    ax.set_title('C) gnomAD Common vs Rare\n(chr2, ~1M variants)',
                 fontsize=11, fontweight='bold')

    plt.suptitle('Comprehensive Variant Effect Prediction Benchmark',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, outfile), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ============================================================================
# Figure 6: Per-consequence heatmap (songlab ClinVar)
# ============================================================================
def plot_clinvar_consequence_heatmap(results_per_cons, consequences,
                                     outfile='fig6_clinvar_consequence_heatmap.png'):
    """Heatmap of AUC per model × consequence type on songlab ClinVar."""
    models = list(results_per_cons.keys())
    if not models:
        print('  [SKIP] No per-consequence results.')
        return

    # Sort models by their 'all' AUC
    models_sorted = sorted(models,
                           key=lambda m: results_per_cons[m].get('all', 0),
                           reverse=True)

    matrix = np.full((len(models_sorted), len(consequences)), np.nan)
    for i, m in enumerate(models_sorted):
        for j, c in enumerate(consequences):
            if c in results_per_cons[m]:
                matrix[i, j] = results_per_cons[m][c]

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')

    for i in range(len(models_sorted)):
        for j in range(len(consequences)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='white' if val > 0.85 or val < 0.55 else 'black')

    cons_display = ['All'] + [c.replace('_', ' ').title() for c in consequences[1:]]
    ax.set_xticks(np.arange(len(consequences)))
    ax.set_xticklabels(cons_display, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(np.arange(len(models_sorted)))
    ax.set_yticklabels(models_sorted, fontsize=10)
    ax.set_title('ClinVar Pathogenic vs Benign: AUC by Variant Consequence\n'
                 '(songlab/clinvar_vs_benign)',
                 fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('AUC', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, outfile), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ============================================================================
# Results table
# ============================================================================
def save_results_table(clinvar_songlab, clinvar_graphylo, gnomad,
                       outfile='benchmark_results_summary.csv'):
    """Save a combined results table across all benchmarks."""
    rows = []
    for name, info in clinvar_songlab.items():
        rows.append({
            'Model': name, 'Benchmark': 'ClinVar (songlab)',
            'AUC': info['auc'], 'N_variants': info['n']
        })
    for name, info in clinvar_graphylo.items():
        rows.append({
            'Model': name, 'Benchmark': 'ClinVar (GraphyloVar)',
            'AUC': info['auc'], 'N_variants': info['n']
        })
    for name, info in gnomad.items():
        rows.append({
            'Model': name, 'Benchmark': 'gnomAD Balanced',
            'AUC': info['auc'], 'N_variants': info['n']
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(['Benchmark', 'AUC'], ascending=[True, False])
    outpath = os.path.join(OUTPUT_DIR, outfile)
    df.to_csv(outpath, index=False)
    print(f'  Saved: {outfile}')

    # Pretty print
    print('\n' + '=' * 80)
    print('BENCHMARK RESULTS SUMMARY')
    print('=' * 80)
    for bench in df['Benchmark'].unique():
        print(f'\n--- {bench} ---')
        sub = df[df['Benchmark'] == bench].reset_index(drop=True)
        for _, row in sub.iterrows():
            auc_str = f'{row["AUC"]:.4f}' if not np.isnan(row["AUC"]) else 'N/A'
            print(f'  {row["Model"]:<40s} AUC={auc_str}  (n={row["N_variants"]:,})')
    print('=' * 80)

    return df


# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('=' * 70)
    print(' Comprehensive Variant Effect Prediction Benchmark')
    print(' GraphyloVar vs Evo2, GPN-Star, CADD, AlphaMissense, and more')
    print('=' * 70)

    # ------------------------------------------------------------------
    # Benchmark 1: songlab ClinVar Pathogenic vs Benign
    # ------------------------------------------------------------------
    print('\n' + '-' * 70)
    df_songlab = load_clinvar_songlab()
    results_songlab = run_clinvar_songlab_benchmark(df_songlab)
    results_songlab_cons, cons_songlab = run_clinvar_songlab_per_consequence(df_songlab)

    print('\n  Generating Figure 1: ClinVar songlab AUC...')
    plot_clinvar_songlab_auc(results_songlab)

    print('  Generating Figure 2: ClinVar songlab ROC...')
    plot_clinvar_songlab_roc(df_songlab, results_songlab)

    print('  Generating Figure 6: ClinVar per-consequence heatmap...')
    plot_clinvar_consequence_heatmap(results_songlab_cons, cons_songlab)

    # ------------------------------------------------------------------
    # Benchmark 2: User's ClinVar with GraphyloVar
    # ------------------------------------------------------------------
    print('\n' + '-' * 70)
    df_graphylo = load_graphylovar_clinvar()
    results_graphylo = run_graphylovar_clinvar_benchmark(df_graphylo)

    print('\n  Generating Figure 3: GraphyloVar ClinVar AUC...')
    plot_graphylovar_clinvar_auc(results_graphylo)

    # ------------------------------------------------------------------
    # Benchmark 3: gnomAD Balanced
    # ------------------------------------------------------------------
    print('\n' + '-' * 70)
    df_gnomad = load_gnomad_balanced(chroms=[2])
    results_gnomad = run_gnomad_benchmark(df_gnomad)
    results_gnomad_cons, cons_gnomad = run_gnomad_per_consequence(df_gnomad)

    print('\n  Generating Figure 4: gnomAD AUC heatmap...')
    plot_gnomad_heatmap(results_gnomad_cons, cons_gnomad)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print('\n' + '-' * 70)
    print('  Generating Figure 5: Summary comparison...')
    plot_summary_comparison(results_songlab, results_graphylo, results_gnomad)

    print('\n  Saving results table...')
    save_results_table(results_songlab, results_graphylo, results_gnomad)

    print('\n' + '=' * 70)
    print(' All figures saved to:', OUTPUT_DIR)
    print('=' * 70)


if __name__ == '__main__':
    main()
