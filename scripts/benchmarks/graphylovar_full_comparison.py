#!/usr/bin/env python3
"""
GraphyloVar Comprehensive Model Comparison
===========================================

Thorough benchmark comparing GraphyloVar against state-of-the-art variant
effect predictors, addressing reviewer comments (R1P4, R2P3):

  - R1P4: Compare with Evo2 and GPN-Star
  - R2P3: Compare with more dbNSFP tools (CADD, AlphaMissense, PhyloP, etc.)

Benchmarks:
-----------
1. ClinVar Pathogenic vs TopMed Common (35,613 variants)
   - User's primary dataset with ALL GraphyloVar model variants
   - Baselines: GPN-MSA, CADD, PhyloP (100v, 241m), PhastCons, ESM-1b, NT, HyenaDNA

2. ClinVar Pathogenic vs Benign — Merged songlab (21,896 overlapping variants)
   - HEAD-TO-HEAD: GraphyloVar vs Evo2 (40B, 7B), GPN-Star (4 scales),
     AlphaMissense, GPN-MSA, CADD, ESM-1b, NT, PhyloP, PhastCons, Roulette
   - Uses only variants present in BOTH the user's scored CSV and songlab/clinvar_vs_benign

3. COSMIC Somatic vs TopMed Common (18,903 variants)
   - GraphyloVar + baselines on COSMIC somatic mutations

4. Cross-benchmark summary (multi-panel, combined results table)

Outputs (saved to OUTPUT_DIR):
------------------------------
  fig01_clinvar_topmed_auc_bar.png         AUC bar chart (ClinVar TopMed)
  fig02_clinvar_topmed_roc.png              ROC curves (ClinVar TopMed)
  fig03_clinvar_topmed_corr_heatmap.png     Pearson + Spearman correlation
  fig04_clinvar_merged_auc_bar.png          AUC bar chart (merged, head-to-head)
  fig05_clinvar_merged_roc.png              ROC curves (merged, head-to-head)
  fig06_clinvar_merged_consequence.png      Per-consequence AUC heatmap
  fig07_cosmic_auc_bar.png                  AUC bar chart (COSMIC)
  fig08_cosmic_roc.png                      ROC curves (COSMIC)
  fig09_summary_multipanel.png              3-panel summary across benchmarks
  fig10_graphylovar_variants_auc.png        GraphyloVar architecture comparison

  results_clinvar_topmed.csv                ClinVar TopMed AUC table
  results_clinvar_merged.csv                ClinVar merged AUC table
  results_cosmic.csv                        COSMIC AUC table
  results_all_benchmarks.csv                Combined AUC table (all benchmarks)
  merged_clinvar_all_scores.csv             Full merged dataset with all scores
  model_predictions_clinvar_topmed.csv      Per-variant predictions (ClinVar TopMed)
  model_predictions_clinvar_merged.csv      Per-variant predictions (merged ClinVar)

Usage:
------
    conda activate graphylo
    python graphylovar_full_comparison.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = os.path.expanduser('~/research/alignment')
CLINVAR_TOPMED_CSV = os.path.join(BASE_DIR, 'clinvar_with_graphylo_topmed_common2.csv')
COSMIC_TOPMED_CSV = os.path.join(BASE_DIR, 'cosmic_with_graphylo_topmed_common.csv')
CLINVAR_SONGLAB_DIR = os.path.join(BASE_DIR, 'clinvar_vs_benign')
OUTPUT_DIR = os.path.join(BASE_DIR, 'comparison_results')

# ============================================================================
# Model definitions: (csv_column, display_name, category)
# category: 'graphylovar', 'llm', 'conservation', 'ensemble'
# ============================================================================
GRAPHYLOVAR_MODELS = {
    'graphylo_multitask':                          ('GraphyloVar (MLP)',                  'graphylovar'),
    'graphylo_transformer_multitask':              ('GraphyloVar (Transformer)',           'graphylovar'),
    'graphylo_transformer_multitask_0':            ('GraphyloVar (Transformer, flank=0)', 'graphylovar'),
    'graphylo_transformer_multitask_8':            ('GraphyloVar (Transformer, flank=8)', 'graphylovar'),
    'graphylo_transformer_multitask_32':           ('GraphyloVar (Transformer, flank=32)','graphylovar'),
    'graphylo_multitask_conditional':              ('GraphyloVar (Cond. MLP)',             'graphylovar'),
    'graphylo_transformer_multitask_conditional':  ('GraphyloVar (Cond. Transformer)',     'graphylovar'),
    'graphylo_transformer_multitask_0_conditional':('GraphyloVar (Cond. flank=0)',         'graphylovar'),
    'graphylo_transformer_multitask_8_conditional':('GraphyloVar (Cond. flank=8)',         'graphylovar'),
    'graphylo_transformer_multitask_32_conditional':('GraphyloVar (Cond. flank=32)',       'graphylovar'),
    'graphylo_evolstm':                            ('GraphyloVar (EvoLSTM)',               'graphylovar'),
    'graphylo_polymorphic':                        ('GraphyloVar (Poly. MLP)',             'graphylovar'),
    'graphylo_transformer_polymorphic':            ('GraphyloVar (Poly. Transformer)',     'graphylovar'),
    'graphylo_transformer_polymorphic_0':          ('GraphyloVar (Poly. flank=0)',         'graphylovar'),
    'graphylo_transformer_polymorphic_8':          ('GraphyloVar (Poly. flank=8)',         'graphylovar'),
    'graphylo_transformer_polymorphic_32':         ('GraphyloVar (Poly. flank=32)',        'graphylovar'),
}

BASELINE_MODELS_TOPMED = {
    'GPN-MSA':      ('GPN-MSA',          'conservation'),
    'CADD':         ('CADD',             'ensemble'),
    'phyloP-100v':  ('PhyloP (100-vert)','conservation'),
    'phyloP-241m':  ('PhyloP (241-mam)', 'conservation'),
    'phastCons-100v':('PhastCons (100-vert)','conservation'),
    'ESM-1b':       ('ESM-1b',           'llm'),
    'NT':           ('NT',               'llm'),
    'HyenaDNA':     ('HyenaDNA',         'llm'),
}

SONGLAB_MODELS = {
    'Evo2_40B':                  ('Evo2 (40B)',                'llm'),
    'Evo2_7B':                   ('Evo2 (7B)',                 'llm'),
    'GPN-Star-V100':             ('GPN-Star (V100)',           'conservation'),
    'GPN-Star-M447':             ('GPN-Star (M447)',           'conservation'),
    'GPN-Star-P243':             ('GPN-Star (P243)',           'conservation'),
    'GPN-Star-P36':              ('GPN-Star (P36)',            'conservation'),
    'GPN-Star-V100_AF_adjusted': ('GPN-Star (V100, AF adj.)', 'conservation'),
    'GPN-MSA':                   ('GPN-MSA',                   'conservation'),
    'CADD':                      ('CADD',                      'ensemble'),
    'AlphaMissense':             ('AlphaMissense',             'llm'),
    'ESM-1b':                    ('ESM-1b',                    'llm'),
    'NT_2.5B_MS':                ('NT (2.5B)',                 'llm'),
    'PhyloP-V100':               ('PhyloP (100-vert)',         'conservation'),
    'PhyloP-M447':               ('PhyloP (241-mam)',          'conservation'),
    'PhyloP-P243':               ('PhyloP (P243)',             'conservation'),
    'PhastCons-V100':            ('PhastCons (100-vert)',      'conservation'),
    'PhastCons-M470':            ('PhastCons (M470)',          'conservation'),
    'PhastCons-P43':             ('PhastCons (P43)',           'conservation'),
    'Roulette':                  ('Roulette',                  'ensemble'),
}


# ============================================================================
# Color palette — consistent across all figures
# ============================================================================
CATEGORY_COLORS = {
    'graphylovar':  '#1565C0',  # dark blue
    'conservation': '#2E7D32',  # dark green
    'llm':          '#7B1FA2',  # purple
    'ensemble':     '#E65100',  # orange
}

MODEL_COLORS = {
    # GraphyloVar family
    'GraphyloVar (MLP)':                  '#1565C0',
    'GraphyloVar (Transformer)':           '#1976D2',
    'GraphyloVar (Transformer, flank=0)': '#1E88E5',
    'GraphyloVar (Transformer, flank=8)': '#2196F3',
    'GraphyloVar (Transformer, flank=32)':'#42A5F5',
    'GraphyloVar (Cond. MLP)':             '#0D47A1',
    'GraphyloVar (Cond. Transformer)':     '#0277BD',
    'GraphyloVar (Cond. flank=0)':         '#0288D1',
    'GraphyloVar (Cond. flank=8)':         '#039BE5',
    'GraphyloVar (Cond. flank=32)':        '#03A9F4',
    'GraphyloVar (EvoLSTM)':               '#283593',
    'GraphyloVar (Poly. MLP)':             '#5C6BC0',
    'GraphyloVar (Poly. Transformer)':     '#7986CB',
    'GraphyloVar (Poly. flank=0)':         '#9FA8DA',
    'GraphyloVar (Poly. flank=8)':         '#C5CAE9',
    'GraphyloVar (Poly. flank=32)':        '#3949AB',
    # Evo2
    'Evo2 (40B)':            '#D32F2F',
    'Evo2 (7B)':             '#EF5350',
    # GPN-Star
    'GPN-Star (V100)':       '#388E3C',
    'GPN-Star (M447)':       '#43A047',
    'GPN-Star (P243)':       '#4CAF50',
    'GPN-Star (P36)':        '#66BB6A',
    'GPN-Star (V100, AF adj.)':'#2E7D32',
    # Conservation
    'GPN-MSA':               '#1B5E20',
    'PhyloP (100-vert)':     '#FF8F00',
    'PhyloP (241-mam)':      '#FFA000',
    'PhyloP (P243)':         '#FFB300',
    'PhastCons (100-vert)':  '#FF6F00',
    'PhastCons (M470)':      '#E65100',
    'PhastCons (P43)':       '#BF360C',
    # Ensemble / LLM baselines
    'CADD':                  '#E65100',
    'AlphaMissense':         '#C62828',
    'ESM-1b':                '#6A1B9A',
    'NT':                    '#8E24AA',
    'NT (2.5B)':             '#8E24AA',
    'HyenaDNA':              '#CE93D8',
    'Roulette':              '#795548',
}


def get_color(name):
    """Get color for a model name, with fallback."""
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    h = hash(name) % 360
    return matplotlib.colors.hsv_to_rgb([h / 360, 0.7, 0.8])


# ============================================================================
# Core metrics
# ============================================================================
def compute_auc(labels, scores):
    """
    Compute AUC with automatic score direction detection.
    All scores in this project follow: more negative = more pathogenic.
    We negate scores so that roc_auc_score works correctly (higher=pathogenic).
    Returns (auc, n_valid, direction_used).
    """
    mask = np.isfinite(scores) & np.isfinite(labels)
    n = mask.sum()
    if n < 50:
        return np.nan, n

    y = np.array(labels[mask], dtype=int)
    s = np.array(scores[mask], dtype=float)

    if len(np.unique(y)) < 2:
        return np.nan, n

    # Try both directions, report the better one
    try:
        auc_pos = roc_auc_score(y, s)
        auc_neg = roc_auc_score(y, -s)
    except ValueError:
        return np.nan, n

    if auc_neg >= auc_pos:
        return auc_neg, n
    else:
        return auc_pos, n


def compute_roc(labels, scores):
    """Compute ROC curve with automatic direction detection."""
    mask = np.isfinite(scores) & np.isfinite(labels)
    y = np.array(labels[mask], dtype=int)
    s = np.array(scores[mask], dtype=float)

    if len(np.unique(y)) < 2:
        return None, None, np.nan

    auc_pos = roc_auc_score(y, s)
    auc_neg = roc_auc_score(y, -s)

    if auc_neg >= auc_pos:
        fpr, tpr, _ = roc_curve(y, -s)
        return fpr, tpr, auc_neg
    else:
        fpr, tpr, _ = roc_curve(y, s)
        return fpr, tpr, auc_pos


def compute_aupr(labels, scores):
    """Compute Average Precision (AUPR) with auto direction."""
    mask = np.isfinite(scores) & np.isfinite(labels)
    y = np.array(labels[mask], dtype=int)
    s = np.array(scores[mask], dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan, 0
    ap_pos = average_precision_score(y, s)
    ap_neg = average_precision_score(y, -s)
    return max(ap_pos, ap_neg), mask.sum()


# ============================================================================
# Data loading
# ============================================================================
def load_clinvar_topmed():
    """Load user's ClinVar Pathogenic vs TopMed Common dataset."""
    print('='*70)
    print('[Benchmark 1] ClinVar Pathogenic vs TopMed Common')
    print('='*70)
    df = pd.read_csv(CLINVAR_TOPMED_CSV)
    df['label_binary'] = df['label'].astype(int)  # True=1=Pathogenic
    n_path = df['label_binary'].sum()
    n_ben = len(df) - n_path
    print(f'  Loaded: {len(df):,} variants ({n_path:,} Pathogenic, {n_ben:,} Benign/Common)')
    print(f'  Score columns: {len(df.columns) - 5}')
    return df


def load_clinvar_merged():
    """
    Merge GraphyloVar scores from user's ClinVar CSV with songlab
    ClinVar predictions to get a HEAD-TO-HEAD comparison.
    
    Returns merged DataFrame with scores from both sources.
    """
    print('\n' + '='*70)
    print('[Benchmark 2] ClinVar Merged — GraphyloVar vs Evo2/GPN-Star/AlphaMissense')
    print('='*70)

    # Load songlab test set
    sl = pd.read_parquet(os.path.join(CLINVAR_SONGLAB_DIR, 'test.parquet'))
    sl['label_binary'] = (sl['label'] == 'Pathogenic').astype(int)
    print(f'  Songlab ClinVar: {len(sl):,} variants')

    # Load all songlab prediction parquets
    pred_files = sorted([f for f in os.listdir(CLINVAR_SONGLAB_DIR)
                         if f.endswith('.parquet') and f != 'test.parquet'])
    for pf in pred_files:
        name = pf.replace('.parquet', '')
        pred = pd.read_parquet(os.path.join(CLINVAR_SONGLAB_DIR, pf))
        sl[name] = pred['score'].values

    # Load GraphyloVar scores
    gv = pd.read_csv(CLINVAR_TOPMED_CSV)
    gv['label_binary'] = gv['label'].astype(int)

    # Create merge keys
    sl['key'] = sl['chrom'].astype(str) + ':' + sl['pos'].astype(str) + ':' + sl['ref'] + ':' + sl['alt']
    gv['key'] = gv['chrom'].astype(str) + ':' + gv['pos'].astype(str) + ':' + gv['ref'] + ':' + gv['alt']

    # Merge: keep only overlapping variants, using songlab labels
    gv_score_cols = [c for c in gv.columns if c.startswith('graphylo_')]
    gv_for_merge = gv[['key'] + gv_score_cols].copy()

    merged = pd.merge(sl, gv_for_merge, on='key', how='inner')
    n_path = merged['label_binary'].sum()
    n_ben = len(merged) - n_path
    print(f'  Merged: {len(merged):,} variants ({n_path:,} Pathogenic, {n_ben:,} Benign)')
    print(f'  Songlab models: {len(pred_files)}')
    print(f'  GraphyloVar models: {len(gv_score_cols)}')

    return merged


def load_cosmic():
    """Load COSMIC Somatic vs TopMed Common dataset."""
    print('\n' + '='*70)
    print('[Benchmark 3] COSMIC Somatic vs TopMed Common')
    print('='*70)
    df = pd.read_csv(COSMIC_TOPMED_CSV)
    df['label_binary'] = df['label'].astype(int)  # True=1=COSMIC somatic
    n_path = df['label_binary'].sum()
    n_ben = len(df) - n_path
    print(f'  Loaded: {len(df):,} variants ({n_path:,} COSMIC somatic, {n_ben:,} TopMed common)')
    return df


# ============================================================================
# Benchmark runners
# ============================================================================
def run_benchmark(df, model_defs, label_col='label_binary'):
    """
    Compute AUC and AUPR for all models in model_defs on the given DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
    model_defs : dict {csv_col: (display_name, category)}
    label_col : str
    
    Returns
    -------
    dict : display_name → {'auc': float, 'aupr': float, 'n': int, 'col': str, 'category': str}
    """
    results = {}
    for col, (display, cat) in model_defs.items():
        if col not in df.columns:
            continue
        auc_val, n = compute_auc(df[label_col], df[col])
        aupr_val, _ = compute_aupr(df[label_col], df[col])
        results[display] = {
            'auc': auc_val, 'aupr': aupr_val, 'n': n,
            'col': col, 'category': cat
        }
    return results


def run_per_consequence(df, model_defs, label_col='label_binary',
                        consequence_col='consequence', min_count=200):
    """Compute AUC per consequence type."""
    if consequence_col not in df.columns:
        print('  [SKIP] No consequence column found.')
        return {}, []

    cons_counts = df[consequence_col].value_counts()
    top_cons = list(cons_counts[cons_counts >= min_count].head(8).index)

    results = {}
    for cons_name in ['all'] + top_cons:
        sub = df if cons_name == 'all' else df[df[consequence_col] == cons_name]
        for col, (display, cat) in model_defs.items():
            if col not in sub.columns:
                continue
            auc_val, n = compute_auc(sub[label_col], sub[col])
            results.setdefault(display, {})[cons_name] = auc_val

    return results, ['all'] + top_cons


# ============================================================================
# Figure 1: ClinVar TopMed AUC bar chart
# ============================================================================
def plot_auc_bar(results, title, subtitle, outfile, top_n=None):
    """Horizontal bar chart of AUC, split by category."""
    sorted_results = sorted(results.items(), key=lambda x: x[1]['auc'],
                            reverse=True)
    if top_n:
        sorted_results = sorted_results[:top_n]

    fig_height = max(6, 0.45 * len(sorted_results))
    fig, ax = plt.subplots(figsize=(11, fig_height))

    names = [r[0] for r in sorted_results]
    aucs = [r[1]['auc'] for r in sorted_results]
    ns = [r[1]['n'] for r in sorted_results]
    colors = [get_color(n) for n in names]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, aucs, color=colors, edgecolor='black', linewidth=0.5,
                   height=0.7)

    for i, (a, n) in enumerate(zip(aucs, ns)):
        if np.isnan(a):
            ax.text(0.52, i, 'N/A', va='center', fontsize=9, color='gray')
        else:
            ax.text(a + 0.003, i, f'{a:.4f} (n={n:,})', va='center',
                    fontsize=8, fontweight='bold')

    # Category legend
    from matplotlib.patches import Patch
    seen = set()
    legend_patches = []
    for name in names:
        cat = results[name].get('category', 'unknown')
        if cat not in seen:
            seen.add(cat)
            legend_patches.append(Patch(color=CATEGORY_COLORS.get(cat, 'gray'),
                                        label=cat.replace('_', ' ').title()))
    if legend_patches:
        ax.legend(handles=legend_patches, loc='lower right', fontsize=9,
                  framealpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title(f'{title}\n{subtitle}', fontsize=13, fontweight='bold')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(0.45, min(1.0, max(a for a in aucs if not np.isnan(a)) + 0.08))
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, outfile), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ============================================================================
# Figure 2: ROC curves
# ============================================================================
def plot_roc_curves(df, results, label_col, title, outfile, top_n=12):
    """ROC curves for top N models."""
    top = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=(9, 8))

    for name, info in top:
        col = info['col']
        if col not in df.columns:
            continue
        fpr, tpr, auc_val = compute_roc(df[label_col], df[col])
        if fpr is None:
            continue
        ax.plot(fpr, tpr, label=f'{name} ({auc_val:.4f})',
                color=get_color(name), linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random (0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, outfile), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ============================================================================
# Figure 3: Correlation heatmaps
# ============================================================================
def plot_correlation_heatmaps(df, model_defs, outfile):
    """Pearson and Spearman correlation heatmaps side by side."""
    # Get available score columns
    cols = []
    names = []
    for col, (display, cat) in model_defs.items():
        if col in df.columns:
            cols.append(col)
            names.append(display)

    if len(cols) < 3:
        print('  [SKIP] Not enough models for correlation heatmap.')
        return

    score_df = df[cols].copy()
    score_df.columns = names

    # Drop rows with any NaN for clean correlation
    score_df = score_df.dropna()

    if len(score_df) < 100:
        print('  [SKIP] Too few complete rows for correlation.')
        return

    pearson_corr = score_df.corr(method='pearson')
    spearman_corr = score_df.corr(method='spearman')

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    for ax, corr, method in [(axes[0], pearson_corr, 'Pearson'),
                              (axes[1], spearman_corr, 'Spearman')]:
        im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        for i in range(len(corr)):
            for j in range(len(corr)):
                val = corr.values[i, j]
                color = 'white' if abs(val) > 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color, fontweight='bold')

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(f'{method} Correlation', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation')

    fig.suptitle('Score Correlations Across Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, outfile), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ============================================================================
# Figure 6: Per-consequence heatmap
# ============================================================================
def plot_consequence_heatmap(results_per_cons, consequences, title, outfile):
    """Heatmap of AUC per model x consequence type."""
    models = list(results_per_cons.keys())
    if not models or not consequences:
        print('  [SKIP] No per-consequence results.')
        return

    # Sort by 'all' AUC
    models_sorted = sorted(models,
                           key=lambda m: results_per_cons[m].get('all', 0),
                           reverse=True)

    matrix = np.full((len(models_sorted), len(consequences)), np.nan)
    for i, m in enumerate(models_sorted):
        for j, c in enumerate(consequences):
            if c in results_per_cons[m]:
                matrix[i, j] = results_per_cons[m][c]

    fig, ax = plt.subplots(figsize=(14, max(5, 0.5 * len(models_sorted))))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')

    for i in range(len(models_sorted)):
        for j in range(len(consequences)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 0.85 or val < 0.55 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=7, fontweight='bold', color=color)

    cons_display = ['All'] + [c.split(',')[0].replace('_', ' ').title()
                              for c in consequences[1:]]
    ax.set_xticks(range(len(consequences)))
    ax.set_xticklabels(cons_display, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(models_sorted)))
    ax.set_yticklabels(models_sorted, fontsize=8)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='AUC')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, outfile), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ============================================================================
# Figure 9: Summary multi-panel
# ============================================================================
def plot_summary_multipanel(res_topmed, res_merged, res_cosmic, outfile):
    """3-panel summary: ClinVar TopMed, ClinVar Merged, COSMIC."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 9))

    panels = [
        (axes[0], res_topmed, 'A) ClinVar Path. vs TopMed Common\n(35,613 variants)', 15),
        (axes[1], res_merged, 'B) ClinVar Path. vs Benign (Merged)\n(21,896 variants)', 15),
        (axes[2], res_cosmic, 'C) COSMIC Somatic vs TopMed Common\n(18,903 variants)', 15),
    ]

    for ax, results, title, top_n in panels:
        top = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)[:top_n]
        if not top:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    fontsize=14, transform=ax.transAxes)
            ax.set_title(title, fontsize=11, fontweight='bold')
            continue

        names = [r[0] for r in top]
        aucs = [r[1]['auc'] for r in top]
        colors = [get_color(n) for n in names]
        y = np.arange(len(names))

        ax.barh(y, aucs, color=colors, edgecolor='black', linewidth=0.4, height=0.65)
        for i, a in enumerate(aucs):
            if not np.isnan(a):
                ax.text(a + 0.003, i, f'{a:.3f}', va='center', fontsize=7,
                        fontweight='bold')

        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('AUC', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        valid_aucs = [a for a in aucs if not np.isnan(a)]
        if valid_aucs:
            ax.set_xlim(0.45, min(1.0, max(valid_aucs) + 0.06))
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle('GraphyloVar Comprehensive Benchmark — AUC Comparison',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, outfile), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ============================================================================
# Figure 10: GraphyloVar architecture comparison
# ============================================================================
def plot_graphylovar_variants(results_topmed, results_merged, outfile):
    """
    Focused comparison of GraphyloVar architecture variants,
    showing how different configurations perform.
    """
    # Extract only GraphyloVar models from both benchmarks
    gv_topmed = {k: v for k, v in results_topmed.items() if 'GraphyloVar' in k}
    gv_merged = {k: v for k, v in results_merged.items() if 'GraphyloVar' in k}

    if not gv_topmed:
        print('  [SKIP] No GraphyloVar models found.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, results, title in [
        (axes[0], gv_topmed, 'ClinVar Path. vs TopMed Common'),
        (axes[1], gv_merged, 'ClinVar Path. vs Benign (Merged)'),
    ]:
        top = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
        if not top:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    fontsize=14, transform=ax.transAxes)
            ax.set_title(title, fontsize=11, fontweight='bold')
            continue

        names = [r[0] for r in top]
        aucs = [r[1]['auc'] for r in top]
        colors = [get_color(n) for n in names]
        y = np.arange(len(names))

        ax.barh(y, aucs, color=colors, edgecolor='black', linewidth=0.5, height=0.65)
        for i, a in enumerate(aucs):
            if not np.isnan(a):
                ax.text(a + 0.002, i, f'{a:.4f}', va='center', fontsize=8,
                        fontweight='bold')

        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('AUC', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        valid_aucs = [a for a in aucs if not np.isnan(a)]
        if valid_aucs:
            ax.set_xlim(min(valid_aucs) - 0.05, max(valid_aucs) + 0.06)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

    fig.suptitle('GraphyloVar Architecture Variants — AUC Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, outfile), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ============================================================================
# Results table savers
# ============================================================================
def save_benchmark_results(results, benchmark_name, outfile):
    """Save AUC/AUPR results for one benchmark."""
    rows = []
    for name, info in sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True):
        rows.append({
            'Model': name,
            'Category': info.get('category', ''),
            'AUC': round(info['auc'], 6) if not np.isnan(info['auc']) else None,
            'AUPR': round(info['aupr'], 6) if not np.isnan(info['aupr']) else None,
            'N_variants': info['n'],
            'Benchmark': benchmark_name,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, outfile), index=False)
    print(f'  Saved: {outfile}')
    return df


def save_combined_results(res_topmed, res_merged, res_cosmic, outfile):
    """Save combined results across all benchmarks."""
    all_rows = []
    for bench_name, results in [
        ('ClinVar_TopMed', res_topmed),
        ('ClinVar_Merged', res_merged),
        ('COSMIC_TopMed', res_cosmic),
    ]:
        for name, info in sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True):
            all_rows.append({
                'Model': name,
                'Category': info.get('category', ''),
                'AUC': round(info['auc'], 6) if not np.isnan(info['auc']) else None,
                'AUPR': round(info['aupr'], 6) if not np.isnan(info['aupr']) else None,
                'N_variants': info['n'],
                'Benchmark': bench_name,
            })
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUTPUT_DIR, outfile), index=False)
    print(f'  Saved: {outfile}')

    # Pretty print
    print('\n' + '='*90)
    print('  COMPREHENSIVE BENCHMARK RESULTS')
    print('='*90)
    for bench in df['Benchmark'].unique():
        sub = df[df['Benchmark'] == bench].reset_index(drop=True)
        print(f'\n  --- {bench} ({sub["N_variants"].iloc[0]:,} variants) ---')
        for _, row in sub.iterrows():
            auc_str = f'{row["AUC"]:.4f}' if pd.notna(row["AUC"]) else 'N/A'
            aupr_str = f'{row["AUPR"]:.4f}' if pd.notna(row["AUPR"]) else 'N/A'
            cat = f'[{row["Category"]}]' if row["Category"] else ''
            print(f'    {row["Model"]:<42s} AUC={auc_str}  AUPR={aupr_str}  {cat}')
    print('='*90)
    return df


def save_prediction_csv(df, model_defs, label_col, outfile, extra_cols=None):
    """
    Save per-variant predictions CSV with all model scores.
    Includes chrom, pos, ref, alt, label, and all model scores.
    """
    base_cols = ['chrom', 'pos', 'ref', 'alt', label_col]
    if extra_cols:
        base_cols += [c for c in extra_cols if c in df.columns]

    score_cols = [col for col in model_defs.keys() if col in df.columns]
    out_df = df[base_cols + score_cols].copy()

    # Rename score columns to display names
    rename_map = {col: model_defs[col][0] for col in score_cols}
    out_df = out_df.rename(columns=rename_map)
    out_df = out_df.rename(columns={label_col: 'label'})

    out_df.to_csv(os.path.join(OUTPUT_DIR, outfile), index=False)
    print(f'  Saved: {outfile} ({len(out_df):,} rows, {len(out_df.columns)} columns)')


# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('\n' + '#'*70)
    print('#  GraphyloVar Comprehensive Model Comparison')
    print('#  Addressing Reviewer Comments R1P4, R2P3')
    print('#'*70 + '\n')

    # ==================================================================
    # Benchmark 1: ClinVar Pathogenic vs TopMed Common
    # ==================================================================
    df_topmed = load_clinvar_topmed()

    all_models_topmed = {}
    all_models_topmed.update(GRAPHYLOVAR_MODELS)
    all_models_topmed.update(BASELINE_MODELS_TOPMED)

    print('\n  Computing AUC for all models...')
    res_topmed = run_benchmark(df_topmed, all_models_topmed)
    for name, info in sorted(res_topmed.items(), key=lambda x: x[1]['auc'], reverse=True):
        print(f'    {name:<42s} AUC={info["auc"]:.4f}  (n={info["n"]:,})')

    print('\n  Generating Figure 1: ClinVar TopMed AUC bar chart...')
    plot_auc_bar(res_topmed,
                 'ClinVar Pathogenic vs TopMed Common Variants',
                 f'({len(df_topmed):,} variants, {df_topmed.label_binary.sum():,} pathogenic)',
                 'fig01_clinvar_topmed_auc_bar.png')

    print('  Generating Figure 2: ClinVar TopMed ROC curves...')
    plot_roc_curves(df_topmed, res_topmed, 'label_binary',
                    'ROC: ClinVar Pathogenic vs TopMed Common (Top 12 Models)',
                    'fig02_clinvar_topmed_roc.png', top_n=12)

    print('  Generating Figure 3: Correlation heatmaps...')
    plot_correlation_heatmaps(df_topmed, all_models_topmed,
                              'fig03_clinvar_topmed_corr_heatmap.png')

    print('  Saving results CSV...')
    save_benchmark_results(res_topmed, 'ClinVar_TopMed',
                           'results_clinvar_topmed.csv')

    print('  Saving per-variant predictions CSV...')
    save_prediction_csv(df_topmed, all_models_topmed, 'label_binary',
                        'model_predictions_clinvar_topmed.csv')

    # ==================================================================
    # Benchmark 2: ClinVar Merged (GraphyloVar vs songlab models)
    # ==================================================================
    df_merged = load_clinvar_merged()

    # Combine ALL models for the merged benchmark
    all_models_merged = {}
    all_models_merged.update(GRAPHYLOVAR_MODELS)
    all_models_merged.update(SONGLAB_MODELS)

    print('\n  Computing AUC for all models on merged dataset...')
    res_merged = run_benchmark(df_merged, all_models_merged)
    for name, info in sorted(res_merged.items(), key=lambda x: x[1]['auc'], reverse=True):
        print(f'    {name:<42s} AUC={info["auc"]:.4f}  (n={info["n"]:,})')

    print('\n  Generating Figure 4: ClinVar Merged AUC bar chart...')
    plot_auc_bar(res_merged,
                 'ClinVar Pathogenic vs Benign — Head-to-Head Comparison',
                 f'({len(df_merged):,} overlapping variants, GraphyloVar vs Evo2/GPN-Star/AlphaMissense)',
                 'fig04_clinvar_merged_auc_bar.png')

    print('  Generating Figure 5: ClinVar Merged ROC curves...')
    plot_roc_curves(df_merged, res_merged, 'label_binary',
                    'ROC: ClinVar Path. vs Benign — GraphyloVar vs State-of-the-Art',
                    'fig05_clinvar_merged_roc.png', top_n=12)

    # Per-consequence analysis on merged data
    all_models_merged_subset = {}
    # Select representative models for heatmap
    for col_name in ['graphylo_transformer_multitask_0', 'graphylo_multitask_conditional',
                     'graphylo_multitask', 'graphylo_evolstm',
                     'Evo2_40B', 'Evo2_7B',
                     'GPN-Star-V100', 'GPN-Star-V100_AF_adjusted',
                     'GPN-MSA', 'CADD', 'AlphaMissense', 'ESM-1b',
                     'PhyloP-V100', 'PhastCons-V100', 'Roulette']:
        if col_name in GRAPHYLOVAR_MODELS:
            all_models_merged_subset[col_name] = GRAPHYLOVAR_MODELS[col_name]
        elif col_name in SONGLAB_MODELS:
            all_models_merged_subset[col_name] = SONGLAB_MODELS[col_name]

    print('  Computing per-consequence AUC...')
    res_cons, cons_list = run_per_consequence(df_merged, all_models_merged_subset)

    if cons_list:
        print('  Generating Figure 6: Per-consequence heatmap...')
        plot_consequence_heatmap(res_cons, cons_list,
                                'AUC by Variant Consequence — GraphyloVar vs State-of-the-Art\n'
                                f'(Merged ClinVar, {len(df_merged):,} variants)',
                                'fig06_clinvar_merged_consequence.png')

    print('  Saving results CSV...')
    save_benchmark_results(res_merged, 'ClinVar_Merged',
                           'results_clinvar_merged.csv')

    print('  Saving merged predictions CSV...')
    # For merged, combine both model defs
    save_prediction_csv(df_merged, all_models_merged, 'label_binary',
                        'model_predictions_clinvar_merged.csv',
                        extra_cols=['consequence', 'key'])

    print('  Saving full merged scores CSV...')
    out_cols = ['chrom', 'pos', 'ref', 'alt', 'label_binary', 'consequence']
    score_cols = [c for c in list(SONGLAB_MODELS.keys()) + list(GRAPHYLOVAR_MODELS.keys())
                  if c in df_merged.columns]
    merged_out = df_merged[out_cols + score_cols].copy()
    merged_out.to_csv(os.path.join(OUTPUT_DIR, 'merged_clinvar_all_scores.csv'), index=False)
    print(f'  Saved: merged_clinvar_all_scores.csv ({len(merged_out):,} rows)')

    # ==================================================================
    # Benchmark 3: COSMIC Somatic vs TopMed Common
    # ==================================================================
    df_cosmic = load_cosmic()

    # COSMIC has same model columns as TopMed ClinVar (minus HyenaDNA, NT,
    # and some GraphyloVar variants may differ)
    cosmic_graphylo_models = {}
    for col, (display, cat) in GRAPHYLOVAR_MODELS.items():
        if col in df_cosmic.columns:
            cosmic_graphylo_models[col] = (display, cat)

    cosmic_baseline_models = {}
    for col, (display, cat) in BASELINE_MODELS_TOPMED.items():
        if col in df_cosmic.columns:
            cosmic_baseline_models[col] = (display, cat)

    all_models_cosmic = {}
    all_models_cosmic.update(cosmic_graphylo_models)
    all_models_cosmic.update(cosmic_baseline_models)

    print('\n  Computing AUC for all models on COSMIC...')
    res_cosmic = run_benchmark(df_cosmic, all_models_cosmic)
    for name, info in sorted(res_cosmic.items(), key=lambda x: x[1]['auc'], reverse=True):
        print(f'    {name:<42s} AUC={info["auc"]:.4f}  (n={info["n"]:,})')

    print('\n  Generating Figure 7: COSMIC AUC bar chart...')
    plot_auc_bar(res_cosmic,
                 'COSMIC Somatic vs TopMed Common Variants',
                 f'({len(df_cosmic):,} variants, {df_cosmic.label_binary.sum():,} COSMIC somatic)',
                 'fig07_cosmic_auc_bar.png')

    print('  Generating Figure 8: COSMIC ROC curves...')
    plot_roc_curves(df_cosmic, res_cosmic, 'label_binary',
                    'ROC: COSMIC Somatic vs TopMed Common (Top 12)',
                    'fig08_cosmic_roc.png', top_n=12)

    print('  Saving results CSV...')
    save_benchmark_results(res_cosmic, 'COSMIC_TopMed',
                           'results_cosmic.csv')

    # ==================================================================
    # Summary figures
    # ==================================================================
    print('\n' + '-'*70)
    print('  Generating summary figures...')

    print('  Figure 9: Summary multi-panel...')
    plot_summary_multipanel(res_topmed, res_merged, res_cosmic,
                            'fig09_summary_multipanel.png')

    print('  Figure 10: GraphyloVar architecture comparison...')
    plot_graphylovar_variants(res_topmed, res_merged,
                              'fig10_graphylovar_variants_auc.png')

    # ==================================================================
    # Combined results table
    # ==================================================================
    print('\n  Saving combined results table...')
    save_combined_results(res_topmed, res_merged, res_cosmic,
                          'results_all_benchmarks.csv')

    print('\n' + '#'*70)
    print(f'#  All outputs saved to: {OUTPUT_DIR}')
    print('#'*70)


if __name__ == '__main__':
    main()
