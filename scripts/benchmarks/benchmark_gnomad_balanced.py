#!/usr/bin/env python
"""
Benchmark GraphyloVar on the gnomad_balanced dataset from songlab/GPN-Star.

This is a STANDARD benchmark (12M gnomAD variants, balanced common vs rare)
with precomputed scores for: CADD, GPN-MSA, GPN-Star (3 scales), PhastCons,
PhyloP, Roulette. We add GraphyloVar scores and compare all methods.

This script:
1. Loads gnomad_balanced test.parquet + all prediction parquets
2. Scores chr2 variants with GraphyloVar using the user's model + alignment data
3. Generates AUC comparison figure (new supplementary figure)

NOTE: Steps 2 requires the GraphyloVar model and alignment pickle files.
      If not available, the script can be run in --compare-only mode using
      the matched CSV from add_gpnstar_comparison.py.

Usage:
    # Full benchmark (requires GraphyloVar model + alignments):
    python benchmark_gnomad_balanced.py --model-dir models/graphylo_mutation_transformer_flank32_multitask_topmed

    # Compare-only mode (uses pre-scored CSV):
    python benchmark_gnomad_balanced.py --compare-only --scored-csv df_with_gpnstar_chr2.csv

Author: Dongjoon Lim
Date: 2025-02
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GNOMAD_DIR = os.path.join(BASE_DIR, 'gnomad_balanced')

HF_BASE_URL = 'https://huggingface.co/datasets/songlab/gnomad_balanced/resolve/main'

ALL_PREDICTION_FILES = {
    'CADD': f'{HF_BASE_URL}/predictions/CADD.parquet',
    'GPN-MSA': f'{HF_BASE_URL}/predictions/GPN-MSA.parquet',
    'GPN-Star-V100': f'{HF_BASE_URL}/predictions/GPN-Star-V100.parquet',
    'GPN-Star-M447': f'{HF_BASE_URL}/predictions/GPN-Star-M447.parquet',
    'GPN-Star-P243': f'{HF_BASE_URL}/predictions/GPN-Star-P243.parquet',
    'PhastCons-V100': f'{HF_BASE_URL}/predictions/PhastCons-V100.parquet',
    'PhyloP-V100': f'{HF_BASE_URL}/predictions/PhyloP-V100.parquet',
    'Roulette': f'{HF_BASE_URL}/predictions/Roulette.parquet',
}

# Model definitions for comparison
MODELS = [
    {'column': 'GraphyloVar',     'display': 'GraphyloVar',       'reverse_sign': False, 'color': '#1f77b4'},
    {'column': 'GPN-Star-V100',   'display': 'GPN-Star (Vert.)',  'reverse_sign': False, 'color': '#8c564b'},
    {'column': 'GPN-Star-M447',   'display': 'GPN-Star (Mam.)',   'reverse_sign': False, 'color': '#e377c2'},
    {'column': 'GPN-Star-P243',   'display': 'GPN-Star (Prim.)',  'reverse_sign': False, 'color': '#17becf'},
    {'column': 'GPN-MSA',         'display': 'GPN-MSA',           'reverse_sign': False, 'color': '#9467bd'},
    {'column': 'CADD',            'display': 'CADD',              'reverse_sign': True,  'color': '#ff7f0e'},
    {'column': 'PhastCons-V100',  'display': 'PhastCons',         'reverse_sign': True,  'color': '#2ca02c'},
    {'column': 'PhyloP-V100',     'display': 'PhyloP',            'reverse_sign': True,  'color': '#d62728'},
    {'column': 'Roulette',        'display': 'Roulette',          'reverse_sign': False, 'color': '#bcbd22'},
]


def download_all_predictions():
    """Download all prediction parquets from HuggingFace."""
    os.makedirs(GNOMAD_DIR, exist_ok=True)
    for name, url in ALL_PREDICTION_FILES.items():
        local = os.path.join(GNOMAD_DIR, f'{name}.parquet')
        if os.path.exists(local) and os.path.getsize(local) > 100_000:
            continue
        print(f'  Downloading {name}...')
        os.system(f'wget -q -O "{local}" "{url}"')


def load_benchmark_data(chroms=None):
    """Load gnomad_balanced with all prediction columns."""
    print('Loading gnomad_balanced...')
    df = pd.read_parquet(os.path.join(GNOMAD_DIR, 'test.parquet' if os.path.exists(
        os.path.join(GNOMAD_DIR, 'test.parquet')) else 'test_sample.parquet'))

    if chroms:
        df = df[df['chrom'].isin([str(c) for c in chroms])].copy()
        print(f'  Filtered to chrom(s) {chroms}: {len(df):,}')

    # Add prediction columns
    for name in ALL_PREDICTION_FILES.keys():
        pf = os.path.join(GNOMAD_DIR, f'{name}.parquet')
        if os.path.exists(pf):
            pred = pd.read_parquet(pf)
            if chroms:
                pred = pred.iloc[df.index]
            df[name] = pred['score'].values
            print(f'  {name}: loaded ({df[name].notna().sum():,} valid)')
        else:
            print(f'  {name}: NOT FOUND (skipping)')

    return df


def compute_auc_per_consequence(df, models, top_k=None):
    """Compute AUC for each model × consequence type."""
    consequences = ['all'] + list(df['consequence'].value_counts().head(6).index)
    results = {}

    for consequence in consequences:
        subdf = df if consequence == 'all' else df[df['consequence'] == consequence]
        if top_k and len(subdf) > top_k:
            subdf = subdf.sample(n=top_k, random_state=42)

        for m in models:
            col = m['column']
            if col not in subdf.columns:
                results.setdefault(m['display'], []).append(np.nan)
                continue
            data = subdf[[col, 'label']].dropna()
            if len(data) < 50:
                results.setdefault(m['display'], []).append(np.nan)
                continue

            scores = -data[col] if m['reverse_sign'] else data[col]
            try:
                auc = roc_auc_score(data['label'].astype(int), scores)
            except ValueError:
                auc = np.nan
            results.setdefault(m['display'], []).append(auc)

    return results, consequences


def plot_benchmark_auc(results, consequences, outfile='gnomad_benchmark_auc.png'):
    """Plot AUC comparison on gnomad_balanced benchmark."""
    fig, ax = plt.subplots(figsize=(16, 6))

    models_with_data = [k for k, v in results.items() if not all(np.isnan(x) for x in v)]
    n_models = len(models_with_data)
    bar_width = 0.8 / max(n_models, 1)
    index = np.arange(len(consequences))

    # Get colors from MODELS list
    color_map = {m['display']: m.get('color', f'C{i}') for i, m in enumerate(MODELS)}

    for i, name in enumerate(models_with_data):
        vals = results[name]
        color = color_map.get(name, f'C{i}')
        ax.bar(index + i * bar_width, vals, bar_width,
               label=name, color=color, edgecolor='black', linewidth=0.5)

    consequence_display = ['All'] + [c.replace('_', ' ').title() for c in consequences[1:]]
    ax.set_xlabel('Consequence', fontsize=12)
    ax.set_ylabel('AUC (Common vs Rare)', fontsize=12)
    ax.set_title('gnomAD Balanced Benchmark: AUC for Common vs Rare Classification', fontsize=13)
    ax.set_xticks(index + bar_width * n_models / 2)
    ax.set_xticklabels(consequence_display, fontsize=10, rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9, ncol=2)
    ax.set_ylim(0.45, 0.75)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def main():
    parser = argparse.ArgumentParser(description='Benchmark on gnomad_balanced')
    parser.add_argument('--compare-only', action='store_true',
                        help='Skip GraphyloVar scoring, use existing scores')
    parser.add_argument('--scored-csv', default=None,
                        help='Pre-scored CSV (for --compare-only mode)')
    parser.add_argument('--chroms', nargs='+', default=['2'],
                        help='Chromosomes to analyze (default: 2)')
    parser.add_argument('--download', action='store_true',
                        help='Download all prediction parquets')
    args = parser.parse_args()

    os.chdir(BASE_DIR)
    print('=' * 70)
    print('gnomAD Balanced Benchmark — GraphyloVar vs GPN-Star et al.')
    print('=' * 70)

    # Download predictions
    if args.download:
        print('\nDownloading prediction parquets...')
        download_all_predictions()

    # Load benchmark data
    print('\nLoading benchmark data...')
    df = load_benchmark_data(chroms=args.chroms)

    print(f'\nDataset: {len(df):,} variants')
    print(f'  Common (label=True): {df["label"].sum():,}')
    print(f'  Rare (label=False): {(~df["label"]).sum():,}')

    # TODO: Add GraphyloVar scoring here when model + alignments are available
    # For now, skip GraphyloVar and compare available models
    if not args.compare_only and 'GraphyloVar' not in df.columns:
        print('\n[NOTE] GraphyloVar scores not available for gnomad_balanced variants.')
        print('  To score with GraphyloVar, run the model on these variants using')
        print('  the seqDictPad alignment files + your trained model.')
        print('  Proceeding with other models only.')

    # Filter to models that have data
    available_models = [m for m in MODELS if m['column'] in df.columns]
    if not available_models:
        print('\n[ERROR] No model predictions found. Run with --download first.')
        sys.exit(1)

    print(f'\nAvailable models: {[m["display"] for m in available_models]}')

    # Compute AUC
    print('\nComputing AUC...')
    results, consequences = compute_auc_per_consequence(df, available_models)

    print('\nAUC Results (All variants):')
    for name, vals in results.items():
        print(f'  {name}: {vals[0]:.4f}')

    # Plot
    plot_benchmark_auc(results, consequences)

    # Save results table
    auc_df = pd.DataFrame(results, index=consequences).T
    auc_df.columns = ['All'] + [c.replace('_', ' ').title() for c in consequences[1:]]
    auc_df.to_csv('gnomad_benchmark_auc_table.csv')
    print(f'\nSaved: gnomad_benchmark_auc_table.csv')
    print(auc_df.to_string())

    print('\n' + '=' * 70)
    print('Done!')
    print('=' * 70)


if __name__ == '__main__':
    main()
