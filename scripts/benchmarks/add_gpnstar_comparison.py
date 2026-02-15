#!/usr/bin/env python
"""
Add GPN-Star (and optionally Evo2) scores to the MAF comparison pipeline.

This script:
1. Loads the pre-scored dbSNP chr2 CSV (df_with_graphylo_context8_context32_chr2.csv)
2. Downloads GPN-Star precomputed scores from songlab/gnomad_balanced (HuggingFace)
3. Matches dbSNP variants to gnomAD by (chrom, pos, ref, alt)
4. Computes all existing scores (CADD, PhastCons, PhyloP, GPN-MSA) + new GPN-Star
5. Generates updated Figure 1A (AUC bar chart) and Figure 1B (MAF bin plot)

Requirements:
    conda activate graphylo
    # Packages: pandas, numpy, pyBigWig, pysam, scipy, sklearn, matplotlib, pyarrow

Usage:
    python add_gpnstar_comparison.py [--chr CHR] [--download-only] [--skip-download]

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
import pyBigWig
import pysam
from io import StringIO
from scipy.stats import t as t_dist
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATHS = {
    'scored_csv': os.path.join(BASE_DIR, 'df_with_graphylo_context8_context32_chr2.csv'),
    'cadd_bw': os.path.join(BASE_DIR, 'data', 'CADD_GRCh38-v1.7.bw'),
    'phastcons_bw': os.path.join(BASE_DIR, 'data', 'hg38.phastCons100way.bw'),
    'phylop_bw': os.path.join(BASE_DIR, 'data', 'hg38.phyloP100way.bw'),
    'gpnmsa_bgz': os.path.join(BASE_DIR, 'GPNMSA', 'scores.tsv.bgz'),
    'gpnmsa_tbi': os.path.join(BASE_DIR, 'GPNMSA', 'scores.tsv.bgz.tbi'),
    'coding_regions': os.path.join(BASE_DIR, 'coding_regions.csv'),
    '3utr': os.path.join(BASE_DIR, '3utr.bed'),
    'ccres': os.path.join(BASE_DIR, 'data', 'ccres.csv'),
    'transposable': os.path.join(BASE_DIR, 'data', 'transposable.bed'),
}

GNOMAD_DIR = os.path.join(BASE_DIR, 'gnomad_balanced')

HF_BASE_URL = 'https://huggingface.co/datasets/songlab/gnomad_balanced/resolve/main'
GNOMAD_FILES = {
    'test': f'{HF_BASE_URL}/test.parquet',
    'GPN-Star-V100': f'{HF_BASE_URL}/predictions/GPN-Star-V100.parquet',
    'GPN-Star-M447': f'{HF_BASE_URL}/predictions/GPN-Star-M447.parquet',
}

# Score definitions for figures
# 'reverse_sign': True means higher score = more conserved (negate for AUC where common=positive)
SCORES_FIGURE1A = [
    {'column': 'graphylo_conditional_score_flank32', 'display': 'GraphyloVar',   'reverse_sign': False, 'color': '#1f77b4'},
    {'column': 'cadd_score',                         'display': 'CADD',          'reverse_sign': True,  'color': '#ff7f0e'},
    {'column': 'phastcons_score',                    'display': 'PhastCons',     'reverse_sign': True,  'color': '#2ca02c'},
    {'column': 'phylop_score',                       'display': 'PhyloP',        'reverse_sign': True,  'color': '#d62728'},
    {'column': 'gpnmsa_score',                       'display': 'GPN-MSA',       'reverse_sign': False, 'color': '#9467bd'},
    {'column': 'gpnstar_v100_score',                 'display': 'GPN-Star (V)',  'reverse_sign': False, 'color': '#8c564b'},
    {'column': 'gpnstar_m447_score',                 'display': 'GPN-Star (M)',  'reverse_sign': False, 'color': '#e377c2'},
]

SCORES_FIGURE1B = [
    {'column': 'minorAlleleFreq_topmed',             'display': 'Ground Truth MAF', 'reverse_sign': False},
    {'column': 'graphylo_conditional_score_flank32', 'display': 'GraphyloVar',      'reverse_sign': False},
    {'column': 'graphylo_SNP_flank32',               'display': 'GraphyloVar SNP',  'reverse_sign': False},
    {'column': 'cadd_score',                         'display': 'CADD',             'reverse_sign': True},
    {'column': 'phastcons_score',                    'display': 'PhastCons',        'reverse_sign': True},
    {'column': 'phylop_score',                       'display': 'PhyloP',           'reverse_sign': True},
    {'column': 'gpnmsa_score',                       'display': 'GPN-MSA',          'reverse_sign': False},
    {'column': 'gpnstar_v100_score',                 'display': 'GPN-Star (V)',     'reverse_sign': False},
]

ANNOTATIONS = ['all', 'coding', '3utr', 'ccre', 'transposable', 'others']
ANNOTATION_DISPLAY = ['All', 'Coding', "3'UTR", 'cCREs', 'TE', 'Others']


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_nth_element(series, n):
    """Extract nth element from comma-separated string series."""
    return series.str.split(',').str[n].str.strip()


def safe_float(x):
    """Convert to float, handling -inf and NaN."""
    try:
        if x == '-inf' or pd.isna(x):
            return np.nan
        return float(x)
    except (ValueError, TypeError):
        return np.nan


def compute_graphylo_score(row, flank):
    """Compute log-ratio score: log(P(alt) / P(ref))."""
    ref_allele = row['ref']
    minor_allele = row['minorAllele_topmed']
    major_allele = row['majorAllele_topmed']

    if ref_allele == minor_allele:
        alt_allele = major_allele
    elif ref_allele == major_allele:
        alt_allele = minor_allele
    else:
        return np.nan

    allele_to_col = {b: f'graphylo_{b}_flank{flank}' for b in ['A', 'C', 'G', 'T']}
    if ref_allele not in allele_to_col or alt_allele not in allele_to_col:
        return np.nan

    ref_score = max(row[allele_to_col[ref_allele]], 1e-10)
    alt_score = max(row[allele_to_col[alt_allele]], 1e-10)

    if pd.isna(ref_score) or pd.isna(alt_score):
        return np.nan

    score = np.log(alt_score / ref_score)
    return np.nan if (np.isinf(score) or np.isnan(score)) else score


def compute_graphylo_conditional_score(row, flank):
    """Compute conditional score: log(P_snp * P(alt) / ((1-P_snp) + P_snp * P(ref)))."""
    ref_allele = row['ref']
    minor_allele = row['minorAllele_topmed']
    major_allele = row['majorAllele_topmed']
    graphylo_snp = row[f'graphylo_SNP_flank{flank}']

    if ref_allele == minor_allele:
        alt_allele = major_allele
    elif ref_allele == major_allele:
        alt_allele = minor_allele
    else:
        return np.nan

    allele_to_col = {b: f'graphylo_{b}_flank{flank}' for b in ['A', 'C', 'G', 'T']}
    if ref_allele not in allele_to_col or alt_allele not in allele_to_col:
        return np.nan

    ref_score_mt = max(row[allele_to_col[ref_allele]], 1e-10)
    alt_score_mt = max(row[allele_to_col[alt_allele]], 1e-10)
    graphylo_snp = max(graphylo_snp, 1e-10)

    numerator = graphylo_snp * alt_score_mt
    denominator = (1 - graphylo_snp) + graphylo_snp * ref_score_mt
    if denominator <= 0:
        return np.nan

    score = np.log(numerator / denominator)
    return np.nan if (np.isinf(score) or np.isnan(score)) else score


def get_bigwig_score(bw, chrom, end):
    """Look up a single-position score from a BigWig file."""
    try:
        c = f'chr{chrom}' if not str(chrom).startswith('chr') else str(chrom)
        end = int(end)
        start = end - 1
        vals = bw.values(c, start, end)
        if vals and len(vals) == 1 and not np.isnan(vals[0]):
            return vals[0]
        return np.nan
    except Exception:
        return np.nan


def annotate_variants(df, cds, utr3, ccre, trans, chrom='chr2'):
    """Assign genomic region annotations to variants."""
    genome_size = 250_000_000  # approximate chr2 length
    coding_arr = np.zeros(genome_size, dtype=bool)
    utr3_arr = np.zeros(genome_size, dtype=bool)
    ccre_arr = np.zeros(genome_size, dtype=bool)
    trans_arr = np.zeros(genome_size, dtype=bool)

    # Coding regions
    cds_chr = cds[cds['chrom'] == chrom] if 'chrom' in cds.columns else cds[cds.iloc[:, 0] == chrom]
    for _, row in cds_chr.iterrows():
        s, e = int(row.iloc[1]), int(row.iloc[2])
        coding_arr[s:e] = True

    # 3'UTR
    utr3_chr = utr3[utr3.iloc[:, 0] == chrom]
    for _, row in utr3_chr.iterrows():
        s, e = int(row.iloc[1]), int(row.iloc[2])
        utr3_arr[s:e] = True

    # cCREs
    ccre_chr = ccre[ccre.iloc[:, 0] == chrom] if ccre.iloc[:, 0].dtype == object else ccre[ccre['chrom'] == chrom]
    for _, row in ccre_chr.iterrows():
        s, e = int(row.iloc[1]), int(row.iloc[2])
        ccre_arr[s:e] = True

    # Transposable elements
    trans_chr = trans[trans.iloc[:, 0] == chrom]
    for _, row in trans_chr.iterrows():
        s, e = int(row.iloc[1]), int(row.iloc[2])
        trans_arr[s:e] = True

    # Assign priority: coding > 3utr > ccre > transposable > others
    def get_annotation(pos):
        p = int(pos)
        if p >= genome_size:
            return 'others'
        if coding_arr[p]:
            return 'coding'
        if utr3_arr[p]:
            return '3utr'
        if ccre_arr[p]:
            return 'ccre'
        if trans_arr[p]:
            return 'transposable'
        return 'others'

    df = df.copy()
    df['annotation'] = df['chromEnd'].apply(get_annotation)
    return df


def compute_ci(group, column='minorAlleleFreq_topmed'):
    """Compute mean + 95% CI via t-distribution."""
    n = len(group)
    mean = group[column].mean()
    std = group[column].std()
    if n < 2 or std == 0:
        return pd.Series({'mean': mean, 'ci_lower': mean, 'ci_upper': mean})
    sem = std / np.sqrt(n)
    ci = t_dist.interval(0.95, df=n - 1, loc=mean, scale=sem)
    return pd.Series({'mean': mean, 'ci_lower': ci[0], 'ci_upper': ci[1]})


# ---------------------------------------------------------------------------
# Download GPN-Star scores
# ---------------------------------------------------------------------------

def download_gnomad_files():
    """Download gnomad_balanced parquets from HuggingFace if not present."""
    os.makedirs(GNOMAD_DIR, exist_ok=True)

    for name, url in GNOMAD_FILES.items():
        local = os.path.join(GNOMAD_DIR, f'{name}.parquet')
        if os.path.exists(local) and os.path.getsize(local) > 1_000_000:
            print(f'  [skip] {name}.parquet already exists ({os.path.getsize(local) / 1e6:.1f} MB)')
            continue
        print(f'  [download] {name}.parquet from {url}')
        ret = os.system(f'wget -q -O "{local}" "{url}"')
        if ret != 0:
            print(f'  [ERROR] Failed to download {name}')
            sys.exit(1)
        print(f'  [done] {os.path.getsize(local) / 1e6:.1f} MB')


# ---------------------------------------------------------------------------
# Load and merge GPN-Star scores
# ---------------------------------------------------------------------------

def load_gpnstar_scores(chrom='2'):
    """Load gnomad_balanced + GPN-Star predictions, filter to chromosome."""
    print(f'Loading gnomad_balanced test.parquet...')
    gnomad = pd.read_parquet(os.path.join(GNOMAD_DIR, 'test.parquet'))
    print(f'  Total variants: {len(gnomad):,}')

    # Load GPN-Star predictions (index-aligned with test.parquet)
    for model in ['GPN-Star-V100', 'GPN-Star-M447']:
        pf = os.path.join(GNOMAD_DIR, f'{model}.parquet')
        if os.path.exists(pf):
            pred = pd.read_parquet(pf)
            col = model.lower().replace('-', '_') + '_score'  # gpn_star_v100_score
            # Simplified column names
            if 'V100' in model:
                col = 'gpnstar_v100_score'
            elif 'M447' in model:
                col = 'gpnstar_m447_score'
            gnomad[col] = pred['score'].values
            print(f'  Added {col}: min={pred["score"].min():.3f}, max={pred["score"].max():.3f}')

    # Filter to target chromosome
    gnomad_chr = gnomad[gnomad['chrom'] == str(chrom)].copy()
    print(f'  Chromosome {chrom}: {len(gnomad_chr):,} variants')

    return gnomad_chr


def merge_gpnstar_with_dbsnp(dbsnp_df, gnomad_chr):
    """
    Match dbSNP variants to gnomAD by (chrom, pos, ref, alt).

    dbSNP uses 'chr2' and 'chromEnd' for position.
    gnomAD uses '2' and 'pos' for position.
    """
    # Prepare dbSNP join keys
    dbsnp_df = dbsnp_df.copy()
    dbsnp_df['chrom_nochr'] = dbsnp_df['chrom'].astype(str).str.replace('chr', '', regex=False)

    # Determine alt allele for dbSNP (minor or major that differs from ref)
    def get_alt_allele(row):
        ref = row['ref']
        minor = row.get('minorAllele_topmed', '')
        major = row.get('majorAllele_topmed', '')
        if ref != minor and isinstance(minor, str) and len(minor) == 1:
            return minor
        elif ref != major and isinstance(major, str) and len(major) == 1:
            return major
        return np.nan

    dbsnp_df['alt_for_merge'] = dbsnp_df.apply(get_alt_allele, axis=1)

    # Merge
    merged = dbsnp_df.merge(
        gnomad_chr[['pos', 'ref', 'alt', 'gpnstar_v100_score', 'gpnstar_m447_score']].rename(
            columns={'ref': 'gnomad_ref', 'alt': 'gnomad_alt'}
        ),
        left_on=['chromEnd', 'alt_for_merge'],
        right_on=['pos', 'gnomad_alt'],
        how='left'
    )

    matched = merged['gpnstar_v100_score'].notna().sum()
    total = len(merged)
    print(f'  GPN-Star match: {matched:,} / {total:,} ({100 * matched / total:.1f}%)')

    # Drop temporary columns
    merged = merged.drop(columns=['chrom_nochr', 'alt_for_merge', 'pos', 'gnomad_ref', 'gnomad_alt'],
                         errors='ignore')
    return merged


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_all_scores(df):
    """Compute GraphyloVar + external scores for all variants."""
    print('\nComputing GraphyloVar scores...')

    # Convert graphylo columns to numeric
    for flank in [8, 32]:
        for allele in ['A', 'C', 'G', 'T']:
            col = f'graphylo_{allele}_flank{flank}'
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        snp_col = f'graphylo_SNP_flank{flank}'
        if snp_col in df.columns:
            df[snp_col] = pd.to_numeric(df[snp_col], errors='coerce')

    df['graphylo_score_flank32'] = df.apply(compute_graphylo_score, axis=1, flank=32)
    df['graphylo_conditional_score_flank32'] = df.apply(
        compute_graphylo_conditional_score, axis=1, flank=32
    )
    print(f'  GraphyloVar conditional_flank32: {df["graphylo_conditional_score_flank32"].notna().sum():,} valid')

    # BigWig scores
    print('Loading BigWig scores...')
    for name, path, col in [
        ('CADD', DATA_PATHS['cadd_bw'], 'cadd_score'),
        ('PhastCons', DATA_PATHS['phastcons_bw'], 'phastcons_score'),
        ('PhyloP', DATA_PATHS['phylop_bw'], 'phylop_score'),
    ]:
        if col not in df.columns:
            if os.path.exists(path):
                bw = pyBigWig.open(path)
                df[col] = df.apply(lambda r: get_bigwig_score(bw, r['chrom'], r['chromEnd']), axis=1)
                bw.close()
                print(f'  {name}: {df[col].notna().sum():,} valid')
            else:
                print(f'  [WARN] {name} BigWig not found: {path}')
                df[col] = np.nan
        else:
            print(f'  {name}: already in dataframe')

    # GPN-MSA via tabix
    if 'gpnmsa_score' not in df.columns:
        bgz = DATA_PATHS['gpnmsa_bgz']
        if os.path.exists(bgz):
            print('Loading GPN-MSA scores via tabix...')
            tabix_file = pysam.TabixFile(bgz, parser=None)

            def get_alt_nuc(row):
                if row['ref'] != row['minorAllele_topmed']:
                    return row['minorAllele_topmed']
                return row['majorAllele_topmed']

            df['alt_nuc'] = df.apply(get_alt_nuc, axis=1)
            records = []
            for chrom, pos in zip(df['chrom'], df['chromEnd']):
                try:
                    c = str(chrom).replace('chr', '')
                    for record in tabix_file.fetch(c, int(pos) - 1, int(pos)):
                        records.append(record)
                except Exception:
                    continue
            tabix_file.close()

            if records:
                gpnmsa_df = pd.read_csv(
                    StringIO('\n'.join(records)),
                    sep='\t',
                    names=['gpnmsa_chrom', 'pos', 'ref', 'alt', 'score']
                )
                df['chrom_nochr_tmp'] = df['chrom'].str.replace('chr', '', regex=False).astype(str)
                df = df.merge(
                    gpnmsa_df[['gpnmsa_chrom', 'pos', 'alt', 'score']],
                    left_on=['chrom_nochr_tmp', 'chromEnd', 'alt_nuc'],
                    right_on=['gpnmsa_chrom', 'pos', 'alt'],
                    how='left'
                )
                df = df.rename(columns={'score': 'gpnmsa_score'})
                df = df.drop(columns=['chrom_nochr_tmp', 'gpnmsa_chrom', 'pos', 'alt_nuc'],
                             errors='ignore')
                print(f'  GPN-MSA: {df["gpnmsa_score"].notna().sum():,} valid')
            else:
                df['gpnmsa_score'] = np.nan
        else:
            print(f'  [WARN] GPN-MSA tabix not found: {bgz}')
            df['gpnmsa_score'] = np.nan
    else:
        print(f'  GPN-MSA: already in dataframe')

    return df


# ---------------------------------------------------------------------------
# AUC computation
# ---------------------------------------------------------------------------

def compute_auc_table(df, scores, annotations=ANNOTATIONS):
    """Compute AUC for each score × annotation combination."""
    results = {}
    for annotation in annotations:
        if annotation == 'all':
            df_sub = df
        else:
            df_sub = df[df['annotation'] == annotation]

        for s in scores:
            col = s['column']
            if col not in df.columns:
                continue
            data = df_sub[[col, 'label']].dropna()
            if len(data) < 10:
                results.setdefault(s['display'], []).append(np.nan)
                continue
            score_vals = -data[col] if s['reverse_sign'] else data[col]
            try:
                auc = roc_auc_score(data['label'], score_vals)
            except ValueError:
                auc = np.nan
            results.setdefault(s['display'], []).append(auc)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_auc_bar_chart(auc_data, scores, annotations, annotation_display,
                       outfile='combined_auc_bar_chart_with_gpnstar.png'):
    """Generate grouped bar chart (Figure 1A style) with GPN-Star included."""
    fig, ax = plt.subplots(figsize=(16, 6))

    n_scores = len([s for s in scores if s['display'] in auc_data])
    n_annot = len(annotations)
    bar_width = 0.8 / n_scores
    index = np.arange(n_annot)

    for i, s in enumerate(scores):
        if s['display'] not in auc_data:
            continue
        vals = auc_data[s['display']]
        color = s.get('color', plt.cm.tab10(i / n_scores))
        ax.bar(index + i * bar_width, vals, bar_width,
               label=s['display'], color=color, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Genomic Region', fontsize=12)
    ax.set_ylabel('AUC (Common vs Rare)', fontsize=12)
    ax.set_title('AUC for Distinguishing Common (MAF>0.01) vs Rare Variants', fontsize=14)
    ax.set_xticks(index + bar_width * n_scores / 2)
    ax.set_xticklabels(annotation_display, fontsize=11)
    ax.set_ylim(0.48, 0.72)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_maf_bins(df, scores, annotation_label='all', n_bins=20,
                  outfile_prefix='scores_ci_subplots_with_gpnstar'):
    """Generate MAF bin subplot (Figure 1B style) with GPN-Star included."""
    y_column = 'minorAlleleFreq_topmed'

    if annotation_label != 'all':
        df = df[df['annotation'] == annotation_label]

    valid_scores = [s for s in scores if s['column'] in df.columns]
    n_plots = len(valid_scores)
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), sharex=True, sharey=True)
    axes_flat = axes.flatten() if n_plots > 1 else [axes]

    for idx, score_info in enumerate(valid_scores):
        col = score_info['column']
        data = df[[col, y_column]].dropna()
        data = data[data[y_column] > 0]  # exclude MAF=0

        if len(data) < n_bins * 5:
            axes_flat[idx].set_title(score_info['display'] + ' (insufficient data)')
            continue

        score_values = -data[col] if score_info['reverse_sign'] else data[col]
        ranks = score_values.rank(ascending=True)

        try:
            data['bin'] = pd.qcut(ranks, q=n_bins, labels=range(1, n_bins + 1)[::-1], duplicates='drop')
        except ValueError:
            axes_flat[idx].set_title(score_info['display'] + ' (binning failed)')
            continue

        stats = data.groupby('bin').apply(lambda g: compute_ci(g, y_column)).reset_index()
        stats['bin'] = stats['bin'].astype(int)
        stats = stats.sort_values('bin')

        ax = axes_flat[idx]
        ax.errorbar(
            stats['bin'], stats['mean'],
            yerr=[stats['mean'] - stats['ci_lower'], stats['ci_upper'] - stats['mean']],
            fmt='o-', capsize=3, markersize=4, linewidth=1.5
        )
        ax.set_yscale('log')
        ax.set_title(score_info['display'], fontsize=12)
        ax.set_xlabel('Score Bin (1=highest)', fontsize=10)
        ax.set_ylabel('Mean MAF', fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle(f'MAF vs Score Bins — {annotation_label.capitalize()}', fontsize=14, y=1.01)
    plt.tight_layout()
    outfile = f'{outfile_prefix}_{annotation_label}.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Add GPN-Star to MAF comparison')
    parser.add_argument('--chr', default='2', help='Chromosome (default: 2)')
    parser.add_argument('--download-only', action='store_true', help='Only download data')
    parser.add_argument('--skip-download', action='store_true', help='Skip download step')
    parser.add_argument('--skip-bigwig', action='store_true', help='Skip BigWig score recomputation')
    args = parser.parse_args()

    os.chdir(BASE_DIR)
    print('=' * 70)
    print('GraphyloVar + GPN-Star MAF Comparison Pipeline')
    print('=' * 70)

    # Step 1: Download GPN-Star data
    if not args.skip_download:
        print('\n[1/6] Downloading GPN-Star scores from HuggingFace...')
        download_gnomad_files()

    if args.download_only:
        print('Download complete. Exiting.')
        return

    # Step 2: Load scored CSV
    print('\n[2/6] Loading pre-scored dbSNP CSV...')
    df = pd.read_csv(DATA_PATHS['scored_csv'])
    print(f'  Loaded {len(df):,} variants')

    # Extract TOPMed allele frequencies
    df['minorAlleleFreq_topmed'] = get_nth_element(df['minorAlleleFreq'], 2)
    df['majorAllele_topmed'] = get_nth_element(df['majorAllele'], 2)
    df['minorAllele_topmed'] = get_nth_element(df['minorAllele'], 2)
    df['minorAlleleFreq_topmed'] = df['minorAlleleFreq_topmed'].apply(safe_float)
    df['majorAlleleFreq_topmed'] = 1 - df['minorAlleleFreq_topmed']
    df['majorAlleleFreq_topmed'] = df['majorAlleleFreq_topmed'].fillna(-1)
    df = df.dropna(subset=['minorAlleleFreq_topmed'])

    # Binary label: common (MAF > 0.01) = 1, rare = 0
    df['label'] = np.where(df['minorAlleleFreq_topmed'] > 0.01, 1, 0)
    print(f'  After filtering: {len(df):,} variants')
    print(f'  Common (MAF>0.01): {(df["label"]==1).sum():,}, Rare: {(df["label"]==0).sum():,}')

    # Step 3: Load and merge GPN-Star scores
    print('\n[3/6] Loading and merging GPN-Star scores...')
    gnomad_chr = load_gpnstar_scores(chrom=args.chr)
    df = merge_gpnstar_with_dbsnp(df, gnomad_chr)

    # Step 4: Compute all scores
    print('\n[4/6] Computing model scores...')
    df = compute_all_scores(df)

    # Drop rows with too many NaN
    required_cols = ['graphylo_conditional_score_flank32', 'label']
    df = df.dropna(subset=required_cols)
    print(f'\n  Final dataset: {len(df):,} variants')
    print(f'  Score coverage:')
    for s in SCORES_FIGURE1A:
        col = s['column']
        if col in df.columns:
            n_valid = df[col].notna().sum()
            print(f'    {s["display"]}: {n_valid:,} ({100 * n_valid / len(df):.1f}%)')

    # Step 5: Annotate genomic regions
    print('\n[5/6] Annotating genomic regions...')
    try:
        cds = pd.read_csv(DATA_PATHS['coding_regions'])
        utr3 = pd.read_csv(DATA_PATHS['3utr'], sep='\t', header=None)
        ccre = pd.read_csv(DATA_PATHS['ccres'])
        trans = pd.read_csv(DATA_PATHS['transposable'], sep='\t', header=None)
        df = annotate_variants(df, cds, utr3, ccre, trans)
        print(f'  Annotations: {df["annotation"].value_counts().to_dict()}')
    except FileNotFoundError as e:
        print(f'  [WARN] Annotation file not found: {e}')
        print(f'  Using "all" annotation only.')
        df['annotation'] = 'others'

    # Step 6: Generate figures
    print('\n[6/6] Generating comparison figures...')

    # Figure 1A: AUC bar chart
    print('  Computing AUC scores...')
    auc_data = compute_auc_table(df, SCORES_FIGURE1A)
    print('  AUC results:')
    for name, vals in auc_data.items():
        all_auc = vals[0] if vals else np.nan
        print(f'    {name}: All={all_auc:.4f}')

    plot_auc_bar_chart(auc_data, SCORES_FIGURE1A, ANNOTATIONS, ANNOTATION_DISPLAY)

    # Figure 1B: MAF bin plots
    for annotation in ['all', 'coding', 'ccre']:
        plot_maf_bins(df, SCORES_FIGURE1B, annotation_label=annotation)

    # Save augmented CSV
    out_csv = f'df_with_gpnstar_chr{args.chr}.csv'
    df.to_csv(out_csv, index=False)
    print(f'\n  Saved augmented CSV: {out_csv}')

    print('\n' + '=' * 70)
    print('Done! New figures saved with GPN-Star included.')
    print('=' * 70)


if __name__ == '__main__':
    main()
