#!/usr/bin/env python
"""
Score variants with Evo2 via NVIDIA NIM API.

Evo2 requires:
  - GPU: Compute Capability 8.9+ (Ada/Hopper) — NOT available on RTX 6000 (7.5)
  - Python 3.12 — server has 3.8
  - So we use the NVIDIA hosted API instead.

This script takes a CSV of variants (chr, pos, ref, alt) and computes
zero-shot variant effect scores using Evo2's log-likelihood ratio:
    score = logP(sequence_with_alt) - logP(sequence_with_ref)

Usage:
    # Set your NVIDIA API key
    export NVCF_RUN_KEY="your-api-key-from-build.nvidia.com"

    python score_evo2_api.py \\
        --input df_with_graphylo_context8_context32_chr2.csv \\
        --output evo2_scores_chr2.csv \\
        --context-size 512 \\
        --batch-size 100

    # Get API key from: https://build.nvidia.com/arc/evo2-40b
    # (Free tier available with rate limits)

NOTE: The NVIDIA API has rate limits. For 716K variants, this would require
significant API usage. Consider scoring a representative subset first.
For the revision, we suggest either:
1. Scoring a stratified sample of ~10K variants (enough for robust AUC/MAF analysis)
2. Using a collaborator's machine with Ada/Hopper GPU
3. Noting Evo2 comparison as future work (it's trained on all domains of life,
   not specifically optimized for human variant effect prediction)

Author: Dongjoon Lim
Date: 2025-02
"""

import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import requests


NVIDIA_API_URL = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/logprob"
# Alternative: use evo2-7b for faster processing
NVIDIA_API_URL_7B = "https://health.api.nvidia.com/v1/biology/arc/evo2-7b/logprob"


def get_reference_sequence(chrom, start, end, genome_fasta=None):
    """
    Get reference sequence from local genome FASTA.

    If no local FASTA, falls back to UCSC DAS API (slow but works).
    """
    if genome_fasta:
        try:
            import pysam
            fasta = pysam.FastaFile(genome_fasta)
            seq = fasta.fetch(chrom, start, end)
            fasta.close()
            return seq.upper()
        except Exception as e:
            print(f'  [WARN] pysam FASTA fetch failed: {e}')

    # Fallback: UCSC DAS API
    url = f'https://api.genome.ucsc.edu/getData/sequence?genome=hg38&chrom={chrom}&start={start}&end={end}'
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        return data.get('dna', '').upper()
    except Exception as e:
        print(f'  [WARN] UCSC API failed: {e}')
        return None


def score_variant_evo2_api(sequence_ref, sequence_alt, api_key, model='40b'):
    """
    Compute Evo2 zero-shot VEP score via NVIDIA NIM API.

    Returns log-likelihood ratio: logP(alt_seq) - logP(ref_seq)
    """
    url = NVIDIA_API_URL if model == '40b' else NVIDIA_API_URL_7B
    headers = {"Authorization": f"Bearer {api_key}"}

    scores = {}
    for label, seq in [('ref', sequence_ref), ('alt', sequence_alt)]:
        payload = {"sequence": seq}
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                result = r.json()
                # The API returns per-position log probabilities
                logprobs = result.get('logprobs', [])
                scores[label] = sum(logprobs) if logprobs else None
            elif r.status_code == 429:
                # Rate limited — wait and retry
                time.sleep(5)
                r = requests.post(url, headers=headers, json=payload, timeout=60)
                if r.status_code == 200:
                    result = r.json()
                    logprobs = result.get('logprobs', [])
                    scores[label] = sum(logprobs) if logprobs else None
            else:
                print(f'  API error {r.status_code}: {r.text[:200]}')
                return None
        except Exception as e:
            print(f'  Request failed: {e}')
            return None

    if scores.get('ref') is not None and scores.get('alt') is not None:
        return scores['alt'] - scores['ref']
    return None


def main():
    parser = argparse.ArgumentParser(description='Score variants with Evo2 via NVIDIA API')
    parser.add_argument('--input', required=True, help='Input CSV with variants')
    parser.add_argument('--output', default='evo2_scores.csv', help='Output CSV')
    parser.add_argument('--context-size', type=int, default=512,
                        help='Flanking context (bp) on each side of variant')
    parser.add_argument('--n-sample', type=int, default=None,
                        help='Score only a random sample of N variants')
    parser.add_argument('--genome-fasta', default=None,
                        help='Path to hg38.fa for local sequence retrieval')
    parser.add_argument('--model', choices=['7b', '40b'], default='7b',
                        help='Evo2 model size (default: 7b)')
    args = parser.parse_args()

    api_key = os.environ.get('NVCF_RUN_KEY')
    if not api_key:
        print('ERROR: Set NVCF_RUN_KEY environment variable.')
        print('Get your key at: https://build.nvidia.com/arc/evo2-40b')
        sys.exit(1)

    # Load variants
    df = pd.read_csv(args.input)
    print(f'Loaded {len(df):,} variants from {args.input}')

    if args.n_sample and args.n_sample < len(df):
        df = df.sample(n=args.n_sample, random_state=42)
        print(f'Sampled {args.n_sample:,} variants')

    # Score each variant
    results = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 100 == 0:
            print(f'  Progress: {i}/{total} ({100 * i / total:.1f}%)')

        chrom = row['chrom'] if 'chrom' in row.index else f"chr{row.get('chrom_nochr', '2')}"
        pos = int(row.get('chromEnd', row.get('pos', 0)))
        ref = row['ref']

        # Determine alt allele
        if 'minorAllele_topmed' in row.index:
            minor = row['minorAllele_topmed']
            major = row.get('majorAllele_topmed', '')
            alt = minor if ref != minor else major
        elif 'alt' in row.index:
            alt = row['alt']
        else:
            results.append(np.nan)
            continue

        if not isinstance(alt, str) or len(alt) != 1:
            results.append(np.nan)
            continue

        # Get flanking sequence
        start = pos - args.context_size - 1  # 0-based
        end = pos + args.context_size
        seq = get_reference_sequence(chrom, start, end, args.genome_fasta)

        if not seq or len(seq) < 2 * args.context_size:
            results.append(np.nan)
            continue

        # Create ref and alt sequences
        center = args.context_size
        seq_ref = seq
        seq_alt = seq[:center] + alt + seq[center + 1:]

        # Score via API
        score = score_variant_evo2_api(seq_ref, seq_alt, api_key, model=args.model)
        results.append(score)

        # Rate limit protection
        time.sleep(0.1)

    df['evo2_score'] = results
    valid = sum(1 for r in results if r is not None and not np.isnan(r))
    print(f'\nCompleted: {valid}/{total} variants scored')

    df.to_csv(args.output, index=False)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
