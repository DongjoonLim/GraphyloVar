#!/usr/bin/env python
# Copyright (c) 2025 Dongjoon Lim, McGill University
# Licensed under the MIT License. See LICENSE file in the project root.
"""
Extract chromosome-specific TOPMed labels from UCSC dbSNP bigBed.

The script streams records from `dbSnp155.bb` via `bigBedToBed`, derives the
TOPMed allele frequency fields from the third source slot used in the existing
notebooks, and writes compact per-chromosome TSV files.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphylovar.topmed import allele_to_index, parse_chromosome_spec  # noqa: E402


DEFAULT_BB = "https://hgdownload.soe.ucsc.edu/gbdb/hg38/snp/dbSnp155.bb"


def get_nth_element(text: str, index: int) -> str:
    parts = str(text).split(",")
    if index < len(parts):
        return parts[index].strip()
    return ""


def safe_float(value: str) -> float | None:
    value = str(value).strip()
    if not value or value.lower() in {"nan", "na", "none"}:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if math.isinf(out) or math.isnan(out):
        return None
    return out


def build_vector(major_allele: str, minor_allele: str, minor_freq: float) -> list[float]:
    vector = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    major_freq = max(0.0, 1.0 - minor_freq)
    vector[allele_to_index(major_allele)] = major_freq
    if minor_freq > 0:
        idx = allele_to_index(minor_allele)
        vector[idx] += minor_freq
    return vector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract TOPMed labels from UCSC dbSNP bigBed")
    parser.add_argument("--chromosomes", default="1-22")
    parser.add_argument("--dbsnp_bb", default=DEFAULT_BB)
    parser.add_argument("--bigbed_to_bed",
                        default="/home/mcb/users/dlim63/research/alignment/tools/bigBedToBed")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gzip", action="store_true", default=True)
    return parser


def process_chromosome(chromosome: int, dbsnp_bb: str, bigbed_to_bed: str, output_dir: str) -> dict:
    chrom_label = f"chr{chromosome}"
    out_path = Path(output_dir) / f"topmed_chr{chromosome}.tsv.gz"
    summary = {
        "chromosome": chromosome,
        "output": str(out_path),
        "rows_seen": 0,
        "snv_rows": 0,
        "topmed_rows": 0,
        "written": 0,
    }

    cmd = [bigbed_to_bed, f"-chrom={chrom_label}", dbsnp_bb, "stdout"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    header = [
        "chrom", "chromStart", "chromEnd", "name", "ref", "class",
        "majorAllele_topmed", "minorAllele_topmed",
        "majorAlleleFreq_topmed", "minorAlleleFreq_topmed",
        "allele_frequency_vector",
    ]

    with gzip.open(out_path, "wt", encoding="utf-8") as out_handle:
        out_handle.write("\t".join(header) + "\n")
        assert proc.stdout is not None
        for line in proc.stdout:
            summary["rows_seen"] += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 17:
                continue

            chrom, chrom_start, chrom_end, name = fields[:4]
            ref = fields[4]
            minor_freqs = fields[9]
            major_alleles = fields[10]
            minor_alleles = fields[11]
            cls = fields[13]

            if cls.lower() != "snv":
                continue
            summary["snv_rows"] += 1

            minor_freq = safe_float(get_nth_element(minor_freqs, 2))
            major_allele = get_nth_element(major_alleles, 2)
            minor_allele = get_nth_element(minor_alleles, 2)
            if minor_freq is None or not major_allele or not minor_allele:
                continue

            summary["topmed_rows"] += 1
            major_freq = max(0.0, 1.0 - minor_freq)
            vector = build_vector(major_allele, minor_allele, minor_freq)
            out_handle.write(
                f"{chrom}\t{chrom_start}\t{chrom_end}\t{name}\t{ref}\t{cls}\t"
                f"{major_allele}\t{minor_allele}\t{major_freq}\t{minor_freq}\t"
                f"{json.dumps(vector)}\n"
            )
            summary["written"] += 1

    stderr = proc.communicate()[1]
    if proc.returncode != 0:
        raise RuntimeError(f"bigBedToBed failed for {chrom_label}: {stderr}")

    with open(Path(output_dir) / f"topmed_chr{chromosome}.summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(
        f"chr{chromosome}: seen={summary['rows_seen']:,} snv={summary['snv_rows']:,} "
        f"topmed={summary['topmed_rows']:,} written={summary['written']:,}"
    )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    summaries = []
    for chromosome in parse_chromosome_spec(args.chromosomes):
        summaries.append(
            process_chromosome(
                chromosome=chromosome,
                dbsnp_bb=args.dbsnp_bb,
                bigbed_to_bed=args.bigbed_to_bed,
                output_dir=args.output_dir,
            )
        )

    with open(Path(args.output_dir) / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)


if __name__ == "__main__":
    main()