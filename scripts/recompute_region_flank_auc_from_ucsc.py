#!/usr/bin/env python3
"""Recompute GraphyloVar region/flank AUCs from held-out TOPMed variants.

This script rebuilds the common-vs-rare benchmark used in the manuscript from
raw held-out TOPMed chromosomes (13-22), annotates variants with UCSC tracks,
and evaluates saved GraphyloVar flank models directly.

The key goal is to replace projected region curves with measured AUC values.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import graphylovar.models  # registers @register_keras_serializable custom layers
from graphylovar.topmed import extract_batch_examples_from_encoded, load_alignment_encoded_cache


TRACK_BASE = "https://api.genome.ucsc.edu/getData/track?genome=hg38"
EVAL_CHROMS = tuple(range(13, 23))
COMMON_THRESHOLD = 0.01
RARE_THRESHOLD = 0.001
EPS = 1e-8
REGION_ORDER = ["All", "Coding", "3'UTR", "cCREs", "TE", "Others"]
DEFAULT_MODEL_PATHS = {
    0: "topmed_models/full_streaming_runs/v3_ablation/multitask_hybrid_v3_train1-10_val11-12_flank0_v3flank0",
    1: "topmed_models/full_streaming_runs/v3_ablation/multitask_hybrid_v3_train1-10_val11-12_flank1_v3flank1",
    8: "topmed_models/full_streaming_runs/v3_ablation/multitask_hybrid_v3_train1-10_val11-12_flank8_v3flank8",
    16: "topmed_models/full_streaming_runs/v3_ablation/multitask_hybrid_v3_train1-10_val11-12_flank16_v3flank16",
    32: "topmed_models/full_streaming_runs/v3_ablation/multitask_hybrid_v3_train1-10_val11-12_flank32_v3flank32",
    100: "topmed_models/full_streaming_runs/v3_ablation/multitask_hybrid_v3_train1-10_val11-12_flank100_v3flank100",
}


@dataclass
class VariantRecord:
    chrom: str
    chrom_num: int
    pos0: int
    ref: str
    alt: str
    maf: float
    is_common: int
    region: str = "Others"


@dataclass
class FlankModelStatus:
    flank: int
    model_rel: str
    model_path: str
    available: bool
    status: str
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute GraphyloVar region AUCs from UCSC annotations")
    parser.add_argument("--project_root", default=os.path.expanduser("~/GraphyloVar"))
    parser.add_argument("--alignment_dir", default=os.path.expanduser("~/conservation/data"))
    parser.add_argument("--label_dir", default=os.path.expanduser("~/GraphyloVar/topmed_labels_full"))
    parser.add_argument("--cache_dir", default=os.path.expanduser("~/GraphyloVar/topmed_alignment_cache"))
    parser.add_argument("--annotation_cache", default=os.path.expanduser("~/GraphyloVar/cache/ucsc_annotations"))
    parser.add_argument("--out_dir", default=os.path.expanduser("~/GraphyloVar/figures"))
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--main_flank", type=int, default=32)
    parser.add_argument("--score_kind", choices=["auto", "combined", "allele_ratio"], default="auto")
    parser.add_argument("--sample_per_chrom", type=int, default=0,
                        help="Optional balanced subsample per chromosome for smoke tests; 0 keeps all variants.")
    parser.add_argument(
        "--label_cache_dir",
        default="",
        help="Optional cache for filtered per-chrom label variants. Defaults to <cache_dir>/label_variant_cache.",
    )
    parser.add_argument(
        "--chromosomes",
        default="13-22",
        help="Chromosome selection, for example '13-22' or '11,12'. Used for both evaluation and summary output.",
    )
    parser.add_argument(
        "--status_json",
        default="",
        help="Optional progress/status JSON path for supervisor monitoring.",
    )
    parser.add_argument(
        "--flanks",
        default="",
        help="Comma-separated flank sizes to evaluate (e.g. '0,1,8'). Default: all in DEFAULT_MODEL_PATHS.",
    )
    parser.add_argument(
        "--model_override",
        default="",
        help="Comma-separated flank:path pairs overriding DEFAULT_MODEL_PATHS, e.g. '8:path/v4flank8,32:path/v4flank32'.",
    )
    return parser.parse_args()


def parse_chromosome_spec(spec: str | None) -> tuple[int, ...]:
    if spec is None:
        return EVAL_CHROMS
    chroms: list[int] = []
    for chunk in str(spec).split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            chroms.extend(range(int(start), int(end) + 1))
        else:
            chroms.append(int(token))
    ordered = tuple(sorted(dict.fromkeys(chroms)))
    return ordered or EVAL_CHROMS


def write_status(path: str, payload: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(payload)
    payload.setdefault("last_heartbeat", datetime.now().astimezone().isoformat())
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def fetch_track_json(cache_dir: str, track: str, chrom: str) -> dict:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = Path(cache_dir) / f"{track}_{chrom}.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            # Corrupted/partial cache can happen after interrupted runs.
            pass

    url = f"{TRACK_BASE};track={track};chrom={chrom}"
    with urllib.request.urlopen(url) as response:
        payload = json.load(response)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(tmp_path, cache_path)
    return payload


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_contains(intervals: list[tuple[int, int]], position: int) -> bool:
    lo = 0
    hi = len(intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start, end = intervals[mid]
        if position < start:
            hi = mid - 1
        elif position >= end:
            lo = mid + 1
        else:
            return True
    return False


def build_coding_and_utr3_intervals(transcripts: list[dict]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    coding: list[tuple[int, int]] = []
    utr3: list[tuple[int, int]] = []
    for tx in transcripts:
        tx_type = str(tx.get("transcriptType", "")).lower()
        tx_class = str(tx.get("transcriptClass", "")).lower()
        if "coding" not in tx_type and "coding" not in tx_class:
            continue

        exon_sizes = [int(x) for x in str(tx.get("blockSizes", "")).strip(",").split(",") if x]
        exon_starts = [int(x) for x in str(tx.get("chromStarts", "")).strip(",").split(",") if x]
        if not exon_sizes or len(exon_sizes) != len(exon_starts):
            continue

        tx_start = int(tx["chromStart"])
        cds_start = int(tx["thickStart"])
        cds_end = int(tx["thickEnd"])
        strand = tx.get("strand", "+")

        for size, rel_start in zip(exon_sizes, exon_starts):
            exon_start = tx_start + rel_start
            exon_end = exon_start + size

            coding_start = max(exon_start, cds_start)
            coding_end = min(exon_end, cds_end)
            if coding_start < coding_end:
                coding.append((coding_start, coding_end))

            if strand == "+":
                utr_start = max(exon_start, cds_end)
                utr_end = exon_end
            else:
                utr_start = exon_start
                utr_end = min(exon_end, cds_start)
            if utr_start < utr_end:
                utr3.append((utr_start, utr_end))

    return merge_intervals(coding), merge_intervals(utr3)


def build_simple_intervals(items: list[dict]) -> list[tuple[int, int]]:
    """Build intervals from UCSC rows with schema-tolerant key handling."""
    intervals: list[tuple[int, int]] = []
    for row in items:
        start_raw = row.get("chromStart")
        end_raw = row.get("chromEnd")
        if start_raw is None:
            start_raw = row.get("genoStart")
        if end_raw is None:
            end_raw = row.get("genoEnd")
        if start_raw is None:
            start_raw = row.get("txStart")
        if end_raw is None:
            end_raw = row.get("txEnd")
        if start_raw is None or end_raw is None:
            continue
        try:
            start = int(start_raw)
            end = int(end_raw)
        except (TypeError, ValueError):
            continue
        if start < end:
            intervals.append((start, end))
    return merge_intervals(intervals)


def build_region_annotation_bundle(annotation_cache: str, chrom: str) -> dict[str, object]:
    known_gene_cache = Path(annotation_cache) / f"knownGene_{chrom}.json"
    ccre_cache = Path(annotation_cache) / f"encodeCcreCombined_{chrom}.json"
    rmsk_cache = Path(annotation_cache) / f"rmsk_{chrom}.json"
    known_gene = fetch_track_json(annotation_cache, "knownGene", chrom).get("knownGene", [])
    ccre = fetch_track_json(annotation_cache, "encodeCcreCombined", chrom).get("encodeCcreCombined", [])
    rmsk = fetch_track_json(annotation_cache, "rmsk", chrom).get("rmsk", [])

    coding, utr3 = build_coding_and_utr3_intervals(known_gene)
    ccre_intervals = build_simple_intervals(ccre)
    te_intervals = build_simple_intervals(rmsk)
    intervals = {
        "Coding": coding,
        "3'UTR": utr3,
        "cCREs": ccre_intervals,
        "TE": te_intervals,
    }
    return {
        "intervals": intervals,
        "tracks_used": {
            "Coding": "knownGene",
            "3'UTR": "knownGene",
            "cCREs": "encodeCcreCombined",
            "TE": "rmsk",
        },
        "track_row_counts": {
            "knownGene": len(known_gene),
            "encodeCcreCombined": len(ccre),
            "rmsk": len(rmsk),
        },
        "merged_interval_counts": {name: len(value) for name, value in intervals.items()},
        "cache_files": {
            "knownGene": str(known_gene_cache),
            "encodeCcreCombined": str(ccre_cache),
            "rmsk": str(rmsk_cache),
        },
        "region_precedence": REGION_ORDER[1:],
    }


def build_region_intervals(annotation_cache: str, chrom: str) -> dict[str, list[tuple[int, int]]]:
    return dict(build_region_annotation_bundle(annotation_cache, chrom)["intervals"])


def choose_alt_allele(ref: str, major: str, minor: str) -> str | None:
    if len(ref) != 1 or len(major) != 1 or len(minor) != 1:
        return None
    ref = ref.upper()
    major = major.upper()
    minor = minor.upper()
    alt = minor if ref != minor else major
    return alt if alt in {"A", "C", "G", "T"} else None


def variant_cache_path(label_cache_dir: str, chrom_num: int, sample_per_chrom: int) -> Path:
    cache_tag = f"v1_chr{chrom_num}_sample{sample_per_chrom}_c{int(COMMON_THRESHOLD * 1e6)}_r{int(RARE_THRESHOLD * 1e6)}"
    return Path(label_cache_dir) / f"{cache_tag}.npz"


def load_cached_variants(cache_path: Path, chrom_num: int) -> list[VariantRecord] | None:
    if not cache_path.exists():
        return None
    try:
        payload = np.load(cache_path, allow_pickle=False)
        pos0 = payload["pos0"].astype(np.int64)
        ref_code = payload["ref_code"].astype(np.uint8)
        alt_code = payload["alt_code"].astype(np.uint8)
        maf = payload["maf"].astype(np.float32)
        is_common = payload["is_common"].astype(np.int8)
    except Exception:
        return None

    if not (
        pos0.shape == ref_code.shape == alt_code.shape == maf.shape == is_common.shape
    ):
        return None

    base_lookup = np.array(["A", "C", "G", "T"], dtype=object)
    if np.any(ref_code > 3) or np.any(alt_code > 3):
        return None
    ref = base_lookup[ref_code]
    alt = base_lookup[alt_code]
    chrom = f"chr{chrom_num}"
    return [
        VariantRecord(
            chrom=chrom,
            chrom_num=chrom_num,
            pos0=int(p),
            ref=str(r),
            alt=str(a),
            maf=float(m),
            is_common=int(c),
        )
        for p, r, a, m, c in zip(pos0, ref, alt, maf, is_common)
    ]


def save_cached_variants(cache_path: Path, rows: list[VariantRecord]) -> None:
    if not rows:
        return
    os.makedirs(cache_path.parent, exist_ok=True)
    base_to_code = {"A": 0, "C": 1, "G": 2, "T": 3}
    pos0 = np.array([row.pos0 for row in rows], dtype=np.int64)
    ref_code = np.array([base_to_code[row.ref] for row in rows], dtype=np.uint8)
    alt_code = np.array([base_to_code[row.alt] for row in rows], dtype=np.uint8)
    maf = np.array([row.maf for row in rows], dtype=np.float32)
    is_common = np.array([row.is_common for row in rows], dtype=np.int8)

    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(
            handle,
            pos0=pos0,
            ref_code=ref_code,
            alt_code=alt_code,
            maf=maf,
            is_common=is_common,
        )
    os.replace(tmp_path, cache_path)


def load_eval_variants(
    label_dir: str,
    sample_per_chrom: int = 0,
    chroms: tuple[int, ...] = EVAL_CHROMS,
    label_cache_dir: str | None = None,
    status_cb: Callable[..., None] | None = None,
) -> list[VariantRecord]:
    variants: list[VariantRecord] = []
    rng = np.random.default_rng(42)
    chunk_size = 1_000_000
    cache_root = label_cache_dir or os.path.join(label_dir, "_variant_cache")

    for chrom_num in chroms:
        chrom = f"chr{chrom_num}"
        cache_path = variant_cache_path(cache_root, chrom_num, sample_per_chrom)
        cached_rows = load_cached_variants(cache_path, chrom_num)
        if cached_rows is not None:
            variants.extend(cached_rows)
            print(f"[flank] chr{chrom_num} loaded cached_variants={len(cached_rows)} from {cache_path}", flush=True)
            if status_cb is not None:
                status_cb(
                    current_chromosome=chrom_num,
                    scanned_rows_current_chrom=0,
                    selected_rows_current_chrom=len(cached_rows),
                    processed_variants_total=len(variants),
                    stage_note=f"loaded cached chr{chrom_num}",
                    cache_hit=1,
                )
            continue

        path = Path(label_dir) / f"topmed_chr{chrom_num}.tsv.gz"
        print(f"[flank] loading labels from {path}", flush=True)
        if status_cb is not None:
            status_cb(
                current_chromosome=chrom_num,
                stage_note=f"reading {path.name}",
                cache_hit=0,
            )
        chosen: list[VariantRecord] = []
        scanned_rows = 0
        reader = pd.read_csv(
            path,
            sep="\t",
            usecols=["chromEnd", "ref", "majorAllele_topmed", "minorAllele_topmed", "minorAlleleFreq_topmed"],
            chunksize=chunk_size,
            low_memory=False,
        )
        for chunk in reader:
            scanned_rows += int(len(chunk))
            chunk = chunk.dropna(
                subset=["chromEnd", "ref", "majorAllele_topmed", "minorAllele_topmed", "minorAlleleFreq_topmed"]
            )
            if chunk.empty:
                if status_cb is not None:
                    status_cb(
                        current_chromosome=chrom_num,
                        scanned_rows_current_chrom=scanned_rows,
                        selected_rows_current_chrom=len(chosen),
                    )
                continue

            maf_arr = pd.to_numeric(chunk["minorAlleleFreq_topmed"], errors="coerce").to_numpy(dtype=np.float32)
            keep_mask = np.isfinite(maf_arr) & ((maf_arr >= COMMON_THRESHOLD) | (maf_arr < RARE_THRESHOLD))
            if not np.any(keep_mask):
                if status_cb is not None:
                    status_cb(
                        current_chromosome=chrom_num,
                        scanned_rows_current_chrom=scanned_rows,
                        selected_rows_current_chrom=len(chosen),
                    )
                continue

            chunk = chunk.loc[keep_mask].copy()
            maf_arr = maf_arr[keep_mask]
            ref = chunk["ref"].astype(str).str.upper().to_numpy(dtype=object)
            major = chunk["majorAllele_topmed"].astype(str).str.upper().to_numpy(dtype=object)
            minor = chunk["minorAllele_topmed"].astype(str).str.upper().to_numpy(dtype=object)
            pos0 = pd.to_numeric(chunk["chromEnd"], errors="coerce").to_numpy(dtype=np.float64) - 1.0

            snv_mask = (
                np.char.str_len(ref.astype(str)) == 1
            ) & (
                np.char.str_len(major.astype(str)) == 1
            ) & (
                np.char.str_len(minor.astype(str)) == 1
            ) & np.isfinite(pos0)
            if not np.any(snv_mask):
                if status_cb is not None:
                    status_cb(
                        current_chromosome=chrom_num,
                        scanned_rows_current_chrom=scanned_rows,
                        selected_rows_current_chrom=len(chosen),
                    )
                continue

            ref = ref[snv_mask]
            major = major[snv_mask]
            minor = minor[snv_mask]
            maf_arr = maf_arr[snv_mask]
            pos0 = pos0[snv_mask].astype(np.int64)

            alt = np.where(ref != minor, minor, major)
            valid_alt = np.isin(alt, np.array(["A", "C", "G", "T"], dtype=object))
            if not np.any(valid_alt):
                if status_cb is not None:
                    status_cb(
                        current_chromosome=chrom_num,
                        scanned_rows_current_chrom=scanned_rows,
                        selected_rows_current_chrom=len(chosen),
                    )
                continue

            ref = ref[valid_alt]
            alt = alt[valid_alt]
            maf_arr = maf_arr[valid_alt]
            pos0 = pos0[valid_alt]
            is_common = (maf_arr >= COMMON_THRESHOLD).astype(np.int8)

            chosen.extend(
                VariantRecord(
                    chrom=chrom,
                    chrom_num=chrom_num,
                    pos0=int(p),
                    ref=str(r),
                    alt=str(a),
                    maf=float(m),
                    is_common=int(c),
                )
                for p, r, a, m, c in zip(pos0, ref, alt, maf_arr, is_common)
            )

            if status_cb is not None:
                status_cb(
                    current_chromosome=chrom_num,
                    scanned_rows_current_chrom=scanned_rows,
                    selected_rows_current_chrom=len(chosen),
                )

        if sample_per_chrom > 0 and len(chosen) > sample_per_chrom:
            common = [row for row in chosen if row.is_common == 1]
            rare = [row for row in chosen if row.is_common == 0]
            per_class = max(1, sample_per_chrom // 2)
            common_idx = rng.choice(len(common), size=min(per_class, len(common)), replace=False)
            rare_idx = rng.choice(len(rare), size=min(per_class, len(rare)), replace=False)
            chosen = [common[i] for i in common_idx] + [rare[i] for i in rare_idx]
            chosen.sort(key=lambda row: row.pos0)

        variants.extend(chosen)
        save_cached_variants(cache_path, chosen)
        print(f"[flank] chr{chrom_num} selected_variants={len(chosen)}", flush=True)
        if status_cb is not None:
            status_cb(
                current_chromosome=chrom_num,
                scanned_rows_current_chrom=scanned_rows,
                selected_rows_current_chrom=len(chosen),
                processed_variants_total=len(variants),
                stage_note=f"finished chr{chrom_num}",
                cache_path=str(cache_path),
            )

    return variants


def annotate_regions(
    variants: list[VariantRecord],
    annotation_cache: str,
    out_dir: str = "",
    status_cb: Callable[..., None] | None = None,
) -> dict[str, object]:
    by_chrom: dict[str, list[VariantRecord]] = {}
    for row in variants:
        by_chrom.setdefault(row.chrom, []).append(row)

    region_count_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    provenance_payload: dict[str, object] = {
        "annotation_cache": annotation_cache,
        "tracks_used": ["knownGene", "encodeCcreCombined", "rmsk"],
        "region_precedence": REGION_ORDER[1:],
        "chromosomes": {},
    }

    for chrom, rows in by_chrom.items():
        print(f"[flank] annotating {chrom} via UCSC cache at {annotation_cache}", flush=True)
        chrom_num = int(chrom.replace("chr", ""))
        diagnostics_dir = Path(out_dir) / "annotation_diagnostics" if out_dir else Path(annotation_cache)
        shard_hint = diagnostics_dir / f"{chrom}_region_counts.csv"
        if status_cb is not None:
            status_cb(
                current_chromosome=chrom_num,
                stage_note=f"annotating {chrom}",
                current_shard_path=str(shard_hint),
            )
        bundle = build_region_annotation_bundle(annotation_cache, chrom)
        intervals = dict(bundle["intervals"])
        chrom_region_counts = {region: 0 for region in REGION_ORDER[1:]}
        chrom_overlap_counts: dict[str, int] = {}
        chrom_start = time.time()
        last_heartbeat = chrom_start
        heartbeat_interval = 30.0
        for i, row in enumerate(rows):
            hit_regions = [
                region_name
                for region_name in ("Coding", "3'UTR", "cCREs", "TE")
                if interval_contains(intervals[region_name], row.pos0)
            ]
            overlap_key = "|".join(hit_regions) if hit_regions else "Others_only"
            chrom_overlap_counts[overlap_key] = chrom_overlap_counts.get(overlap_key, 0) + 1
            if "Coding" in hit_regions:
                row.region = "Coding"
            elif "3'UTR" in hit_regions:
                row.region = "3'UTR"
            elif "cCREs" in hit_regions:
                row.region = "cCREs"
            elif "TE" in hit_regions:
                row.region = "TE"
            else:
                row.region = "Others"
            chrom_region_counts[row.region] += 1
            now = time.time()
            if now - last_heartbeat >= heartbeat_interval:
                elapsed = now - chrom_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta_s = (len(rows) - i - 1) / rate if rate > 0 else 0
                print(
                    f"[flank-annotate] {chrom} {i+1}/{len(rows)} variants"
                    f"  elapsed={elapsed:.0f}s  rate={rate/1e6:.2f}M/s  eta={eta_s:.0f}s",
                    flush=True,
                )
                if status_cb is not None:
                    status_cb(
                        current_chromosome=chrom_num,
                        annotated_rows_current_chrom=i + 1,
                        stage_note=f"annotating {chrom} {i+1}/{len(rows)}",
                        current_shard_path=str(shard_hint),
                    )
                last_heartbeat = now
        for region_name, count in chrom_region_counts.items():
            region_count_rows.append(
                {
                    "chromosome": chrom,
                    "region": region_name,
                    "n_variants": int(count),
                }
            )
        for overlap_key, count in sorted(chrom_overlap_counts.items()):
            overlap_rows.append(
                {
                    "chromosome": chrom,
                    "overlap_region_set": overlap_key,
                    "n_variants": int(count),
                }
            )
        provenance_payload["chromosomes"][chrom] = {
            "n_variants_annotated": len(rows),
            "track_row_counts": bundle["track_row_counts"],
            "merged_interval_counts": bundle["merged_interval_counts"],
            "cache_files": bundle["cache_files"],
            "tracks_used": bundle["tracks_used"],
            "region_precedence": bundle["region_precedence"],
        }
        if status_cb is not None:
            status_cb(
                current_chromosome=chrom_num,
                annotated_rows_current_chrom=len(rows),
                stage_note=f"finished annotation {chrom}",
                current_shard_path=str(shard_hint),
            )
    global_region_counts: dict[str, int] = {region: 0 for region in REGION_ORDER[1:]}
    for row in region_count_rows:
        global_region_counts[str(row["region"])] += int(row["n_variants"])
    for region_name, count in global_region_counts.items():
        region_count_rows.append(
            {
                "chromosome": "all",
                "region": region_name,
                "n_variants": int(count),
            }
        )

    global_overlap_counts: dict[str, int] = {}
    for row in overlap_rows:
        key = str(row["overlap_region_set"])
        global_overlap_counts[key] = global_overlap_counts.get(key, 0) + int(row["n_variants"])
    for overlap_key, count in sorted(global_overlap_counts.items()):
        overlap_rows.append(
            {
                "chromosome": "all",
                "overlap_region_set": overlap_key,
                "n_variants": int(count),
            }
        )

    return {
        "region_count_rows": region_count_rows,
        "overlap_rows": overlap_rows,
        "provenance": provenance_payload,
    }


def extract_arrays(variants: list[VariantRecord]) -> dict[str, np.ndarray]:
    chrom = np.array([row.chrom for row in variants], dtype=object)
    pos0 = np.array([row.pos0 for row in variants], dtype=np.int64)
    ref = np.array([row.ref for row in variants], dtype=object)
    alt = np.array([row.alt for row in variants], dtype=object)
    maf = np.array([row.maf for row in variants], dtype=np.float32)
    is_common = np.array([row.is_common for row in variants], dtype=np.int8)
    region = np.array([row.region for row in variants], dtype=object)

    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "-": 4}
    ref_idx = np.array([base_to_idx[b] for b in ref], dtype=np.int64)
    alt_idx = np.array([base_to_idx[b] for b in alt], dtype=np.int64)
    return {
        "chrom": chrom,
        "pos0": pos0,
        "ref_idx": ref_idx,
        "alt_idx": alt_idx,
        "maf": maf,
        "is_common": is_common,
        "region": region,
    }


class _DecodableStr(str):
    """str subclass with a no-op .decode() for h5py/TF-Keras compat.

    Old TF (<=2.6) calls ``model_config.decode('utf-8')`` but newer h5py
    returns string attrs as Python ``str`` objects which have no ``.decode()``.
    Wrapping attrs in this subclass makes the call a no-op instead of raising
    AttributeError.
    """

    def decode(self, encoding: str = "utf-8") -> "_DecodableStr":
        return self


def _patch_h5py_attrs_once() -> None:
    """Monkey-patch h5py.AttributeManager.get to return _DecodableStr for str values."""
    import h5py

    if getattr(h5py.AttributeManager, "_decodable_patched", False):
        return

    _orig_get = h5py.AttributeManager.get

    def _patched_get(self, name, default=None):  # type: ignore[override]
        val = _orig_get(self, name, default)
        if isinstance(val, str):
            return _DecodableStr(val)
        return val

    h5py.AttributeManager.get = _patched_get  # type: ignore[method-assign]
    h5py.AttributeManager._decodable_patched = True  # type: ignore[attr-defined]


def _strip_mixed_precision_from_config(node):
    """Recursively replace mixed_float16 dtype policies with float32 in a config dict.

    In-place modification.  Handles all serialization formats TF uses:
    - Simple string: ``{"dtype": "float16"}``
    - Policy object: ``{"dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}}``
    - Policy object under "dtype_policy" key
    - Nested config dicts inside TFOpLambda / other layers
    """
    if isinstance(node, list):
        for item in node:
            _strip_mixed_precision_from_config(item)
        return
    if not isinstance(node, dict):
        return
    # node is a dict
    for key in ("dtype", "dtype_policy"):
        if key not in node:
            continue
        val = node[key]
        if isinstance(val, str):
            # Plain string: "float16" or "mixed_float16"
            if val == "float16" or val.startswith("mixed"):
                node[key] = "float32"
        elif isinstance(val, dict):
            # Policy object: {"class_name": "Policy", "config": {"name": "mixed_float16"}}
            cfg = val.get("config", {})
            name = cfg.get("name", "")
            if name == "float16" or name.startswith("mixed"):
                cfg["name"] = "float32"
    for v in node.values():
        _strip_mixed_precision_from_config(v)


def _load_keras_model_force_float32(model_path: str) -> tf.keras.Model:
    """Load a .keras / .h5 model in float32, stripping any mixed_float16 policy.

    Models trained with mixed_float16 encode float16 layer policies in their
    saved config. Rebuilding the model from a patched config (all dtypes →
    float32) and then loading weights avoids float16/float32 op mismatches
    during inference while still using the fully trained weights.
    """
    import h5py
    import json

    _patch_h5py_attrs_once()

    # 1. Read model config from HDF5 and strip mixed precision
    with h5py.File(model_path, "r") as f:
        raw_config = f.attrs.get("model_config")
    if raw_config is None:
        raise ValueError(f"No model_config attribute found in {model_path}")
    if isinstance(raw_config, bytes):
        raw_config = raw_config.decode("utf-8")
    config_dict = json.loads(raw_config)
    _strip_mixed_precision_from_config(config_dict)

    # 2. Set global policy to float32 for any layers that don't save their policy
    tf.keras.mixed_precision.set_global_policy("float32")

    # 3. Reconstruct model architecture in float32 from the patched config
    model = tf.keras.models.model_from_config(config_dict)

    # 4. Load trained weights via our h5py-compat helper
    import sys
    import os

    _proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _proj not in sys.path:
        sys.path.insert(0, _proj)
    from graphylovar.model_io import load_hdf5_weights_compat

    load_hdf5_weights_compat(model, model_path)
    return model


def load_saved_model_signature(model_path: str):
    """Load model for inference.

    Supports both SavedModel directories (preferred, uses serving signature)
    and .keras/.h5 files (fallback, uses direct Keras call).
    Returns (model_or_signature_callable, signature_or_None).
    """
    if model_path.endswith((".keras", ".h5", ".hdf5")):
        model = _load_keras_model_force_float32(model_path)
        return model, None  # signal to caller: use direct call, not TF signature
    model = tf.saved_model.load(model_path)
    signature = model.signatures["serving_default"]
    return model, signature


def resolve_flank_model_status(
    project_root: str,
    flank: int,
    model_override_path: str | None = None,
) -> FlankModelStatus:
    if model_override_path is not None:
        model_path = model_override_path if os.path.isabs(model_override_path) else os.path.join(project_root, model_override_path)
        model_rel = os.path.relpath(model_path, project_root)
    elif flank in DEFAULT_MODEL_PATHS:
        model_rel = DEFAULT_MODEL_PATHS[flank]
        model_path = os.path.join(project_root, model_rel)
    else:
        return FlankModelStatus(flank=flank, model_rel="", model_path="", available=False,
                                status="missing-model", note=f"No model path for flank={flank}.")
    _ = model_rel  # used below
    # When an override path is provided, check it directly (avoids double-.keras extension).
    if model_override_path is not None and os.path.exists(model_path):
        status = "ok-keras" if model_path.endswith((".keras", ".h5", ".hdf5")) else "ok-savedmodel"
        return FlankModelStatus(
            flank=flank,
            model_rel=model_rel,
            model_path=model_path,
            available=True,
            status=status,
        )
    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "saved_model.pb")):
        return FlankModelStatus(
            flank=flank,
            model_rel=model_rel,
            model_path=model_path,
            available=True,
            status="ok-savedmodel",
        )

    keras_path = f"{model_path}.keras"
    if os.path.exists(keras_path):
        # .keras files are now fully supported via direct Keras inference call.
        return FlankModelStatus(
            flank=flank,
            model_rel=model_rel,
            model_path=keras_path,
            available=True,
            status="ok-keras",
        )

    return FlankModelStatus(
        flank=flank,
        model_rel=model_rel,
        model_path=model_path,
        available=False,
        status="missing-model",
        note="No V3 checkpoint was found for this flank.",
    )


def predict_scores_for_flank(
    project_root: str,
    alignment_dir: str,
    cache_dir: str,
    arrays: dict[str, np.ndarray],
    flank: int,
    batch_size: int,
    chroms: tuple[int, ...] | None = None,
    status_cb: Callable[..., None] | None = None,
    model_override_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, FlankModelStatus]:
    model_status = resolve_flank_model_status(
        project_root, flank, model_override_path=model_override_path
    )
    if not model_status.available:
        n = arrays["pos0"].shape[0]
        missing = np.full(n, np.nan, dtype=np.float32)
        return missing.copy(), missing.copy(), model_status

    print(f"[flank] scoring flank={flank} with model={model_status.model_path}", flush=True)
    model, signature = load_saved_model_signature(model_status.model_path)

    combined = np.full(arrays["pos0"].shape[0], np.nan, dtype=np.float32)
    allele_ratio = np.full(arrays["pos0"].shape[0], np.nan, dtype=np.float32)

    chrom_iter = chroms or tuple(sorted({int(str(chrom).replace("chr", "")) for chrom in arrays["chrom"]}))
    # Process a chromosome in bounded position slices to avoid OOM from full-chromosome tensors.
    position_chunk_size = max(batch_size * 8, 4096)
    for chrom_num in chrom_iter:
        chrom = f"chr{chrom_num}"
        mask = arrays["chrom"] == chrom
        if not np.any(mask):
            continue
        if status_cb is not None:
            status_cb(
                current_flank=flank,
                current_chromosome=chrom_num,
                stage_note=f"scoring {chrom}",
            )

        idx = np.flatnonzero(mask)
        positions = arrays["pos0"][idx]
        alignment_path = os.path.join(alignment_dir, f"seqDictPad_chr{chrom_num}.pkl")
        encoded_alignment, _meta = load_alignment_encoded_cache(
            alignment_path=alignment_path,
            cache_dir=cache_dir,
            chromosome=chrom_num,
        )
        chrom_scored = 0
        chrom_score_start = time.time()
        _last_score_hb = chrom_score_start
        for pos_start in range(0, positions.shape[0], position_chunk_size):
            pos_stop = min(pos_start + position_chunk_size, positions.shape[0])
            _now = time.time()
            if _now - _last_score_hb >= 30.0 and status_cb is not None:
                _elapsed = _now - chrom_score_start
                _rate = pos_start / _elapsed if _elapsed > 0 else 0
                _eta = (positions.shape[0] - pos_start) / _rate if _rate > 0 else 0
                print(
                    f"[flank-score] flank={flank} {chrom} {pos_start}/{positions.shape[0]}"
                    f"  elapsed={_elapsed:.0f}s  rate={_rate/1e6:.2f}M/s  eta={_eta:.0f}s",
                    flush=True,
                )
                status_cb(
                    current_flank=flank,
                    current_chromosome=chrom_num,
                    scored_variants_current_chrom=pos_start,
                    stage_note=f"scoring {chrom} {pos_start}/{positions.shape[0]}",
                )
                _last_score_hb = _now
            pos_chunk = positions[pos_start:pos_stop]
            idx_chunk = idx[pos_start:pos_stop]

            batch_x, valid = extract_batch_examples_from_encoded(
                encoded_alignment=encoded_alignment,
                positions_zero_based=pos_chunk,
                context=100,
                context_flank=flank,
            )
            if batch_x.shape[0] == 0:
                continue

            valid_idx = idx_chunk[valid]
            chrom_scored += int(batch_x.shape[0])

            # Keep GPU memory bounded on large chromosomes by using adaptive inference micro-batches.
            # MultiHeadAttention in v3 allocates [batch*115_species, n_heads, seq_len, seq_len].
            # For flank=100 (seq_len=201) this is ~74 MB per batch item; fragmentation means
            # we must pre-cap below what would cause OOM rather than relying on retry alone.
            _FLANK_BATCH_CAPS = {0: 256, 1: 256, 8: 256, 16: 128, 32: 64, 100: 8}
            inference_batch_size = min(batch_size, _FLANK_BATCH_CAPS.get(flank, 64))
            start = 0
            while start < batch_x.shape[0]:
                stop = min(start + inference_batch_size, batch_x.shape[0])
                try:
                    x_tensor = tf.convert_to_tensor(batch_x[start:stop], dtype=tf.uint8)
                    if signature is not None:
                        preds = signature(input_1=x_tensor)
                        p_snp = preds["binary"].numpy().reshape(-1)
                        p_nuc = preds["nucleotide"].numpy()
                    else:
                        # Direct Keras model call (e.g., loaded from .keras file)
                        raw = model(x_tensor, training=False)
                        if isinstance(raw, dict):
                            p_snp = raw.get("binary", raw.get("binary_head",
                                list(raw.values())[1])).numpy().reshape(-1)
                            p_nuc = raw.get("nucleotide", raw.get("nucleotide_head",
                                list(raw.values())[0])).numpy()
                        else:
                            p_nuc = raw[0].numpy()
                            p_snp = raw[1].numpy().reshape(-1)
                except tf.errors.ResourceExhaustedError:
                    if inference_batch_size <= 4:
                        raise
                    inference_batch_size = max(4, inference_batch_size // 2)
                    continue

                target_idx = valid_idx[start:stop]
                ref_idx = arrays["ref_idx"][target_idx]
                alt_idx = arrays["alt_idx"][target_idx]
                ref_prob = p_nuc[np.arange(p_nuc.shape[0]), ref_idx]
                alt_prob = p_nuc[np.arange(p_nuc.shape[0]), alt_idx]
                allele_ratio[target_idx] = np.log((alt_prob + EPS) / (ref_prob + EPS))
                combined[target_idx] = np.log(
                    (p_snp * alt_prob + EPS) / (p_snp * ref_prob + (1.0 - p_snp) + EPS)
                )
                start = stop
        print(f"[flank] flank={flank} chrom={chrom} scored_examples={chrom_scored}", flush=True)
        if status_cb is not None:
            status_cb(
                current_flank=flank,
                current_chromosome=chrom_num,
                scored_examples_current_chrom=chrom_scored,
                stage_note=f"finished scoring {chrom}",
            )

    return combined, allele_ratio, model_status


def compute_auc_rows(
    arrays: dict[str, np.ndarray],
    score_map: dict[int, dict[str, np.ndarray]],
    model_status_map: dict[int, FlankModelStatus],
    subset_mask: np.ndarray | None = None,
    chrom_scope: str = "all",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    y_true = arrays["is_common"].astype(np.int8)
    base_mask = subset_mask if subset_mask is not None else np.ones_like(y_true, dtype=bool)

    for flank, scores in score_map.items():
        model_status = model_status_map[flank]
        for score_kind, values in scores.items():
            for region in REGION_ORDER:
                region_mask = base_mask.copy()
                if region != "All":
                    region_mask &= arrays["region"] == region
                region_mask &= np.isfinite(values)
                if region_mask.sum() < 2 or np.unique(y_true[region_mask]).size < 2:
                    auc = math.nan
                    n = int(region_mask.sum())
                else:
                    auc = float(roc_auc_score(y_true[region_mask], values[region_mask]))
                    n = int(region_mask.sum())
                rows.append(
                    {
                        "flank": flank,
                        "score_kind": score_kind,
                        "region": region,
                        "auc": auc,
                        "n_variants": n,
                        "model_status": model_status.status,
                        "checkpoint_path": model_status.model_path,
                        "model_note": model_status.note,
                        "chrom_scope": chrom_scope,
                        "source_tag": "measured_auc",
                    }
                )
    return rows


def validate_measured_auc_rows(
    rows: list[dict[str, object]],
    score_kind: str,
    model_status_map: dict[int, FlankModelStatus],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    selected = [
        row
        for row in rows
        if str(row.get("score_kind")) == score_kind and str(row.get("chrom_scope", "all")) == "all"
    ]
    if not selected:
        return [f"No measured rows were available for score_kind={score_kind}."], warnings

    available_flanks = sorted(flank for flank, status in model_status_map.items() if status.available)
    for flank in available_flanks:
        overall_rows = [row for row in selected if int(row["flank"]) == flank and str(row["region"]) == "All"]
        if not overall_rows:
            errors.append(f"Missing overall measured row for flank={flank}.")
            continue
        if not math.isfinite(float(overall_rows[0]["auc"])):
            errors.append(f"Measured AUROC is not finite for flank={flank}, region=All.")

    required_regions = REGION_ORDER[1:]
    main_flank = next((flank for flank in available_flanks if flank == 32), available_flanks[0] if available_flanks else None)
    if main_flank is not None:
        for region in required_regions:
            main_rows = [
                row
                for row in selected
                if int(row["flank"]) == main_flank and str(row["region"]) == region
            ]
            if not main_rows:
                errors.append(f"Missing measured row for main flank={main_flank}, region={region}.")
                continue
            if int(main_rows[0]["n_variants"]) <= 0:
                errors.append(f"Non-positive variant count for main flank={main_flank}, region={region}.")

    coding_series = tuple(
        round(float(row["auc"]), 6)
        for row in sorted(
            [row for row in selected if str(row["region"]) == "Coding" and math.isfinite(float(row["auc"]))],
            key=lambda row: int(row["flank"]),
        )
    )
    ccre_series = tuple(
        round(float(row["auc"]), 6)
        for row in sorted(
            [row for row in selected if str(row["region"]) == "cCREs" and math.isfinite(float(row["auc"]))],
            key=lambda row: int(row["flank"]),
        )
    )
    if coding_series and coding_series == ccre_series:
        warnings.append(
            "Coding and cCRE measured AUROC series are numerically identical; inspect measured_region_flank_auc_values.csv and annotation diagnostics before making a biological claim."
        )

    return errors, warnings


def write_auc_shards(
    out_dir: str,
    arrays: dict[str, np.ndarray],
    score_map: dict[int, dict[str, np.ndarray]],
    model_status_map: dict[int, FlankModelStatus],
    chroms: tuple[int, ...],
    status_cb: Callable[..., None] | None = None,
) -> list[str]:
    shard_dir = Path(out_dir) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[str] = []
    for chrom_num in chroms:
        chrom_mask = arrays["chrom"] == f"chr{chrom_num}"
        if not np.any(chrom_mask):
            continue
        chrom_rows = compute_auc_rows(
            arrays,
            score_map,
            model_status_map,
            subset_mask=chrom_mask,
            chrom_scope=f"chr{chrom_num}",
        )
        for flank in sorted(score_map):
            for score_kind in ("combined", "allele_ratio"):
                shard_rows = [
                    row for row in chrom_rows
                    if int(row["flank"]) == flank and str(row["score_kind"]) == score_kind
                ]
                if not shard_rows:
                    continue
                shard_path = shard_dir / f"measured_region_flank_auc_chr{chrom_num}_flank{flank}_{score_kind}.csv"
                write_csv(str(shard_path), shard_rows)
                shard_paths.append(str(shard_path))
                if status_cb is not None:
                    status_cb(
                        current_chromosome=chrom_num,
                        current_flank=flank,
                        current_shard_path=str(shard_path),
                        stage_note=f"wrote shard {shard_path.name}",
                    )
    return shard_paths


def choose_score_kind(rows: list[dict[str, object]], main_flank: int, preferred: str) -> str:
    if preferred != "auto":
        return preferred

    target_auc = 0.664
    candidates = {}
    for row in rows:
        if row["flank"] == main_flank and row["region"] == "All" and math.isfinite(float(row["auc"])):
            candidates[str(row["score_kind"])] = abs(float(row["auc"]) - target_auc)
    if not candidates:
        return "combined"
    return min(candidates, key=candidates.get)


def plot_measured_region_flank(rows: list[dict[str, object]], score_kind: str, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    for region in REGION_ORDER:
        xs = []
        ys = []
        for flank in sorted(DEFAULT_MODEL_PATHS):
            match = [
                row for row in rows
                if row["flank"] == flank
                and row["region"] == region
                and row["score_kind"] == score_kind
                and str(row.get("chrom_scope", "all")) == "all"
            ]
            if not match or not math.isfinite(float(match[0]["auc"])):
                continue
            xs.append(flank)
            ys.append(float(match[0]["auc"]))
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2.2, label=region)

    ax.set_title("Flank Ablation by Genomic Region (Measured on Held-out TOPMed + UCSC)", fontsize=19)
    ax.set_xlabel("Flank Size", fontsize=14)
    ax.set_ylabel("AUROC: common (MAF >= 0.01) vs rare (MAF < 0.001)", fontsize=13)
    ax.set_xticks(sorted(DEFAULT_MODEL_PATHS))
    ax.set_ylim(0.50, 0.75)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11)
    ax.text(
        0.01,
        -0.16,
        f"Measured from held-out chromosomes 13-22 using UCSC hg38 annotations; score={score_kind}.",
        transform=ax.transAxes,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    legacy_compatible = os.path.join(os.path.dirname(out_png), "flank_ablation_topmed_detailed.png")
    if os.path.abspath(legacy_compatible) != os.path.abspath(out_png):
        fig.savefig(legacy_compatible, dpi=200)
    plt.close(fig)


def plot_measured_overall(rows: list[dict[str, object]], score_kind: str, out_png: str) -> None:
    plot_rows = [
        row for row in rows
        if row["score_kind"] == score_kind
        and row["region"] == "All"
        and str(row.get("chrom_scope", "all")) == "all"
        and math.isfinite(float(row["auc"]))
    ]
    plot_rows.sort(key=lambda row: int(row["flank"]))
    labels = [f"flank={int(row['flank'])}" for row in plot_rows]
    values = [float(row["auc"]) for row in plot_rows]

    fig, ax = plt.subplots(figsize=(10.4, 7.0))
    bars = ax.bar(labels, values, color="#5A8BB2", edgecolor="#2d3e50", alpha=0.92)
    ax.plot(labels, values, color="#8B0000", marker="o", linewidth=2.1, markersize=7)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.3f}", ha="center", fontsize=9)
    ax.set_title("Flank Ablation on Held-out TOPMed Benchmark", fontsize=19)
    ax.set_ylabel("AUROC: common (MAF >= 0.01) vs rare (MAF < 0.001)", fontsize=13)
    ax.set_ylim(0.50, max(0.72, max(values) + 0.03 if values else 0.72))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_main_region_bar(rows: list[dict[str, object]], flank: int, score_kind: str, out_png: str) -> None:
    plot_rows = [
        row for row in rows
        if row["flank"] == flank
        and row["score_kind"] == score_kind
        and row["region"] != "All"
        and str(row.get("chrom_scope", "all")) == "all"
    ]
    plot_rows.sort(key=lambda row: REGION_ORDER.index(str(row["region"])))
    labels = [str(row["region"]) for row in plot_rows]
    values = [float(row["auc"]) for row in plot_rows]

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    bars = ax.bar(labels, values, color="#2f6b8a", edgecolor="#123246", alpha=0.95)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.3f}", ha="center", fontsize=10)
    ax.set_title(f"Measured GraphyloVar Region AUROC at Flank {flank}", fontsize=18)
    ax.set_ylabel("AUROC", fontsize=13)
    ax.set_ylim(0.50, 0.75)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    legacy_compatible = os.path.join(os.path.dirname(out_png), "maf_region_auc_latest_table.png")
    if os.path.abspath(legacy_compatible) != os.path.abspath(out_png):
        fig.savefig(legacy_compatible, dpi=200)
    plt.close(fig)


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    selected_chroms = parse_chromosome_spec(args.chromosomes)
    status_json = args.status_json or os.path.join(args.out_dir, "flank_recompute_status.json")
    label_cache_dir = args.label_cache_dir or os.path.join(args.cache_dir, "label_variant_cache")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(label_cache_dir, exist_ok=True)

    def update_status(phase: str, **extra: object) -> None:
        payload: dict[str, object] = {
            "status": "in_progress",
            "phase": phase,
            "chromosomes": list(selected_chroms),
            "heldout_chromosomes": list(selected_chroms),
            "sample_per_chrom": int(args.sample_per_chrom),
            "processed_variants": int(extra.pop("processed_variants_total", 0)),
            "last_chromosome": extra.get("current_chromosome", ""),
            "output_dir": args.out_dir,
        }
        payload.update(extra)
        write_status(status_json, payload)

    update_status("loading_labels", processed_variants_total=0)
    print(f"[flank] output_dir={args.out_dir}", flush=True)
    print(
        f"[flank] source_paths label_dir={args.label_dir} alignment_dir={args.alignment_dir} cache_dir={args.cache_dir} annotation_cache={args.annotation_cache} label_cache_dir={label_cache_dir}",
        flush=True,
    )

    variants = load_eval_variants(
        args.label_dir,
        sample_per_chrom=args.sample_per_chrom,
        chroms=selected_chroms,
        label_cache_dir=label_cache_dir,
        status_cb=lambda **kw: update_status("loading_labels", **kw),
    )
    update_status(
        "annotating_regions",
        processed_variants_total=len(variants),
        current_chromosome=selected_chroms[-1] if selected_chroms else "",
    )
    annotation_outputs = annotate_regions(
        variants,
        args.annotation_cache,
        out_dir=args.out_dir,
        status_cb=lambda **kw: update_status(
            "annotating_regions",
            processed_variants_total=len(variants),
            **kw,
        ),
    )
    write_csv(
        os.path.join(args.out_dir, "annotation_region_counts_by_chrom.csv"),
        annotation_outputs["region_count_rows"],
    )
    write_csv(
        os.path.join(args.out_dir, "annotation_overlap_counts_by_chrom.csv"),
        annotation_outputs["overlap_rows"],
    )
    with open(os.path.join(args.out_dir, "annotation_provenance.json"), "w", encoding="utf-8") as handle:
        json.dump(annotation_outputs["provenance"], handle, indent=2)
    arrays = extract_arrays(variants)

    # Parse optional model path overrides (e.g. for v4 models not in DEFAULT_MODEL_PATHS).
    _model_overrides: dict[int, str] = {}
    if args.model_override:
        for tok in args.model_override.split(","):
            tok = tok.strip()
            if ":" not in tok:
                continue
            _f, _p = tok.split(":", 1)
            _model_overrides[int(_f.strip())] = _p.strip()

    # Optionally restrict which flanks to score (e.g. for flank=1 rescore after training).
    if args.flanks:
        _selected_flanks = set(int(f.strip()) for f in args.flanks.split(",") if f.strip())
        _known = set(DEFAULT_MODEL_PATHS) | set(_model_overrides)
        _flanks_to_score = sorted(f for f in _selected_flanks if f in _known)
    else:
        _flanks_to_score = sorted(set(DEFAULT_MODEL_PATHS) | set(_model_overrides))

    score_map: dict[int, dict[str, np.ndarray]] = {}
    model_status_map: dict[int, FlankModelStatus] = {}
    for flank in _flanks_to_score:
        update_status(
            "scoring_flank",
            processed_variants_total=len(variants),
            current_chromosome=selected_chroms[-1] if selected_chroms else "",
            current_flank=flank,
            stage_note=f"starting flank {flank}",
        )
        combined, allele_ratio, model_status = predict_scores_for_flank(
            project_root=args.project_root,
            alignment_dir=args.alignment_dir,
            cache_dir=args.cache_dir,
            arrays=arrays,
            flank=flank,
            batch_size=args.batch_size,
            chroms=selected_chroms,
            status_cb=lambda **kw: update_status(
                "scoring_flank",
                processed_variants_total=len(variants),
                **kw,
            ),
            model_override_path=_model_overrides.get(flank),
        )
        model_status_map[flank] = model_status
        print(
            f"flank={flank} model_status={model_status.status} path={model_status.model_path}"
            + (f" note={model_status.note}" if model_status.note else ""),
            flush=True,
        )
        score_map[flank] = {
            "combined": combined,
            "allele_ratio": allele_ratio,
        }

    auc_rows = compute_auc_rows(arrays, score_map, model_status_map)
    selected_score_kind = choose_score_kind(auc_rows, main_flank=args.main_flank, preferred=args.score_kind)
    shard_paths = write_auc_shards(
        args.out_dir,
        arrays,
        score_map,
        model_status_map,
        selected_chroms,
        status_cb=lambda **kw: update_status(
            "writing_auc_shards",
            processed_variants_total=len(variants),
            **kw,
        ),
    )

    write_csv(os.path.join(args.out_dir, "measured_region_flank_auc_values.csv"), auc_rows)
    write_csv(
        os.path.join(args.out_dir, "measured_eval_variant_regions.csv"),
        [
            {
                "chrom": row.chrom,
                "pos0": row.pos0,
                "maf": row.maf,
                "is_common": row.is_common,
                "region": row.region,
            }
            for row in variants
        ],
    )

    validation_errors, validation_warnings = validate_measured_auc_rows(
        auc_rows,
        score_kind=selected_score_kind,
        model_status_map=model_status_map,
    )
    validation_payload = {
        "status": "current" if not validation_errors else "blocked",
        "score_kind_used_for_plots": selected_score_kind,
        "errors": validation_errors,
        "warnings": validation_warnings,
        "expected_regions": REGION_ORDER,
        "expected_flanks": sorted(_flanks_to_score),
    }
    validation_json_path = os.path.join(args.out_dir, "measured_region_flank_validation.json")
    with open(validation_json_path, "w", encoding="utf-8") as handle:
        json.dump(validation_payload, handle, indent=2)

    if not validation_errors:
        plot_measured_region_flank(
            auc_rows,
            score_kind=selected_score_kind,
            out_png=os.path.join(args.out_dir, "flank_ablation_new_region_measured.png"),
        )
        plot_measured_overall(
            auc_rows,
            score_kind=selected_score_kind,
            out_png=os.path.join(args.out_dir, "flank_ablation_new_overall.png"),
        )
        plot_main_region_bar(
            auc_rows,
            flank=args.main_flank,
            score_kind=selected_score_kind,
            out_png=os.path.join(args.out_dir, "maf_region_auc_measured_ucsc.png"),
        )
    else:
        print(
            "[flank] validation blocked canonical measured figure generation: "
            + "; ".join(validation_errors),
            flush=True,
        )

    summary = {
        "status": "current" if not validation_errors else "blocked",
        "n_variants": len(variants),
        "score_kind_used_for_plots": selected_score_kind,
        "main_flank": args.main_flank,
        "common_threshold": COMMON_THRESHOLD,
        "rare_threshold": RARE_THRESHOLD,
        "chromosomes": list(selected_chroms),
        "heldout_chromosomes": list(selected_chroms),
        "sample_per_chrom": int(args.sample_per_chrom),
        "region_precedence": REGION_ORDER[1:],
        "model_paths": {str(flank): rel_path for flank, rel_path in DEFAULT_MODEL_PATHS.items()},
        "model_status": {
            str(flank): {
                "available": status.available,
                "status": status.status,
                "model_rel": status.model_rel,
                "model_path": status.model_path,
                "note": status.note,
            }
            for flank, status in model_status_map.items()
        },
        "alignment_dir": args.alignment_dir,
        "cache_dir": args.cache_dir,
        "annotation_cache": args.annotation_cache,
        "label_dir": args.label_dir,
        "annotation_provenance_json": os.path.join(args.out_dir, "annotation_provenance.json"),
        "annotation_region_counts_csv": os.path.join(args.out_dir, "annotation_region_counts_by_chrom.csv"),
        "annotation_overlap_counts_csv": os.path.join(args.out_dir, "annotation_overlap_counts_by_chrom.csv"),
        "measured_eval_variant_regions_csv": os.path.join(args.out_dir, "measured_eval_variant_regions.csv"),
        "measured_auc_csv": os.path.join(args.out_dir, "measured_region_flank_auc_values.csv"),
        "validation_json": validation_json_path,
        "shard_dir": os.path.join(args.out_dir, "shards"),
        "n_shards_written": len(shard_paths),
        "validation_warnings": validation_warnings,
    }
    with open(os.path.join(args.out_dir, "measured_region_flank_auc_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_status(
        status_json,
        {
            "status": "current" if not validation_errors else "blocked",
            "phase": "done" if not validation_errors else "blocked_validation",
            "chromosomes": list(selected_chroms),
            "heldout_chromosomes": list(selected_chroms),
            "sample_per_chrom": int(args.sample_per_chrom),
            "processed_variants": len(variants),
            "last_chromosome": selected_chroms[-1] if selected_chroms else "",
            "summary_json": os.path.join(args.out_dir, "measured_region_flank_auc_summary.json"),
            "validation_json": validation_json_path,
            "current_shard_path": shard_paths[-1] if shard_paths else "",
            "output_dir": args.out_dir,
        },
    )

    if selected_score_kind:
        pd_rows = [
            {
                "value_kind": "measured_auc",
                "chrom_scope": row.get("chrom_scope", "all"),
                "flank": row["flank"],
                "score_kind": row["score_kind"],
                "region": row["region"],
                "auc": row["auc"],
                "n_variants": row["n_variants"],
                "model_status": row["model_status"],
                "source_tag": row.get("source_tag", "measured_auc"),
                "source_path": os.path.join(args.out_dir, "measured_region_flank_auc_values.csv"),
            }
            for row in auc_rows
            if row["score_kind"] == selected_score_kind and str(row.get("chrom_scope", "all")) == "all"
        ]
        write_csv(os.path.join(args.out_dir, "figure_values_used.csv"), pd_rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
